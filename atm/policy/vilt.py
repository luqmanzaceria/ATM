import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as T

from einops import rearrange, repeat

from atm.model import *
from atm.model.track_patch_embed import TrackPatchEmbed
from atm.policy.vilt_modules.transformer_modules import *
from atm.policy.vilt_modules.rgb_modules import *
from atm.policy.vilt_modules.language_modules import *
from atm.policy.vilt_modules.extra_state_modules import ExtraModalityTokens
from atm.policy.vilt_modules.policy_head import *
from atm.utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video

###############################################################################
#
# A ViLT Policy
#
###############################################################################


class BCViLTPolicy(nn.Module):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, obs_cfg, img_encoder_cfg, language_encoder_cfg, extra_state_encoder_cfg, track_cfg,
                 spatial_transformer_cfg, temporal_transformer_cfg,
                 policy_head_cfg, load_path=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.spatial_transformer_use_text = spatial_transformer_cfg.get('use_language_token', True)
        print(f"spatial_transformer_use_text: {self.spatial_transformer_use_text}")
    
        self._process_obs_shapes(**obs_cfg)

        # 1. encode image
        self._setup_image_encoder(**img_encoder_cfg)

        # 2. encode language (spatial)
        self.language_encoder_spatial = self._setup_language_encoder(output_size=self.spatial_embed_size, **language_encoder_cfg)

        # 3. Track Transformer module
        self._setup_track(**track_cfg)

        # 3. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 4. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg)

        # 6. encode language (temporal), this will also act as the TEMPORAL_TOKEN, i.e., CLS token for action prediction
        self.language_encoder_temporal = self._setup_language_encoder(output_size=self.temporal_embed_size, **language_encoder_cfg)

        # 7. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)

        # 8. define policy head
        self._setup_policy_head(**policy_head_cfg)

        if load_path is not None:
            self.load(load_path)
            self.track.load(f"{track_cfg.track_fn}/model_best.ckpt")
            
        # self.additional_features_projection = nn.Linear(6144, 128)
        self.additional_features_projection = nn.Linear(384, 128)

        self.track_patch_pos_embed = nn.Parameter(torch.randn(1, self.num_track_patches, self.spatial_embed_size-self.track_id_embed_dim))

    def _process_obs_shapes(self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.policy_num_track_ts = obs_shapes["tracks"][0]
        self.policy_num_track_ids = obs_shapes["tracks"][1]
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.max_seq_len = max_seq_len
        # define buffer queue for encoded latent features
        self.latent_queue = deque(maxlen=max_seq_len)
        self.track_obs_queue = deque(maxlen=max_seq_len)

    def _setup_image_encoder(self, network_name, patch_size, embed_size, no_patch_embed_bias):
        self.spatial_embed_size = embed_size
        self.image_encoders = []
        for _ in range(self.num_views):
            input_shape = self.obs_shapes["rgb"]
            self.image_encoders.append(eval(network_name)(input_shape=input_shape, patch_size=patch_size,
                                                          embed_size=self.spatial_embed_size,
                                                          no_patch_embed_bias=no_patch_embed_bias))
        self.image_encoders = nn.ModuleList([encoder.to(self.device) for encoder in self.image_encoders])

        self.img_num_patches = sum([x.num_patches for x in self.image_encoders])

    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _setup_track(self, track_fn, policy_track_patch_size=None, use_zero_track=False):
        """
        track_fn: path to the track model
        policy_track_patch_size: The patch size of TrackPatchEmbedding in the policy, if None, it will be assigned the same patch size as TrackTransformer by default
        use_zero_track: whether to zero out the tracks (ie use only the image)
        """
        track_cfg = OmegaConf.load(f"{track_fn}/config.yaml")
        self.use_zero_track = use_zero_track

        track_cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
        track_cls = eval(track_cfg.model_name)
        self.track = track_cls(**track_cfg.model_cfg)
        # freeze
        self.track.eval()
        for param in self.track.parameters():
            param.requires_grad = False

        self.num_track_ids = self.track.num_track_ids
        self.num_track_ts = self.track.num_track_ts
        self.policy_track_patch_size = self.track.track_patch_size if policy_track_patch_size is None else policy_track_patch_size


        self.track_proj_encoder = TrackPatchEmbed(
            num_track_ts=self.policy_num_track_ts,
            num_track_ids=self.num_track_ids,
            patch_size=self.policy_track_patch_size,
            in_dim=2 + self.num_views,  # X, Y, one-hot view embedding
            embed_dim=self.spatial_embed_size)

        self.track = self.track.to(self.device)
        self.track_proj_encoder = self.track_proj_encoder.to(self.device)


        self.track_id_embed_dim = 16
        self.num_track_patches_per_view = self.track_proj_encoder.num_patches_per_track
        self.num_track_patches = self.num_track_patches_per_view * self.num_views

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(torch.randn(1, 1, self.spatial_embed_size))  # SPATIAL_TOKEN
        img_patch_pos_embed = nn.Parameter(torch.randn(1, self.img_num_patches, self.spatial_embed_size))
        track_patch_pos_embed = nn.Parameter(torch.randn(1, self.num_track_patches, self.spatial_embed_size-self.track_id_embed_dim))
        modality_embed = nn.Parameter(
            torch.randn(1, len(self.image_encoders) + self.num_views + 1, self.spatial_embed_size)
        )  # IMG_PATCH_TOKENS + TRACK_PATCH_TOKENS + SENTENCE_TOKEN
    
        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("img_patch_pos_embed", img_patch_pos_embed)
        self.register_parameter("track_patch_pos_embed", track_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)
    
        # for selecting modality embed
        modality_idx = []
        for i, encoder in enumerate(self.image_encoders):
            modality_idx += [i] * encoder.num_patches
        for i in range(self.num_views):
            modality_idx += [len(self.image_encoders) + i] * self.num_track_ids * self.num_track_patches_per_view
        
        use_language_token = getattr(self, 'spatial_transformer_use_text', True)
        if use_language_token:
            modality_idx += [len(self.image_encoders) + self.num_views]  # for sentence embedding
        
        self.modality_idx = torch.LongTensor(modality_idx).to(self.device)
        
        # Move parameters to the correct device
        self.spatial_token.data = self.spatial_token.data.to(self.device)
        self.img_patch_pos_embed.data = self.img_patch_pos_embed.data.to(self.device)
        self.track_patch_pos_embed.data = self.track_patch_pos_embed.data.to(self.device)
        self.modality_embed.data = self.modality_embed.data.to(self.device)

    def _setup_extra_state_encoder(self, **extra_state_encoder_cfg):
        if len(self.extra_state_keys) == 0:
            return None
        else:
            return ExtraModalityTokens(
                use_joint=("joint_states" in self.extra_state_keys),
                use_gripper=("gripper_states" in self.extra_state_keys),
                use_ee=("ee_states" in self.extra_state_keys),
                **extra_state_encoder_cfg
            )

    def _setup_spatial_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout,
                                   spatial_downsample, spatial_downsample_embed_size, use_language_token=True):
        self.spatial_transformer = TransformerDecoder(
            input_size=self.spatial_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )

        if spatial_downsample:
            self.temporal_embed_size = spatial_downsample_embed_size
            self.spatial_downsample = nn.Linear(self.spatial_embed_size, self.temporal_embed_size)
        else:
            self.temporal_embed_size = self.spatial_embed_size
            self.spatial_downsample = nn.Identity()

        self.spatial_transformer_use_text = use_language_token

    def _setup_temporal_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout, use_language_token=True):
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(input_size=self.temporal_embed_size)

        self.temporal_transformer = TransformerDecoder(
            input_size=self.temporal_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,)
        self.temporal_transformer_use_text = use_language_token

        action_cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_embed_size))
        nn.init.normal_(action_cls_token, std=1e-6)
        self.register_parameter("action_cls_token", action_cls_token)

    # def _setup_policy_head(self, network_name, **policy_head_kwargs):
    #     policy_head_kwargs["input_size"] \
    #         = self.temporal_embed_size + self.num_views * self.policy_num_track_ts * self.policy_num_track_ids * 2

    #     action_shape = policy_head_kwargs["output_size"]
    #     self.act_shape = action_shape
    #     self.out_shape = np.prod(action_shape)
    #     policy_head_kwargs["output_size"] = self.out_shape
    #     self.policy_head = eval(network_name)(**policy_head_kwargs)

    # _setup_policy_head with intermediate outputs
    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        # The total input size is 2112 based on the shape of policy_input
        total_input_size = 2112
        
        policy_head_kwargs["input_size"] = total_input_size
        
        action_shape = policy_head_kwargs["output_size"]
        self.act_shape = action_shape
        self.out_shape = np.prod(action_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        self.policy_head = eval(network_name)(**policy_head_kwargs)
    
        print(f"Policy head input size: {total_input_size}")
        print(f"Policy head output size: {self.out_shape}")
            
    @torch.no_grad()
    def preprocess(self, obs, track, action):
        """
        Preprocess observations, according to an observation dictionary.
        Return the feature and state.
        """
        b, v, t, c, h, w = obs.shape

        action = action.reshape(b, t, self.out_shape)

        obs = self._preprocess_rgb(obs)

        return obs, track, action

    @torch.no_grad()
    def _preprocess_rgb(self, rgb):
        rgb = self.img_normalizer(rgb / 255.)
        return rgb

    def _get_view_one_hot(self, tr):
        """ tr: b, v, t, tl, n, d -> (b, v, t), tl n, d + v"""
        b, v, t, tl, n, d = tr.shape
        tr = rearrange(tr, "b v t tl n d -> (b t tl n) v d")
        one_hot = torch.eye(v, device=tr.device, dtype=tr.dtype)[None, :, :].repeat(tr.shape[0], 1, 1)
        tr_view = torch.cat([tr, one_hot], dim=-1)  # (b t tl n) v (d + v)
        tr_view = rearrange(tr_view, "(b t tl n) v c -> b v t tl n c", b=b, v=v, t=t, tl=tl, n=n, c=d + v)
        return tr_view

    # def track_encode(self, track_obs, task_emb):
    #     """
    #     Args:
    #         track_obs: b v t tt_fs c h w
    #         task_emb: b e
    #     Returns: b v t track_len n 2
    #     """
    #     assert self.num_track_ids == 32
    #     b, v, t, *_ = track_obs.shape

    #     if self.use_zero_track:
    #         recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
    #     else:
    #         track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")

    #         grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
    #         grid_sampled_track = repeat(grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=self.num_track_ts)
    #         grid_sampled_track = rearrange(grid_sampled_track, "b v t tl n d -> (b v t) tl n d")

    #         expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
    #         expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
    #         with torch.no_grad():
    #             pred_tr, _ = self.track.reconstruct(track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
    #             recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

    #     recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]  # truncate the track to a shorter one
    #     _recon_tr = recon_tr.clone()  # b v t tl n 2
    #     with torch.no_grad():
    #         tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c

    #     tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
    #     tr = self.track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
    #     tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=self.num_track_ids)  # (b t n) (v patch_num) d

    #     return tr, _recon_tr

    # track_encode with intermediate outputs
    def track_encode(self, track_obs, task_emb):
        assert self.num_track_ids == 32
        b, v, t, *_ = track_obs.shape
    
        if self.use_zero_track:
            recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
            intermediate_outputs = []
        else:
            track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")
            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
    
            # Create a dummy grid since we're not using it
            dummy_grid = torch.zeros((b*v*t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
    
            with torch.no_grad():
                pred_tr, _, intermediate_outputs = self.track.reconstruct(
                    track_obs_to_pred, 
                    dummy_grid,  # Pass the dummy grid
                    expand_task_emb,
                    p_img=0  # Set p_img to 0 or another appropriate value
                )
                recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)
    
        recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]
        _recon_tr = recon_tr.clone()
        with torch.no_grad():
            tr_view = self._get_view_one_hot(recon_tr)
        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)
        tr = rearrange(tr, "(b v t) pn n d -> b t (v n pn) d", b=b, v=v, t=t, n=self.num_track_ids)
    
        if intermediate_outputs:
            additional_features = torch.cat([output.mean(dim=1) for output in intermediate_outputs], dim=-1)
        else:
            additional_features = None
    
        return tr, _recon_tr, additional_features

    # def spatial_encode(self, obs, track_obs, task_emb, extra_states, return_recon=False):
    #     """
    #     Encode the images separately in the videos along the spatial axis.
    #     Args:
    #         obs: b v t c h w
    #         track_obs: b v t tt_fs c h w, (0, 255)
    #         task_emb: b e
    #         extra_states: {k: b t n}
    #     Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
    #     """
    #     # 1. encode image
    #     img_encoded = []
    #     for view_idx in range(self.num_views):
    #         img_encoded.append(
    #             rearrange(
    #                 TensorUtils.time_distributed(
    #                     obs[:, view_idx, ...], self.image_encoders[view_idx]
    #                 ),
    #                 "b t c h w -> b t (h w) c",
    #             )
    #         )  # (b, t, num_patches, c)

    #     img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
    #     img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
    #     B, T = img_encoded.shape[:2]

    #     # 2. encode task_emb
    #     text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
    #     text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c)

    #     # 3. encode track
    #     track_encoded, _recon_track = self.track_encode(track_obs, task_emb)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)
    #     # patch position embedding
    #     tr_feat, tr_id_emb = track_encoded[:, :, :-self.track_id_embed_dim], track_encoded[:, :, -self.track_id_embed_dim:]
    #     tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
    #     # track id embedding
    #     tr_id_emb[:, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :1, -self.track_id_embed_dim:]  # guarantee the permutation invariance
    #     track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
    #     track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, 2*num_track*num_track_patch, c)

    #     # 3. concat img + track + text embs then add modality embeddings
    #     if self.spatial_transformer_use_text:
    #         img_track_text_encoded = torch.cat([img_encoded, track_encoded, text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 1, c)
    #         img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
    #     else:
    #         img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
    #         img_track_text_encoded += self.modality_embed[None, :, self.modality_idx[:-1], :]

    #     # 4. add spatial token
    #     spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
    #     encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)

    #     # 5. pass through transformer
    #     encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
    #     out = self.spatial_transformer(encoded)
    #     out = out[:, 0]  # extract spatial token as summary at o_t
    #     out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

    #     # 6. encode extra states
    #     if self.extra_encoder is None:
    #         extra = None
    #     else:
    #         extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')

    #     # 7. encode language, treat it as action token
    #     text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
    #     text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c')
    #     action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
    #     if self.temporal_transformer_use_text:
    #         out_seq = [action_cls_token, text_encoded_, out]
    #     else:
    #         out_seq = [action_cls_token, out]

    #     if self.extra_encoder is not None:
    #         out_seq.append(extra)
    #     output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

    #     if return_recon:
    #         output = (output, _recon_track)

    #     return output

    # spatial_encode with intermediate_outputs
    def spatial_encode(self, obs, track_obs, task_emb, extra_states, return_recon=False):
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)
    
        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]
    
        text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c)
    
        track_encoded, _recon_track, intermediate_outputs = self.track_encode(track_obs, task_emb)

        if isinstance(intermediate_outputs, torch.Tensor):
            intermediate_outputs = [intermediate_outputs]  # Convert single tensor to list
        
        if intermediate_outputs:
            print(f"Number of intermediate outputs: {len(intermediate_outputs)}")
            print(f"Shape of first intermediate output: {intermediate_outputs[0].shape}")
            
            # Concatenate all intermediate outputs
            additional_features = torch.cat(intermediate_outputs, dim=1)
            print(f"Shape of additional_features after concatenation: {additional_features.shape}")
            
            # Project the features to the desired dimension
            additional_features = self.additional_features_projection(additional_features)
            print(f"Shape of additional_features after projection: {additional_features.shape}")
            
            # Average across the first dimension (320)
            additional_features = additional_features.mean(dim=0)
            print(f"Shape of additional_features after averaging: {additional_features.shape}")
            
            batch_size, time_dim, num_track_ids, _ = track_encoded.shape
            additional_features = additional_features.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, time_dim, num_track_ids, -1)
            print(f"Shape of additional_features after expansion: {additional_features.shape}")
            
            track_encoded = torch.cat([track_encoded, additional_features], dim=-1)

        # if intermediate_outputs:
        #     print(f"Number of intermediate outputs: {len(intermediate_outputs)}")
        #     print(f"Shape of final layer output: {intermediate_outputs[0].shape}")
            
        #     # Get the last layer output
        #     additional_features = intermediate_outputs[0]
        #     print(f"Shape of additional_features before projection: {additional_features.shape}")
            
        #     # Project each token independently
        #     additional_features = self.additional_features_projection(additional_features)
        #     print(f"Shape of additional_features after projection: {additional_features.shape}")
            
        #     # Average across the sequence dimension (dim=1)
        #     additional_features = additional_features.mean(dim=1)
        #     print(f"Shape of additional_features after averaging: {additional_features.shape}")
            
        #     batch_size, time_dim, num_track_ids, *_ = track_encoded.shape
        #     additional_features = additional_features.unsqueeze(1).unsqueeze(2).expand(batch_size, time_dim, num_track_ids, -1)
        #     print(f"Shape of additional_features after expansion: {additional_features.shape}")
            
        #     track_encoded = torch.cat([track_encoded, additional_features], dim=-1)
        
        print(f"Final shape of track_encoded: {track_encoded.shape}")

    
        tr_feat, tr_id_emb = track_encoded[:, :, :, :-self.track_id_embed_dim], track_encoded[:, :, :, -self.track_id_embed_dim:]
        
        print(f"Shape of tr_feat: {tr_feat.shape}")
        print(f"Shape of self.track_patch_pos_embed: {self.track_patch_pos_embed.shape}")
        
        b, t, n, d = tr_feat.shape
        position = torch.arange(n, dtype=torch.float, device=tr_feat.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float, device=tr_feat.device) * (-math.log(10000.0) / d))
        pos_embed = torch.zeros(n, d, device=tr_feat.device)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)
        
        print(f"Shape of new pos_embed: {pos_embed.shape}")
        
        tr_feat += pos_embed
        tr_id_emb[:, :, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :, :1, -self.track_id_embed_dim:]
        track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
        
        track_encoded = rearrange(track_encoded, "b t n (v pn d) -> b t (n v pn) d", v=self.num_views, pn=self.num_track_patches_per_view)
    
        print(f"Shape of track_encoded after reshaping: {track_encoded.shape}")
    
        if self.spatial_transformer_use_text:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded, text_encoded], -2)
        else:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)
    
        print(f"Shape of img_track_text_encoded: {img_track_text_encoded.shape}")
        print(f"Shape of self.modality_embed: {self.modality_embed.shape}")
        print(f"Shape of self.modality_idx: {self.modality_idx.shape}")
    
        # Move tensors to the same device as img_track_text_encoded
        device = img_track_text_encoded.device
        b, t, n, d = img_track_text_encoded.shape
        modality_embed_resized = self.modality_embed.to(device).expand(b, t, -1, -1)
        
        # Adjust modality_idx_expanded to match the size of img_track_text_encoded
        modality_idx_expanded = self.modality_idx.to(device)
        if modality_idx_expanded.shape[0] < n:
            padding = torch.zeros(n - modality_idx_expanded.shape[0], dtype=torch.long, device=device)
            modality_idx_expanded = torch.cat([modality_idx_expanded, padding])
        modality_idx_expanded = modality_idx_expanded[:n].unsqueeze(0).unsqueeze(0).expand(b, t, -1)
    
        modality_embed_selected = torch.gather(modality_embed_resized, 2, modality_idx_expanded.unsqueeze(-1).expand(-1, -1, -1, d))
    
        print(f"Shape of modality_embed_selected: {modality_embed_selected.shape}")
        print(f"Shape of img_track_text_encoded before addition: {img_track_text_encoded.shape}")
    
        img_track_text_encoded += modality_embed_selected
    
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)
    
        encoded = rearrange(encoded, "b t n c -> (b t) n c")
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')
    
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')
    
        text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c')
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [action_cls_token, text_encoded_, out]
        else:
            out_seq = [action_cls_token, out]
    
        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')
    
        if return_recon:
            return output, _recon_track, intermediate_outputs
        return output
    
    def temporal_encode(self, x):
        """
        Args:
            x: b, t, num_modality, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, 2+num_extra, c)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (b, t*num_modality, c)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # (b, t, num_modality, c)
        return x[:, :, 0]  # (b, t, c)

    # def forward(self, obs, track_obs, track, task_emb, extra_states):
    #     """
    #     Return feature and info.
    #     Args:
    #         obs: b v t c h w
    #         track_obs: b v t tt_fs c h w
    #         track: b v t track_len n 2, not used for training, only preserved for unified interface
    #         extra_states: {k: b t e}
    #     """
    #     x, recon_track = self.spatial_encode(obs, track_obs, task_emb, extra_states, return_recon=True)  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
    #     x = self.temporal_encode(x)  # (b, t, c)

    #     recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
    #     x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)

    #     dist = self.policy_head(x)  # only use the current timestep feature to predict action
    #     return dist

    # forward with intermediate_outputs
    def forward(self, obs, track_obs, track, task_emb, extra_states):
        print(f"Input obs shape: {obs.shape}")
        print(f"Input track_obs shape: {track_obs.shape}")
        print(f"Input task_emb shape: {task_emb.shape}")
        x, recon_track, intermediate_outputs = self.spatial_encode(obs, track_obs, task_emb, extra_states, return_recon=True)
        x = self.temporal_encode(x)  # (b, t, c)
    
        recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        
        if isinstance(intermediate_outputs, torch.Tensor) and intermediate_outputs.numel() > 0:
            additional_features = intermediate_outputs.view(x.shape[0], x.shape[1], -1)
            print(f"Shape of additional features/intermediate outputs: {additional_features.shape}")
            policy_input = torch.cat([x, recon_track, additional_features], dim=-1)
        else:
            print(f"Shape of intermediate outputs LOOK HERE: {intermediate_outputs[0].shape}")
            policy_input = torch.cat([x, recon_track], dim=-1)
    
        print(f"Shape of policy_input before reshaping: {policy_input.shape}")
    
        # Use only the last timestep for action prediction
        policy_input = policy_input[:, -1, :]  # (b, input_size)
    
        print(f"Shape of policy_input after reshaping: {policy_input.shape}")
    
        action = self.policy_head(policy_input)
        return action, policy_input

    # def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action):
    #     """
    #     Args:
    #         obs: b v t c h w
    #         track_obs: b v t tt_fs c h w
    #         track: b v t track_len n 2, not used for training, only preserved for unified interface
    #         task_emb: b emb_size
    #         action: b t act_dim
    #     """
    #     obs, track, action = self.preprocess(obs, track, action)
    #     dist = self.forward(obs, track_obs, track, task_emb, extra_states)
    #     loss = self.policy_head.loss_fn(dist, action, reduction="mean")

    #     ret_dict = {
    #         "bc_loss": loss.sum().item(),
    #     }

    #     if not self.policy_head.deterministic:
    #         # pseudo loss
    #         sampled_action = dist.sample().detach()
    #         mse_loss = F.mse_loss(sampled_action, action)
    #         ret_dict["pseudo_sampled_action_mse_loss"] = mse_loss.sum().item()

    #     ret_dict["loss"] = ret_dict["bc_loss"]
    #     return loss.sum(), ret_dict

    def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action):
        obs, track, action = self.preprocess(obs, track, action)
        pred_action, _ = self.forward(obs, track_obs, track, task_emb, extra_states)
        loss = self.policy_head.loss_fn(pred_action, action[:, -1], reduction="mean")
    
        ret_dict = {
            "bc_loss": loss.item(),
            "loss": loss.item(),
        }
    
        return loss, ret_dict

    def forward_vis(self, obs, track_obs, track, task_emb, extra_states, action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2
            task_emb: b emb_size
        Returns:
        """
        _, track, _ = self.preprocess(obs, track, action)
        track = track[:, :, 0, :, :, :]  # Use the first timestep's track for visualization.
    
        b, v, t, track_obs_t, c, h, w = track_obs.shape
        if t >= self.num_track_ts:
            track_obs = track_obs[:, :, :self.num_track_ts, ...]
            track = track[:, :, :self.num_track_ts, ...]
        else:
            last_obs = track_obs[:, :, -1:, ...]
            pad_obs = repeat(last_obs, "b v 1 track_obs_t c h w -> b v t track_obs_t c h w", t=self.num_track_ts-t)
            track_obs = torch.cat([track_obs, pad_obs], dim=2)
            last_track = track[:, :, -1:, ...]
            pad_track = repeat(last_track, "b v 1 n d -> b v tl n d", tl=self.num_track_ts-t)
            track = torch.cat([track, pad_track], dim=2)
    
        all_ret_dict = {}
        combined_images = []
        combined_track_vids = []
        for view in range(self.num_views):
            # Create a dummy grid since we're not using it
            dummy_grid = torch.zeros((1, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
            
            _, ret_dict = self.track.forward_vis(track_obs[:1, view, 0, :, ...], dummy_grid, task_emb[:1], p_img=0)
            
            for k, v in ret_dict.items():
                if k in all_ret_dict:
                    all_ret_dict[k].extend(v if isinstance(v, list) else [v])
                else:
                    all_ret_dict[k] = v if isinstance(v, list) else [v]
            
            if "combined_image" in ret_dict:
                combined_images.append(ret_dict["combined_image"])
            if "track_vid" in ret_dict:
                combined_track_vids.append(ret_dict["track_vid"])
    
        # Process and calculate mean for numeric values
        for k, values in all_ret_dict.items():
            if k != "combined_image" and all(isinstance(x, (torch.Tensor, np.ndarray)) for x in values):
                values = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in values]
                all_ret_dict[k] = np.mean(values)
            elif k != "combined_image" and all(isinstance(x, (float, int, np.number)) for x in values):
                all_ret_dict[k] = np.mean(values)
    
        # Combine images from all views
        if combined_images:
            all_ret_dict["combined_image"] = np.concatenate(combined_images, axis=1)
        else:
            all_ret_dict["combined_image"] = np.array([])
    
        # Combine track videos from all views
        if combined_track_vids:
            all_ret_dict["combined_track_vid"] = np.concatenate(combined_track_vids, axis=2)  # Assuming the videos are numpy arrays
        else:
            all_ret_dict["combined_track_vid"] = None
    
        return None, all_ret_dict, None

    # def act(self, obs, task_emb, extra_states):
    #     """
    #     Args:
    #         obs: (b, v, h, w, c)
    #         task_emb: (b, em_dim)
    #         extra_states: {k: (b, state_dim,)}
    #     """
    #     self.eval()
    #     B = obs.shape[0]

    #     # expand time dimenstion
    #     obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()
    #     extra_states = {k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()}

    #     dtype = next(self.parameters()).dtype
    #     device = next(self.parameters()).device
    #     obs = torch.Tensor(obs).to(device=device, dtype=dtype)
    #     task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
    #     extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}

    #     if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (obs.shape[-1] != self.obs_shapes["rgb"][-1]):
    #         obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
    #         obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
    #         obs = rearrange(obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views)

    #     while len(self.track_obs_queue) < self.max_seq_len:
    #         self.track_obs_queue.append(torch.zeros_like(obs))
    #     self.track_obs_queue.append(obs.clone())
    #     track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
    #     track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")

    #     obs = self._preprocess_rgb(obs)

    #     with torch.no_grad():
    #         x, rec_tracks = self.spatial_encode(obs, track_obs, task_emb=task_emb, extra_states=extra_states, return_recon=True)  # x: (b, 1, 4, c), recon_track: (b, v, 1, tl, n, 2)
    #         self.latent_queue.append(x)
    #         x = torch.cat(list(self.latent_queue), dim=1)  # (b, t, 4, c)
    #         x = self.temporal_encode(x)  # (b, t, c)

    #         feat = torch.cat([x[:, -1], rearrange(rec_tracks[:, :, -1, :, :, :], "b v tl n d -> b (v tl n d)")], dim=-1)

    #         action = self.policy_head.get_action(feat)  # only use the current timestep feature to predict action
    #         action = action.detach().cpu()  # (b, act_dim)

    #     action = action.reshape(-1, *self.act_shape)
    #     action = torch.clamp(action, -1, 1)
    #     return action.float().cpu().numpy(), (None, rec_tracks[:, :, -1, :, :, :])  # (b, *act_shape)

    # act with intermediate outputs
    def act(self, obs, task_emb, extra_states):
        self.eval()
        B = obs.shape[0]
    
        obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()
        extra_states = {k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()}
    
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        obs = torch.Tensor(obs).to(device=device, dtype=dtype)
        task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}
    
        if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (obs.shape[-1] != self.obs_shapes["rgb"][-1]):
            obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
            obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
            obs = rearrange(obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views)
    
        while len(self.track_obs_queue) < self.max_seq_len:
            self.track_obs_queue.append(torch.zeros_like(obs))
        self.track_obs_queue.append(obs.clone())
        track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
        track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")
    
        obs = self._preprocess_rgb(obs)
    
        with torch.no_grad():
            action, _ = self.forward(obs, track_obs, None, task_emb, extra_states)
            action = action.detach().cpu()  # (b, act_dim)
    
        action = action.reshape(-1, *self.act_shape)
        action = torch.clamp(action, -1, 1)
        return action.float().cpu().numpy(), (None, None)  # (b, *act_shape)
    
    def reset(self):
        self.latent_queue.clear()
        self.track_obs_queue.clear()

    def save(self, path):
        torch.save(self.state_dict(), path)

    # def load(self, path):
    #     self.load_state_dict(torch.load(path, map_location="cpu"))

    def load(self, path):
        state_dict = torch.load(path, map_location="cpu")
        model_state_dict = self.state_dict()
        
        # Filter out mismatched keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        # Update model state dict
        model_state_dict.update(filtered_state_dict)
        
        # Load the filtered state dict
        self.load_state_dict(model_state_dict, strict=False)
        
        print(f"Loaded checkpoint from {path}")
        print(f"Missed keys: {set(model_state_dict.keys()) - set(filtered_state_dict.keys())}")

    def train(self, mode=True):
        super().train(mode)
        self.track.eval()

    def eval(self):
        super().eval()
        self.track.eval()