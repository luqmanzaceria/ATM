import hydra
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import lightning
from lightning.fabric import Fabric

import os
import wandb
import json
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atm.dataloader import BCDataset, get_dataloader
from atm.policy import *
from atm.utils.train_utils import setup_optimizer, setup_lr_scheduler, init_wandb
from atm.utils.log_utils import MetricLogger, BestAvgLoss
from atm.utils.env_utils import build_env
from engine.utils import rollout, merge_results


@hydra.main(config_path="../conf/train_bc", version_base="1.3")
def main(cfg: DictConfig):
        
    # Put the import here so that running on slurm does not have import error
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    train_dataset = BCDataset(dataset_dir=cfg.train_dataset, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset,
                                      mode="train",
                                      num_workers=cfg.num_workers,
                                      batch_size=cfg.batch_size)

    train_vis_dataset = BCDataset(dataset_dir=cfg.train_dataset, vis=True, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_vis_dataloader = get_dataloader(train_vis_dataset,
                                              mode="train",
                                              num_workers=1,
                                              batch_size=1)

    val_dataset = BCDataset(dataset_dir=cfg.val_dataset, num_demos=cfg.val_num_demos, **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    val_vis_dataset = BCDataset(dataset_dir=cfg.val_dataset, num_demos=cfg.val_num_demos, vis=True, **cfg.dataset_cfg, aug_prob=0.)
    val_vis_dataloader = get_dataloader(val_vis_dataset, mode="train", num_workers=1, batch_size=1)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), precision="bf16-mixed" if cfg.mix_precision else None, strategy="deepspeed")
    fabric.launch()

    if fabric.is_global_zero and not cfg.dry:
        try:
            init_wandb(cfg)
            print("Wandb initialized successfully")
        except Exception as e:
            print(f"Error initializing wandb: {e}")
    else:
        print(f"Skipping wandb initialization. Is global zero: {fabric.is_global_zero}, Dry run: {cfg.dry}")

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    # initialize the environments in each rank
    cfg.env_cfg.render_gpu_ids = cfg.env_cfg.render_gpu_ids[fabric.global_rank] if isinstance(cfg.env_cfg.render_gpu_ids, list) else cfg.env_cfg.render_gpu_ids
    env_num_each_rank = math.ceil(len(cfg.env_cfg.env_name) / fabric.world_size)
    env_idx_start_end = (env_num_each_rank * fabric.global_rank,  min(env_num_each_rank * (fabric.global_rank + 1), len(cfg.env_cfg.env_name)))
    rollout_env = build_env(img_size=cfg.img_size, env_idx_start_end=env_idx_start_end, **cfg.env_cfg)
    rollout_horizon = cfg.env_cfg.get("horizon", None)

    fabric.barrier()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    # Pick ckpt based on  the average of the last 5 epochs
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

    fabric.barrier()
    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            cfg,
            cfg.clip_grad,
            mix_precision=cfg.mix_precision,
            scheduler=scheduler,
        )
    
        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        print(f"Is global zero: {fabric.is_global_zero}")
        print(f"Dry run: {cfg.dry}")
            
        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)
    
            # Run validation and log metrics every epoch
            print("Starting validation...")
            val_metrics, val_task_losses = evaluate(model,
                                                    val_loader,
                                                    mix_precision=cfg.mix_precision,
                                                    tag="val")
            print("Finished validation")
    
            # Save best checkpoint
            metric_logger.update(**val_metrics)
    
            loss_metric = val_metrics["val/loss"]
            is_best = best_loss_logger.update_best(loss_metric, epoch)
    
            if is_best:
                model.save(f"{work_dir}/model_best.ckpt")
                with open(f"{work_dir}/best_epoch.txt", "w") as f:
                    f.write(
                        "Best epoch: %d, Best %s: %.4f"
                        % (epoch, "loss", best_loss_logger.best_loss)
                    )
            
            # Log validation metrics to wandb
            if not cfg.dry:
                try:
                    # Log general validation loss
                    wandb.log({"val/loss": val_metrics["val/loss"]}, step=epoch)
                    
                    # Log per-task validation losses
                    for task_name, task_loss in val_task_losses.items():
                        wandb.log({f"val/task_loss/{task_name}": task_loss}, step=epoch)
                    
                    # Log best loss so far
                    wandb.log({"val/best_loss": best_loss_logger.best_loss}, step=epoch)

                    print("Successfully logged validation metrics to wandb")
                except Exception as e:
                    print(f"Error logging to wandb: {e}")
    
            print(f"Validation metrics at epoch {epoch}:")
            print(f"General val loss: {val_metrics['val/loss']:.4f}")
            for task_name, task_loss in val_task_losses.items():
                print(f"Val loss for {task_name}: {task_loss:.4f}")

        if epoch % cfg.save_freq == 0:
            model.save(f"{work_dir}/model_{epoch}.ckpt")

            def vis_and_log(model, vis_dataloader, mode="train"):
                eval_dict = visualize(model, vis_dataloader, mix_precision=cfg.mix_precision)

                caption = f"reconstruction (right) @ epoch {epoch}; \n Track MSE: {eval_dict['track_loss']:.4f}; Img MSE: {eval_dict['img_loss']:.4f}"
                wandb_image = wandb.Image(eval_dict["combined_image"], caption=caption)
                wandb_vid_rollout = wandb.Video(eval_dict["combined_track_vid"], fps=24, format="mp4", caption=caption)
                None if cfg.dry else wandb.log({f"{mode}/first_frame": wandb_image,
                                                f"{mode}/rollout_track": wandb_vid_rollout},
                                                step=epoch)

            if fabric.is_global_zero and hasattr(model, "forward_vis"):
                vis_and_log(model, train_vis_dataloader, mode="train")
                vis_and_log(model, val_vis_dataloader, mode="val")

            gathered_results = [{} for _ in range(fabric.world_size)]
            results = rollout(rollout_env, model, 20 // cfg.env_cfg.vec_env_num, horizon=rollout_horizon)
            fabric.barrier()
            dist.all_gather_object(gathered_results, results)
            if fabric.is_global_zero:
                gathered_results = merge_results(gathered_results)
                None if cfg.dry else wandb.log(gathered_results, step=epoch)

                for k in list(results.keys()):
                    if k.startswith("rollout/vis_"):
                        results.pop(k)

                metric_logger.update(**results)
        fabric.barrier()

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(fabric, model, dataloader, optimizer, cfg, clip_grad=1.0, mix_precision=False, scheduler=None):
    tot_loss_dict, tot_items = {}, 0
    task_losses = {task_name: [] for task_name in dataloader.dataset.task_names}
    lang_grad_norms = []
    total_loss = 0
    total_bc_loss = 0

    model.train()
    for batch_idx, (obs, track_obs, track, task_emb, action, extra_states, task_ids) in enumerate(tqdm(dataloader)):
        if mix_precision:
            obs, track_obs, track, task_emb, action = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}

        # Print tasks in this batch
        tasks_in_batch = [dataloader.dataset.task_names[task_id.item()] for task_id in task_ids]
        # print(f"Batch {batch_idx}, Tasks: {tasks_in_batch}")

        # Ensure task_emb requires grad
        task_emb.requires_grad_(True)

        loss, ret_dict = model.forward_loss(obs, track_obs, track, task_emb, extra_states, action)
        optimizer.zero_grad()
        fabric.backward(loss)

        # Calculate gradient magnitude on language input
        lang_grad_norm = torch.norm(task_emb.grad).item() if task_emb.grad is not None else 0
        lang_grad_norms.append(lang_grad_norm)

        # Print task embedding and its gradient for the first few batches
        # if batch_idx < 5:
            # print(f"Batch {batch_idx}:")
            # print(f"Task embedding shape: {task_emb.shape}")
            # print(f"Task embedding (first sample): {task_emb[0, :10]}")
            # print(f"Task embedding grad (first sample): {task_emb.grad[0, :10] if task_emb.grad is not None else 'None'}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

        # Accumulate losses for each task
        for i, task_id in enumerate(task_ids):
            task_name = dataloader.dataset.task_names[task_id.item()]
            task_loss = ret_dict['bc_loss'] / len(task_ids)
            task_losses[task_name].append(task_loss)

        # Accumulate total loss and bc_loss
        total_loss += loss.item()
        total_bc_loss += ret_dict['bc_loss']

    # Calculate and log average losses for each task over the entire epoch
    avg_task_losses = {k: sum(v) / len(v) for k, v in task_losses.items() if v}
    
    # Calculate average total loss and bc_loss
    avg_total_loss = total_loss / tot_items
    avg_bc_loss = total_bc_loss / tot_items

    if fabric.is_global_zero and not cfg.dry:
        # Log general metrics
        wandb.log({
            "train/epoch_avg_loss": avg_total_loss,
            "train/epoch_avg_bc_loss": avg_bc_loss,
        })
        
        # Log task-specific metrics
        wandb.log({f"train/epoch_avg_bc_loss_{k}": v for k, v in avg_task_losses.items()})

    # Log average gradient magnitude on language input for the entire epoch
    avg_lang_grad_norm = sum(lang_grad_norms) / len(lang_grad_norms)
    if fabric.is_global_zero:
        if not cfg.dry:
            wandb.log({"train/epoch_avg_lang_grad_norm": avg_lang_grad_norm})
        print(f"Average language gradient norm: {avg_lang_grad_norm}")

    out_dict = {}
    for k, v in tot_loss_dict.items():
        out_dict[f"train/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    if scheduler is not None:
        scheduler.step()

    return out_dict
    
@torch.no_grad()
def evaluate(model, dataloader, mix_precision=False, tag="val"):
    tot_loss_dict, tot_items = {}, 0
    task_losses = {task_name: [] for task_name in dataloader.dataset.task_names}
    model.eval()

    for obs, track_obs, track, task_emb, action, extra_states, task_ids in tqdm(dataloader):
        obs, track_obs, track, task_emb, action = obs.cuda(), track_obs.cuda(), track.cuda(), task_emb.cuda(), action.cuda()
        extra_states = {k: v.cuda() for k, v in extra_states.items()}
        if mix_precision:
            obs, track_obs, track, task_emb, action = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}

        _, ret_dict = model.forward_loss(obs, track_obs, track, task_emb, extra_states, action)

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

        # Accumulate losses for each task
        for i, task_id in enumerate(task_ids):
            task_name = dataloader.dataset.task_names[task_id.item()]
            task_loss = ret_dict['bc_loss'] / len(task_ids)
            task_losses[task_name].append(task_loss)

    out_dict = {}
    for k, v in tot_loss_dict.items():
        out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    # Calculate and add average losses for each task
    avg_task_losses = {k: sum(v) / len(v) for k, v in task_losses.items() if v}
    out_dict.update({f"{tag}/avg_bc_loss_{k}": v for k, v in avg_task_losses.items()})

    return out_dict, avg_task_losses


@torch.no_grad()
def visualize(model, dataloader, mix_precision=False):
    model.eval()
    keep_eval_dict = None

    for obs, track_obs, track, task_emb, action, extra_states, task_ids in dataloader:
        obs, track_obs, track, task_emb = obs.cuda(), track_obs.cuda(), track.cuda(), task_emb.cuda()
        extra_states = {k: v.cuda() for k, v in extra_states.items()}
        if mix_precision:
            obs, track_obs, track, task_emb = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}
        _, eval_dict = model.forward_vis(obs, track_obs, track, task_emb, extra_states, action)
        keep_eval_dict = eval_dict
        break

    return keep_eval_dict


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()