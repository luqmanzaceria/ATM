# atm/utils/dims_vars_utils.py

import csv
import matplotlib.pyplot as plt
import os

def log_dims_and_vars(filename, epoch, step, **kwargs):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':  # New file, write header
            writer.writerow(['epoch', 'step'] + list(kwargs.keys()))
        writer.writerow([epoch, step if step is not None else 'epoch_end'] + list(kwargs.values()))

def plot_vars(filename, output_dir):
    # Load data from the CSV file
    data = {'epoch': [], 'step': []}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                if key == 'epoch':
                    data[key].append(int(value))
                elif 'var' in key:
                    try:
                        data[key].append(float(value))
                    except ValueError:
                        print(f"Warning: Could not convert '{value}' to float for key '{key}'. Skipping this value.")
                        continue

    # Plot only variances
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot the variables related to 'var'
    for key in data.keys():
        if key not in ['epoch', 'step'] and 'var' in key:
            ax.plot(data['epoch'], data[key], label=key)
    
    ax.set_title('Variances over epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Variance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vars_plot.png')
    plt.close()

    print(f"Variance plot saved to {output_dir}/vars_plot.png")