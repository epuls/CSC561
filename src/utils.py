import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['opt_state'])

def plot_sequence(inp, tgt, pred=None, figsize=(12,8)):
    """
    inp: (T_in, C, H, W)
    tgt: (T_out, C, H, W)
    pred: optional, same shape as tgt
    """
    T_in, C, H, W = inp.shape
    T_out = tgt.shape[0]
    cols = max(T_in, T_out) * (3 if pred is not None else 2)
    rows = 1 + (1 if pred is None else 2)
    fig, axes = plt.subplots(rows, max(T_in, T_out), figsize=figsize)
    # Inputs
    for i in range(T_in):
        ax = axes[0,i] if rows>1 else axes[i]
        img = inp[i][:3].permute(1,2,0).cpu().numpy()  # RGB
        ax.imshow((img - img.min())/(img.max()-img.min()))
        ax.set_title(f"Input {i}")
        ax.axis('off')
    # Targets
    for i in range(T_out):
        ax = axes[1,i]
        img = tgt[i][:3].permute(1,2,0).cpu().numpy()
        ax.imshow((img - img.min())/(img.max()-img.min()))
        ax.set_title(f"Target {i}")
        ax.axis('off')
    # Predictions
    if pred is not None:
        for i in range(T_out):
            ax = axes[2,i]
            img = pred[i][:3].permute(1,2,0).cpu().numpy()
            ax.imshow((img - img.min())/(img.max()-img.min()))
            ax.set_title(f"Pred {i}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()
