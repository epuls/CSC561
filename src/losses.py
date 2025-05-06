import torch
import torch.nn.functional as F

def physics_informed_loss(preds, targets, inputs, a=1.0, b=1.0):
    """
    preds, targets, inputs: 
      either (B, C, H, W) or (B, T, C, H, W)
    a: weight for the momentum term
    b: weight for the mass-conservation term
    """
    # ---- 1) flatten time if needed ----
    if preds.dim() == 5:
        B, T, C, H, W = preds.shape
        # merge B and T into a single batch dimension
        preds   = preds.  reshape(B * T, C, H, W)
        targets = targets.reshape(B * T, C, H, W)
        inputs  = inputs. reshape(B * T, C, H, W)
    elif preds.dim() == 4:
        # already (B,C,H,W)
        pass
    else:
        raise ValueError(f"Expected 4D or 5D tensors, got {preds.dim()}D")

    # ---- 2) delta prediction losses ----
    delta_true = targets - inputs   # (N, C, H, W)
    delta_pred = preds   - inputs

    # MSE on *all* channels (you can change to only select some if you like)
    mse_term = F.mse_loss(delta_pred, delta_true)

    # ---- 3) momentum loss on channels 0 (r_x) & 1 (r_y) ----
    mom_x = F.mse_loss(delta_pred[:, 0], delta_true[:, 0])
    mom_y = F.mse_loss(delta_pred[:, 1], delta_true[:, 1])
    momentum_term = mom_x + mom_y

    # ---- 4) mass-conservation (continuity) on channel 2 (h) ----
    # continuity: delta_h + div(v) ≈ 0
    delta_h = delta_pred[:, 2]   # (N, H, W)
    vel_x   = preds[:, 0]        # (N, H, W)
    vel_y   = preds[:, 1]

    # ∂v_x/∂x  → difference along width dim (last dim)
    div_x = vel_x[:, :, 1:] - vel_x[:, :, :-1]     # (N, H, W-1)
    # pad width: (pad_left, pad_right, pad_top, pad_bottom)
    div_x = F.pad(div_x, (0, 1, 0, 0))              # → (N, H, W)

    # ∂v_y/∂y  → difference along height dim (second-to-last dim)
    div_y = vel_y[:, 1:, :] - vel_y[:, :-1, :]     # (N, H-1, W)
    # pad height: (pad_left, pad_right, pad_top, pad_bottom)
    div_y = F.pad(div_y, (0, 0, 0, 1))              # → (N, H, W)

    continuity_residual = delta_h + div_x + div_y  # (N, H, W)
    mass_term = torch.mean(continuity_residual**2)

    # ---- 5) combine ----
    return mse_term + a * momentum_term + b * mass_term
