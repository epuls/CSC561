import torch
import torch.nn.functional as F
import torch.autograd as autograd
import wandb
from datasets import load_pt_dataset
from models import ConvLSTMSeq2Seq, ConvLSTMSeq2SeqNoAlpha, SequenceDiscriminator, ViTRefiner, CascadePredictor, BiConvLSTMSeq2Seq
from utils import set_seed, save_checkpoint, load_checkpoint
from argparse import Namespace
import io
import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from utils import plot_sequence
from torch.amp import autocast, GradScaler
import argparse, yaml
from PIL import Image
from torchvision.transforms import v2
import random
import imageio
import numpy as np
from losses import physics_informed_loss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from datasets import PTSequenceDataset
from torch.utils.data import Dataset, DataLoader
import OpenEXR
import Imath
import os


def save_exr(path, tensor):
    """
    Save a C×H×W float32 tensor to an EXR, including alpha if present.
    """
    arr = tensor.numpy()
    C, H, W = arr.shape

    header = OpenEXR.Header(W, H)
    pix = Imath.PixelType(Imath.PixelType.FLOAT)

    if C == 1:
        # single-channel
        header['channels'] = {'Y': Imath.Channel(pix)}
        out = OpenEXR.OutputFile(path, header)
        out.writePixels({'Y': arr[0].astype(np.float32).tobytes()})

    elif C >= 3:
        # RGB + optional A
        chan = {
            'R': Imath.Channel(pix),
            'G': Imath.Channel(pix),
            'B': Imath.Channel(pix)
        }
        if C >= 4:
            chan['A'] = Imath.Channel(pix)
        header['channels'] = chan

        out = OpenEXR.OutputFile(path, header)
        data = {
            'R': arr[0].astype(np.float32).tobytes(),
            'G': arr[1].astype(np.float32).tobytes(),
            'B': arr[2].astype(np.float32).tobytes()
        }
        if C >= 4:
            data['A'] = arr[3].astype(np.float32).tobytes()

        out.writePixels(data)

    else:
        raise ValueError(f"Unsupported channel count: {C}")

    out.close()

def print_batch_progress(epoch, batch_idx, total_batches):
    # \r returns to start of line, end='' keeps it on one line
    print(f"Epoch {epoch} — Batch {batch_idx}/{total_batches}", end='\r', flush=True)


def compute_gradient_penalty(D, real_seq, fake_seq, device):
       # Interpolate between real and fake
    alpha = torch.rand(real_seq.size(0), 1, 1, 1, 1, device=device)
    interp = (alpha * real_seq + (1 - alpha) * fake_seq).requires_grad_(True)

    # Score on the interpolated batch
    d_interp = D(interp)

    # Compute gradients w.r.t. the interpolated inputs
    grads = autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]  # shape (B, T, C, H, W)

    # Flatten per-sample without needing contiguity
    grads = grads.reshape(grads.size(0), -1)

    # L2 norm per sample
    grad_norm = grads.norm(2, dim=1)

    # WGAN-GP penalty
    return ((grad_norm - 1) ** 2).mean()

def train_epoch_seq2seq_amp(model, loader, optimizer, scaler, device, config, threshold=0.01):
    model.train()
    tot_loss, tot_acc, total_rgb_loss, cnt = 0.0, 0.0, 0.0, 0
    for seq, tgt in loader:
        seq, tgt = seq.to(device), tgt.to(device)
        optimizer.zero_grad()
        # forward/backward in mixed-precision
        with autocast('cuda'):
            pred = model(seq)
            if(config['physics_informed_loss']):
                if config['skip_alpha_channel'] and not config['convlstm_only_skip']:
                    loss = physics_informed_loss(pred[..., :3, :, :], tgt[..., :3, :, :], seq[..., :3, :, :], config['momentum_weight'], config['mass_weight'])
                else:
                    loss = physics_informed_loss(pred, tgt, seq, config['momentum_weight'], config['mass_weight'])
            else:
                if config['skip_alpha_channel'] and not config['convlstm_only_skip']:
                    if config['only_deltas']:
                        loss = F.mse_loss(pred[..., :3, :, :], seq-tgt[..., :3, :, :])
                    else:
                        loss = F.mse_loss(pred[..., :3, :, :], tgt[..., :3, :, :])
                else:
                    if config['only_deltas']:
                        loss = F.mse_loss(pred, seq-tgt)
                    else:
                        loss = F.mse_loss(pred, tgt)
            rgb_loss = F.mse_loss(pred[..., :3, :, :], tgt[..., :3, :, :])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc = ((pred - tgt).abs() < threshold).float().mean()
        tot_loss += loss.item(); tot_acc += acc.item(); cnt += 1
        total_rgb_loss += rgb_loss.item()
    return tot_loss/cnt, tot_acc/cnt, total_rgb_loss/cnt

def train_epoch_seq2seq(model, loader, optimizer, device, config, threshold=0.01):
    model.train()
    total_loss, total_acc, total_rgb_loss, cnt = 0.0, 0.0, 0.0, 0
    

    for seq, tgt in loader:
        seq, tgt = seq.to(device), tgt.to(device)
        optimizer.zero_grad()
        pred = model(seq)
        if(config['physics_informed_loss']):
            loss = physics_informed_loss(pred, tgt, seq, config['momentum_weight'], config['mass_weight'])
        else:
            if config['skip_alpha']:
                if config['only_deltas']:
                    loss = F.mse_loss(pred[..., :3, :, :], seq-tgt[..., :3, :, :])
                else:
                    loss = F.mse_loss(pred[..., :3, :, :], tgt[..., :3, :, :])
            else:
                if config['only_deltas']:
                    loss = F.mse_loss(pred, seq-tgt)
                else:
                    loss = F.mse_loss(pred, tgt)
        rgb_loss = F.mse_loss(pred[..., :3, :, :], tgt[..., :3, :, :])
        loss.backward()
        optimizer.step()
        # only RGB channels
        acc = ((pred[..., :3, :, :] - tgt[..., :3, :, :]).abs() < threshold)\
        .float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
        total_rgb_loss += rgb_loss.item()
        cnt += 1
    return total_loss/cnt, total_acc/cnt, total_rgb_loss/cnt

def validate_seq2seq(model, loader, device, config, threshold=0.01):
    model.eval()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    total_loss, total_acc, total_rgb_loss, cnt = 0.0, 0.0, 0.0, 0
    total_psnr = 0.0
    total_ssim = 0.0


    with torch.no_grad():
        for seq, tgt in loader:
            seq, tgt = seq.to(device), tgt.to(device)
            pred = model(seq)
            if(config['physics_informed_loss']):
                if config['skip_alpha_channel'] and not config['convlstm_only_skip']:
                    loss = physics_informed_loss(pred[..., :3, :, :], tgt[..., :3, :, :], seq[..., :3, :, :], config['momentum_weight'], config['mass_weight'])
                else:
                    loss = physics_informed_loss(pred, tgt, seq, config['momentum_weight'], config['mass_weight'])
            else:
                if config['skip_alpha_channel'] and not config['convlstm_only_skip']:
                    if config['only_deltas']:
                        loss = F.mse_loss(pred[..., :3, :, :], seq-tgt[..., :3, :, :])
                    else:
                        loss = F.mse_loss(pred[..., :3, :, :], tgt[..., :3, :, :])
                else:
                    if config['only_deltas']:
                        loss = F.mse_loss(pred, seq-tgt)
                    else:
                        loss = F.mse_loss(pred, tgt)
                
            # only RGB channels
            rgb_loss = F.mse_loss(pred[..., :3, :, :], tgt[..., :3, :, :]) 
            acc = ((pred[..., :3, :, :] - tgt[..., :3, :, :]).abs() < threshold)\
            .float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            total_rgb_loss += rgb_loss.item()

            # preds, tgt: (B, T, C, H, W)
            B, T, C, H, W = pred.shape

            # (1) If you only care about RGB, drop the height channel:
            pred_rgb = pred[..., :3, :, :]    # → (B, T, 3, H, W)
            tgt_rgb  = tgt[...,  :3, :, :]

            # (2) Collapse B and T into a single batch dimension:
            pred_flat = pred_rgb.reshape(B * T, 3, H, W)
            tgt_flat  = tgt_rgb. reshape(B * T, 3, H, W)

            total_psnr += psnr_metric(pred_flat, tgt_flat).item()
            total_ssim += ssim_metric(pred_flat, tgt_flat).item()

            cnt += 1

    return total_loss/cnt, total_acc/cnt, total_rgb_loss/cnt, total_psnr/cnt, total_ssim/cnt

def test_and_log(model, test_loader, device, config, threshold=0.01, log=True, filename="test_mse_map_example.png", plotName="test", test_loader_idx = 1):
    """model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for seq, tgt in test_loader:
            seq, tgt = seq.to(device), tgt.to(device)
            pred = model(seq)
            #tot_loss += F.mse_loss(pred, tgt).item()
            tot_loss += F.mse_loss(pred[..., :3, :, :], tgt[..., :3, :, :]).item()
            tot_acc  += ((pred[..., :3, :, :] - tgt[..., :3, :, :])
                         .abs() < threshold).float().mean().item()
            n += 1

    avg_loss = tot_loss / n
    avg_acc  = tot_acc  / n"""
    test_loss, test_acc, test_rgb_loss, test_psnr, test_ssim = validate_seq2seq(
        model, test_loader, device, config, threshold=threshold
    )
    
    print(f"Test Loss: {test_loss:.5f},  Test Acc: {test_acc:.5f}")
    print(f"Test RGB Loss: {test_rgb_loss:.5f}")
    print(f"Test PSNR: {test_psnr:.5f},  Test SSIM: {test_ssim:.5f}")
    if(log):
        wandb.log({"test/mse": test_loss, "test/acc": test_acc})

    loader_iter = iter(test_loader)           # create the iterator once
    for i in range(test_loader_idx + 1):      # go up through your target index
        seq, tgt = next(loader_iter)
    # after this loop, (seq, tgt) is the batch at index test_loader_idx

    seq, tgt = seq.to(device), tgt.to(device)
    with torch.no_grad():
        pred = model(seq)

    # undo NormalizeMinMax if needed (same as before)...
    ds        = getattr(test_loader.dataset, 'dataset', test_loader.dataset)
    transform = getattr(ds, 'transform', None)
    if transform and hasattr(transform, 'scale') and hasattr(transform, 'shift'):
        inv_scale = 1.0 / transform.scale
        inv_shift = -transform.shift * inv_scale
        seq  = seq  * inv_scale.to(device) + inv_shift.to(device)
        tgt  = tgt  * inv_scale.to(device) + inv_shift.to(device)
        pred = pred * inv_scale.to(device) + inv_shift.to(device)
        caption = "De-normalized 5→5 forecast"
    else:
        caption = "5→5 forecast (normalized)"

    # pick the first sample
    inp = seq[0].cpu()   # (T_in, 4, H, W)
    tar = tgt[0].cpu()   # (T_out,4,H,W)
    pr  = pred[0].cpu()  # (T_out,4,H,W)

    T_in  = inp.shape[0]
    T_out = tar.shape[0]
    cols  = max(T_in, T_out)


    # Normalize all based on first image in input seq
    first = inp[0][:3].permute(1,2,0).numpy()
    norm_max = first.max()
    norm_min = first.min()
    # make 4 rows: inputs, targets, preds, error‐maps
    fig, axes = plt.subplots(4, cols, figsize=(4*cols, 4*4))
    #plt.title(plotName)
    # Row 1: inputs
    for i in range(T_in):
        ax = axes[0, i]
        # print(f'img shape: {inp[i].shape}')
        img = inp[i][:3].permute(1,2,0).numpy()
        # img = inp[i].permute(1,2,0).numpy()
        # disp = (img - img.min())/(img.max()-img.min())
        disp = (img - norm_min)/(norm_max-img.min())
        #disp = np.clip(img, 0.0, 1.0)
        ax.imshow(disp); ax.set_title(f"In {i}"); ax.axis('off')

    # Row 2: targets
    for i in range(T_out):
        ax = axes[1, i]
        img = tar[i][:3].permute(1,2,0).numpy()
        #disp = (img - img.min())/(img.max()-img.min())
        #disp = np.clip(img, 0.0, 1.0)
        disp = (img - norm_min)/(norm_max-img.min())
        #disp = np.clip(img, 0.0, 1.0)
        ax.imshow(disp); ax.set_title(f"Tgt {i}"); ax.axis('off')

    # Row 3: predictions
    for i in range(T_out):
        ax = axes[2, i]
        img = pr[i][:3].permute(1,2,0).numpy()
        #disp = (img - img.min())/(img.max()-img.min())
        #disp = np.clip(img, 0.0, 1.0)
        disp = (img - norm_min)/(norm_max-img.min())
        #disp = np.clip(img, 0.0, 1.0)
        ax.imshow(disp); ax.set_title(f"Pred {i}"); ax.axis('off')

    # Row 4: per‐pixel MSE error maps (RGB only)
    for i in range(T_out):
        ax = axes[3, i]
        #ax.colorbar()
        # compute mean squared error over RGB channels only
        err_map = ((pr[i][:3] - tar[i][:3])**2).mean(dim=0).numpy()
        # normalize for display
        #e_min, e_max = err_map.min(), err_map.max()
        #disp = (err_map - e_min)/(e_max - e_min + 1e-8)
        #disp= np.clip(err_map, 0.0, 1.0)
        im = ax.imshow(err_map, cmap='magma')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        err = err_map.mean()
        ax.set_title(f"MSE Loss Pred {i}: {err:.4f}")
        ax.axis('off')

    fig.suptitle(f"{plotName} MSE | PSNR: {test_psnr:.4f} | SSIM: {test_ssim:.4f} | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}\n\n", fontsize=18, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf)

    if not log:
        pil_img.save(filename)
        pil_img.show()

    if(log):
        wandb.log({
            "tested_forecast": wandb.Image(pil_img, caption=caption),
            "test/mse_map_example": wandb.Image(pil_img, caption="Inputs/Targets/Preds/Error")
        })


class RandomFlip:
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, seq: torch.Tensor, tgt: torch.Tensor):
        # seq: (T, C, H, W), tgt: (T', C, H, W)
        if random.random() < self.p_h:
            # horizontal flip (width axis)
            seq = torch.flip(seq, dims=[-1])
            tgt = torch.flip(tgt, dims=[-1])
        if random.random() < self.p_v:
            # vertical flip (height axis)
            seq = torch.flip(seq, dims=[-2])
            tgt = torch.flip(tgt, dims=[-2])
        return seq, tgt

def get_transforms(config):
    config_tforms = config['transforms']
    p_hflip = config_tforms['HorizontalFlip']['p']
    p_vflip = config_tforms['VerticalFlip']['p']
    if (p_hflip == 0 and p_vflip == 0):
        return None

    print(f"Using RandomFlip with p_h={p_hflip} and p_v={p_vflip}")
    return RandomFlip(p_h=p_hflip, p_v=p_vflip)

def build_model_and_optimizer(config):
    # Build model & optimizer
    device = config['device']
    if(config['use_refiner']):
        convlstm_cfg = {
            'in_ch':           4,
            'hid_chs':         [config['hidden_channels'],config['hidden_channels']],
            'ks':              [config['kernel_size'],config['kernel_size']],
            'input_length':    5,
            'output_length':   5,
        }
        refiner_cfg = {
            'in_ch':        8,             # 4 rough + 4 last_input
            'out_ch':       4,
            'img_size':     (256, 256),
            'patch_size':   config['patch_size'],
            'embed_dim':    config['embed_dim'],
            'depth':        config['depth'],
            'num_heads':    config['num_heads'],
            'mlp_ratio':    config['mlp_ratio'],
            'dropout':      config['refiner_dropout'],
        }

        model = CascadePredictor(convlstm_cfg, refiner_cfg).to(device)
        if(config['use_pretrained_convlstm']):
            ckpt = torch.load(config['pretrained_convlstm_path'], map_location=device, weights_only=True)
            convlstm_state = ckpt['model_state']
            model.convlstm.load_state_dict(convlstm_state)
            for param in model.convlstm.parameters():
                param.requires_grad = False
            
            print("Loaded pretrained ConvLSTM weights")

            # 3) Build an optimizer only for the refiner if frozen
            if config['freeze_convlstm']:
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config['lr'], weight_decay=config['weight_decay']
                )
                print("ConvLSTM frozen, only training refiner")
                return model, optimizer


    else:
        model = ConvLSTMSeq2Seq(
            in_ch         = 4,
            hid_chs       = [config['hidden_channels'],config['hidden_channels']],
            ks            = [config['kernel_size'],config['kernel_size']],
            input_length  = config['input_length'],
            output_length = config['output_length']
        ).to(device)

        if(config['use_pretrained_convlstm']):
            ckpt = torch.load(config['pretrained_convlstm_path'], map_location=device, weights_only=True)
            convlstm_state = ckpt['model_state']
            model.load_state_dict(convlstm_state)
            print("Loaded pretrained ConvLSTM weights")
            


    # Get Optimizer
    if config['use_refiner'] == False:
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        elif config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    return model, optimizer

def train_model(config):
    seed = config['seed']
    use_fixed_seed = config['use_fixed_seed']
    if use_fixed_seed:
        set_seed(seed)
    else:
        # Use a random seed for each run
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        config['seed'] = seed
        set_seed(seed)
    
    best_val_acc = -float('inf')
    best_accuracy_ckpt = None

    best_val_loss = float('inf')
    best_loss_ckpt = None

    project   = config['project']
    device    = config['device']
    use_amp   = config['use_amp']

 
    pt_dir          = config['data_dir']
    batch_size      = config['batch_size']
    train_split     = config['train_split']
    val_split       = config['val_split']
    num_workers     = config['num_workers']
    print_validation= config['print_validation']
    data_normalize = config['normalize']

 
    lr        = config['lr']
    epochs    = config['epochs']
    threshold = config['accuracy_threshold']

    # Initialize W&B
    wandb.init(project=project, group=config['group'], config=config)
    wandb.log(config)

    print(f"wandb config: {wandb.config}")

    # Load data
    train_loader, val_loader, test_loader = load_pt_dataset(
        pt_dir,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        transform=get_transforms(config),
        num_workers=num_workers,
        print_validation=print_validation, 
        normalize=data_normalize,
        seed=seed
    )

    # Build model & optimizer
    if(config['use_refiner']):
        convlstm_cfg = {
            'in_ch':           4,
            'hid_chs':         [config['hidden_channels'],config['hidden_channels']],
            'ks':              [config['kernel_size'],config['kernel_size']],
            'input_length':    5,
            'output_length':   5,
        }
        refiner_cfg = {
            'in_ch':        8,             # 4 rough + 4 last_input
            'out_ch':       4,
            'img_size':     (256, 256),
            'patch_size':   config['patch_size'],
            'embed_dim':    config['embed_dim'],
            'depth':        config['depth'],
            'num_heads':    config['num_heads'],
            'mlp_ratio':    config['mlp_ratio'],
            'dropout':      config['refiner_dropout'],
        }

        model = CascadePredictor(convlstm_cfg, refiner_cfg, config['skip_alpha_channel']).to(device)
        if(config['use_pretrained_convlstm']):
            ckpt = torch.load(config['pretrained_convlstm_path'], map_location=device, weights_only=True)
            convlstm_state = ckpt['model_state']
            model.convlstm.load_state_dict(convlstm_state)
            for param in model.convlstm.parameters():
                param.requires_grad = False
            
            # 3) Build an optimizer only for the refiner if frozen
            if config['freeze_convlstm']:
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config['lr'], weight_decay=config['weight_decay']
                )
            print("Loaded pretrained ConvLSTM weights")

        
    else:

        if(config['skip_alpha_channel']):
            model = ConvLSTMSeq2SeqNoAlpha(
            in_ch         = 4,
            hid_chs       = [config['hidden_channels'],config['hidden_channels']],
            ks            = [config['kernel_size'],config['kernel_size']],
            input_length  = config['input_length'],
            output_length = config['output_length']
        ).to(device)
        else:
            model = ConvLSTMSeq2Seq(
                in_ch         = 4,
                hid_chs       = [config['hidden_channels'],config['hidden_channels']],
                ks            = [config['kernel_size'],config['kernel_size']],
                input_length  = config['input_length'],
                output_length = config['output_length']
                #merge_mode='sum'
            ).to(device)

            if(config['use_pretrained_convlstm']):
                ckpt = torch.load(config['pretrained_convlstm_path'], map_location=device, weights_only=True)
                convlstm_state = ckpt['model_state']
                model.load_state_dict(convlstm_state)
                print("Loaded pretrained ConvLSTM weights")
            

    # Build discriminator if using GAN
    if (config['use_gan']):
        discriminator = SequenceDiscriminator(
            in_ch         = 4,
            T             = config['output_length'],
            base_ch       = config['disc_base_channels']
        ).to(device)
        wandb.watch(discriminator, log="all", log_graph=True)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=config['disc_lr'],
            weight_decay=config['weight_decay']
        )
        lambda_gp = config['lambda_gp']
        adv_weight = config['adv_weight']
        n_critic = config['n_critic']


    # Get Optimizer
    if config['use_refiner'] == False:
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")


    # Get Scheduler
    if config['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
    elif config['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['lr_scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'], min_lr=config['min_lr'])

    if use_amp: 
        scaler = GradScaler()
        if(config['use_gan']):
            d_scaler = GradScaler()
    # Training loop (No GAN)
    for epoch in range(1, epochs + 1):
        if(config['use_gan']): 
            break
        print("Training model...")
        if use_amp:
            train_loss, train_acc, train_rgb_loss = train_epoch_seq2seq_amp(
                model, train_loader, optimizer, scaler, device, config=config, threshold=threshold
            )
        else:
            train_loss, train_acc, train_rgb_loss = train_epoch_seq2seq(
                model, train_loader, optimizer, device, config=config, threshold=threshold
            )

        val_loss, val_acc, val_rgb_loss, val_psnr, val_ssim     = validate_seq2seq(
            model, val_loader, device, config=config, threshold=threshold
        )

        wandb.log({
            'epoch':      epoch,
            'train/loss': train_loss,
            'train/rgb_loss': train_rgb_loss,
            'train/acc':  train_acc,
            'val/loss':   val_loss,
            'val/rgb_loss': val_rgb_loss,
            'val/acc':    val_acc,
            'val/psnr':   val_psnr,
            'val/ssim':   val_ssim,
            'lr':         optimizer.param_groups[0]['lr'],
            'test/val_loss_delta': abs(train_loss-val_loss)
        })

        print(f"[Epoch {epoch:2d}]  "
              f"Train L={train_loss:.4f} A={train_acc:.4f}  |  "
              f"Val L={val_loss:.4f} A={val_acc:.4f}")
        
        save_checkpoint(model, optimizer, f"{wandb.run.dir}/last.pt")
        print(f"Saved checkpoint to {wandb.run.dir}/last.pt")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # build a checkpoint dict

            # overwrite on‐disk best‐model file
            if config['save_checkpoints']:
                save_checkpoint(model, optimizer, f"{wandb.run.dir}/best_val_accuracy_model.pt")
                print(f"→ New best val_acc={val_acc:.4f} (epoch {epoch}), saving checkpoint")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # build a checkpoint dict


            # overwrite on‐disk best‐model file
            if config['save_checkpoints']:
                save_checkpoint(model, optimizer, f"{wandb.run.dir}/best_val_loss_model.pt")
                print(f"→ New best val loss={val_loss:.4f} (epoch {epoch}), saving checkpoint")
        
        if config['lr_scheduler'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
    test = 0
    # Discriminator training (if using GAN)
    for epoch in range(1, epochs + 1):
        if not config['use_gan']:
            break
        print("Training GAN...")
        discriminator.train()
        model.train()
        running = {'g_loss':0, 'd_loss':0, 'rec_loss':0, 'adv_loss':0, 'cnt':0}

        for seq, tgt in train_loader:
            seq, tgt = seq.to(device), tgt.to(device)
            print_batch_progress(epoch, test, len(train_loader))
            test += 1

             # --- 0) Precompute one fake
            with autocast(device_type='cuda'):
                fake = model(seq)

            # --- 1) Update Discriminator n_critic times, re-using fake
            for _ in range(n_critic):
                discriminator_optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    d_real = discriminator(tgt)
                    d_fake = discriminator(fake.detach())
                    loss_D  = d_fake.mean() - d_real.mean()
                # GP in FP32
                gp = compute_gradient_penalty(discriminator, tgt, fake.detach(), device)
                loss_D = loss_D + lambda_gp * gp
                d_scaler.scale(loss_D).backward()
                d_scaler.step(discriminator_optimizer)
                d_scaler.update()

            # --- 2) Update Generator once
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                # If you want fresh fakes for G, call G(seq) again.
                # But you can also re-use the same fake from above:
                fake2 = model(seq)
                d_fake   = discriminator(fake2)       # NOTE: no detach here
                rec_loss = F.mse_loss(fake2, tgt)
                adv_loss = -d_fake.mean()
                loss_G   = rec_loss + adv_weight * adv_loss

            # ‣ Scale + backward + step
            scaler.scale(loss_G).backward()
            scaler.step(optimizer)
            scaler.update()

            # accumulate for logging
            running['d_loss']   += loss_D.item()
            running['g_loss']   += loss_G.item()
            running['rec_loss'] += rec_loss.item()
            running['adv_loss'] += adv_loss.item()
            running['cnt']      += 1

        # epoch‐level logging…
        n = running['cnt']
        wandb.log({
        'D/loss': running['d_loss']/n,
        'G/loss': running['g_loss']/n,
        'G/rec':  running['rec_loss']/n,
        'G/adv':  running['adv_loss']/n,
        'epoch':  epoch
        })
        print(f"[{epoch}] D_loss={running['d_loss']/n:.4f} "
            f"G_rec={running['rec_loss']/n:.4f} G_adv={running['adv_loss']/n:.4f}")

    # Final test & checkpoint
    if(config['use_gan']):
            # --- Final test and plotting ---
        test_and_log(model, test_loader, device, threshold=config['accuracy_threshold'])
        save_checkpoint(model, optimizer, f"{wandb.run.dir}/G_final.pt")
        wandb.save("G_final.pt")
        wandb.finish()
        return

    # register accuracy model as a WandB artifact
    # artifact = wandb.Artifact('best_val_accuracy_model', type='model')
    # artifact.add_file(f"{wandb.run.dir}/best_val_accuracy_model.pt")
    # wandb.log_artifact(artifact)
    
    # # register loss model as a WandB artifact
    # artifact = wandb.Artifact('best_val_loss_model', type='model')
    # artifact.add_file(f"{wandb.run.dir}/best_val_loss_model.pt")
    # wandb.log_artifact(artifact)

    test_loss, test_acc, test_rgb, test_psnr, test_ssim = validate_seq2seq(
        model, test_loader, device, config=config, threshold=threshold
    )
    print(f"Test L={test_loss:.4f} A={test_acc:.4f}")
    print(f"Test PSNR={test_psnr:.4f} SSIM={test_ssim:.4f}")
    

    save_checkpoint(model, optimizer, f"{wandb.run.dir}/final.pt")
    wandb.save(f"{wandb.run.name}.pt")
    test_and_log(model, test_loader, device, threshold=threshold)
    wandb.finish()

def diy_sweep(config, key, values):
    default = config[key]
    
    for i in range(0, len(values)):
        config[key] = values[i]
        print(f"Running with {key}={values[i]}")
        train_model(config)
    
    config[key] = default

def plot_autoregressive(test_loader_idx, model, test_loader, device, num_iters=10, filename="autoregressive_predictions.png", make_gif=True, plotName="plt", exr_dir="final_evals_norm/comparison", exr_only=True):

    """
    Samples one random sequence from test_loader, runs the cascade model
    autoregressively num_iters times, then plots:
      row 0: the T_in real inputs
      row i: the i-th block of T_out predictions
    """
    model.eval()
    loader_iter = iter(test_loader)          # create the iterator once
    if (test_loader_idx == 0):
        seq_batch, __ = next(loader_iter)     # get the first batch
        print("WARNING: test_loader_idx is 0, will sample the first batch")
    else:
        for i in range(test_loader_idx + 1):      # go up through your target index
            seq_batch, __ = next(loader_iter)


    #for __ in range(test_loader_idx):
    #    seq_batch, _ = next(iter(test_loader))    # seq: (B, T_in, C, H, W)
    bsz, T_in, C, H, W = seq_batch.shape
    idx = random.randrange(bsz)
    curr_seq = seq_batch[idx : idx+1].to(device)  # (1, T_in, C, H, W)

    # 2) stash the real frames
    frames = [ curr_seq[0, t].cpu() for t in range(T_in) ]
    print(f"Input shape: {curr_seq.shape}, frames: {len(frames)}, frames shape: {frames[0].shape}")

    # 3) autoregressive generation
    with torch.no_grad():
        for it in range(1, num_iters+1):
            pred = model(curr_seq)  # → (1, T_out, C, H, W)
            T_out = pred.shape[1]
            # print(f"Iter {it} - Pred shape: {pred.shape}")
            for t in range(T_out):
                frames.append(pred[0, t].cpu())
            # next input = *only* the most recent block
            curr_seq = pred


    if exr_only:
        for i, frame in enumerate(frames, start=0):
            exr_path = os.path.join(exr_dir, f"0_{i}.exr")
            save_exr(exr_path, frame)
        print(f"Saved {len(frames)} predictions to '{exr_dir}/'")
        return

    
    # 4) plot the grid
    total_rows = num_iters + 1
    cols = T_in  # assumes T_out == T_in
    fig, axes = plt.subplots(total_rows, cols, figsize=(2*cols, 2*total_rows))

    for r in range(total_rows):
        for c in range(cols):
            ax = axes[r, c]
            img = frames[r*cols + c][:3].permute(1,2,0).numpy()  # RGB only
            #disp = (img - img.min())/(img.max() - img.min() + 1e-8)
            disp = np.clip(img, 0.0, 1.0)
            ax.imshow(disp)
            ax.axis('off')
            if r == 0:
                ax.set_title(f"In {c}")
            else:
                ax.set_title(f"Pred {r}·{c}")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

     # optionally save GIF of sequential frames

    if make_gif:
        gif_frames = []
        normalized_gif_frames = []
        first_im = frames[0][:3].permute(1,2,0).numpy()
        first_frame_max = first_im.max()
        first_frame_min = first_im.min()
        terrain = frames[0][-1].numpy()
        # quantize terrain to 16-bit PNG
        quantized_terrain = (terrain * 65535.0).round().astype(np.uint16)
        imageio.imwrite(filename[:-4] + "_terrain.png", quantized_terrain)

        for frame in frames:
            img = frame[:3].permute(1, 2, 0).numpy()
            disp_norm = (img - first_frame_min) / (first_frame_max-first_frame_min + 1e-8)
            disp = np.clip(img, 0.0, 1.0)
            uint8_img = (disp * 255).astype(np.uint8)
            uint8_img_norm = (disp_norm * 255).astype(np.uint8)
            gif_frames.append(uint8_img)
            normalized_gif_frames.append(uint8_img_norm)
        # Save as GIF
        print(f'total frames: {len(gif_frames)}')
        imageio.mimsave(filename[:-4] + ".gif", gif_frames, fps=60)
        imageio.mimsave(filename[:-4] + "_normalized.gif", normalized_gif_frames, fps=60)


def eval_cascade_checkpoint(test_loader_idx, plotTitle, config, checkpoint_path, filename, autoregressive=False, autoregressive_iters=10, exr_dir="final_evals_norm/comparison"):
    ckpnt = torch.load(checkpoint_path, weights_only=True)


    convlstm_cfg = {
            'in_ch':           4,
            'hid_chs':         [config['hidden_channels'],config['hidden_channels']],
            'ks':              [config['kernel_size'],config['kernel_size']],
            'input_length':    5,
            'output_length':   5,
            }
    refiner_cfg = {
        'in_ch':        8,             # 4 rough + 4 last_input
        'out_ch':       4,
        'img_size':     (256, 256),
        'patch_size':   config['patch_size'],
        'embed_dim':    config['embed_dim'],
        'depth':        config['depth'],
        'num_heads':    config['num_heads'],
        'mlp_ratio':    config['mlp_ratio'],
        'dropout':      config['refiner_dropout'],
        }
    
    model = CascadePredictor(convlstm_cfg, refiner_cfg, config['skip_alpha_channel']).to(config['device'])
    model.load_state_dict(ckpnt['model_state'])
    

    test_dataset = PTSequenceDataset(pt_dir=config['data_dir'], compute_norm=False, transform=None)

    shuffle = False
    if autoregressive:
        filename = filename[:-4] + "_autoregressive.png"
        #shuffle = True

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=config['num_workers'], pin_memory=True)
    print(f"Test loader size: {len(test_loader)}")

    if autoregressive:
        plot_autoregressive(test_loader_idx, model, test_loader, config['device'], num_iters=autoregressive_iters, filename=filename, plotName=plotTitle, exr_dir=exr_dir)
    else:
        test_and_log(model, test_loader, config['device'], config, threshold=config['accuracy_threshold'], log=False, filename=filename, plotName=plotTitle, test_loader_idx=test_loader_idx)

    #test_and_log(model, test_loader, config['device'], threshold=config['accuracy_threshold'], log=False, filename=filename)


def eval_convlstm_checkpoint(test_loader_idx, plotTitle, config, checkpoint_path, filename, autoregressive=False, skip_alpha=False, autoregressive_iters=10, exr_dir="final_evals_norm/comparison"):
    ckpnt = torch.load(checkpoint_path, weights_only=True)

    if skip_alpha:
        model = ConvLSTMSeq2SeqNoAlpha(
            in_ch         = 4,
            hid_chs       = [config['hidden_channels'],config['hidden_channels']],
            ks            = [config['kernel_size'],config['kernel_size']],
            input_length  = config['input_length'],
            output_length = config['output_length']
        ).to(config['device'])
    else:
        model = ConvLSTMSeq2Seq(
            in_ch         = 4,
            hid_chs       = [config['hidden_channels'],config['hidden_channels']],
            ks            = [config['kernel_size'],config['kernel_size']],
            input_length  = config['input_length'],
            output_length = config['output_length'],
            inference=True
        ).to(config['device'])

    model.load_state_dict(ckpnt['model_state'])
    test_dataset = PTSequenceDataset(pt_dir=config['data_dir'], compute_norm=False, transform=None)

    shuffle = False
    if autoregressive:
        filename = filename[:-4] + "_autoregressive.png"
        #shuffle = True

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=config['num_workers'], pin_memory=True)
    print(f"Test loader size: {len(test_loader)}")


    if autoregressive:
        plot_autoregressive(test_loader_idx, model, test_loader, config['device'], num_iters=autoregressive_iters, filename=filename, plotName=plotTitle, exr_dir=exr_dir)
    else:
        test_and_log(model, test_loader, config['device'], config, threshold=config['accuracy_threshold'], log=False, filename=filename, plotName=plotTitle, test_loader_idx=test_loader_idx)


def final_evals(test_loader_idx, plotNamePostFix, config, im_folder, test_data_path, autoregressive, autoregressive_iters=300):
    #-----------------------------NOT SKIPPING ALPHA CHANNEL----------------------------------
    config['data_dir'] = test_data_path
    config['skip_alpha_channel'] = False
    config['kernel_size'] = 5
    config['hidden_channels'] = 64
    print('-----------------------------------------------------------------------------')
    
    # PLEASANT-LAKE-95 ID: 5y6871ie
    # Kernel size: 5, Hidden channels: 64
    # 692 Samples
    print(f'Pleasant Lake 95 ID: 5y6871ie')
    eval_convlstm_checkpoint(test_loader_idx, f"Pleasant Lake {plotNamePostFix}", config, "wandb/run-20250427_214853-5y6871ie/files/final.pt", f"{im_folder}/k5ch64-pleasant-lake-best-val-acc.png", autoregressive=autoregressive, skip_alpha=False, autoregressive_iters=autoregressive_iters, exr_dir=f"EXR_COMPS/pleasant/{plotNamePostFix}")
    print('-----------------------------------------------------------------------------')
    
    # EAGER-GORGE-107 ID: 18ldmrto
    # Kernel size: 5, Hidden channels: 64
    # 1500 Samples
    print(f'Eager Gorge 107 ID: 18ldmrto')
    eval_convlstm_checkpoint(test_loader_idx, f"Eager Gorge {plotNamePostFix}", config, "wandb/run-20250428_230054-18ldmrto/files/best_val_accuracy_model.pt", f"{im_folder}/k5ch64-eager-gorge-best-val-acc.png", autoregressive=autoregressive, skip_alpha=False, autoregressive_iters=autoregressive_iters, exr_dir=f"EXR_COMPS/gorge/{plotNamePostFix}")
    print('-----------------------------------------------------------------------------')

    # SCRUFFY-LOOKING-EMPEROR-144 ID: 99ivze5y
    # Kernel size: 5, Hidden channels: 64
    # 1500 Samples
    # DELTAS ONLY
    print(f'Scruffy Looking Emperor 144 ID: 99ivze5y')
    eval_convlstm_checkpoint(test_loader_idx, f"Scruffy Looking Emperor {plotNamePostFix}", config, "wandb/run-20250503_220848-99ivze5y/files/best_val_accuracy_model.pt", f"{im_folder}/k5ch64-scruffy-looking-emp-best-val-acc.png", autoregressive=autoregressive, autoregressive_iters=autoregressive_iters, exr_dir=f"EXR_COMPS/scruffyemp/{plotNamePostFix}")

    print('-----------------------------------------------------------------------------')
    
    
    #-----------------------------SKIPPING ALPHA CHANNEL----------------------------------
    # SKINNY-PLANT-142 ID: 8sipbxo9
    # Kernel size: 5, Hidden channels: 64
    # 1500 Samples
    # Used ViT refiner
    # Skipped alpha only on convlstm
    config['use_refiner'] = True
    config['skip_alpha_channel'] = True
    config['convlstm_only_skip'] = True
    print(f'Skinny Plant 142 ID: 8sipbxo9')
    eval_cascade_checkpoint(test_loader_idx, f"Skinny Plant {plotNamePostFix}", config, "wandb/run-20250502_180848-8sipbxo9/files/best_val_accuracy_model.pt", f"{im_folder}/casc-skinny-plant-best-val-acc.png", autoregressive=autoregressive, autoregressive_iters=autoregressive_iters, exr_dir=f"EXR_COMPS/plant/{plotNamePostFix}")
    config['use_refiner'] = False
    config['convlstm_only_skip'] = False
    print('-----------------------------------------------------------------------------')

    # JUMPING-FIREFLY-121 ID: 9ng657y4
    # Kernel size: 5, Hidden channels: 64
    # 1500 Samples
    print(f'Jumping Firefly 121 ID: 9ng657y4')
    eval_convlstm_checkpoint(test_loader_idx, f"Jumping Firefly {plotNamePostFix}", config, "wandb/run-20250430_105837-9ng657y4/files/best_val_accuracy_model.pt", f"{im_folder}/k5ch64-jumping-firefly-best-val-acc.png", autoregressive=autoregressive, skip_alpha=True, autoregressive_iters=autoregressive_iters, exr_dir=f"EXR_COMPS/firefly/{plotNamePostFix}")

    #print('-----------------------------------------------------------------------------')

    


def final_evals_sweep(subfolder, config, iters, test_loader_idx):
    final_evals(test_loader_idx, "Normal", config, f"final_evals_norm/single_pred/{subfolder}", "data/Test/test_pt_norm", autoregressive=False)
    final_evals(test_loader_idx, "Normal", config, f"final_evals_norm/autoregressive/{subfolder}", "data/Test/test_pt_norm", autoregressive=True, autoregressive_iters=iters)


    final_evals(test_loader_idx, "Hard", config, f"final_evals_hard/single_pred/{subfolder}", "data/Test/test_pt_hard", autoregressive=False)
    final_evals(test_loader_idx, "Hard", config, f"final_evals_hard/autoregressive/{subfolder}", "data/Test/test_pt_hard", autoregressive=True, autoregressive_iters=iters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)


    final_evals(65, "norm", config, f"final_evals_norm/autoregressive/tmp", "data/Test/test_pt_norm", autoregressive=True, autoregressive_iters=150)
    final_evals(65, "hard", config, f"final_evals_hard/autoregressive/tmp", "data/Test/test_pt_hard", autoregressive=True, autoregressive_iters=150)
    """
    test_loader_idx = 3
    final_evals_sweep("/1", config, 150, test_loader_idx)
    test_loader_idx = 65
    final_evals_sweep("/2", config, 150, test_loader_idx)
    test_loader_idx = 20
    final_evals_sweep("/3", config, 150, test_loader_idx)
    test_loader_idx = 39
    final_evals_sweep("/4", config, 150, test_loader_idx)
    test_loader_idx = 77
    final_evals_sweep("/5", config, 150, test_loader_idx)
    """
    

    





    #eval_cascade_checkpoint(config, "wandb/run-20250502_180848-8sipbxo9/files/best_val_loss_model.pt", "cascade_final.png", autoregressive=True, autoregressive_iters=100)
    #eval_convlstm_checkpoint(config, "pretrained/archive/best_val_accuracy_model.pt", "convlstm_gorge_acc_regressive_TERR_300.png", autoregressive=True, skip_alpha=False, autoregressive_iters=300)
    #eval_convlstm_checkpoint(config, "pretrained/best_val_loss_model.pt", "convlstm_firefly_loss_autoregressive_300_2.png", autoregressive=True, skip_alpha=True, autoregressive_iters=300)
    #eval_convlstm_checkpoint(config, "pretrained/last.pt", "convlstm_firefly_final.png", autoregressive=False, skip_alpha=True)
    #eval_convlstm_checkpoint(config, "pretrained/last.pt", "convlstm_firefly_final_regressive.png", autoregressive=True, skip_alpha=True)

    

if __name__ == "__main__":
    main()