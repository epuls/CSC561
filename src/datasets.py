import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import OpenEXR, Imath
import numpy as np
from sklearn.model_selection import train_test_split

def set_start_method():
    # Use spawn to avoid fork issues in notebooks/scripts
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)



def compute_channel_min_max(pt_dir, batch_size=8, num_workers=4):
    ds = PTSequenceDataset(pt_dir, transform=None)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    # Initialize mins and maxs with extreme values
    channel_min = torch.full((4,), float('inf'))
    channel_max = torch.full((4,), float('-inf'))

    for seq, tgt in loader:
        # seq: (B, T_in, C, H, W); tgt: (B, T_out, C, H, W)
        # merge inputs and targets into one tensor of shape (B, T_in+T_out, C, H, W)
        all_frames = torch.cat([seq, tgt], dim=1)

        # Compute per-batch per-channel min/max:
        # collapse B, T_in+T_out, H, W into one dimension
        # result: (C,)
        batch_min = all_frames.amin(dim=(0,1,3,4))
        batch_max = all_frames.amax(dim=(0,1,3,4))

        channel_min = torch.minimum(channel_min, batch_min)
        channel_max = torch.maximum(channel_max, batch_max)

    return channel_min, channel_max

class NormalizeMinMax:
    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor):
        # mins/maxs: shape (C,)
        self.mins = mins.view(1,1,-1,1,1)   # broadcast over B,T,H,W
        self.maxs = maxs.view(1,1,-1,1,1)
        self.scale = 2.0 / (self.maxs - self.mins)
        self.shift = -1.0 - self.scale * self.mins

    def __call__(self, seq: torch.Tensor, tgt: torch.Tensor):
        # applies to both input and target sequences
        seq_norm = seq * self.scale + self.shift
        tgt_norm = tgt * self.scale + self.shift
        return seq_norm, tgt_norm


def read_exr(path):
    """Load a single RGBA EXR as a (4, H, W) float32 tensor."""
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [np.frombuffer(exr.channel(ch, pt), dtype=np.float32)
                .reshape(H, W)
                for ch in ("R","G","B","A")]
    arr = np.stack(channels, axis=0)
    return torch.from_numpy(arr)

class EXRSequenceDataset(Dataset):
    def __init__(self, root_dir, input_length=5, output_length=5, transform=None):
        """
        Expects files named <seq>_<t>.exr under root_dir/Inputs.
        Groups by <seq>, sorts by t, then splits into
        first input_length frames → input sequence,
        next output_length frames → target sequence.
        """
        self.transform = transform
        inp_paths = sorted(glob.glob(os.path.join(root_dir, "Inputs", "*.exr")))
        # group by sequence index
        seq_dict = {}
        for p in inp_paths:
            fname = os.path.basename(p)
            seq, t = fname.split(".")[0].split("_")
            seq = int(seq); t = int(t)
            seq_dict.setdefault(seq, {})[t] = p
        # build list of (input_list, target_list)
        self.seqs = []
        for seq_idx in sorted(seq_dict):
            frames = seq_dict[seq_idx]
            # ensure we have at least input+output frames
            if max(frames) >= input_length + output_length - 1:
                inp = [frames[i] for i in range(input_length)]
                tgt = [frames[i] for i in range(input_length, input_length+output_length)]
                self.seqs.append((inp, tgt))
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        inp_paths, tgt_paths = self.seqs[idx]
        inp_seq = torch.stack([read_exr(p) for p in inp_paths], dim=0)   # (T_in,4,H,W)
        tgt_seq = torch.stack([read_exr(p) for p in tgt_paths], dim=0)   # (T_out,4,H,W)
        if self.transform:
            inp_seq, tgt_seq = self.transform(inp_seq, tgt_seq)
        return inp_seq, tgt_seq

class PTSequenceDataset(Dataset):
    def __init__(self, pt_dir, compute_norm=False, transform=None):
        """
        Expects files named seq_{i}.pt, each containing {'sequence', 'target'}.
        """
        self.files = sorted(glob.glob(os.path.join(pt_dir, "seq_*.pt")))
        self.transform = transform
        if compute_norm:
            # run once to get global mins/maxs
            mins, maxs = compute_channel_min_max(pt_dir)
            # override to the “fixed” normalizer
            self.transform = NormalizeMinMax(mins, maxs)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        seq, tgt = data['sequence'], data['target']
        if self.transform:
            seq, tgt = self.transform(seq, tgt)
        return seq, tgt

def load_dataset(root_dir, batch_size, train_split=0.8, val_split=0.1,
                 input_length=5, output_length=5, transform=None,
                 print_validation=False, num_workers=0):
    ds = EXRSequenceDataset(root_dir, input_length, output_length, transform)
    n = len(ds)
    idx_train, idx_tmp = train_test_split(range(n), train_size=train_split, shuffle=True, random_state=42)
    val_size = int(n * val_split)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=val_split/(1-train_split), shuffle=True, random_state=42)
    def make_loader(idx_list, shuffle):
        subset = torch.utils.data.Subset(ds, idx_list)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    train_loader = make_loader(idx_train, True)
    val_loader   = make_loader(idx_val,   False)
    test_loader  = make_loader(idx_test,  False)

    if print_validation:
        print(f"Total sequences: {n}")
        for name, loader in [("Train",train_loader),("Val",val_loader),("Test",test_loader)]:
            print(f"{name}: {len(loader.dataset)} seqs, {len(loader)} batches, bs={batch_size}")
            seq, tgt = next(iter(loader))
            print(f"  sample shapes: inp {seq.shape}, tgt {tgt.shape}")

    return train_loader, val_loader, test_loader

def load_pt_dataset(pt_dir, batch_size, train_split=0.8, val_split=0.1,
                    transform=None, print_validation=False, num_workers=0, normalize=True, seed=42):
    
    ds = PTSequenceDataset(pt_dir=pt_dir, transform=transform, compute_norm=normalize)
    n = len(ds)
    idx_train, idx_tmp = train_test_split(range(n), train_size=train_split, shuffle=True, random_state=seed)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, shuffle=True, random_state=seed)

    def make_loader(idx_list, shuffle):
        subset = torch.utils.data.Subset(ds, idx_list)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    
    train_loader = make_loader(idx_train, True)
    val_loader   = make_loader(idx_val,   False)
    test_loader  = make_loader(idx_test,  False)

    if print_validation:
        print(f"[PT] Total samples: {n}")
        for name, loader in [("Train",train_loader),("Val",val_loader),("Test",test_loader)]:
            print(f"{name}: {len(loader.dataset)} seqs, {len(loader)} batches, bs={batch_size}")
            seq, tgt = next(iter(loader))
            print(f"  sample shapes: inp {seq.shape}, tgt {tgt.shape}")

    return train_loader, val_loader, test_loader
