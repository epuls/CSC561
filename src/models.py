import torch
import torch.nn as nn


class ResBlock2(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        pad = k // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=pad)

    def forward(self, x, h, c):
        # x: (B,in_ch,H,W), h,c: (B,hid_ch,H,W)
        combined = torch.cat([x, h], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(self.conv(combined), 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMSeq2Seq(nn.Module):
    def __init__(self, in_ch=4, hid_chs=[64,64], ks=[3,3], input_length=5, output_length=5, inference=False):
        super().__init__()
        assert len(hid_chs) == len(ks)
        # May need to comment this out when loading older models
        self.skip_alpha = False
        self.inference = inference
        self.cells = nn.ModuleList()
        ch = in_ch
        for h, k in zip(hid_chs, ks):
            self.cells.append(ConvLSTMCell(ch, h, k))
            ch = h
        self.output_length = output_length
        self.conv_out = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T_in, C, H, W = x.size()
        alpha_out = x[:, :, -1, ...] 
        # init h,c
        hs = [torch.zeros(B, cell.hid_ch, H, W, device=x.device) for cell in self.cells]
        cs = [torch.zeros_like(h) for h in hs]
        # encoder
        for t in range(T_in):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(inp, hs[i], cs[i])
                inp = hs[i]
        # decoder (autoregressive)
        outputs = []
        prev = x[:, -1]
        for _ in range(self.output_length):
            inp = prev
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(inp, hs[i], cs[i])
                inp = hs[i]
            out = self.conv_out(inp)
            outputs.append(out)
            prev = out
        
        final = torch.stack(outputs, dim=1)  # (B, T_out, C, H, W)
        if self.inference:
            # overwrite the last channel with alpha_out
            final[:, :, -1, ...] = alpha_out

        return final  # (B, T_out, C, H, W)


class BiConvLSTMSeq2Seq(nn.Module):
    def __init__(self,
                 in_ch=4,
                 hid_chs=[64, 64],
                 ks=[3, 3],
                 input_length=5,
                 output_length=5,
                 merge_mode='sum'  # or 'cat'
                 ):
        super().__init__()
        assert len(hid_chs) == len(ks), "hid_chs and ks must be same length"
        self.input_length = input_length
        self.output_length = output_length
        self.merge_mode = merge_mode

        # --- build two encoder stacks: forward and backward ---
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        ch = in_ch
        for h, k in zip(hid_chs, ks):
            self.fwd_cells.append(ConvLSTMCell(ch, h, k))
            self.bwd_cells.append(ConvLSTMCell(ch, h, k))
            ch = h

        # --- decoder uses the forward stack’s weights ---
        self.decoder_cells = self.fwd_cells

        # if concatenating states, hidden‐dim doubles after merge
        if merge_mode == 'cat':
            ch = hid_chs[-1] * 2
        else:
            ch = hid_chs[-1]

        # final conv to map back to in_ch channels
        self.conv_out = nn.Conv2d(ch, in_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T_in, C, H, W = x.size()

        # init forward states
        hs_f = [torch.zeros(B, cell.hid_ch, H, W, device=x.device)
                for cell in self.fwd_cells]
        cs_f = [h.clone() for h in hs_f]

        # init backward states
        hs_b = [torch.zeros(B, cell.hid_ch, H, W, device=x.device)
                for cell in self.bwd_cells]
        cs_b = [h.clone() for h in hs_b]

        # --- forward encoding pass ---
        for t in range(T_in):
            inp = x[:, t]
            for i, cell in enumerate(self.fwd_cells):
                hs_f[i], cs_f[i] = cell(inp, hs_f[i], cs_f[i])
                inp = hs_f[i]

        # --- backward encoding pass ---
        for t in reversed(range(T_in)):
            inp = x[:, t]
            for i, cell in enumerate(self.bwd_cells):
                hs_b[i], cs_b[i] = cell(inp, hs_b[i], cs_b[i])
                inp = hs_b[i]

        # --- merge final states ---
        hs, cs = [], []
        for hf, hb, cf, cb in zip(hs_f, hs_b, cs_f, cs_b):
            if self.merge_mode == 'sum':
                hs.append(hf + hb)
                cs.append(cf + cb)
            else:  # 'cat'
                hs.append(torch.cat([hf, hb], dim=1))
                cs.append(torch.cat([cf, cb], dim=1))

        # --- autoregressive decoding ---
        outputs = []
        prev = x[:, -1]  # start from last input frame
        for _ in range(self.output_length):
            inp = prev
            for i, cell in enumerate(self.decoder_cells):
                hs[i], cs[i] = cell(inp, hs[i], cs[i])
                inp = hs[i]
            out = self.conv_out(inp)
            outputs.append(out)
            prev = out

        # (B, T_out, C, H, W)
        return torch.stack(outputs, dim=1)

class ConvLSTMSeq2SeqNoAlpha(nn.Module):
    def __init__(
        self,
        in_ch=4,            # input channels (R,G,B,Height)
        hid_chs=[64,64],
        ks=[3,3],
        input_length=5,
        output_length=5,
        inference=False
    ):
        super().__init__()
        assert len(hid_chs) == len(ks)
        self.skip_alpha = True
        self.input_length  = input_length
        self.output_length = output_length
        self.inference = inference
        
        # how many channels we actually predict (drop the Height)
        self.predict_ch = in_ch - 1  # 4→3
        
        # build the ConvLSTM stack
        self.cells = nn.ModuleList()
        ch = in_ch
        for h, k in zip(hid_chs, ks):
            self.cells.append(ConvLSTMCell(ch, h, k))
            ch = h
        
        # final conv to map last hidden → only 3 channels
        self.conv_out = nn.Conv2d(ch, self.predict_ch, kernel_size=1)
        print(f'ConvLSTMSeq2SeqNoAlpha Inference Only: {self.inference}')

    def forward(self, x):
        """
        x: (B, T_in, 4, H, W)  # channels = [R,G,B,Height]
        returns: (B, T_out, 4, H, W)  # we re-attach Height
        """
        B, T_in, C, H, W = x.size()
        assert C == self.predict_ch + 1
        
        # 1) pull off the static Height map once
        #    (we assume terrain is the same at all timesteps)
        height = x[:, 0, 3:4, ...]           # → (B,1,H,W)
        
        # 2) run the usual encoder
        hs = [torch.zeros(B, c.hid_ch, H, W, device=x.device)
              for c in self.cells]
        cs = [h.clone() for h in hs]
        
        for t in range(T_in):
            inp = x[:, t]                   # (B,4,H,W)
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(inp, hs[i], cs[i])
                inp = hs[i]
        
        # 3) autoregressive decode, but only predict 3-ch → then re-attach Height
        outputs = []
        # start from last RGB of the encoder inputs
        prev_rgb = x[:, -1, : self.predict_ch, ...]  # (B,3,H,W)
        prev = torch.cat([prev_rgb, height], dim=1)  # back to (B,4,H,W)
        
        for _ in range(self.output_length):
            inp = prev
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(inp, hs[i], cs[i])
                inp = hs[i]
            
            rgb_pred = self.conv_out(inp)                 # (B,3,H,W)
            full_out = torch.cat([rgb_pred, height], dim=1)
            outputs.append(full_out)
            
            prev = full_out
        
        return torch.stack(outputs, dim=1)  # (B, T_out, 4, H, W)

class SequenceDiscriminator(nn.Module):
    def __init__(self, in_ch=4, T=5, base_ch=32):
        super().__init__()
        # 3D conv: in_ch × T channels collapsed into one volume
        self.net = nn.Sequential(
            nn.Conv3d(in_ch,   base_ch, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_ch, base_ch*2, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_ch*2, 1, kernel_size=(3,3,3), padding=(1,1,1))
        )

    def forward(self, seq):
        # seq: (B, T, C, H, W) → reshape to (B, C, T, H, W)
        B, T, C, H, W = seq.size()
        x = seq.permute(0,2,1,3,4)
        out = self.net(x)
        # global average pool
        return out.mean([2,3,4])


class ViTRefiner(nn.Module):
    """
    A Vision-Transformer–based spatial refiner that maps a coarse 4→4 image
    (optionally concatenated with context) back to a refined 4-channel output.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        img_size: tuple[int,int],
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, "Image dims must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.embed_dim = embed_dim
        self.out_ch = out_ch
        self.norm = nn.LayerNorm(embed_dim)

        # 1) Patch‐embedding via conv
        self.patch_embed = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 2) Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim*mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # 4) Linear head to map each token back to a patch of pixels
        #    each token -> out_ch * (patch_size**2)
        self.patch_deembed = nn.Linear(
            embed_dim,
            out_ch * patch_size * patch_size
        )

        # 5) Fold to reassemble patches into an image
        self.fold = nn.Fold(
            output_size=(H, W),
            kernel_size=patch_size,
            stride=patch_size
        )


        self.smooth = nn.Sequential(
            ResBlock2(out_ch, groups=4),
            ResBlock2(out_ch, groups=4),
            
        )

        self.smooth2 = nn.Sequential(
            ResBlock2(out_ch, groups=4)
        )

        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,in_ch,H,W)
        # keep a copy for the skip
        skip = x[:, :self.out_ch]         # assume first out_ch channels are the coarse preds
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_embed
        x = self.norm(x)                  # pre-norm
        x = self.transformer(x)
        x = self.patch_deembed(x).permute(0,2,1)
        x = self.fold(x)                  # (B,out_ch,H,W)
        # residual
        skip2 = x
        x = self.smooth(x)
        x = x + skip2
        skip3 = x
        x = self.smooth2(x)
        x = x + skip3
        return x + skip


class CascadePredictor(nn.Module):
    """
    Stage-wise cascade:
      1) ConvLSTMSeq2Seq produces rough (B, T_out, C, H, W)
      2) ViTRefiner polishes each frame t using:
         - the last K true input frames
         - the first t+1 rough predictions
    """

    def __init__(
        self,
        convlstm_cfg: dict,
        refiner_cfg:  dict,
        skip_alpha: bool = False,
        inference: bool = False,
        history_length: int = None
    ):
        super().__init__()
        self.inference = inference
        # 1) your pretrained (or to‐be‐trained) ConvLSTM
        if not skip_alpha:
            self.convlstm = ConvLSTMSeq2Seq(**convlstm_cfg, inference=inference)
        else: 
            self.convlstm = ConvLSTMSeq2SeqNoAlpha(**convlstm_cfg, inference=inference)

        # 2) how many of the last inputs to feed in
        self.K = convlstm_cfg['input_length']

        # 3) set up the ViTRefiner to expect
        #    in_ch = K*C  +  T_out*C  (we feed all rough frames each time)
        C = convlstm_cfg['in_ch']
        T = convlstm_cfg['output_length']
        refiner_cfg = refiner_cfg.copy()
        refiner_cfg['in_ch'] = (self.K + T) * C
        refiner_cfg['out_ch'] = T * C   # it will spit back the whole refined sequence

        self.refiner = ViTRefiner(**refiner_cfg)

        self.final_conv = nn.Conv2d(
            in_channels=T*C,
            out_channels=T*C,
            kernel_size=1
            )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T_in, C, H, W)
        returns: (B, T_out, C, H, W)
        """
        B, T_in, C, H, W = seq.shape
        T_out = self.convlstm.output_length

        # 1) rough prediction
        rough = self.convlstm(seq)                 # (B, T_out, C, H, W)

        # 2) build the two parts of the refiner input
        # 2a) last K inputs, flattened to (B, K*C, H, W)
        hist = seq[:, -self.K:]                    # (B, K, C, H, W)
        hist = hist.reshape(B, self.K * C, H, W)

        # 2b) all rough frames, flattened to (B, T_out*C, H, W)
        whole_rough = rough.reshape(B, T_out * C, H, W)

        # 3) concatenate once and refine _all_ frames in one pass
        x_in = torch.cat([hist, whole_rough], dim=1)  # (B, (K+T_out)*C, H, W)
        # refiner returns (B, T_out*C, H, W)
        refined = self.refiner(x_in)
        refined = self.final_conv(refined)
        B, _, H, W = refined.shape
        # 4) reshape back to sequence: (B, T_out, C, H, W)
        refined = refined.view(B, T_out, C, H, W)
        return refined