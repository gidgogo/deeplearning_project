"""
Degradation-aware NAFNet via FiLM conditioning (config D / E).

A degradation-type id -> embedding -> small MLP -> (gamma, beta) modulates the
intro (first-layer) features:   x <- x * (1 + gamma) + beta.

The type id is injected through set_deg(), which SIDLCondModel.feed_data calls
each step, so the standard NAFNet forward path net_g(lq) stays untouched.
The FiLM head is zero-initialised, so at iter 0 the model is identical to the
plain baseline (gamma=beta=0) and conditioning is learned from there.

Drop this file into  NAFNet/basicsr/models/archs/  (auto-imported as *_arch.py).
Use in YAML:
  network_g:
    type: NAFNetFiLMLocal
    width: 32
    enc_blk_nums: [1, 1, 1, 8]
    middle_blk_num: 1
    dec_blk_nums: [1, 1, 1, 1]
    num_types: 6
    emb_dim: 64
"""

import torch
import torch.nn as nn

from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.models.archs.local_arch import Local_Base


class NAFNetFiLM(NAFNet):
    def __init__(self, *args, num_types=6, emb_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.intro.out_channels
        self.deg_embed = nn.Embedding(num_types, emb_dim)
        self.film = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True),
            nn.Linear(emb_dim, 2 * width),
        )
        # zero-init -> FiLM starts as identity (gamma=0, beta=0)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self._deg = None

    def set_deg(self, deg):
        """deg: LongTensor (B,) of degradation-type ids (see TYPE_TO_ID)."""
        self._deg = deg

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        if self._deg is not None:
            deg = self._deg
            if deg.shape[0] != x.shape[0]:               # safety broadcast
                deg = deg[:1].expand(x.shape[0])
            e = self.deg_embed(deg.to(x.device).long())  # (B, emb_dim)
            gamma, beta = self.film(e).chunk(2, dim=1)    # (B, width) each
            x = x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]


class NAFNetFiLMLocal(Local_Base, NAFNetFiLM):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetFiLM.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
