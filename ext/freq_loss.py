"""
Frequency-domain losses for SIDL dirty-lens restoration (config C / E).

NAFNet's ImageRestorationModel builds a single `pixel_opt` loss via
    getattr(basicsr.models.losses, <type>)
so we expose new loss classes from the basicsr.models.losses package namespace.

Injection (done by the Colab SETUP cell):
  1. copy this file ->  NAFNet/basicsr/models/losses/freq_loss.py
  2. append          ->  "from .freq_loss import *"  to losses/__init__.py

Then use in YAML:
  pixel_opt:
    type: PSNRFFTLoss
    loss_weight: 1
    reduction: mean
    fft_weight: 0.05      # lambda for the frequency term (tune in Table-2 sweep)

Design notes
------------
* The base term reuses NAFNet's PSNRLoss (log-MSE), so config C stays directly
  comparable to the config-A baseline (same base loss + an added FFT term).
* The FFT term is an L1 distance between the 2-D real FFTs of prediction and
  target. Using the *complex* difference captures both amplitude and phase
  errors, which is what blur / low-frequency dirty-lens degradations distort.
"""

import torch
import torch.nn as nn

from .losses import PSNRLoss


def _fft_l1(pred, target):
    """L1 distance between 2-D FFTs (complex) of pred and target.

    pred, target: (N, C, H, W) in [0, 1]. norm='ortho' keeps the scale stable
    and roughly comparable across patch sizes.
    """
    pf = torch.fft.rfft2(pred, norm='ortho')
    tf = torch.fft.rfft2(target, norm='ortho')
    return (pf - tf).abs().mean()


class FFTLoss(nn.Module):
    """Pure frequency-domain L1 loss (for analysis / standalone use)."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * _fft_l1(pred, target)


class PSNRFFTLoss(nn.Module):
    """Config-C loss: PSNRLoss (spatial) + fft_weight * FFT-L1 (frequency).

    Keeps the exact PSNRLoss base of the baseline and adds a frequency term,
    so the only controlled change vs. config A is the added FFT supervision.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False,
                 fft_weight=0.05):
        super().__init__()
        self.psnr = PSNRLoss(loss_weight=loss_weight, reduction=reduction,
                             toY=toY)
        self.fft_weight = fft_weight

    def forward(self, pred, target):
        return self.psnr(pred, target) + self.fft_weight * _fft_l1(pred, target)


class PSNRSSIMLoss(nn.Module):
    """Optional variant for the Table-2 loss-type ablation: PSNR + (1-SSIM).

    Lightweight single-scale SSIM (Gaussian window) so it has no extra deps.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False,
                 ssim_weight=0.1, window_size=11):
        super().__init__()
        self.psnr = PSNRLoss(loss_weight=loss_weight, reduction=reduction,
                             toY=toY)
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.register_buffer('window', self._make_window(window_size))

    @staticmethod
    def _gaussian(win, sigma=1.5):
        coords = torch.arange(win).float() - win // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _make_window(self, win):
        g = self._gaussian(win)
        w2 = g[:, None] @ g[None, :]
        return w2[None, None]  # (1,1,win,win)

    def _ssim(self, pred, target):
        c = pred.shape[1]
        w = self.window.to(pred.dtype).to(pred.device).expand(c, 1, self.window_size, self.window_size)
        pad = self.window_size // 2
        mu1 = torch.nn.functional.conv2d(pred, w, padding=pad, groups=c)
        mu2 = torch.nn.functional.conv2d(target, w, padding=pad, groups=c)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = torch.nn.functional.conv2d(pred * pred, w, padding=pad, groups=c) - mu1_sq
        s2 = torch.nn.functional.conv2d(target * target, w, padding=pad, groups=c) - mu2_sq
        s12 = torch.nn.functional.conv2d(pred * target, w, padding=pad, groups=c) - mu12
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        return self.psnr(pred, target) + self.ssim_weight * (1.0 - self._ssim(pred, target))
