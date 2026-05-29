"""
SIDL all-in-one (multi-task) paired dataset for NAFNet / BasicSR.

Reads degraded(input) / clean(target) pairs from MULTIPLE contamination-type
folders at once (e.g. finger, dust, water, scratch) and trains a single
"all-in-one" restoration model.

Each sample additionally carries a `degradation_type` integer label, which is
ignored by the plain baseline model but consumed later by the
degradation-aware (FiLM) variant.

Drop this file into  NAFNet/basicsr/data/  --- any file ending with
`_dataset.py` is auto-imported by basicsr/data/__init__.py, so the
`@DATASET_REGISTRY.register()` below makes `SIDLMultiTaskDataset` usable via
    type: SIDLMultiTaskDataset
in the YAML config.

Author: course project (SIDL dirty-lens restoration)
"""

import os
import os.path as osp
import random

import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor

# NOTE: NAFNet's basicsr does NOT use a DATASET_REGISTRY. Datasets are looked
# up by class name via getattr() over auto-imported `*_dataset.py` modules.
# So we just define the class at module level with the right name; no decorator.


# Canonical ordering of degradation types -> integer id.
# Keep this fixed across train / val / test and all experiments.
TYPE_TO_ID = {
    'clean': 0,
    'finger': 1,
    'dust': 2,
    'water': 3,
    'scratch': 4,
    'mixed': 5,
}

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
            '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF', '.TIFF')


def _list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(IMG_EXTS)])


def _build_pairs(task_dirs):
    """task_dirs: list of dicts {type, lq_root, gt_root}.

    Returns list of dicts {lq_path, gt_path, type_id, type_name}.
    Pairs input<->target by sorted filename within each task folder.
    """
    pairs = []
    for td in task_dirs:
        tname = td['type']
        lq_root, gt_root = td['lq_root'], td['gt_root']
        assert osp.isdir(lq_root), f'LQ folder not found: {lq_root}'
        assert osp.isdir(gt_root), f'GT folder not found: {gt_root}'
        lq_files = _list_images(lq_root)
        gt_files = _list_images(gt_root)
        assert len(lq_files) == len(gt_files), (
            f'[{tname}] #input({len(lq_files)}) != #target({len(gt_files)}) '
            f'in\n  {lq_root}\n  {gt_root}')
        for lqf, gtf in zip(lq_files, gt_files):
            pairs.append({
                'lq_path': osp.join(lq_root, lqf),
                'gt_path': osp.join(gt_root, gtf),
                'type_id': TYPE_TO_ID.get(tname, -1),
                'type_name': tname,
            })
    return pairs


class SIDLMultiTaskDataset(data.Dataset):
    """All-in-one paired dataset over several contamination types.

    Required YAML keys (under datasets.train / datasets.val):
        dataroot:   base dir that contains  <type>/input  and  <type>/target
        tasks:      list of type names, e.g. [finger, dust, water, scratch]
        (optional, validation) difficulty: easy|medium|hard  -- if set, looks
                    for  <type>/<difficulty>/input|target  instead.

    Training-only keys:
        gt_size, use_flip (use_hflip), use_rot
        synth_aug: false   # placeholder for config-B synthetic dirty aug

    Common:
        io_backend: {type: disk}
        scale: 1
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.scale = opt.get('scale', 1)
        self.gt_size = opt.get('gt_size', None)

        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})
        self.file_client = None

        dataroot = opt['dataroot']
        tasks = opt['tasks']
        difficulty = opt.get('difficulty', None)

        task_dirs = []
        for t in tasks:
            if difficulty:  # val / test layout: <type>/<difficulty>/{input,target}
                base = osp.join(dataroot, t, difficulty)
            else:           # train layout: <type>/{input,target}
                base = osp.join(dataroot, t)
            task_dirs.append({
                'type': t,
                'lq_root': osp.join(base, 'input'),
                'gt_root': osp.join(base, 'target'),
            })

        self.pairs = _build_pairs(task_dirs)
        assert len(self.pairs) > 0, f'No image pairs found under {dataroot} for tasks {tasks}'

        # synthetic dirty augmentation flag (used by config B / E)
        self.synth_aug = opt.get('synth_aug', False)

    def _maybe_synth_aug(self, lq, gt):
        """Lightweight synthetic dirty-lens augmentation applied on top of the
        real degraded input. Disabled unless synth_aug=True in the config.

        These are cheap, label-preserving perturbations that mimic extra lens
        contamination (soft blobs / local haze / faint scratches). Kept simple
        on purpose; tune in config-B experiments.
        """
        if not self.synth_aug:
            return lq
        h, w = lq.shape[:2]
        out = lq.copy()
        r = random.random
        # 1) soft circular haze blobs (water/dust-like)
        if r() < 0.5:
            n = random.randint(1, 3)
            for _ in range(n):
                cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
                rad = random.randint(max(2, w // 16), max(3, w // 6))
                yy, xx = np.ogrid[:h, :w]
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad * rad
                alpha = random.uniform(0.1, 0.35)
                out[mask] = (1 - alpha) * out[mask] + alpha * 1.0
        # 2) faint straight scratch lines
        if r() < 0.3:
            n = random.randint(1, 2)
            for _ in range(n):
                x0, y0 = random.randint(0, w - 1), random.randint(0, h - 1)
                x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
                steps = max(abs(x1 - x0), abs(y1 - y0), 1)
                xs = np.linspace(x0, x1, steps).astype(int).clip(0, w - 1)
                ys = np.linspace(y0, y1, steps).astype(int).clip(0, h - 1)
                val = random.uniform(0.6, 1.0)
                out[ys, xs] = val
        return np.clip(out, 0.0, 1.0)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        item = self.pairs[index]
        gt_path, lq_path = item['gt_path'], item['lq_path']

        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

        if self.phase == 'train':
            gt_size = self.gt_size
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size,
                                                self.scale, gt_path)
            img_lq = self._maybe_synth_aug(img_lq, img_gt)
            img_gt, img_lq = augment(
                [img_gt, img_lq],
                self.opt.get('use_flip', self.opt.get('use_hflip', False)),
                self.opt.get('use_rot', False))

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'degradation_type': torch.tensor(item['type_id'], dtype=torch.long),
        }

    def __len__(self):
        return len(self.pairs)
