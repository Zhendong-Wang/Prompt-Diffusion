from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from annotator.util import HWC3


class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        prompt_option: str = 'edit',
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

        self.prompt_option = prompt_option

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        example_seed = seeds[torch.randint(0, len(seeds), ()).item()]
        seed = seeds[torch.randint(0, len(seeds), ()).item()]

        # Load text prompt
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            prompt = prompt['output']

        # Load input and output images; shape -> h w c
        image_0 = Image.open(propt_dir.joinpath(f"{example_seed}_0.jpg"))
        image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = 2 * torch.tensor(np.array(image_0)).float() / 255. - 1
        image_1 = 2 * torch.tensor(np.array(image_1)).float() / 255. - 1

        # Load Controls; shape -> h w c
        task = np.random.choice(['inv_seg', 'inv_depth', 'inv_hed', 'seg', 'depth', 'hed'])
        txt_log = task
        if task == 'inv_seg':
            image_seg = Image.open(propt_dir.joinpath(f"{example_seed}_0_seg.jpg"))
            image_seg = image_seg.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_seg = 2 * torch.tensor(np.array(image_seg)).float() / 255. - 1

            example_pair = torch.cat((image_seg, image_0), dim=2) # h w c

            image_query = Image.open(propt_dir.joinpath(f"{seed}_1_seg.jpg"))
            image_query = image_query.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_query = 2 * torch.tensor(np.array(image_query)).float() / 255. - 1

            image_target = image_1

        elif task == 'seg':
            image_seg = Image.open(propt_dir.joinpath(f"{example_seed}_0_seg.jpg"))
            image_seg = image_seg.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_seg = 2 *  torch.tensor(np.array(image_seg)).float() / 255. - 1

            example_pair = torch.cat((image_0, image_seg), dim=2)  # h w c
            image_query = image_1
            prompt = 'segmentation map'

            image_target = Image.open(propt_dir.joinpath(f"{seed}_1_seg.jpg"))
            image_target = image_target.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_target = 2 * torch.tensor(np.array(image_target)).float() / 255. - 1

        elif task == 'inv_depth':
            image_depth = Image.open(propt_dir.joinpath(f"{example_seed}_0_depth.jpg"))
            image_depth = image_depth.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_depth = 2 * torch.tensor(HWC3(np.array(image_depth))).float() / 255. - 1

            example_pair = torch.cat((image_depth, image_0), dim=2)  # h w c

            image_query = Image.open(propt_dir.joinpath(f"{seed}_1_depth.jpg"))
            image_query = image_query.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_query = 2 * torch.tensor(HWC3(np.array(image_query))).float() / 255. - 1

            image_target = image_1

        elif task == 'depth':
            image_depth = Image.open(propt_dir.joinpath(f"{example_seed}_0_depth.jpg"))
            image_depth = image_depth.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_depth = 2 * torch.tensor(HWC3(np.array(image_depth))).float() / 255. - 1

            example_pair = torch.cat((image_0, image_depth), dim=2)  # h w c
            image_query = image_1
            prompt = 'depth map'

            image_target = Image.open(propt_dir.joinpath(f"{seed}_1_depth.jpg"))
            image_target = image_target.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_target = 2 * torch.tensor(HWC3(np.array(image_target))).float() / 255. - 1

        elif task == 'inv_hed':

            image_hed = Image.open(propt_dir.joinpath(f"{example_seed}_0_hed.jpg"))
            image_hed = image_hed.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_hed = 2 * torch.tensor(HWC3(np.array(image_hed))).float() / 255. - 1

            example_pair = torch.cat((image_hed, image_0), dim=2)  # h w c

            image_query = Image.open(propt_dir.joinpath(f"{seed}_1_hed.jpg"))
            image_query = image_query.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_query = 2 * torch.tensor(HWC3(np.array(image_query))).float() / 255. - 1

            image_target = image_1

        elif task == 'hed':

            image_hed = Image.open(propt_dir.joinpath(f"{example_seed}_0_hed.jpg"))
            image_hed = image_hed.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_hed = 2 * torch.tensor(HWC3(np.array(image_hed))).float() / 255. - 1

            example_pair = torch.cat((image_0, image_hed), dim=2)  # h w c
            image_query = image_1
            prompt = 'hed map'

            image_target = Image.open(propt_dir.joinpath(f"{seed}_1_hed.jpg"))
            image_target = image_target.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_target = 2 * torch.tensor(HWC3(np.array(image_target))).float() / 255. - 1

        return dict(jpg=image_target, txt=prompt, query=image_query, example_pair=example_pair, txt_log=txt_log)

