import torch

import numpy as np
import argparse, os, sys

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

from PIL import Image
import einops
from edit_dataset import EditDatasetEval

from . import distributed as dist


def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--image_dir", type=str, default='image_log', help="path to saving original/generated images")
    parser.add_argument("--ckpt", type=str, help="path to the saved ckpt")
    parser.add_argument("--task", type=str, help="task to test")
    return parser


@torch.no_grad()
def main():
    dist.init()

    sys.path.append(os.getcwd())
    opt, _ = get_parser().parse_known_args()

    # Load Models and Sampler
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(opt.ckpt, location='cpu'))
    ddim_sampler = DDIMSampler(model)
    device = torch.device('cuda')
    model = model.to(device)

    # Data
    dataset = EditDatasetEval(path='/v-zhendwang/datasets/IP2P/clip-filtered-dataset',
                              split='test', prompt_option='output', task=opt.task)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1)

    # Sampling Parameters
    ddim_steps = 100
    eta = 0.0
    scale = 9.0

    a_prompt = ', best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    batch_ids = np.arange(100)[dist.get_rank() :: dist.get_world_size()]
    for i, batch in enumerate(dataloader):
        if i in batch_ids:
            aug_prompt = batch['txt']
            assert isinstance(aug_prompt, list)
            aug_prompt = [p + a_prompt for p in aug_prompt]
            batch['txt'] = aug_prompt

            # Generation
            _, cond = model.get_input(batch, model.first_stage_key, drop_rate=0.)
            un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * len(aug_prompt))],
                       "hint":        cond['hint'],
                       "control":     cond['control']}

            b, c, h, w = cond["control"][0].shape
            shape = (model.channels, h // 8, w // 8)
            samples, _ = ddim_sampler.sample(ddim_steps, b, shape, cond, verbose=False, eta=eta,
                                             unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            # Log original images
            ori_images = (batch['jpg'] * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            for j, (y, x) in enumerate(zip(ori_images, x_samples)):
                image_id = f'batch_{i}_{j}'
                y_dir = os.path.join(opt.image_dir, opt.task, 'original_images')
                os.makedirs(y_dir, exist_ok=True)
                y_image_path = os.path.join(y_dir, f'{image_id}.png')
                Image.fromarray(y).save(y_image_path)

                x_dir = os.path.join(opt.image_dir, opt.task, 'generated_images')
                os.makedirs(x_dir, exist_ok=True)
                x_image_path = os.path.join(x_dir, f'{image_id}.png')
                Image.fromarray(x).save(x_image_path)

            print(f'Batch {i} Finished')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

