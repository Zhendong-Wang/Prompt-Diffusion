import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from PIL import Image

from annotator.hed import HEDdetector
from annotator.uniformer import UniformerDetector
from annotator.midas import MidasDetector

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="logs", help="directory for logging dat shit")
parser.add_argument("--i_start", type=int, default=0, help="Start Seed")
parser.add_argument("--i_end", type=int, default=100, help="End Seed")
args = parser.parse_args()


path = args.path
i_start = args.i_start
i_end = args.i_end

with open(Path(path, "seeds.json")) as f:
    seeds = json.load(f)

print(f"Length of Seeds: {len(seeds)}")
i_end = min(len(seeds), i_end)

hed_annotator = HEDdetector()
uniformer_annotator = UniformerDetector()
midas_annotator = MidasDetector()


N = int(i_end - i_start)
for i in tqdm(range(i_start, i_end), total=N, miniters=int(N/100), maxinterval=600):
    name, i_seeds = seeds[i]
    propt_dir = Path(path, name)
    for seed in i_seeds:
        image = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

        hed_path = propt_dir.joinpath(f"{seed}_0_hed.jpg")
        image_hed = hed_annotator(np.array(image))
        Image.fromarray(image_hed).save(hed_path)

        seg_path = propt_dir.joinpath(f"{seed}_0_seg.jpg")
        image_seg = uniformer_annotator(np.array(image))
        Image.fromarray(image_seg).save(seg_path)

        depth_path = propt_dir.joinpath(f"{seed}_0_depth.jpg")
        normal_path = propt_dir.joinpath(f"{seed}_0_normal.jpg")
        image_depth, image_normal = midas_annotator(np.array(image))
        Image.fromarray(image_depth).save(depth_path)
        Image.fromarray(image_normal).save(normal_path)


