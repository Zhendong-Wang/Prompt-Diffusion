import argparse
from PIL import Image
import blobfile as bf
import numpy as np

def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--source_dir", type=str, help="path to the original images, e.g., ./image_log/original_images")
    parser.add_argument("--task", type=str, default='inv_hed', help="task to test")
    return parser

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "pt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def main():
    opt, _ = get_parser().parse_known_args()

    all_files = _list_image_files_recursively(opt.source)

    mse = []
    for x_path in all_files:
        cur_path = x_path.split('/')
        cur_path[-2] = 'generated_images'
        y_path = '/'.join(cur_path)

        x = np.array(Image.open(x_path)) / 255.
        y = np.array(Image.open(y_path)) / 255.

        mse.append(((x - y) ** 2).mean())

    rmse = np.sqrt(np.mean(mse))

    print(f'RMSE of {opt.task}: {rmse}')






