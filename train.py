from share import *
import numpy as np
import argparse, os, sys
from functools import partial
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from cldm.model import create_model, load_state_dict

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default='./models/cldm_v15.yaml',
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--only_mid_control",
        action="store_true",
        default=False,
        help="only_mid_control for control net",
    )
    parser.add_argument(
        "--sd_locked",
        action="store_false",
        default=True,
        help="sd_locked for control net",
    )
    # Training
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus for training",
    )
    parser.add_argument(
        "--nnode",
        type=int,
        default=1,
        help="number of nodes for training",
    )

    # Prompt Engineering
    parser.add_argument(
        "--data_config",
        type=str,
        help="path to data config files",
        default='./models/data.yaml',
    )
    parser.add_argument(
        "--sd_v2",
        action="store_true",
        default=False,
        help="if use stable diffusion 2.0",
    )

    return parser


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, persistent_workers=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)



if __name__ == "__main__":
    sys.path.append(os.getcwd())
    # parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)

    opt, _ = get_parser().parse_known_args()

    nowname = f"{opt.name}"
    logdir = os.path.join(opt.logdir, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)

    # Configs
    resume_path = './models/control_sd15_ini.ckpt' if not opt.sd_v2 else './models/control_sd21_ini.ckpt'
    # batch_size = 32
    learning_rate = 1e-4

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(opt.base).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = opt.sd_locked
    model.only_mid_control = opt.only_mid_control

    # Data
    data_config = OmegaConf.load(opt.data_config)
    dataloader = instantiate_from_config(data_config.data)
    dataloader.prepare_data()
    dataloader.setup()
    print("#### Data #####")
    for k in dataloader.datasets:
        print(f"{k}, {dataloader.datasets[k].__class__.__name__}, {len(dataloader.datasets[k])}")

    # Callbacks
    callbacks_cfg = {
        "checkpoint_callback": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}-{step:09}",
                "verbose": True,
                'save_top_k': -1,
                'every_n_train_steps': 1000,
                'save_weights_only': True,
                "save_last": True,
            }
        },
        "image_logger": {
            "target": "cldm.logger.ImageLogger",
            "params": {
                "batch_frequency": 500,
                "max_images": 16,
                "clamp": True,
                "log_images_kwargs": {'N': 16,
                                      'unconditional_guidance_scale': 9.0}
            }
        },
    }

    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # Trainer
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=logdir)
    trainer = pl.Trainer(gpus=opt.gpus, accelerator='ddp', num_nodes=opt.nnode,
                         max_steps=10000, check_val_every_n_epoch=2, accumulate_grad_batches=4,
                         precision=32, callbacks=callbacks, logger=tb_logger)

    # Train!
    trainer.fit(model, dataloader)
