"""Shared helpers for train.py and train_slab.py."""

import logging
import math
import os
import os.path as osp
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UNIDB_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "UniDB-fork", "UniDB-GOU"))
sys.path.insert(0, _UNIDB_ROOT)
sys.path.insert(0, os.path.join(_UNIDB_ROOT, "tasks", "sr"))

import options as option
import utils as util
from data import create_dataloader, create_dataset


def prepare_image_for_tb(img):
    """Convert a uint8 image (H,W) or (H,W,C) to a [1,H,W] float for TensorBoard."""
    if img.ndim == 2:
        plane = img.astype(np.float64)
    elif img.shape[2] == 5:
        plane = img[:, :, 2].astype(np.float64)
    elif img.shape[2] == 10:
        plane = np.abs(img[:, :, 4].astype(np.float64) + 1j * img[:, :, 5].astype(np.float64))
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    return ((plane - plane.min()) / (plane.max() - plane.min() + 1e-8))[None]


def log_visuals_to_tb(tb_logger, prefix, model, epoch):
    """Log output/input/GT images to TensorBoard."""
    visuals = model.get_current_visuals()
    for tag, key in [("output", "Output"), ("input", "Input"), ("GT", "GT")]:
        img = util.tensor2img(visuals[key].squeeze())
        tb_logger.add_image(f"{prefix}/{tag}", prepare_image_for_tb(img), epoch)


def expand_env_vars(obj):
    """Recursively expand ${VAR} in all strings of a nested dict/list."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


def parse_options(default_yml):
    """Parse CLI args and return the fully-resolved option dict."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, help="Path to option YAML file.",
        default=osp.join(_SCRIPT_DIR, "options", default_yml),
    )
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt = expand_env_vars(opt)

    opt["path"]["root"] = _SCRIPT_DIR
    root = osp.join(_SCRIPT_DIR, "experiments", opt["name"])
    opt["path"].update({
        "experiments_root": root,
        "models": osp.join(root, "models"),
        "training_state": osp.join(root, "training_state"),
        "log": root,
        "val_images": osp.join(root, "val_images"),
    })
    return option.dict_to_nonedict(opt)


def build_dataloaders(opt, logger):
    """Create train/val dataloaders from the option dict."""
    train_loader = val_loader = None
    for phase, ds_opt in opt["datasets"].items():
        ds = create_dataset(ds_opt)
        if phase == "train":
            train_loader = create_dataloader(ds, ds_opt, opt, None)
            iters_per_epoch = int(math.ceil(len(ds) / ds_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / iters_per_epoch))
            logger.info(f"Train: {len(ds):,d} images, {iters_per_epoch:,d} iters/epoch")
            logger.info(f"Total: {total_epochs:d} epochs for {total_iters:,d} iters")
        elif phase == "val":
            val_loader = create_dataloader(ds, ds_opt, opt, None)
            logger.info(f"Val [{ds_opt['name']}]: {len(ds):d} images")
        else:
            raise NotImplementedError(f"Unknown phase: {phase}")
    assert train_loader and val_loader
    return train_loader, val_loader, total_iters, total_epochs


def setup_experiment(opt):
    """Create dirs, loggers, and TensorBoard writer. Returns (logger, tb_logger)."""

    resume_state = opt["path"].get("resume_state")
    if not resume_state:
        util.mkdir_and_rename(opt["path"]["experiments_root"])
        util.mkdirs(opt["path"]["models"])

    util.setup_logger("base", opt["path"]["log"], "train_" + opt["name"],
                      level=logging.INFO, screen=False, tofile=True)
    util.setup_logger("val", opt["path"]["log"], "val_" + opt["name"],
                      level=logging.INFO, screen=False, tofile=True)
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    tb_logger = None
    if opt["use_tb_logger"] and "debug" not in opt["name"]:
        tb_logger = SummaryWriter(
            log_dir=osp.join(opt["path"]["experiments_root"], "tb_logger")
        )

    return logger, tb_logger