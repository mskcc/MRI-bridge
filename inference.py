"""inference.py — Run inference with the MRI-bridge + discNet."""

import argparse
import logging
import os
import os.path as osp
import sys
import time

import numpy as np
import scipy.io as sio
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UNIDB_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "UniDB-fork", "UniDB-GOU"))
sys.path.insert(0, _UNIDB_ROOT)
sys.path.insert(0, os.path.join(_UNIDB_ROOT, "tasks", "sr"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "evaluation"))

import options as option
from models import create_model
from models.slab_model import SlabModel
import utils as util
from data import create_dataloader, create_dataset
from utils.img_utils import get_results
from evaluation.eval_helpers import save_all_test_outputs
from training_utils import expand_env_vars

# ===========================================================================
# Helpers
# ===========================================================================
def parse_options():
    """Parse CLI and return the resolved option NoneDict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, help="Path to option YAML file.",
        default=osp.join(_SCRIPT_DIR, "options", "inference.yml"),
    )
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = expand_env_vars(opt)

    # Override paths so results live under MRI-bridge/
    opt["path"]["root"] = _SCRIPT_DIR
    results_root = osp.join(_SCRIPT_DIR, "results")
    opt["path"]["results_root"] = osp.join(results_root, opt["name"])
    opt["path"]["log"] = osp.join(results_root, opt["name"])

    return option.dict_to_nonedict(opt)


def setup_logging(opt):
    """Create output directories and configure loggers."""
    util.mkdirs(
        path for key, path in opt["path"].items()
        if key != "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
    util.setup_logger(
        "base", opt["path"]["log"], "test_" + opt["name"],
        level=logging.INFO, screen=True, tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))
    return logger


def build_models(opt):
    """Instantiate the diffusion model, SDE, and slab model."""
    # Diffusion bridge model
    model = create_model(opt)

    sde = util.UniDB(
        lambda_square=opt["sde"]["lambda_square"],
        gamma_inv=opt["sde"]["gamma_inv"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=model.device,
    )
    sde.set_model(model.model)

    # Slab model (UNet) — config from network_slab in YAML
    slab_cfg = opt["network_slab"]
    slab_opt = dict(opt)
    slab_opt["network_G"] = slab_cfg
    slab_opt["path"] = dict(opt["path"])
    slab_opt["path"]["pretrain_model_G"] = slab_cfg["pretrain_model"]
    slab_opt["train"] = opt["train"]
    slab_opt["path"]["strict_load"] = opt["path"]["strict_load"]
    slab_model = SlabModel(slab_opt)

    return model, sde, slab_model


# ===========================================================================
# Main
# ===========================================================================

def main():
    opt = parse_options()
    logger = setup_logging(opt)

    # Config
    results_folder = osp.join(
        opt["results_folder"], opt["name"], opt["datasets"]["test"]["name"],
    )
    plot_id = opt["slice_id"]
    scale = opt["degradation"]["scale"]
    inferences = opt["inferences"]

    # Test slices to process (None or empty list = process all)
    test_slices = opt["test_slices"]

    # Dataloader
    dataset_opt = opt["datasets"]["test"]
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(f"Test images in [{dataset_opt['name']}]: {len(test_set)}")

    # Models
    model, sde, slab_model = build_models(opt)

    # --- Inference loop ---
    test_set_name = dataset_opt["name"]
    logger.info(f"\nTesting [{test_set_name}]...")

    dataset_dir = osp.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    avg_L1 = 0.0
    test_times = []

    INDEX_PARALLEL = opt["INDEX_PARALLEL"]
    if INDEX_PARALLEL == 0:
        MINUMUM_PARALLEL = 0
        MAXIMUM_PARALLEL = len(test_loader)
    else:
        MINUMUM_PARALLEL = INDEX_PARALLEL
        MAXIMUM_PARALLEL = INDEX_PARALLEL + 50

    test_data_idx = 0
    for test_data in test_loader:
        if test_data_idx < MINUMUM_PARALLEL:
            test_data_idx += 1 
            continue
        if test_data_idx >= MAXIMUM_PARALLEL:
            break
        test_data_idx += 1 
        # Determine sample name
        has_GT = dataset_opt["dataroot_GT"] is not None
        img_path = test_data["GT_path"][0] if has_GT else test_data["LQ_path"][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        if test_slices and img_name not in test_slices:
            continue

        logger.info(f"Processing {img_name}")
        t0 = time.time()

        if len(test_data["LQ"].shape)>4:
            LQ = test_data["LQ"]
            input_img_windows = np.zeros((LQ.shape[1], LQ.shape[2],LQ.shape[3], LQ.shape[4]))
            slab_SR_img_windows = np.zeros((LQ.shape[1], LQ.shape[2],LQ.shape[3], LQ.shape[4]))
            SR_images_windows = np.zeros((LQ.shape[1], opt["inferences"], LQ.shape[2],LQ.shape[3], LQ.shape[4]))
            for windowid in range(LQ.shape[1]):
                test_data["LQ"] = LQ[:, windowid]
                _, input_img_windows[windowid], \
                    ref, slab_SR_img_windows[windowid], _, SR_images_windows[windowid] = (
                    get_results(test_data, model, sde, slab_model, avg_L1, scale, inferences)
                    )
            # Save .mat file
            mat_data = {
                "input":     input_img_windows[:, plot_id, ...],
                "ref":       ref[plot_id],
                "UNet":      slab_SR_img_windows[:, plot_id, ...],
                "SR_images": SR_images_windows[:, :, plot_id, ...]
            }
            os.makedirs(results_folder, exist_ok=True)
            sio.savemat(os.path.join(results_folder, f"{img_name}.mat"), mat_data)

        else:
            SR_img, input_img, ref, slab_SR_img, avg_L1, SR_images = (
                get_results(test_data, model, sde, slab_model, avg_L1, scale, inferences)
            )

            test_times.append(time.time() - t0)

            save_all_test_outputs(
                dataset_dir=dataset_dir,
                img_name=img_name,
                plot_id=plot_id,
                input_img=input_img,
                ref=ref,
                slab_SR_img=slab_SR_img,
                SR_img = SR_img,
                SR_images = SR_images,
                results_folder=results_folder,
                vmax=1.0,
            )

    if test_times:
        logger.info(f"Average test time: {np.mean(test_times):.4f}s")


if __name__ == "__main__":
    main()

