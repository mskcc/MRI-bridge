import logging

import torch

from training_utils import (
    parse_options,
    build_dataloaders,
    setup_experiment,
    log_visuals_to_tb,
    util,
    option,
)
from models import create_model
from utils.file_utils import set_random_seed


# ===========================================================================
# Validation
# ===========================================================================

def validate(model, val_loader, current_step, epoch, tb_logger, logger):
    """Run validation, log metrics, and return (avg_psnr, avg_l1)."""
    avg_psnr = 0.0
    avg_l1 = 0.0
    n = 0

    for val_data in val_loader:
        n += 1
        LQ, GT = val_data["LQ"], val_data["GT"]

        model.feed_data(LQ, GT)
        avg_l1 += model.check_parameters()
        visuals = model.get_current_visuals()

        output = util.tensor2img(visuals["Output"].squeeze())
        gt_img = util.tensor2img(visuals["GT"].squeeze())

        # Log first sample visuals
        if n == 1 and tb_logger:
            log_visuals_to_tb(tb_logger, "val", model, epoch)

        avg_psnr += util.calculate_psnr(output, gt_img)

    avg_psnr /= n
    avg_l1 /= n

    logger.info(f"# Validation # PSNR: {avg_psnr:.6f}")
    logging.getLogger("val").info(
        f"<epoch:{epoch:3d}, iter:{current_step:8,d}, psnr: {avg_psnr:.6f}"
    )
    if tb_logger:
        tb_logger.add_scalar("psnr", avg_psnr, current_step)

    return avg_psnr, avg_l1


# ===========================================================================
# Main
# ===========================================================================

def main():
    opt = parse_options("train_slab.yml")

    set_random_seed(opt["train"]["manual_seed"])
    torch.backends.cudnn.benchmark = True

    # Resume state
    resume_state = None
    if opt["path"].get("resume_state"):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])

    # Directories, loggers, TensorBoard
    logger, tb_logger = setup_experiment(opt)

    # Data
    train_loader, val_loader, total_iters, total_epochs = build_dataloaders(opt, logger)

    # Model
    model = create_model(opt)
    if resume_state:
        logger.info(f"Resuming from epoch {resume_state['epoch']}, iter {resume_state['iter']}")
        start_epoch, current_step = resume_state["epoch"], resume_state["iter"]
        model.resume_training(resume_state)
    else:
        start_epoch, current_step = 0, 0

    # --- Training ---
    logger.info(f"Start training from epoch {start_epoch}, iter {current_step}")
    best_psnr, best_l1, best_iter = 0.0, 0.0, 0

    for epoch in range(start_epoch, total_epochs + 1):
        print(f"Epoch: {epoch}")
        for train_data in train_loader:
            current_step += 1
            if current_step > total_iters:
                break

            LQ, GT = train_data["LQ"], train_data["GT"]

            model.feed_data(LQ, GT)
            model.optimize_parameters()
            model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])

            # Periodic logging
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                msg = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.get_current_learning_rate():.3e}> "
                for k, v in logs.items():
                    msg += f"{k}: {v:.4e} "
                    if tb_logger:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(msg)

        # --- Epoch-end validation ---
        if epoch % opt["train"]["val_freq"] == 0:
            # Log a training sample for visual sanity check
            if tb_logger:
                sample = next(iter(train_loader))
                model.feed_data(sample["LQ"], sample["GT"])
                model.check_parameters(val_data=False)
                log_visuals_to_tb(tb_logger, "train", model, epoch)

            avg_psnr, avg_l1 = validate(
                model, val_loader, current_step, epoch, tb_logger, logger
            )

            if avg_psnr > best_psnr:
                best_psnr, best_iter = avg_psnr, current_step
                model.save_best(epoch)

            if avg_l1 < best_l1:
                model.save_best_L1(epoch)

            logger.info(f"Best PSNR: {best_psnr:.6f} @ iter {best_iter}")
            model.save(epoch)

    # Finish
    model.save("latest")
    logger.info("Training complete.")
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    main()
