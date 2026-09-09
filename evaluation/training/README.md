# Controlled from-scratch comparison

This experiment compares **RAC-1x**, **SSDD-S-F8C4**, and **DiTo-B-F8C4**. It is separate from the
official-checkpoint benchmark because the two experiments answer different questions.

## Fixed protocol

- Data: ImageNet-1K train, 256 px, with the same sampled image order and augmentation seed.
- Validation: the fixed ImageNet-1K validation set exported by `prepare_inputs.py`.
- Latent interface: continuous `4 x 32 x 32` (F8C4).
- Optimizer, schedule, precision, global batch size, and augmentation are matched whenever the
  architecture permits it. Method-specific losses remain unchanged.
- Save at fixed **images-seen** milestones. Also log training FLOPs and GPU-hours.
- Select checkpoints by validation rFID, never by the training loss.
- Report both fixed-data-budget and converged/best-validation results.

## What can and cannot be compared

Raw loss values are recorded for debugging, but are **not ranked across methods**: RAC, SSDD, and
DiTo optimize different objectives and their numerical scales are not commensurate. Shared curves
use validation rFID, PSNR, SSIM, LPIPS, throughput, GPU-hours, and images seen.

The present local datasets contain only a few thousand images. Any run on them is a pipeline pilot,
not a paper result. Start the formal run only after the complete ImageNet training and 50K
validation sets have been mounted.

## Capacity control

The primary pair is RAC-1x (about 49.5M unique parameters because encoding and decoding share one
field) and SSDD-S-F8C4 (48.22M total parameters). DiTo-B is a stronger, larger reference, with
parameter count and compute shown explicitly rather than described as capacity matched.

## Required logging schema

Each checkpoint evaluation writes one JSON line with:

`method, run_id, seed, images_seen, optimizer_steps, train_loss_native, val_rfid, val_psnr,
val_ssim, val_lpips, train_flops, gpu_hours, parameters, encode_nfe, decode_nfe`.

Do not claim a win from a checkpoint that has not reached a validation plateau. If RAC trails, first
separate optimization progress from architecture by comparing fixed-budget and converged curves.
