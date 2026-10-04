# Project Status

## Completed

- Defined the exact ×2 task: `2.0 m → 1.0 m`.
- Fixed patch geometry: `4×36×36 → 4×72×72`.
- Defined same-footprint 72 m × 72 m training pairs.
- Established leakage-safe spatial split before patch extraction.
- Added train-only normalization.
- Kept EHFNet-MS backbone checkpoint-compatible.
- Added residual spectral adapter.
- Added Transformer and Mamba/SSM spectral-context branch.
- Added Haar high-frequency expert.
- Added Band-4 feature expert.
- Added four-way per-pixel Mixture-of-Experts.
- Added PixelShuffle ×2 upsampling.
- Added bicubic global residual.
- Added HR attention refinement.
- Added Band-4 final output specialist.
- Added uncertainty head.
- Added differentiable degradation / LR reconstruction consistency.
- Added composite spectral/spatial/physical losses.
- Added bounded adaptive loss-weight controller.
- Added EMA, AMP, gradient clipping, staged fine-tuning, cosine LR, auto-resume.
- Added best-checkpoint selection, x8 TTA and checkpoint ensemble.
- Generated the full local Jupyter notebook.
- Performed a Python syntax check over all notebook code cells.

## Not completed yet

- Full runtime execution on the target local GPU.
- 1000-epoch training.
- Final benchmark table.
- Sensor-specific measured MTF/PSF replacement.
- Independent verification of physical Red/NIR band identities.
- NDVI-loss activation.
- PPO loss-weight controller integration.
- Final production full-scene GeoTIFF export from a trained checkpoint.

## Current next step

Run the notebook locally from the beginning through:

1. environment/GPU check;
2. exact-x2 dataset validation;
3. architecture QA;
4. pretrained backbone loading;
5. a short runtime smoke test;
6. then launch the full 1000-epoch run.

Do not mark the model as "trained" or publish final metrics until the full local run completes.
