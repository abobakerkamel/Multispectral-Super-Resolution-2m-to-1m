# Local Training Guide

## Recommended hardware

The current defaults target an RTX A6000 48 GB class GPU.

Default settings:

```text
epochs              = 1000
batch_size          = 32
gradient_accum      = 2
effective_batch     = 64
AMP                 = on
EMA                 = on
auto_resume         = on
TTA x8              = on
checkpoint ensemble = on
```

If you hit OOM:

1. reduce `BATCH_SIZE` from 32 to 24 or 16;
2. increase `ACCUM_STEPS` so the effective batch remains similar;
3. keep `FUSION_C=128` initially;
4. do not disable AMP unless debugging.

## Run order

1. Open `notebooks/GRAMM_MAXSR_V2_LOCAL_2m_to_1m.ipynb`.
2. Run the environment cell.
3. Confirm CUDA is available.
4. Confirm the exact-x2 dataset is found or rebuilt.
5. Confirm:
   - LR batch shape = `(B,4,36,36)`
   - GT batch shape = `(B,4,72,72)`
6. Run architecture QA.
7. Confirm whether true `mamba_ssm` is available.
8. Confirm how many pretrained backbone tensors were loaded.
9. Run a short smoke test if desired.
10. Launch the 1000-epoch training cell.

## Checkpoints

Generated under:

```text
workspace/EHFNet_MS_SR_x2_2m_to_1m/
└── training/
    └── MAXSR_V2_COMPOSITE_1000ep/
        └── checkpoints/
```

Important files:

- `best_val_total.pt`
- `best_val_pixel.pt`
- `best_val_band4_rmse.pt`
- `resume_latest.pt`
- periodic `epoch_XXXX.pt`
- `epoch_1000_last.pt`

## Auto-resume

The notebook automatically resumes from:

`resume_latest.pt`

if it exists and `AUTO_RESUME=True`.

## Final evaluation

The test stage ensembles the available best checkpoints and applies x8 TTA when enabled.

Metrics include:

- MAE
- RMSE
- mean PSNR
- mean SSIM
- SAM in degrees
- Band-4 MAE
- Band-4 RMSE
- Band-4 bias
- per-band MAE/RMSE/Bias/PSNR/SSIM

## Full-scene export

After training, set:

```python
RUN_FULL_SCENE_EXPORT = True
```

The notebook will run overlapping tiled inference and write a Float32 1 m GeoTIFF.
