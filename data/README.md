# Data

Expected raw input:

```text
data/raw/GRAMM.tif
```

Large raster files are ignored by Git.

If you already have the prepared exact-x2 dataset, place it under:

```text
workspace/EHFNet_MS_SR_x2_2m_to_1m/
```

Expected prepared arrays:

```text
patches_72_36/
├── train_lr_36x36.npy
├── train_gt_72x72.npy
├── val_lr_36x36.npy
├── val_gt_72x72.npy
├── test_lr_36x36.npy
└── test_gt_72x72.npy
```

The notebook can rebuild these from `GRAMM.tif` if needed.

Do not commit private, licensed, or very large source imagery unless you have the right to publish it.
