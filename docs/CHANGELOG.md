# Changelog

## MAX-SR V2 — current GitHub snapshot

### Added
- finalized maximum-performance composite architecture;
- pretrained-compatible EHFNet-MS backbone reuse;
- residual spectral adapter;
- 2 axial Transformer blocks;
- 2 Mamba/SSM blocks;
- high-frequency Haar expert;
- Band-4 feature expert;
- four-way per-pixel MoE;
- MoE anti-collapse regularization;
- PixelShuffle ×2;
- bicubic global residual;
- HR residual-attention refinement;
- Band-4 final residual specialist;
- uncertainty head;
- differentiable degradation consistency;
- expanded composite objective;
- bounded learnable loss re-weighting;
- staged backbone fine-tuning;
- AMP / gradient accumulation / clipping;
- EMA;
- warmup + cosine LR;
- auto-resume;
- best-total / best-pixel / best-Band4 checkpoints;
- x8 TTA;
- best-checkpoint ensemble;
- optional full-scene Float32 GeoTIFF export;
- GitHub-ready project structure and documentation.

### Preserved
- exact-x2 2 m → 1 m formulation;
- 36×36 LR to 72×72 GT;
- 72 m × 72 m identical footprint;
- spatial split before patches;
- train-only normalization;
- existing exact-x2 checkpoint path conventions.

### Deferred
- PPO loss-weight controller;
- measured sensor-specific MTF/PSF;
- NDVI supervision until Red/NIR identities are verified;
- final 1000-epoch results.
