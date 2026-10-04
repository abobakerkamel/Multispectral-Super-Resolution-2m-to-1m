# Final Architecture — MAX-SR V2

## Objective

Maximize final super-resolution performance for a four-band multispectral exact-×2 task:

- input: `B×4×36×36` at 2 m/pixel
- output: `B×4×72×72` at 1 m/pixel

The design is intentionally composite. It does **not** optimize for minimal parameter count or clean ablation analysis.

## 1. Residual Spectral Adapter

A small residual adapter performs early spectral mixing without destroying the input distribution expected by the pretrained EHFNet-MS backbone:

```text
4 → 32 (1×1)
GELU
32 → 32 (3×3)
GELU
32 → 4 (1×1)
residual add with bounded learnable β
```

The last layer is zero-initialized.

## 2. EHFNet-MS Spatial Backbone

The exact-x2 EHFNet-MS backbone remains checkpoint-compatible:

- 4 input bands
- 48 channels
- 6 EHF blocks
- edge enhancement
- multi-scale attention
- residual body tail

A `48 → 128` projection produces the common fusion width.

## 3. Spectral Context Expert

Two axial Transformer blocks model long-range row/column relationships on the 36×36 LR feature map.

Two Mamba blocks are then applied when `mamba_ssm` is available. Otherwise a pure-PyTorch four-directional SSM-style fallback is used.

## 4. Frequency Expert

A Haar high-frequency branch extracts three high-pass subbands per feature channel:

- LH
- HL
- HH

These are projected back into the common feature width and refined with depthwise pointwise blocks.

## 5. Band-4 Feature Expert

Band 4 is encoded separately and fused with shared spatial features. This gives the model a dedicated feature stream for the band that receives stronger supervision.

## 6. Four-way Mixture of Experts

Experts:

1. spatial EHF features
2. Transformer/Mamba spectral-context features
3. high-frequency features
4. Band-4-specialized features

The gate is spatially varying and produces four softmax weights at each LR location.

A small load-balancing regularizer discourages early expert collapse.

## 7. Learned ×2 Upsampling

The fused LR feature map is upsampled using:

```text
Conv(C → 4C)
PixelShuffle ×2
GELU
```

Then four HR residual-attention blocks refine the 72×72 representation.

## 8. Bicubic Global Residual

The network predicts a four-band residual on top of bicubic LR upsampling:

```text
SR_initial = bicubic(LR, ×2) + learned_residual
```

This protects low-frequency radiometric structure and focuses capacity on missing detail.

## 9. Band-4 Final Refinement

A second Band-4 specialist operates at output resolution using:

- initial Band-4 SR
- upsampled Band-4 expert features
- upsampled fused features
- HR features

It predicts a bounded residual:

```text
Band4_final = Band4_initial + α * tanh(ΔBand4)
```

`α` is learnable and bounded.

## 10. Uncertainty Head

A four-band log-variance map is predicted and used in a heteroscedastic uncertainty objective.

## 11. Differentiable Degradation

Final SR is blurred using the current Gaussian MTF approximation and downsampled by exact 2× area pooling.

The reconstructed LR is compared against the original LR input.

## 12. Losses

The implementation includes:

- all-band Charbonnier
- Band-4 Charbonnier
- Band-4 bias
- SAM
- inter-band consistency
- edge
- frequency
- LR data consistency
- uncertainty
- MoE balance
- optional NDVI

## 13. Training

- 1000 epochs
- AMP
- gradient accumulation
- gradient clipping
- differential backbone/new-branch learning rates
- 10-epoch backbone freeze when pretrained weights load successfully
- LR warmup + cosine decay
- EMA
- auto-resume
- best-total, best-pixel, best-Band4 checkpoints
- periodic checkpoints every 25 epochs

## 14. Final Inference

Final test prediction uses:

- EMA weights
- x8 geometric test-time augmentation
- ensemble across best validation checkpoints

This is intentionally slower than a single forward pass because final metric performance is the priority.
