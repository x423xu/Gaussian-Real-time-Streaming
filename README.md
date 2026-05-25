# Gaussian-Real-time-Streaming

Realtime streaming for 3D/4D Gaussian Splatting.

## Minimal RTGS scaffold

The current runnable scaffold imitates the orchestration style of `CanonicalGS_mono_cano` while keeping the model intentionally tiny. It uses:

- `config/main.yaml` for Hydra-style runtime configuration.
- `src/grts/main.py` as the package entrypoint.
- `src/grts/model/rtgs_model.py` as the initial `rtgs_model`, with exactly two convolution layers.
- `src/grts/training.py` for a minimal smoke-training loop.
- `re10k_unposed` data, which prepares Gaussian-splatting images at `256x256` and DA3 images at `504x504` without running DA3 inside the dataset.

Run from the repo root on `malab`:

```bash
source /data0/xxy/miniconda3/etc/profile.d/conda.sh
PYTHONPATH=src conda run -n rtgs python -m grts.main mode=inspect_dataset runtime.device=cpu
PYTHONPATH=src conda run -n rtgs python -m grts.main mode=inspect_forward runtime.device=cpu
PYTHONPATH=src conda run -n rtgs python -m grts.main mode=train_smoke runtime.device=cpu train.steps=2 output_dir=outputs/rtgs_smoke
```

Run the focused orchestration tests:

```bash
source /data0/xxy/miniconda3/etc/profile.d/conda.sh
conda run -n rtgs python -m pytest tests/test_rtgs_orchestration.py -q
```