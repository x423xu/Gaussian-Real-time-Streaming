# Datasets

This project uses the DepthSplat-style preprocessed chunk format for RE10K and DL3DV.

```text
datasets/
  re10k/
    train/000000.torch
    train/index.json
    test/000000.torch
    test/index.json
  dl3dv/
    train/000000.torch
    train/index.json
    test/000000.torch
    test/index.json
```

Each `.torch` chunk is a list of scene dictionaries with:

- `key`: unique scene id.
- `cameras`: tensor shaped `[views, 18]`; first four values are normalized `fx, fy, cx, cy`, values `6:` are a flattened `3x4` world-to-camera matrix.
- `images`: list of uint8 tensors containing encoded image bytes.

The dataloader yields:

```python
{
    "context": {"extrinsics", "intrinsics", "image", "near", "far", "index"},
    "target": {"extrinsics", "intrinsics", "image", "near", "far", "index"},
    "scene": "scene-id",
    "all_ind": 120,
}
```

Intrinsics are normalized. Extrinsics are OpenCV camera-to-world matrices.

## Usage

```python
from grts.data import DatasetStage, build_dataset, load_dataset_config
from grts.data.dataloader import build_dataloader

cfg = load_dataset_config("config/dataset/re10k.yaml", overrides={"roots": ["/path/to/datasets/re10k"]})
dataset = build_dataset(cfg, DatasetStage.TRAIN)
loader = build_dataloader(dataset, batch_size=1, num_workers=4, seed=1234)
batch = next(iter(loader))
```

Use `config/dataset/re10k.yaml` for RealEstate10K and `config/dataset/dl3dv.yaml` for DL3DV. Override `roots`, `image_shape`, and `view_sampler` values from your experiment config or from Python.

## Verification

Run this in an environment with PyTorch and pytest installed:

```bash
pytest tests/data/test_chunk_datasets.py -q
```