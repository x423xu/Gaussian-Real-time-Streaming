# Dataset Transplant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config-backed RE10K and DL3DV chunk datasets compatible with the DepthSplat data contract.

**Architecture:** Keep the dataset code isolated under `src/grts/data`, with common chunk parsing in `chunk_dataset.py`, dataset-specific defaults in `re10k.py` and `dl3dv.py`, and DepthSplat-style view sampling under `view_sampler/`. YAML config files under `config/dataset` select roots, image shapes, and sampler policy.

**Tech Stack:** Python 3.10+, PyTorch, Pillow, PyYAML, pytest.

---

### Task 1: Tests

- [x] Add synthetic chunk tests for RE10K config loading, DL3DV dataloader batching, and evaluation index sampling.
- [x] Run the tests before implementation. On the default `malab` shell this fails at dependency import because pytest/torch are not installed there.

### Task 2: Dataset Package

- [x] Add `src/grts/data/types.py`, `dataset_config.py`, `chunk_dataset.py`, `re10k.py`, `dl3dv.py`, and `dataloader.py`.
- [x] Preserve the DepthSplat output contract: `context`, `target`, `scene`, and `all_ind` with camera/image/near/far/index tensors.

### Task 3: View Samplers

- [x] Add all, arbitrary, bounded, boundedv2, unbounded, and evaluation samplers.
- [x] Preserve warmup schedules, test-mode deterministic left context, farthest-point context selection, and evaluation JSON indices.

### Task 4: Config And Docs

- [x] Add RE10K/DL3DV YAML configs and sampler YAML files.
- [x] Document expected dataset layout, public API, and smoke-test command.

### Task 5: Verification

- [ ] Run `pytest tests/data/test_chunk_datasets.py -q` inside an environment with torch and pytest.
- [x] Run `python3 -m compileall -q src/grts` on the default remote shell.