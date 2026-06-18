#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-9}"
LOG_DIR="${LOG_DIR:-outputs/rtgs_ablate_logs}"
QUEUE_LOCK="${QUEUE_LOCK:-${LOG_DIR}/gpu${GPU}.lock}"
DRY_RUN="${DRY_RUN:-0}"
DA3_LR="${DA3_LR:-1.0e-5}"
DA3_DEPTH_HEAD_LR="${DA3_DEPTH_HEAD_LR:-1.0e-4}"

mkdir -p "${LOG_DIR}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v flock >/dev/null 2>&1; then
  echo "[RTGS-ABLATE] flock is required for serialized nohup launches." >&2
  exit 1
fi

BASE_ARGS=(
  mode=train_smoke
  runtime.device=cuda:0
  "dataset.roots=[/data0/xxy/data/re10k]"
  dataset.overfit_to_scene=null
  "dataset.da3_image_shape=[336,336]"
  dataset.num_workers=4
  dataset.persistent_workers=true
  dataset.pin_memory=true
  dataset.prefetch_factor=4
  model.vit_type=vit-b
  model.vit_pretrained=true
  model.vit_image_size=252
  model.dpt_feature_channels=128
  model.da3_model_name=depth-anything/DA3-BASE
  model.unfreeze_da3=false
  model.train_depth_head_only=false
  model.intrinsic_embedding.enabled=false
  model.depth_refinement.enabled=false
  model.camera_refinement.enabled=false
  train.steps=100000
  train.batch_size=2
  train.lr=1.0e-4
  train.log_every=100
  train.save_checkpoint=true
  train.checkpoint_every=5000
  eval.evaluation_index_path=assets/evaluation_index_re10k.json
  eval.every_n_steps=2000
  eval.eval_data_interval=10
  eval.max_batches=null
  wandb.enabled=true
  wandb.entity=xxy
  wandb.project=rtgs
  model.gaussian_scale_max=1.0
)

run_ablation() {
  local name="$1"
  shift
  local output_dir="outputs/rtgs_ablate_${name}"
  local log_file="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"
  local wandb_name="rtgs_ablate_${name}"
  local command_args=(
    "${BASE_ARGS[@]}"
    "output_dir=${output_dir}"
    "wandb.name=${wandb_name}"
    "$@"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] ${name}"
    echo "  log: ${log_file}"
    echo "  pid: ${pid_file}"
    printf "  cmd:"
    printf " %q" "${command_args[@]}"
    printf "\n"
    return 0
  fi

  nohup bash -c '
set -euo pipefail
root_dir="$1"
name="$2"
lock_file="$3"
gpu="$4"
shift 4
cd "${root_dir}"
{
  flock -x 200
  echo "[RTGS-ABLATE] START ${name} $(date -Is)"
  echo "[RTGS-ABLATE] GPU ${gpu}"
  printf "[RTGS-ABLATE] CMD:"
  printf " %q" "$@"
  printf "\n"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" DA3_LOG_LEVEL=WARN PYTHONPATH=src python -m rtgs.main "$@"
  status=$?
  set -e
  echo "[RTGS-ABLATE] END ${name} status=${status} $(date -Is)"
  exit "${status}"
} 200>"${lock_file}"
' bash "${ROOT_DIR}" "${name}" "${QUEUE_LOCK}" "${GPU}" "${command_args[@]}" >"${log_file}" 2>&1 &

  echo "$!" > "${pid_file}"
  echo "[RTGS-ABLATE] queued ${name} pid=$(cat "${pid_file}") log=${log_file}"
}

run_ablation base

run_ablation train_depth_head_only \
  model.unfreeze_da3=true \
  model.train_depth_head_only=true \
  "train.da3_lr=${DA3_LR}" \
  "train.da3_depth_head_lr=${DA3_DEPTH_HEAD_LR}"

run_ablation unfreeze_whole_da3 \
  model.unfreeze_da3=true \
  model.train_depth_head_only=false \
  "train.da3_lr=${DA3_LR}" \
  "train.da3_depth_head_lr=${DA3_DEPTH_HEAD_LR}"

run_ablation intrinsic_embedding_only \
  model.intrinsic_embedding.enabled=true

run_ablation depth_rtgs_features \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[]"

run_ablation depth_both_features \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]"

run_ablation depth_both_features_intrinsic \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.intrinsic_embedding.enabled=true

run_ablation camera_refinement_only \
  model.camera_refinement.enabled=true

run_ablation depth_both_features_camera \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.camera_refinement.enabled=true

run_ablation all_refinements_frozen_da3 \
  model.intrinsic_embedding.enabled=true \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.camera_refinement.enabled=true

run_ablation all_refinements_train_depth_head_only \
  model.unfreeze_da3=true \
  model.train_depth_head_only=true \
  "train.da3_lr=${DA3_LR}" \
  "train.da3_depth_head_lr=${DA3_DEPTH_HEAD_LR}" \
  model.intrinsic_embedding.enabled=true \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.camera_refinement.enabled=true

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[RTGS-ABLATE] dry run complete"
else
  echo "[RTGS-ABLATE] all jobs queued with serialized GPU lock: ${QUEUE_LOCK}"
  echo "[RTGS-ABLATE] logs and pid files: ${LOG_DIR}"
fi
