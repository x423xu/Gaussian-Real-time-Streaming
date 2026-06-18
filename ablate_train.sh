#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-outputs/rtgs_ablate_logs}"
DRY_RUN="${DRY_RUN:-0}"
MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-10000}"
GPU_PREFERENCE="${GPU_PREFERENCE:-9,0,1,2,3,4,5,6,7,8}"
CONDA_EXE="${CONDA_EXE:-/data0/xxy/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-rtgs}"
ONLY="${ONLY:-}"
DA3_LR="${DA3_LR:-1.0e-5}"
DA3_DEPTH_HEAD_LR="${DA3_DEPTH_HEAD_LR:-1.0e-4}"

mkdir -p "${LOG_DIR}"

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

JOB_NAMES=()
JOB_ARGS=()

trim_space() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf "%s" "${value}"
}

should_add_job() {
  local name="$1"
  local selected
  if [[ -z "${ONLY}" ]]; then
    return 0
  fi
  IFS=',' read -r -a selected_jobs <<< "${ONLY}"
  for selected in "${selected_jobs[@]}"; do
    selected="$(trim_space "${selected}")"
    if [[ "${selected}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

add_ablation() {
  local name="$1"
  shift
  if ! should_add_job "${name}"; then
    return 0
  fi
  JOB_NAMES+=("${name}")
  JOB_ARGS+=("$*")
}

add_ablation base

add_ablation train_depth_head_only \
  model.unfreeze_da3=true \
  model.train_depth_head_only=true \
  "train.da3_lr=${DA3_LR}" \
  "train.da3_depth_head_lr=${DA3_DEPTH_HEAD_LR}"

add_ablation unfreeze_whole_da3 \
  model.unfreeze_da3=true \
  model.train_depth_head_only=false \
  "train.da3_lr=${DA3_LR}" \
  "train.da3_depth_head_lr=${DA3_DEPTH_HEAD_LR}"

add_ablation intrinsic_embedding_only \
  model.intrinsic_embedding.enabled=true

add_ablation depth_rtgs_features \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[]"

add_ablation depth_both_features \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]"

add_ablation depth_both_features_intrinsic \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.intrinsic_embedding.enabled=true

add_ablation camera_refinement_only \
  model.camera_refinement.enabled=true

add_ablation depth_both_features_camera \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.camera_refinement.enabled=true

add_ablation all_refinements_frozen_da3 \
  model.intrinsic_embedding.enabled=true \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.camera_refinement.enabled=true

add_ablation all_refinements_train_depth_head_only \
  model.unfreeze_da3=true \
  model.train_depth_head_only=true \
  "train.da3_lr=${DA3_LR}" \
  "train.da3_depth_head_lr=${DA3_DEPTH_HEAD_LR}" \
  model.intrinsic_embedding.enabled=true \
  model.depth_refinement.enabled=true \
  "model.depth_refinement.da3_feature_layers=[5,7,9,11]" \
  model.camera_refinement.enabled=true

declare -A GPU_FREE_BY_INDEX=()
declare -A GPU_CAPACITY_BY_INDEX=()

read_gpu_free_memory() {
  local entry index free
  if [[ -n "${GPU_FREE_MEMORY:-}" ]]; then
    IFS=',' read -r -a entries <<< "${GPU_FREE_MEMORY}"
    for entry in "${entries[@]}"; do
      index="$(trim_space "${entry%%:*}")"
      free="$(trim_space "${entry##*:}")"
      [[ -n "${index}" && -n "${free}" ]] || continue
      GPU_FREE_BY_INDEX["${index}"]="${free}"
    done
    return 0
  fi

  while IFS=',' read -r index free; do
    index="$(trim_space "${index}")"
    free="$(trim_space "${free}")"
    [[ -n "${index}" && -n "${free}" ]] || continue
    GPU_FREE_BY_INDEX["${index}"]="${free}"
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
}

build_gpu_assignments() {
  local job_count="$1"
  local ordered_gpus=()
  local gpu free capacity assigned_count
  local -A seen=()
  local assignments=()

  read_gpu_free_memory
  IFS=',' read -r -a preferred_gpus <<< "${GPU_PREFERENCE}"
  for gpu in "${preferred_gpus[@]}"; do
    gpu="$(trim_space "${gpu}")"
    [[ -n "${gpu}" ]] || continue
    seen["${gpu}"]=1
    free="${GPU_FREE_BY_INDEX[${gpu}]:-0}"
    if (( free >= MIN_FREE_VRAM_MB )); then
      capacity=$((free / MIN_FREE_VRAM_MB))
      GPU_CAPACITY_BY_INDEX["${gpu}"]="${capacity}"
      ordered_gpus+=("${gpu}")
    fi
  done

  for gpu in "${!GPU_FREE_BY_INDEX[@]}"; do
    [[ -z "${seen[${gpu}]:-}" ]] || continue
    free="${GPU_FREE_BY_INDEX[${gpu}]}"
    if (( free >= MIN_FREE_VRAM_MB )); then
      capacity=$((free / MIN_FREE_VRAM_MB))
      GPU_CAPACITY_BY_INDEX["${gpu}"]="${capacity}"
      ordered_gpus+=("${gpu}")
    fi
  done

  if (( ${#ordered_gpus[@]} == 0 )); then
    echo "[RTGS-ABLATE] No GPUs have at least ${MIN_FREE_VRAM_MB} MB free VRAM." >&2
    exit 1
  fi

  while (( ${#assignments[@]} < job_count )); do
    assigned_count="${#assignments[@]}"
    for gpu in "${ordered_gpus[@]}"; do
      capacity="${GPU_CAPACITY_BY_INDEX[${gpu}]:-0}"
      if (( capacity <= 0 )); then
        continue
      fi
      assignments+=("${gpu}")
      GPU_CAPACITY_BY_INDEX["${gpu}"]=$((capacity - 1))
      if (( ${#assignments[@]} == job_count )); then
        break
      fi
    done
    if (( ${#assignments[@]} == assigned_count )); then
      echo "[RTGS-ABLATE] Not enough GPU capacity for ${job_count} jobs at ${MIN_FREE_VRAM_MB} MB/job." >&2
      echo "[RTGS-ABLATE] Free memory: ${GPU_FREE_MEMORY:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ';')}" >&2
      exit 1
    fi
  done

  printf "%s\n" "${assignments[@]}"
}

run_ablation() {
  local name="$1"
  local gpu="$2"
  shift 2
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
    echo "[DRY-RUN] ${name} gpu=${gpu}"
    echo "  log: ${log_file}"
    echo "  pid: ${pid_file}"
    printf "  cmd: CUDA_VISIBLE_DEVICES=%q DA3_LOG_LEVEL=WARN PYTHONPATH=src %q run --no-capture-output -n %q python -m rtgs.main" "${gpu}" "${CONDA_EXE}" "${CONDA_ENV}"
    printf " %q" "${command_args[@]}"
    printf "\n"
    return 0
  fi

  nohup bash -c '
set -euo pipefail
root_dir="$1"
name="$2"
gpu="$3"
conda_exe="$4"
conda_env="$5"
shift 5
cd "${root_dir}"
echo "[RTGS-ABLATE] START ${name} $(date -Is)"
echo "[RTGS-ABLATE] PHYSICAL_GPU ${gpu}"
printf "[RTGS-ABLATE] CMD: CUDA_VISIBLE_DEVICES=%q DA3_LOG_LEVEL=WARN PYTHONPATH=src %q run --no-capture-output -n %q python -m rtgs.main" "${gpu}" "${conda_exe}" "${conda_env}"
printf " %q" "$@"
printf "\n"
set +e
CUDA_VISIBLE_DEVICES="${gpu}" DA3_LOG_LEVEL=WARN PYTHONPATH=src "${conda_exe}" run --no-capture-output -n "${conda_env}" python -m rtgs.main "$@"
status=$?
set -e
echo "[RTGS-ABLATE] END ${name} status=${status} $(date -Is)"
exit "${status}"
' bash "${ROOT_DIR}" "${name}" "${gpu}" "${CONDA_EXE}" "${CONDA_ENV}" "${command_args[@]}" >"${log_file}" 2>&1 &

  echo "$!" > "${pid_file}"
  echo "[RTGS-ABLATE] launched ${name} gpu=${gpu} pid=$(cat "${pid_file}") log=${log_file}"
}

mapfile -t GPU_ASSIGNMENTS < <(build_gpu_assignments "${#JOB_NAMES[@]}")
if (( ${#JOB_NAMES[@]} == 0 )); then
  echo "[RTGS-ABLATE] No ablation jobs selected. Check ONLY=${ONLY}." >&2
  exit 1
fi
if (( ${#GPU_ASSIGNMENTS[@]} != ${#JOB_NAMES[@]} )); then
  exit 1
fi

for index in "${!JOB_NAMES[@]}"; do
  name="${JOB_NAMES[${index}]}"
  gpu="${GPU_ASSIGNMENTS[${index}]}"
  read -r -a extra_args <<< "${JOB_ARGS[${index}]}"
  run_ablation "${name}" "${gpu}" "${extra_args[@]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[RTGS-ABLATE] dry run complete"
else
  echo "[RTGS-ABLATE] all jobs launched in parallel"
  echo "[RTGS-ABLATE] logs and pid files: ${LOG_DIR}"
fi
