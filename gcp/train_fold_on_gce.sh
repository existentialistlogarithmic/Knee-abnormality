#!/usr/bin/env bash
# Train one fold on a Google Compute Engine GPU VM, then ship the weights to
# Kaggle and shut the machine down.
#
# Runs THE SAME generated trainer as the Kaggle kernels — kaggle/<n>/run.py,
# byte-identical. Only the paths and the machine change, so a fold trained here
# pools with one trained on Kaggle. A GCE-specific training loop would make
# every number in docs/EXPERIMENTS.md incomparable, which is the entire reason
# this script shells out instead of reimplementing anything.
#
# Used as a startup-script, so it runs unattended and the VM stops itself. A VM
# left running after a 2-hour fold is the expensive failure mode here, and it is
# the one nobody notices until the bill.
#
# Required instance metadata:
#   kaggle-token   the KAGGLE_API_TOKEN value
#   fold           0-4
# Optional:
#   lineage        default v1distil
#   branch         default claude/rsna-knee-abnormality-jahn5n
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

FOLD=$(meta fold);            FOLD=${FOLD:-0}
LINEAGE=$(meta lineage);      LINEAGE=${LINEAGE:-v1distil}
BRANCH=$(meta branch);        BRANCH=${BRANCH:-claude/rsna-knee-abnormality-jahn5n}
TOKEN=$(meta kaggle-token)
[ -n "$TOKEN" ] || { echo "FATAL: no kaggle-token in instance metadata"; exit 1; }

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
# Stop the VM whatever happens. An abort that leaves the machine up costs more
# than the run did.
trap 'log "shutting down"; shutdown -h +1' EXIT

log "fold=$FOLD lineage=$LINEAGE branch=$BRANCH"
nvidia-smi || { echo "FATAL: no GPU on this VM — check the accelerator flag"; exit 1; }

# --- environment ----------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get -qq update && apt-get -qq install -y git python3-pip >/dev/null
pip3 -q install --upgrade pip
pip3 -q install kaggle pandas numpy pyarrow scikit-learn pydicom
python3 -c "import torch" 2>/dev/null || pip3 -q install torch torchvision
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU; wrong image or driver"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
PY

export KAGGLE_API_TOKEN="$TOKEN"
mkdir -p /root/.kaggle && printf '%s' "$TOKEN" > /root/.kaggle/access_token
chmod 600 /root/.kaggle/access_token

# --- code, cache, labels --------------------------------------------------
cd /opt
rm -rf Knee-abnormality
git clone -q https://github.com/existentialistlogarithmic/Knee-abnormality.git
cd Knee-abnormality && git checkout -q "$BRANCH"
# The same gate CI runs. A drifted run.py still trains and is not the pipeline.
python3 eda/generate_kernels.py --check

DATA=/mnt/disks/data
mkdir -p "$DATA"/{cache,labels,out}
if [ ! -f "$DATA/labels/train.csv" ]; then
  log "downloading labels and train.csv"
  case "$LINEAGE" in
    v1distil) DATASET=achelijndiamantidis/knee-phase1-distilled ;;
    v1pub)    DATASET=achelijndiamantidis/knee-phase1-public ;;
    *) echo "FATAL: unknown lineage $LINEAGE"; exit 1 ;;
  esac
  kaggle datasets download "$DATASET" -p "$DATA/labels" --unzip
  kaggle competitions download rsna-knee-abnormality-detection -f train.csv -p "$DATA/labels"
fi

if [ -z "$(ls -A "$DATA/cache" 2>/dev/null)" ]; then
  log "downloading the 192px cache (~10 GB, 4 shards)"
  for shard in 0 1 2 3; do
    kaggle kernels output "achelijndiamantidis/knee-cache-build-$shard" -p "$DATA/cache"
  done
fi
log "cache: $(du -sh "$DATA/cache" | cut -f1), $(ls "$DATA/cache"/*.npy 2>/dev/null | wc -l) volumes"

# --- train ----------------------------------------------------------------
declare -A DIRS=(
  [v1distil0]=kaggle/65_train_v1distil_fold0 [v1distil1]=kaggle/66_train_v1distil_fold1
  [v1distil2]=kaggle/67_train_v1distil_fold2 [v1distil3]=kaggle/68_train_v1distil_fold3
  [v1distil4]=kaggle/69_train_v1distil_fold4
)
DIR=${DIRS[${LINEAGE}${FOLD}]:-}
[ -n "$DIR" ] || { echo "FATAL: no kernel directory for $LINEAGE fold $FOLD"; exit 1; }

OUT="$DATA/out/${LINEAGE}_fold${FOLD}"
mkdir -p "$OUT"
log "training $DIR"
# Nothing about the model is set here on purpose. Epochs, batch, LR, backbone
# and geometry come from the generated script's own constants, which came from
# src/pipeline.py. Overriding any of them is how this stops being comparable.
python3 "$DIR/run.py" \
  --fold "$FOLD" \
  --cache "$DATA/cache" \
  --labels "$DATA/labels" \
  --headers "$DATA/labels" \
  --train-csv "$DATA/labels/train.csv" \
  --out "$OUT" \
  --time-budget $((6 * 3600)) 2>&1 | tee "$OUT/train.log"

# --- ship the weights back ------------------------------------------------
log "uploading checkpoints"
UP=/tmp/upload && mkdir -p "$UP"
cp "$OUT"/*.pt "$OUT"/gold_oof_*.json "$UP"/ 2>/dev/null || true
cat > "$UP/dataset-metadata.json" <<META
{"title": "Knee - GCE-trained ${LINEAGE} checkpoints",
 "id": "achelijndiamantidis/knee-gce-${LINEAGE}",
 "licenses": [{"name": "other"}]}
META
kaggle datasets create -p "$UP" --dir-mode zip \
  || kaggle datasets version -p "$UP" -m "fold $FOLD" --dir-mode zip

log "done — fold $FOLD complete"
