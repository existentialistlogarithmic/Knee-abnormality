# Training a fold on Google Compute Engine

`train_fold_on_gce.sh` runs **the same generated trainer** as the Kaggle
kernels — `kaggle/<n>/run.py`, byte-identical. Only the paths and the machine
change, so a fold trained here pools with one trained on Kaggle. That
constraint is the whole point and it is why the script shells out to `run.py`
rather than reimplementing anything.

## Read this before spending anything

**GCE is not free.** Colab free is; this is not. The honest reasons to use it
anyway are that it has **no session cap** (Colab reclaims sessions mid-fold),
**no 15 GB Drive ceiling** (the cache alone is ~10 GB), and it runs unattended.

**MEASURED 2026-09-03 on this project: `GPUS_ALL_REGIONS = 0`.** No GPU VM
can start, Spot included. The request is a human review taking hours to
days, which on that date was *longer* than the wait it would have avoided —
the Kaggle quota reset was ~2 days out and delivers 30 GPU-h against the
7.5 the distilled lineage needs. File the increase for later weeks; do not
wait on it for a run the weekly reset already covers.

**A new GCP project has zero GPU quota.** `GPUS_ALL_REGIONS` starts at 0 and an
increase is a request that can take hours or days to approve. Check it *first* —
if it is not already granted, waiting for the Kaggle weekly reset is likely to
be faster than waiting for the quota:

```bash
gcloud compute project-info describe --flatten="quotas[]" --format="csv[no-heading](quotas.metric,quotas.limit)" | grep -i gpu
gcloud compute regions describe us-central1 --flatten="quotas[]" --format="csv[no-heading](quotas.metric,quotas.limit)" | grep -i gpu
```

**Two quotas are needed, not one**, and asking for only the first is the usual
way this fails a second time after the wait:

| quota | scope | why |
|---|---|---|
| `GPUS_ALL_REGIONS` | global | necessary, not sufficient |
| `NVIDIA_T4_GPUS` | the region you create in | the VM will not start without it |
| `PREEMPTIBLE_NVIDIA_T4_GPUS` | same region | separate quota, and the create command above uses Spot |

A quota increase on a project with **no billing account is auto-denied**, so
check that first — `gcloud beta billing projects describe PROJECT_ID` must say
`billingEnabled: true`. A free-trial credit counts as an account.

**Costs are approximate and change — check current pricing before committing.**
An n1-standard-8 with one T4 is on the order of $0.70–0.80/hour on demand and
roughly a fifth of that as Spot. A fold is ~2 h on a single T4, so five folds is
~10–12 h of GPU time. A $300 new-account credit covers that comfortably; a
forgotten VM does not.

**The script stops the VM on exit, including on failure.** That trap is the most
important line in it. A machine left running after a two-hour fold is the
expensive failure mode and nobody notices until the bill.

## Create the VM

The Kaggle token goes in as instance metadata and is **never committed**. Pass
it from your own environment:

```bash
export KAGGLE_API_TOKEN=...        # your token, not in this repo

gcloud compute instances create knee-fold0 \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --metadata-from-file=startup-script=gcp/train_fold_on_gce.sh \
  --metadata=fold=0,lineage=v1distil,kaggle-token="$KAGGLE_API_TOKEN"
```

`--provisioning-model=SPOT` is much cheaper and can be preempted. The trainer
checkpoints as it goes and takes `--time-budget`, so a preemption costs the tail
of a run rather than the run. Drop the flag if you would rather pay for
certainty.

Watch it:

```bash
gcloud compute instances get-serial-port-output knee-fold0 --zone=us-central1-a | tail -40
```

Five folds means five VMs (change `fold=` and the name), or one VM run five
times. They are independent.

## What comes back

Checkpoints upload to `achelijndiamantidis/knee-gce-v1distil` as a private
Kaggle dataset. Inference must still run in a Kaggle notebook with internet off
(`FINDINGS.md` 2.7), so mount that dataset from `kaggle/70_infer_v1distil` —
by editing `src/pipeline.py` and regenerating, never by hand-editing `run.py`.

Note `70_infer_v1distil` asserts `MEMBERS_EXPECTED = 5`. It refuses to run on
four checkpoints rather than quietly submitting a four-member ensemble against a
five-member claim (E067).

## The rule that is easy to break here

`FINDINGS.md` 2.7: external training is allowed, only inference must be a Kaggle
notebook. `FINDINGS.md` 2.15, verbatim: *"You cannot sign up to Kaggle from
multiple accounts and therefore you cannot enter or submit from multiple
accounts."* Renting your own GPUs is extra compute for one entrant. A second
Kaggle account is not, whoever pays for it.
