#!/usr/bin/env bash
# Push kernels as concurrency slots free up.
#
# Kaggle allows 5 concurrent batch CPU sessions and rejects the sixth outright
# rather than queueing it, so a wider fan-out needs this. Polls slowly: the API
# rate-limits readily and a cache shard runs for many minutes anyway.
set -u
POLL=${POLL:-90}
for folder in "$@"; do
  slug=$(python3 -c "import json,sys;print(json.load(open('$folder/kernel-metadata.json'))['id'])")
  while true; do
    out=$(kaggle kernels push -p "$folder" 2>&1 | tail -1)
    if echo "$out" | grep -q "successfully pushed"; then
      echo "pushed  $slug"
      break
    fi
    if echo "$out" | grep -q "Maximum batch"; then
      echo "waiting for a slot: $slug"
      sleep "$POLL"
      continue
    fi
    # A weekly-quota refusal also starts with "Maximum" but no amount of waiting
    # fixes it, so it must not be mistaken for a busy slot. Stop the whole queue
    # rather than the current item: every remaining push would fail the same way.
    if echo "$out" | grep -q "weekly.*quota"; then
      echo "QUOTA EXHAUSTED: $out"
      echo "nothing further can run until the weekly allowance resets; "\
           "check the Kaggle account page, the API does not report it"
      exit 2
    fi
    echo "FAILED  $slug: $out"
    break
  done
  sleep 10
done
echo "queue drained"
