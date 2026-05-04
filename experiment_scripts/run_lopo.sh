#!/usr/bin/env bash
# run_lopo.sh — Full LOPO evaluation (E1-E5) with Discord notifications.
#
# Sends a Discord message when:
#   - A seed finishes (with per-fold AUPRC summary)
#   - An experiment finishes (with full metrics)
#   - Something fails (with last few log lines)
#
# Writes status checkpoints to results/lopo_status/
#
# Usage (on the uni machine, from project root):
#   nohup bash experiment_scripts/run_lopo.sh &> lopo.log &
#   nohup bash experiment_scripts/run_lopo.sh e2 e3 e4 e5 &> lopo.log &
#   bash experiment_scripts/run_lopo.sh e3 e4

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv
source "$PROJECT_ROOT/.venv/bin/activate"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-}"
if [ -z "$DISCORD_WEBHOOK" ] && [ -f "$PROJECT_ROOT/.discord_webhook" ]; then
    DISCORD_WEBHOOK="$(cat "$PROJECT_ROOT/.discord_webhook")"
fi
if [ -z "$DISCORD_WEBHOOK" ]; then
    echo "WARNING: No Discord webhook configured. Set DISCORD_WEBHOOK or create .discord_webhook"
fi
STATUS_DIR="$PROJECT_ROOT/results/lopo_status"
SEEDS=(42 123 456)

mkdir -p "$STATUS_DIR"

# --- Discord helpers ---

discord() {
    local msg="$1"
    local color="${2:-3447003}"
    curl -s -H "Content-Type: application/json" \
        -d "{\"embeds\":[{\"title\":\"LOPO\",\"description\":\"$msg\",\"color\":$color}]}" \
        "$DISCORD_WEBHOOK" > /dev/null 2>&1 || true
}

# --- Results extraction (called after each seed/experiment) ---

summarize_seed() {
    local exp="$1"
    local seed="$2"
    python3 -c "
import json, glob, numpy as np, os
os.chdir('$PROJECT_ROOT')
# E2 saves as results_smote.json / results_adasyn.json
patterns = ['results/$exp/seed_$seed/fold_*/results.json',
            'results/$exp/seed_$seed/fold_*/results_smote.json',
            'results/$exp/seed_$seed/fold_*/results_adasyn.json']
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))
if not files:
    print('No fold results found yet')
else:
    auprcs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        auprcs.append(d['test_metrics']['auprc'])
    n = len(auprcs)
    print(f'{n} results — AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f} (min {np.min(auprcs):.4f}, max {np.max(auprcs):.4f})')
" 2>/dev/null || echo "Could not read results"
}

summarize_experiment() {
    local exp="$1"
    python3 -c "
import json, glob, numpy as np, os
os.chdir('$PROJECT_ROOT')
seed_means = []
for seed in [42, 123, 456]:
    files = []
    for pattern in [f'results/$exp/seed_{seed}/fold_*/results.json',
                    f'results/$exp/seed_{seed}/fold_*/results_smote.json',
                    f'results/$exp/seed_{seed}/fold_*/results_adasyn.json']:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        continue
    auprcs = [json.load(open(f))['test_metrics']['auprc'] for f in files]
    seed_means.append(np.mean(auprcs))
if not seed_means:
    print('No results found')
else:
    print(f'Mean AUPRC across seeds: {np.mean(seed_means):.4f} +/- {np.std(seed_means):.4f}')
    print(f'Per-seed means: {\" / \".join(f\"{m:.4f}\" for m in seed_means)}')
    seed_aurocs = []
    for seed in [42, 123, 456]:
        files = []
        for pattern in [f'results/$exp/seed_{seed}/fold_*/results.json',
                        f'results/$exp/seed_{seed}/fold_*/results_smote.json',
                        f'results/$exp/seed_{seed}/fold_*/results_adasyn.json']:
            files.extend(glob.glob(pattern))
        files = sorted(set(files))
        if not files:
            continue
        aurocs = [json.load(open(f))['test_metrics']['auroc'] for f in files]
        seed_aurocs.append(np.mean(aurocs))
    if seed_aurocs:
        print(f'Mean AUROC across seeds: {np.mean(seed_aurocs):.4f} +/- {np.std(seed_aurocs):.4f}')
" 2>/dev/null || echo "Could not read results"
}

# --- Core runner ---

run_experiment() {
    local exp="$1"
    local gen_model="$2"      # timegan/cvae/ldm or "none" for E1, "builtin" for E2
    local aug_flag="$3"       # adasyn/smote or "" for non-E2
    local t_start=$SECONDS

    discord "▶️ **${exp^^}** LOPO started ($(date '+%d %b %H:%M'))" 3447003
    echo "STARTED $(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS_DIR/${exp}.status"

    for seed in "${SEEDS[@]}"; do
        local seed_start=$SECONDS

        # Generator phase (E3-E5)
        if [ "$gen_model" != "none" ] && [ "$gen_model" != "builtin" ]; then
            echo ">>> ${exp^^} seed $seed: generate ($gen_model) — $(date)"

            local gen_cmd="python -m training.generate --model $gen_model --mode lopo --seed $seed"
            if [ "$gen_model" = "ldm" ]; then
                local cvae_ckpt="results/e4/seed_${seed}/fold_00/cvae.pt"
                if [ ! -f "$cvae_ckpt" ]; then
                    discord "❌ **${exp^^}** FAILED — E4 per-fold CVAE checkpoints missing. Run E4 LOPO first." 15158332
                    echo "FAILED $(date '+%Y-%m-%d %H:%M:%S') missing e4 cvae checkpoints" > "$STATUS_DIR/${exp}.status"
                    return 1
                fi
                gen_cmd="$gen_cmd --cvae-checkpoint $cvae_ckpt"
            fi

            if ! eval "$gen_cmd"; then
                local err_tail
                err_tail=$(tail -8 "$PROJECT_ROOT/lopo.log" 2>/dev/null)
                discord "❌ **${exp^^}** FAILED — generate seed=$seed\\n\`\`\`\\n${err_tail}\\n\`\`\`" 15158332
                echo "FAILED $(date '+%Y-%m-%d %H:%M:%S') generate seed=$seed" > "$STATUS_DIR/${exp}.status"
                return 1
            fi
        fi

        # Detector training phase
        echo ">>> ${exp^^} seed $seed: train LOPO — $(date)"

        local train_cmd="python -m training.train --experiment $exp --mode lopo --seeds $seed"
        [ -n "$aug_flag" ] && train_cmd="$train_cmd --augmentation $aug_flag"

        if ! eval "$train_cmd"; then
            local err_tail
            err_tail=$(tail -8 "$PROJECT_ROOT/lopo.log" 2>/dev/null)
            discord "❌ **${exp^^}** FAILED — train seed=$seed\\n\`\`\`\\n${err_tail}\\n\`\`\`" 15158332
            echo "FAILED $(date '+%Y-%m-%d %H:%M:%S') train seed=$seed" > "$STATUS_DIR/${exp}.status"
            return 1
        fi

        # Seed done — summarize and notify
        local seed_elapsed=$(( (SECONDS - seed_start) / 60 ))
        local seed_summary
        seed_summary=$(summarize_seed "$exp" "$seed")

        discord "✔️ **${exp^^}** seed $seed done (${seed_elapsed}min)\\n$seed_summary" 3066993
        echo "  SEED $seed DONE $(date '+%Y-%m-%d %H:%M:%S') (${seed_elapsed}min)" >> "$STATUS_DIR/${exp}.status"
    done

    # Experiment done — full summary
    local exp_elapsed=$(( (SECONDS - t_start) / 60 ))
    local exp_summary
    exp_summary=$(summarize_experiment "$exp")

    discord "✅ **${exp^^}** LOPO COMPLETE — ${exp_elapsed}min\\n$exp_summary" 5763719
    echo "DONE $(date '+%Y-%m-%d %H:%M:%S') (${exp_elapsed}min)" >> "$STATUS_DIR/${exp}.status"
    echo "$exp_summary" >> "$STATUS_DIR/${exp}.status"
}

# --- E2 is special (two augmentation methods) ---

run_e2_lopo() {
    local t_start=$SECONDS
    discord "▶️ **E2** LOPO started — SMOTE + ADASYN ($(date '+%d %b %H:%M'))" 3447003
    echo "STARTED $(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS_DIR/e2.status"

    for seed in "${SEEDS[@]}"; do
        local seed_start=$SECONDS

        echo ">>> E2 SMOTE seed $seed: LOPO — $(date)"
        if ! python -m training.train --experiment e2 --augmentation smote --mode lopo --seeds "$seed"; then
            local err_tail
            err_tail=$(tail -8 "$PROJECT_ROOT/lopo.log" 2>/dev/null)
            discord "❌ **E2 SMOTE** FAILED seed=$seed\\n\`\`\`\\n${err_tail}\\n\`\`\`" 15158332
            echo "FAILED $(date '+%Y-%m-%d %H:%M:%S') smote seed=$seed" > "$STATUS_DIR/e2.status"
            return 1
        fi

        echo ">>> E2 ADASYN seed $seed: LOPO — $(date)"
        if ! python -m training.train --experiment e2 --augmentation adasyn --mode lopo --seeds "$seed"; then
            local err_tail
            err_tail=$(tail -8 "$PROJECT_ROOT/lopo.log" 2>/dev/null)
            discord "❌ **E2 ADASYN** FAILED seed=$seed\\n\`\`\`\\n${err_tail}\\n\`\`\`" 15158332
            echo "FAILED $(date '+%Y-%m-%d %H:%M:%S') adasyn seed=$seed" > "$STATUS_DIR/e2.status"
            return 1
        fi

        local seed_elapsed=$(( (SECONDS - seed_start) / 60 ))
        discord "✔️ **E2** seed $seed done (${seed_elapsed}min) — SMOTE + ADASYN both complete" 3066993
        echo "  SEED $seed DONE $(date '+%Y-%m-%d %H:%M:%S') (${seed_elapsed}min)" >> "$STATUS_DIR/e2.status"
    done

    local exp_elapsed=$(( (SECONDS - t_start) / 60 ))
    discord "✅ **E2** LOPO COMPLETE — ${exp_elapsed}min" 5763719
    echo "DONE $(date '+%Y-%m-%d %H:%M:%S') (${exp_elapsed}min)" >> "$STATUS_DIR/e2.status"
}

# --- Main ---

TOTAL_START=$SECONDS

if [ $# -eq 0 ] || [ "$1" = "all" ]; then
    EXPERIMENTS=(e1 e2 e3 e4 e5)
else
    EXPERIMENTS=("$@")
fi

FILTER="${EXPERIMENTS[*]}"

echo "============================================"
echo "  LOPO Evaluation — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Experiments: $FILTER"
echo "============================================"

# Free disk space: delete single-split synthetic npz files (results already saved).
# These are ~380 MB each and no longer needed — fidelity plots and detector training
# already completed for single-split. Frees ~3.4 GB for LOPO npz files.
FREED=0
for npz in "$PROJECT_ROOT"/results/e*/seed_*/single_split/synthetic_ratio_*.npz; do
    if [ -f "$npz" ]; then
        rm -f "$npz"
        FREED=$((FREED + 1))
    fi
done
if [ $FREED -gt 0 ]; then
    echo "  Freed disk: deleted $FREED single-split synthetic .npz files"
fi

discord "🚀 **LOPO run started** — $FILTER ($(date '+%d %b %H:%M'))\\n3 seeds × 23 folds each" 3447003

FAILED=0
for exp in "${EXPERIMENTS[@]}"; do
    case "$exp" in
        e1) run_experiment e1 none "" ;;
        e2) run_e2_lopo ;;
        e3) run_experiment e3 timegan "" ;;
        e4) run_experiment e4 cvae "" ;;
        e5) run_experiment e5 ldm "" ;;
        *)  echo "Unknown: $exp"; FAILED=1; continue ;;
    esac

    if [ $? -ne 0 ]; then
        FAILED=1
        echo "!!! $exp failed, moving to next"
        continue
    fi
done

TOTAL_ELAPSED=$(( (SECONDS - TOTAL_START) / 60 ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_REMAINING=$(( TOTAL_ELAPSED % 60 ))

if [ $FAILED -eq 0 ]; then
    discord "🏁 **ALL LOPO COMPLETE** — ${TOTAL_HOURS}h${TOTAL_REMAINING}m total ($(date '+%d %b %H:%M'))" 5763719
    echo "ALL_DONE $(date '+%Y-%m-%d %H:%M:%S') (${TOTAL_HOURS}h${TOTAL_REMAINING}m)" > "$STATUS_DIR/overall.status"
else
    discord "⚠️ **LOPO finished with failures** — ${TOTAL_HOURS}h${TOTAL_REMAINING}m. Check \`results/lopo_status/\`" 15105570
    echo "PARTIAL $(date '+%Y-%m-%d %H:%M:%S') (${TOTAL_HOURS}h${TOTAL_REMAINING}m)" > "$STATUS_DIR/overall.status"
fi

echo ""
echo "============================================"
echo "  LOPO finished — ${TOTAL_HOURS}h${TOTAL_REMAINING}m"
echo "  Status: $STATUS_DIR/"
echo "============================================"
