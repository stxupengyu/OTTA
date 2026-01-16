#!/usr/bin/env bash
# SCOTTA full experiment runner
# Runs four datasets with paper-reported hyperparameters

set -euo pipefail

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_DIR="$(dirname "$SCRIPT_DIR")"

# Python interpreter (override with PYTHON env var)
PYTHON="${PYTHON:-python3}"

# Base command
BASE_CMD=("$PYTHON" "$RAG_DIR/main.py")

# Log file
LOG_FILE="$RAG_DIR/scripts/run_all.log"

# Run a single dataset
run_experiment() {
    local dataset=$1
    local model_type=${2:-gpt3.5}  # Default gpt3.5

    echo "=========================================="
    echo "Starting: Dataset=${dataset}, Model=${model_type}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # Dataset-specific hyperparameters (from paper)
    case "$dataset" in
        movie|MOVIE)
            local B=64
            local k=4
            local tau=0.80
            local l3r_eps=1e-8
            local l3r_alpha=5.0
            local lambda1=0.35
            local lambda2=0.35
            local lambda3=0.30
            ;;
        aapd|AAPD)
            local B=128
            local k=6
            local tau=0.72
            local l3r_eps=1e-8
            local l3r_alpha=5.0
            local lambda1=0.40
            local lambda2=0.30
            local lambda3=0.30
            ;;
        rcv|RCV|rcv1|RCV1)
            local B=256
            local k=6
            local tau=0.70
            local l3r_eps=1e-8
            local l3r_alpha=5.0
            local lambda1=0.40
            local lambda2=0.30
            local lambda3=0.30
            ;;
        se|SE|stackexchange|StackExchange)
            local B=512
            local k=8
            local tau=0.62
            local l3r_eps=1e-8
            local l3r_alpha=5.0
            local lambda1=0.50
            local lambda2=0.25
            local lambda3=0.25
            ;;
        *)
            echo "Error: unsupported dataset: $dataset"
            echo "Supported datasets: movie, aapd, rcv, se"
            return 1
            ;;
    esac

    # Build command
    local cmd=(
        "${BASE_CMD[@]}"
        --dataset "$dataset"
        --model-type "$model_type"
        --mode rag
        --bank-type smb
        --use-label-desc
        --cache-size "$B"
        --rag-k "$k"
        --tau "$tau"
        --l3r-eps "$l3r_eps"
        --l3r-alpha "$l3r_alpha"
        --smb-lambda1 "$lambda1"
        --smb-lambda2 "$lambda2"
        --smb-lambda3 "$lambda3"
        --conf-type l3r
        --l3r-agg mean
    )

    echo "Hyperparameters:"
    echo "  Memory capacity B: $B"
    echo "  Retrieved demos k: $k"
    echo "  Confidence threshold tau: $tau"
    echo "  L3R smoothing eps: $l3r_eps"
    echo "  L3R sharpness alpha: $l3r_alpha"
    echo "  SMB weights (lambda1, lambda2, lambda3): ($lambda1, $lambda2, $lambda3)"
    echo ""
    echo "Command:"
    echo "  ${cmd[*]}"
    echo ""

    # Run experiment
    if "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        echo ""
        echo "OK: Dataset=${dataset}, Model=${model_type}"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        return 0
    else
        echo ""
        echo "FAIL: Dataset=${dataset}, Model=${model_type}"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        return 1
    fi
}

# Main
main() {
    local datasets=("movie" "aapd" "rcv" "se")
    local model_type=${1:-gpt3.5}  # First arg is model type

    echo "=========================================="
    echo "SCOTTA full experiment runner"
    echo "=========================================="
    echo "Model type: $model_type"
    echo "Datasets: ${datasets[*]}"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log file: $LOG_FILE"
    echo "=========================================="
    echo ""

    # Create log file
    touch "$LOG_FILE"
    echo "Run start: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "Model type: $model_type" >> "$LOG_FILE"
    echo "Dataset list: ${datasets[*]}" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    local success_count=0
    local fail_count=0
    local failed_datasets=()

    # Iterate datasets
    for dataset in "${datasets[@]}"; do
        if run_experiment "$dataset" "$model_type"; then
            ((success_count++))
        else
            ((fail_count++))
            failed_datasets+=("$dataset")
        fi

        # Short pause between datasets
        if [ "$dataset" != "${datasets[-1]}" ]; then
            echo "Waiting 5 seconds before the next dataset..."
            sleep 5
        fi
    done

    # Summary
    echo "=========================================="
    echo "Summary"
    echo "=========================================="
    echo "Total datasets: ${#datasets[@]}"
    echo "Succeeded: $success_count"
    echo "Failed: $fail_count"
    if [ $fail_count -gt 0 ]; then
        echo "Failed datasets: ${failed_datasets[*]}"
    fi
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    echo "" >> "$LOG_FILE"
    echo "Run end: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "Succeeded: $success_count, Failed: $fail_count" >> "$LOG_FILE"

    # Return non-zero on failure
    if [ $fail_count -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# Run main when executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
