#!/bin/bash
# =============================================================================
# EMAC+ 2-Phase Training Pipeline
# =============================================================================
# Replicates Algorithm 1 from the EMAC+ paper:
#   Phase 1: DAgger with Behavioral Cloning (BC) → produces πref checkpoint
#   Phase 2: DAgger with DPO using πref as frozen reference model
#
# Usage:
#   bash run_full_pipeline.sh
#
# Prerequisites:
#   - conda environment "emac" activated (or available)
#   - AI2-THOR data downloaded
#   - OpenAI API key set for GPT expert
# =============================================================================

set -euo pipefail

PROJECT_DIR="/srv/scratch/z5428797/EMAC-Embodied-Multimodal-Agent-for-Collaborative-Planning-with-VLM-LLM"
DAGGER_SERVER="$PROJECT_DIR/dagger_server.py"
YAML_CONFIG="$PROJECT_DIR/lavis/configs/models/blip2/blip2_emac.yaml"
OUTPUT_BASE="$PROJECT_DIR/output/dagger_server_human_desc"
POLL_INTERVAL=30  # seconds between completion checks

cd "$PROJECT_DIR"

# -----------------------------------------------------------------------------
# Helper: wait for a DONE marker file in the latest output directory
# -----------------------------------------------------------------------------
wait_for_done() {
    local phase_name="$1"
    local pattern="$2"  # glob pattern to find the output dir

    echo "[$phase_name] Waiting for completion..." >&2
    while true; do
        # Find the most recent matching output directory
        local latest_dir
        latest_dir=$(ls -dt $pattern 2>/dev/null | head -1 || true)

        if [[ -n "$latest_dir" && -f "$latest_dir/DONE" ]]; then
            echo "[$phase_name] Complete! Output: $latest_dir" >&2
            echo "$latest_dir"
            return 0
        fi

        sleep "$POLL_INTERVAL"
    done
}

# =============================================================================
# PHASE 1: DAgger with Behavioral Cloning
# =============================================================================
echo "============================================================"
echo "PHASE 1: DAgger with Behavioral Cloning (BC)"
echo "============================================================"

# Ensure BC mode (enable_dpo = False)
sed -i 's/^enable_dpo = True/enable_dpo = False/' "$DAGGER_SERVER"

# Ensure we're not loading a pretrained checkpoint for Phase 1
sed -i 's/^  load_pretrained: True/  load_pretrained: False/' "$YAML_CONFIG"

echo "[Phase 1] Settings:"
grep "enable_dpo" "$DAGGER_SERVER" | head -1
grep "load_pretrained" "$YAML_CONFIG" | head -1

# Launch Phase 1 in background
bash run_dagger.sh --background

# Wait for Phase 1 to finish
BC_OUTPUT_DIR=$(wait_for_done "Phase 1" "$OUTPUT_BASE/with_bc_dpo-False-*")

# Find the latest checkpoint (highest epoch number)
BC_CHECKPOINT=$(ls -t "$BC_OUTPUT_DIR"/emma_checkpoint_*.pth 2>/dev/null | head -1)

if [[ -z "$BC_CHECKPOINT" ]]; then
    echo "ERROR: No checkpoint found in $BC_OUTPUT_DIR"
    exit 1
fi

echo ""
echo "Phase 1 checkpoint (πref): $BC_CHECKPOINT"
echo ""

# Kill Phase 1 tmux session
tmux kill-session -t emac 2>/dev/null || true
sleep 5

# =============================================================================
# PHASE 2: DAgger with DPO (using πref from Phase 1)
# =============================================================================
echo "============================================================"
echo "PHASE 2: DAgger with DPO (using πref from Phase 1)"
echo "============================================================"

# Switch to DPO mode
sed -i 's/^enable_dpo = False/enable_dpo = True/' "$DAGGER_SERVER"

# Enable pretrained loading and set checkpoint paths
sed -i 's/^  load_pretrained: False/  load_pretrained: True/' "$YAML_CONFIG"

# Escape the checkpoint path for sed (handle slashes)
ESCAPED_CKPT=$(echo "$BC_CHECKPOINT" | sed 's/[\/&]/\\&/g')

# Update pretrained and ref_pretrained paths in the YAML config.
# Replace commented-out lines with actual paths from Phase 1.
sed -i '/^  # ref_pretrained:/c\  ref_pretrained: "'"$BC_CHECKPOINT"'"' "$YAML_CONFIG"
sed -i '/^  # pretrained:/c\  pretrained: "'"$BC_CHECKPOINT"'"' "$YAML_CONFIG"
# Also handle the case where they were previously uncommented
sed -i '/^  ref_pretrained:/c\  ref_pretrained: "'"$BC_CHECKPOINT"'"' "$YAML_CONFIG"
sed -i '/^  pretrained:/c\  pretrained: "'"$BC_CHECKPOINT"'"' "$YAML_CONFIG"

echo "[Phase 2] Settings:"
grep "enable_dpo" "$DAGGER_SERVER" | head -1
grep "load_pretrained" "$YAML_CONFIG" | head -1
grep "pretrained" "$YAML_CONFIG" | grep -v "^  #"

# Launch Phase 2 in background
bash run_dagger.sh --background

# Wait for Phase 2 to finish
DPO_OUTPUT_DIR=$(wait_for_done "Phase 2" "$OUTPUT_BASE/with_bc_dpo-True-*")

DPO_CHECKPOINT=$(ls -t "$DPO_OUTPUT_DIR"/emma_checkpoint_*.pth 2>/dev/null | head -1)

# Kill Phase 2 tmux session
tmux kill-session -t emac 2>/dev/null || true

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Phase 1 (BC)  output:     $BC_OUTPUT_DIR"
echo "Phase 1 (BC)  checkpoint: $BC_CHECKPOINT"
echo "Phase 2 (DPO) output:     $DPO_OUTPUT_DIR"
echo "Phase 2 (DPO) checkpoint: $DPO_CHECKPOINT"
echo ""
echo "To verify Phase 2 used DPO, check for W_reward/L_reward in logs:"
echo "  grep 'W_reward' $DPO_OUTPUT_DIR/running_nb01.log"
echo ""

# =============================================================================
# Restore defaults (BC mode, no pretrained) for future manual runs
# =============================================================================
sed -i 's/^enable_dpo = True/enable_dpo = False/' "$DAGGER_SERVER"
sed -i 's/^  load_pretrained: True/  load_pretrained: False/' "$YAML_CONFIG"
echo "Restored dagger_server.py and blip2_emac.yaml to default (BC) settings."
