# EMAC+ Small-Scale Replication — Full Documentation

This documents all changes made to replicate the EMAC+ 2-phase training pipeline
(Algorithm 1, Table 9) at small scale (2 rounds x 3 environments per phase).

---

## 1. Architecture Overview

EMAC+ is a **VLM + LLM** collaborative system for embodied household tasks in ALFWorld (AI2-THOR).

### How It Works

```
                        ┌─────────────────────────────────────────────┐
                        │           EMMA Model (the VLM)              │
  AI2-THOR image ──────>│  ViT ─> Q-Former ─> Projection ─> Vicuna-7B│──> VLM Action
  Text history ────────>│  (frozen)  (frozen)  (TRAINABLE)   (frozen) │    (sent to env)
                        └─────────────────────────────────────────────┘

  Text history ────────> GPT-5.2 Expert (LLM) ──> Expert Action (training label only)
```

1. The **VLM (EMMA)** sees the image + text history and generates an action
2. The **LLM Expert (GPT-5.2)** sees only the text history and generates the "correct" action
3. The VLM's action is sent to the AI2-THOR environment
4. The LLM's action is used as the training label to improve the VLM
5. Only the **projection layer** (768-dim → 4096-dim MLP) is trained — everything else is frozen

### Two LLMs with Different Roles

| Component | Model | Role | Trainable? |
|-----------|-------|------|------------|
| VLM language backbone | Vicuna-7B-v1.1 (Llama 1 fine-tune) | Decoder inside BLIP-2 — generates action text from image tokens + text | No (frozen) |
| Expert teacher | GPT-5.2 (originally `text-davinci-003`) | Provides ground-truth action labels for training | N/A (external API) |
| Only trainable part | Projection layer (MLP) | Maps Q-Former output to Vicuna's embedding space | **Yes** |

### The DAgger Loop

Each environment episode:
1. Client (`alfworld_client.py`) resets AI2-THOR, sends observation + image to server
2. Server (`dagger_server.py`) runs VLM and LLM, sends VLM's action back to client
3. Client executes action in AI2-THOR, sends new observation back
4. Repeat for up to 50 steps
5. After the episode, server trains VLM using LLM's actions as labels

---

## 2. Two-Phase Training Pipeline

### Phase 1 — Behavioral Cloning (BC)

"Copy the expert."

```
Training pair:
  Input:  image + text history
  Label:  "go to countertop 1"  (what the LLM expert said)
  Loss:   cross-entropy between VLM output and expert label
```

The VLM learns by imitating the LLM's actions directly. This produces the
**reference policy checkpoint (πref)**.

### Phase 2 — DPO (Direct Preference Optimization)

"Learn which answer is better vs worse."

```
Training pair:
  Input:   image + text history
  Winner:  "go to countertop 1"  (LLM expert — good action)
  Loser:   "you, a hand, a hand" (VLM's own output — bad action)
  Loss:    increase probability of Winner, decrease Loser
           (relative to frozen reference model πref)
```

DPO refines the model beyond simple imitation by teaching it to distinguish
good actions from bad ones. It uses the Phase 1 checkpoint as a frozen reference
to measure improvement.

### Why Two Phases?

Phase 1 (BC) gets the VLM from random garbage → somewhat reasonable.
Phase 2 (DPO) refines it from reasonable → better.
DPO can't work from scratch because both winner and loser would be garbage.

### Pipeline Summary

```
Phase 1 (BC):  enable_dpo=False, load_pretrained=False
  Round 0: Env 0 → Env 1 → Env 2 → train → checkpoint_0
  Round 1: Env 0 → Env 1 → Env 2 → train → checkpoint_1 (= πref)

Phase 2 (DPO): enable_dpo=True, load_pretrained=True, ref_pretrained=πref
  Round 0: Env 0 → Env 1 → Env 2 → DPO train → checkpoint_0
  Round 1: Env 0 → Env 1 → Env 2 → DPO train → checkpoint_1 (final)
```

---

## 3. Key Configuration

### dagger_server.py — Experiment Settings

| Parameter | Small-scale | Paper Scale | Description |
|-----------|-------------|-------------|-------------|
| `num_rounds` | 2 | 6 (BC) / 12 (DPO) | DAgger training rounds |
| `num_envs` | 3 | 134 | Household tasks per round |
| `enable_dpo` | False (Phase 1) / True (Phase 2) | — | Toggle BC vs DPO |
| `enable_tc` | False | — | Thought chain imitation |
| `batch_size` (BC) | 8 | 128 | Training batch size for BC |
| `batch_size` (DPO) | 1 | 4 | Training batch size for DPO (reduced to avoid OOM) |
| `accum_grad_steps` | 4 | 4 | Gradient accumulation (effective batch = batch_size × accum) |
| `train_iters_per_epoch` | 5 | 5 | Gradient steps per environment |
| `max_steps` (client) | 50 | 50 | Max actions per episode |

### Batch Size Explained

Batch size = how many examples the model processes simultaneously per step.

- **batch_size=1**: Process 1 image+text pair → compute loss → accumulate gradient
- **batch_size=4**: Process 4 pairs at once → compute average loss → accumulate gradient
- **accum_grad_steps=4**: Accumulate 4 mini-batches before updating weights

With batch_size=1 and accum_grad_steps=4, the effective batch size is 4 but memory
usage equals batch_size=1. This is why we reduced DPO batch from 4→1 after OOM.

### blip2_emac.yaml — Model Config

| Setting | Value | Purpose |
|---------|-------|---------|
| `load_basemodel` | True | Load InstructBLIP base weights for ViT + Q-Former |
| `load_pretrained` | False (Phase 1) / True (Phase 2) | Load BC checkpoint for DPO |
| `pretrained` | Path to BC checkpoint | Initializes model weights for Phase 2 |
| `ref_pretrained` | Path to BC checkpoint | Initializes frozen reference model for DPO |
| `llm_model` | vicuna-7b-v1.1 | Frozen language backbone |
| `low_memory` | True | Memory optimization |
| `freeze_vit` | True | ViT encoder is frozen |
| `freeze_qformer` | True | Q-Former is frozen |
| `freeze_proj_layer` | False | **Projection layer is trainable** |

---

## 4. All Files

### Files Created

| File | Purpose |
|------|---------|
| `alfworld_client.py` | AI2-THOR client — runs environments, sends observations/images to server via HTTP |
| `run_dagger.sh` | Launches server + client in tmux. Supports `--background` flag for automation |
| `run_full_pipeline.sh` | Automates full 2-phase pipeline: BC → checkpoint handoff → DPO |
| `test_thor.py` | Diagnostic script for testing AI2-THOR environment reset |
| `xrl_alignment/__init__.py` | Package init |
| `xrl_alignment/utils.py` | Core utilities (see below) |
| `CHANGES.md` | This documentation file |

### Files Modified

| File | Change |
|------|--------|
| `dagger_server.py` | Reduced scale (2×3), GPT-5.2 expert, `DONE` marker, DPO batch_size=1, English comments |
| `generate_reflections.py` | Updated default model to `gpt-5.2-2025-12-11` |
| `lavis/configs/models/blip2/blip2_emac.yaml` | Added `vicuna7b` model type, DPO settings, checkpoint paths |
| `xrl_alignment/utils.py` | System prompt fix for GPT-5.2 chat model compatibility |
| `environment.yml` | Added `h5py`, `tiktoken`, `tenacity` dependencies |

### xrl_alignment/utils.py — Key Components

| Class/Function | Purpose |
|----------------|---------|
| `EnvironmentHistory` | Tracks base prompt + few-shot examples + action/observation history |
| `ReplayBuffer` | Stores (image, text_input, text_output) tuples for BC training |
| `DPOReplayBuffer` | Stores (image, text_input, winner, loser) tuples for DPO training |
| `save_checkpoint()` | Saves model weights, optimizer state, scaler state to `.pth` file |
| `release_memory()` | Clears GPU cache and runs garbage collection |
| `llm_forward()` | Calls LLM expert with retry logic (up to 6 tries with increasing temperature) |
| `get_chat()` | OpenAI ChatCompletion API wrapper with system prompt and token counting |
| `get_completion()` | Legacy OpenAI Completion API wrapper (for text-davinci-003) |

---

## 5. Fixes Applied

### Fix 1: GPT-5.2 Chat Model Compatibility

**Problem**: The original paper used `text-davinci-003` (a completion model) that naturally
continues few-shot text patterns. GPT-5.2 is a chat model that would return:
- Empty strings (started response with `\n`, got truncated by stop-sequence logic)
- Explanatory text (`"tank" refers to the toilet tank...`) instead of actions
- Multi-step trajectories instead of single actions

**Fix**: Added a system message to `get_chat()` in `xrl_alignment/utils.py`:

```python
messages = [
    {
        "role": "system",
        "content": "You are a text-completion engine for an interactive household task game. "
                   "Output ONLY the next single action on one line... "
                   "Do NOT include '>' prefix, explanations, or multiple actions."
    },
    {"role": "user", "content": prompt}
]
```

**Result**: LLM now consistently outputs proper actions (`go to countertop 1`) instead of
empty strings or explanations.

### Fix 2: DPO Out-of-Memory (OOM) Crash

**Problem**: DPO training with batch_size=4 used all 143GB of H200 VRAM and hung silently.
The log showed `Start Training` but no training metrics appeared. GPU showed 99.7% memory
usage and 0% utilization.

**Cause**: DPO runs 4 forward passes per training step (current model × winner/loser +
reference model × winner/loser), compared to BC's single forward pass. batch_size=4 × 4
passes exceeded available memory.

**Fix**: Reduced DPO `batch_size` from 4 to 1 in `dagger_server.py`. With
`accum_grad_steps=4`, the effective batch size remains 4 but memory usage drops to 1/4.

### Fix 3: DONE Completion Marker

**Problem**: `run_full_pipeline.sh` needed to detect when each phase finishes to automate
the checkpoint handoff.

**Fix**: Added code at the end of `dagger_server.py` that writes a `DONE` file to the
output directory when all training rounds complete. The pipeline script polls for this file.

---

## 6. How to Run

### Prerequisites

1. Conda environment `emac` with all dependencies
2. AI2-THOR data at `$ALFWORLD_DATA` (default: `~/.cache/alfworld`)
3. `OPENAI_API_KEY` environment variable set
4. Vicuna-7B weights at `/srv/scratch/z5428797/models/vicuna-7b-v1.1`
5. GPU node (tested on NVIDIA H200 143GB)

### Option A: Full Pipeline (Automated)

```bash
# On a GPU node (e.g., via qsub):
bash run_full_pipeline.sh
```

This automatically runs Phase 1 (BC), finds the checkpoint, configures Phase 2 (DPO),
runs it, and restores default settings.

### Option B: Single Phase (Manual)

**Phase 1 (BC):**
```bash
# In dagger_server.py: enable_dpo = False
# In blip2_emac.yaml: load_pretrained: False
bash run_dagger.sh
```

**Phase 2 (DPO):**
```bash
# In dagger_server.py: enable_dpo = True
# In blip2_emac.yaml:
#   load_pretrained: True
#   pretrained: "<path to Phase 1 checkpoint>"
#   ref_pretrained: "<path to Phase 1 checkpoint>"
bash run_dagger.sh
```

### Monitoring

```bash
# Attach to tmux session (Ctrl+B then D to detach)
tmux attach -t emac

# Or tail the log
tail -f output/dagger_server_human_desc/*/running_nb01.log

# Check GPU usage
nvidia-smi
```

---

## 7. Output Directory Structure

```
output/dagger_server_human_desc/
├── with_bc_dpo-False-tc-False-<TIMESTAMP>/   ← Phase 1 (BC)
│   ├── running_nb01.log                       ← Training log
│   ├── logging_results/
│   │   ├── world.log                          ← Round summaries (success/fail/accuracy)
│   │   ├── trial_0.log                        ← Per-env trajectories (round 0)
│   │   ├── trial_1.log                        ← Per-env trajectories (round 1)
│   │   ├── env_results_trial_0.json           ← Per-env success/failure
│   │   └── env_results_trial_1.json
│   ├── emma_checkpoint_0.pth                  ← Checkpoint after round 0
│   ├── emma_checkpoint_1.pth                  ← Checkpoint after round 1 (= πref)
│   └── DONE                                   ← Completion marker
│
└── with_bc_dpo-True-tc-False-<TIMESTAMP>/    ← Phase 2 (DPO)
    ├── running_nb01.log
    ├── logging_results/
    ├── emma_checkpoint_0.pth
    ├── emma_checkpoint_1.pth                  ← Final model
    └── DONE
```

---

## 8. Understanding the Training Logs

### BC Training Log

```
INFO:dagger_server_running:Task: [0], Iter: [5], Obs: Nothing happens.,
  VLM Action: you are in the middle of a room,    ← VLM output (garbage — untrained)
  LLM Action: go to countertop 1                  ← Expert label (correct action)
```

- **Task [N]**: Which environment (0 to num_envs-1)
- **Iter [N]**: Step within the episode (0 to max_steps)
- **Obs**: What the environment returned after the VLM's action
- **VLM Action**: What the VLM generated (garbage early on — normal)
- **LLM Action**: What GPT-5.2 expert suggested (used as training label)
- **"Nothing happens."**: Environment didn't recognize the VLM's output as a valid action

```
*************Start Training*************
Epoch: [0], Iter: [0], Lr: [0.000000], Loss: [4.7055]
Epoch: [0], Iter: [19], Loss: [4.8352]
```

- **Epoch [N]**: Which round
- **Iter [N]**: Training step (total across all envs)
- **Lr**: Learning rate (starts near 0 due to warmup, increases over steps)
- **Loss**: Cross-entropy loss (lower = VLM output closer to expert label)

### DPO Training Log

```
Epoch: [0], Iter: [0], Loss: [2.4771], W_reward: [0.0000], L_reward: [0.0000], NLL: [3.5679]
```

- **Loss**: DPO loss = mix of preference loss + NLL
- **W_reward**: Log-probability advantage of winner (LLM action) vs reference model
- **L_reward**: Log-probability advantage of loser (VLM action) vs reference model
- **NLL**: Negative log-likelihood (BC component of the mixed loss)

Over training, you want: W_reward to increase (winner becomes more likely),
L_reward to decrease (loser becomes less likely).

---

## 9. Verification Checklist

- [x] **Phase 1 (BC) completes**: Checkpoints saved, DONE marker written
- [x] **Phase 1 loss decreases**: ~4.7 → ~3.76 over training
- [x] **LLM expert outputs valid actions**: "go to countertop 1" (not empty strings)
- [x] **Phase 2 (DPO) starts**: No crash on `ref_query_tokens is None`
- [x] **Phase 2 shows DPO metrics**: W_reward, L_reward, NLL in training logs
- [ ] **Phase 2 completes**: Checkpoints saved, DONE marker written

---

## 10. Common Errors and Fixes

### LLM expert returns empty actions

**Symptom**: Log shows `LLM Action:` (empty) for most iterations.

**Cause**: Chat models (GPT-5.2) don't follow few-shot completion patterns like
`text-davinci-003` did. Responses start with `\n` or contain explanatory text.

**Fix**: System prompt in `xrl_alignment/utils.py:get_chat()` constrains the model
to output only single-line actions. See Fix 1 above.

### DPO training hangs / OOM

**Symptom**: Log shows `Start Training` but no training metrics. `nvidia-smi` shows
99%+ memory usage and 0% GPU utilization.

**Cause**: DPO uses ~4x more GPU memory than BC (4 forward passes per step).

**Fix**: Reduce `batch_size` for DPO in `dagger_server.py`. Currently set to 1
(effective batch = 4 via gradient accumulation). See Fix 2 above.

### `ref_query_tokens is None` crash during DPO

**Cause**: `ref_pretrained` not set in `blip2_emac.yaml`.

**Fix**: Set both `ref_pretrained` and `pretrained` to the Phase 1 checkpoint path.
`run_full_pipeline.sh` handles this automatically.

### AI2-THOR FIFO pipe errors / stale processes

**Cause**: Previous crashed runs leave `/tmp/ai2thor-fifo/*` pipes.

**Fix**: `run_dagger.sh` cleans these automatically. Manual: `rm -f /tmp/ai2thor-fifo/*`

### Port 7860 already in use

**Cause**: Previous server didn't shut down cleanly.

**Fix**: Server calls `release_port()` at startup. Manual: `ss -lptn 'sport = :7860'`
then `kill <PID>`.

### xvfb-run display errors

**Cause**: No X display on headless server.

**Fix**: All commands in `run_dagger.sh` use `xvfb-run -a`.

---

## 11. Scaling Up to Paper Configuration

To replicate the full paper results, change these settings:

**Phase 1 (BC)** — `dagger_server.py`:
```python
num_rounds = 6        # was 2
num_envs = 134        # was 3
batch_size = 128      # was 8 (BC) — needs multi-GPU or gradient accumulation
```

**Phase 2 (DPO)** — `dagger_server.py`:
```python
num_rounds = 12       # was 2
num_envs = 134        # was 3
batch_size = 4        # was 1 — may need more GPU memory or multi-GPU
```

Note: Paper-scale runs will take significantly longer and require more GPU memory
for larger batch sizes. Consider multi-GPU training or further gradient accumulation.
