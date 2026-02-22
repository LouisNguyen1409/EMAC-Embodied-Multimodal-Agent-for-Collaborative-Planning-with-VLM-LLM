# Contribution.md — Research Questions, Contributions, and Implementation Guide

> This document explains: (1) whether your RQs are correct and well-scoped, (2) what the current EMAC+ does vs what you propose, (3) exactly where and how you contribute, (4) how to implement, train, and test each RQ, and (5) why your supervisor flagged ALFWorld as unsuitable for RQ2 and RQ3, plus which alternative environments to use.

---

## Table of Contents

1. [Are my RQs correct? An honest assessment](#1-are-my-rqs-correct-an-honest-assessment)
2. [What EMAC+ currently does](#2-what-emac-currently-does)
3. [RQ1 — Feedback shaping](#3-rq1--feedback-shaping)
4. [RQ2 — Stuck detection and recovery](#4-rq2--stuck-detection-and-recovery)
5. [RQ3 — Memory representation](#5-rq3--memory-representation)
6. [Supervisor feedback — why ALFWorld is unsuitable for RQ2 and RQ3](#6-supervisor-feedback--why-alfworld-is-unsuitable-for-rq2-and-rq3)
7. [Alternative environments — deep analysis](#7-alternative-environments--deep-analysis)
8. [Summary: your contribution map](#8-summary-your-contribution-map)

---

## 1. Are my RQs correct? An honest assessment

### RQ1 — Feedback shaping ✅ Correct and well-scoped

**Your claim:** EMAC+ uses binary feedback ("success" or "failure") during training, and you want to enrich this signal.

**Is this true?** Yes, exactly. Look at `dagger_server.py`:
```python
# Line 350-353: the ONLY signal
if feedback_data.done:
    env_configs[f'env_{cur_task}']['is_success'] = True
    success += 1
```

The only signal from the environment is `done = True/False`. There is no progress signal, no failure type, no reason why something failed. The agent simply acts until `done=True` (success) or time runs out (failure).

**Your three improvements are:** progress-shaped feedback, failure-type feedback, anti-repeat penalty.

**Are they feasible in ALFWorld?** Yes. ALFWorld provides inventory changes, location changes, and action failure messages that can be parsed to construct these signals.

**Is RQ1 correct?** Yes. This is a real gap in the current system, the improvement is principled, and ALFWorld is a suitable testbed.

---

### RQ2 — Stuck detection and recovery ✅ Concept correct, ⚠️ ALFWorld unsuitable

**Your claim:** Agents can be stuck without clear errors. You propose two stuck detectors and two recovery policies.

**Is the concept correct?** Yes, absolutely. LLM agents repeating actions in loops is a known and documented problem.

**Is this feasible in ALFWorld?** This is what your supervisor questioned. The problem is explained in detail in Section 6 below. Short answer: **ALFWorld's observations are too perfect.** The environment tells the agent everything, so there is no real partial observability, and "stuck" loops are detectable by simple exact-match logic that already exists in the code (but is unused).

**Recommendation:** Use AI2-THOR for genuine visual partial observability (egocentric camera — agent cannot see behind walls or around corners).

---

### RQ3 — Memory representation ✅ Concept correct, ⚠️ ALFWorld unsuitable

**Your claim:** Raw trajectory vs Reflexion reflections vs hybrid memory — which helps the agent improve across attempts?

**Is the concept correct?** Yes. The Reflexion paper (Shinn et al., 2023) showed that verbal reflection improves performance. You want to extend this comparison systematically with hybrid memory.

**Is this feasible in ALFWorld?** Again, your supervisor had concerns. ALFWorld's text observations are already perfectly clean and complete. The raw trajectory already contains all information a reflection would contain. So comparing raw vs reflection in ALFWorld may show little difference because both have the same information density.

**Recommendation:** Use ScienceWorld (longer tasks, denser information, clear value for compression).

---

## 2. What EMAC+ currently does

Understanding the baseline is essential before you know what to change.

### The current DAgger loop (simplified)

```
For each round (12 total):
  For each environment (134 total):
    If already solved: skip
    Else:
      Loop:
        Receive observation (image + text)
        Ask LLM: "What action should we take here?"
        Ask VLM: "What action should we take here?"
        Execute the chosen action in ALFWorld
        If done=True: mark as success and stop
      End of episode:
        Store (image, text, LLM_action, VLM_action) in replay buffer
        Train VLM to prefer LLM_action (using DPO or BC loss)
  After all 134 envs:
    For each failed env: ask LLM to generate a reflection string
    Store reflection in env_configs[i]['memory']
    These reflections prepended to next round's prompt
```

### Current feedback signal

| What the agent knows | How it is communicated |
|---------------------|------------------------|
| Task succeeded? | `done = True` signal from environment |
| Task failed? | Episode ends without `done = True` |
| Why it failed? | Nothing — not tracked |
| What progress was made? | Nothing — not tracked |
| Did it repeat an action? | `_is_exhausted` flag in EnvironmentHistory, but NEVER READ |

### Current stuck detection

`EnvironmentHistory.check_is_exhausted()` exists at `utils.py:194` and is set when the same action is repeated. However, in `dagger_server.py`, this method is **never called**. There is no recovery logic.

### Current memory

Reflexion strings are generated after each failed episode and stored in `env_configs[i]['memory']`. The last 3 are prepended to the next attempt's prompt. This is already a good approach, but it has not been systematically compared to alternatives.

---

## 3. RQ1 — Feedback shaping

### What the current system does

Binary: task done or not. The agent receives no explanation of what went wrong.

### What you propose (your slide)

| Improvement | Description |
|------------|-------------|
| **Progress-shaped feedback** | Summarise changes in inventory/location between steps |
| **Failure-type feedback** | Categorise failure: "invalid action", "object not found", "precondition not met" |
| **Anti-repeat penalty** | If same action repeated: inject warning message into the prompt |

### Why this is a real contribution

1. **Progress-shaped feedback** addresses the credit assignment problem. In a 20-step task, if the agent fails at step 18, it should know steps 1-17 were mostly good. Binary feedback doesn't communicate this.

2. **Failure-type feedback** addresses action space misunderstanding. The LLM/VLM often tries valid-sounding but semantically wrong actions (e.g., "heat apple with fridge"). Knowing WHY an action failed helps the agent not repeat the same category of mistake.

3. **Anti-repeat penalty** addresses the looping problem already documented in the reflexion_few_shot_examples.txt (see the mug example where the agent examines the stove 4 times in a row).

### How to implement RQ1 (where in the code)

**File to modify: `dagger_server.py`**

#### Step 1: Add progress tracking

After each action, compare the observation before and after:

```python
# After receiving feedback_data (new observation after action):
def extract_progress_signal(prev_obs, curr_obs, prev_inventory, curr_inventory):
    """
    Return a natural language progress description.
    In ALFWorld: check for location change, inventory change.
    """
    progress_parts = []
    if "You pick up" in curr_obs:
        progress_parts.append("Progress: you acquired an object.")
    if "You put" in curr_obs:
        progress_parts.append("Progress: you placed an object.")
    if "You heat" in curr_obs or "You cool" in curr_obs or "You clean" in curr_obs:
        progress_parts.append("Progress: you completed a required transformation.")
    if not progress_parts:
        progress_parts.append("No visible progress this step.")
    return " ".join(progress_parts)
```

#### Step 2: Add failure-type classification

ALFWorld action failures return specific strings. Parse them:

```python
def classify_failure(obs_after_action):
    """Classify why an action failed."""
    obs_lower = obs_after_action.lower()
    if "nothing happens" in obs_lower:
        return "FAILURE: Invalid action or precondition not met."
    if "you can't" in obs_lower or "can't reach" in obs_lower:
        return "FAILURE: Object not accessible from current location."
    if "there is no" in obs_lower or "i don't see" in obs_lower:
        return "FAILURE: Object not found in current location."
    return "FAILURE: Action did not succeed."
```

#### Step 3: Add anti-repeat penalty

The `_is_exhausted` flag already exists but is never used. Use it:

```python
# In the main action loop, after env_history.add("action", vlm_action):
if env_history.check_is_exhausted():
    # Inject penalty message into the prompt
    penalty_msg = ("WARNING: You attempted the same action twice. "
                   "Do NOT repeat it. Try a completely different approach.")
    env_history.add("observation", penalty_msg)
    # This becomes part of the context seen by both LLM and VLM
```

#### Step 4: Integrate into feedback message to VLM

When constructing the prompt for the VLM, append the progress and failure information:

```python
enhanced_history = feedback_data.history + "\n" + progress_signal + "\n" + failure_type
vlm_action = model.generate(
    {"image": image, "prompt": enhanced_history},
    ...
)[0]
```

### How to train for RQ1

Training loop stays the same (DAgger + DPO). The change is in what gets passed as the prompt to the VLM. The richer feedback becomes part of the training context, so the VLM learns to use it.

### How to test RQ1

1. **Baseline**: run current EMAC+ for N rounds, record success rate and average episode length
2. **Treatment A**: add progress-shaped feedback only
3. **Treatment B**: add failure-type feedback only
4. **Treatment C**: add anti-repeat penalty only
5. **Treatment D**: all three combined

**Metrics:**
- Task success rate (primary)
- Average steps to success (efficiency)
- Frequency of repeated-action loops (anti-repeat specific)

---

## 4. RQ2 — Stuck detection and recovery

### What the current system does

Nothing. No stuck detection. No recovery. The agent keeps acting until it succeeds or time runs out.

### What you propose (your slide)

| Component | Description |
|-----------|-------------|
| **Stuck A detector** | Same action fails twice |
| **Stuck B detector** | No progress for N consecutive steps |
| **Recovery Policy 1** | Forced scan: inject `look` or `examine` action |
| **Recovery Policy 2** | Logical backtracking: revert plan to last progress checkpoint |

### Is the concept correct?

Yes. Your definitions are sound:
- **Stuck A** captures execution loops (agent knows an action failed but tries it again)
- **Stuck B** captures semantic stagnation (agent takes syntactically valid actions that don't advance the task)
- **Recovery Policy 1** is grounded: re-observing the environment refreshes the agent's context
- **Recovery Policy 2** is correctly defined as plan-space backtracking (not physics reset), which is realistic

### Your clarification about backtracking is important

You explicitly state: "backtracking cannot mean reset the simulator state... so my definition is logical backtracking: revert the plan to a previous decision point." This is correct and sensible. In practice, this means:
- Track which step the agent last made clear progress (e.g., picked up an object, moved to a new room)
- If stuck, instruct the agent (via prompt) to try a different branch from that decision point
- This is implemented in the prompt, not the simulator

### Why ALFWorld is unsuitable (see Section 6 for full explanation)

Brief: ALFWorld's `look` command returns a perfectly complete enumerated list of every object in the room. There is no partial observability. Recovery Policy 1 (scan) adds no new information because the agent already knows everything. Recovery Policy 2 is hard to evaluate because tasks are so short (5-15 steps) that true stagnation rarely manifests.

### Alternative environments for RQ2

See Section 7. AI2-THOR is the best fit:
- Egocentric photorealistic camera — agent physically cannot see behind walls or around corners
- Recovery via rotation/navigation actually reveals new objects not previously visible
- Household tasks map naturally onto the stuck-detection and recovery scenarios

### How to implement RQ2

**If staying on ALFWorld** (for proof of concept):

```python
# In dagger_server.py, add tracking variables:
progress_checkpoints = []  # List of (step_idx, observation) when progress was made
last_action_failed = False
failed_action = None
steps_without_progress = 0
STAGNATION_THRESHOLD = 4   # N steps without progress triggers Stuck B

# After each action, check:
def detect_stuck_A(current_action, last_failed_action):
    return current_action == last_failed_action  # Same failed action repeated

def detect_stuck_B(steps_without_progress, threshold=STAGNATION_THRESHOLD):
    return steps_without_progress >= threshold

def recovery_scan(env_history):
    # Force a 'look' action to refresh context
    env_history.add("observation", "[Recovery: re-scanning environment]")
    return "look"

def recovery_backtrack(env_history, progress_checkpoints):
    if not progress_checkpoints:
        return "look"  # Fallback
    last_checkpoint = progress_checkpoints[-1]
    env_history.add("observation",
        f"[Recovery: returning to decision point at step {last_checkpoint['step']}. "
        f"Try a different approach from here.]")
    # Construct alternative plan prompt
    return None  # Signal to ask LLM/VLM for new action
```

**For AI2-THOR** (recommended):

AI2-THOR provides photorealistic household scenes with an egocentric camera. Install via:
```bash
pip install ai2thor
```

The egocentric camera makes recovery meaningful: rotating or navigating into a new room reveals objects that were physically out of view. This is genuine perceptual partial observability — not a software flag like ALFWorld's enumerated `look` list.

### How to test RQ2

| Condition | Stuck Detector | Recovery Policy |
|-----------|---------------|-----------------|
| Baseline | None | None |
| A | Stuck-A (fail twice) | No recovery |
| B | Stuck-B (N steps no progress) | No recovery |
| C | Stuck-A | Policy 1 (scan) |
| D | Stuck-A | Policy 2 (backtrack) |
| E | Stuck-B | Policy 1 (scan) |
| F | Stuck-B | Policy 2 (backtrack) |

**Metrics:**
- Task success rate
- Number of "stuck events" per episode
- Number of loops broken per episode
- Steps to completion (efficiency)

**Hypotheses mapping:**
- H1: C,D,E,F all beat A,B (recovery helps)
- H2: D,F beat C,E (backtracking better than scan)
- H3: Stuck-B rows beat Stuck-A rows for efficiency (stagnation trigger reduces unnecessary recoveries)

---

## 5. RQ3 — Memory representation

### What the current system does

EMAC+ already uses Reflexion-style memory. After each failed episode, `generate_reflections.py` generates a short text lesson and stores it in `env_configs[i]['memory']`. The last 3 reflections are prepended to the next attempt's prompt.

However, the current system uses **only reflections**. It does not:
1. Compare reflections vs raw trajectory
2. Use a hybrid of both
3. Systematically measure how memory type affects cross-attempt learning

### What you propose

| Memory Type | What it stores |
|------------|---------------|
| **Raw recent trajectory** | Last K (observation, action, result) steps verbatim |
| **Reflexion reflection** | LLM-generated 1-3 bullet lessons from the episode |
| **Hybrid** | Short raw trajectory (grounding) + long-term reflections (strategy) |

### Why this is a real contribution

The Reflexion paper showed reflections help. But:
1. Reflections can lose low-level details ("the mug was on countertop 2, not 1")
2. Raw trajectories can be too long to fit in context and too noisy for credit assignment
3. Hybrid memory is a natural solution but has not been systematically tested

### Why ALFWorld is partially unsuitable (see Section 6)

ALFWorld text observations already enumerate every object perfectly. The raw trajectory already reads almost like a summary. Reflections won't add much because there is little ambiguity to resolve. The benefit of hybrid memory is most visible when:
- Observations are long and noisy (hard to extract lessons)
- Tasks take many steps (long trajectories, high credit assignment challenge)

ScienceWorld is much better: tasks take 30-60 steps, observations describe physical states that are harder to reason about.

### How to implement RQ3

**File to modify: `dagger_server.py` and `generate_reflections.py`**

#### Memory Type 1: Raw trajectory

```python
def get_raw_trajectory_memory(env_history, last_k=5):
    """Return the last K steps as a raw text string."""
    history = env_history._history  # List of {label, value} dicts
    recent = history[-last_k * 2:]  # K actions + K observations
    lines = []
    for item in recent:
        if item['label'] == 'action':
            lines.append(f"You did: {item['value']}")
        elif item['label'] == 'observation':
            lines.append(f"Result: {item['value']}")
    return "\n".join(lines)
```

#### Memory Type 2: Reflexion (already implemented in generate_reflections.py)

```python
# Already working. generate_reflections.py creates a text lesson string.
# Stored in env_configs[i]['memory'] as a list of strings.
```

#### Memory Type 3: Hybrid

```python
def get_hybrid_memory(env_history, reflexion_memories, last_k=5):
    """Combine short raw context with long-term lessons."""
    raw = get_raw_trajectory_memory(env_history, last_k)
    reflections = "\n".join(reflexion_memories[-3:])  # Last 3 lessons
    return f"Recent context:\n{raw}\n\nLessons from past attempts:\n{reflections}"
```

Then in `dagger_server.py`, when constructing `base_prompt`:

```python
if memory_type == "raw":
    memory_content = get_raw_trajectory_memory(env_history)
elif memory_type == "reflection":
    memory_content = env_configs[f'env_{cur_task}']['memory']  # existing list
elif memory_type == "hybrid":
    memory_content = get_hybrid_memory(env_history, env_configs[f'env_{cur_task}']['memory'])
```

### How to train for RQ3

The memory type affects what context is given to both the LLM (expert teacher) and the VLM (student). The training loop stays the same. The key is that the prompt changes:

- With raw memory: the VLM sees recent steps → learns to use them for decision making
- With reflection: the VLM sees lessons → learns to use verbal lessons
- With hybrid: the VLM sees both → learns to combine grounding and strategy

### How to test RQ3

Structure: run each memory condition for M attempts per task (e.g., M=3). Record success at attempt 1, 2, 3.

| Memory Type | Attempt 1 | Attempt 2 | Attempt 3 | Improvement |
|------------|-----------|-----------|-----------|-------------|
| None (baseline) | X% | X% | X% | Small |
| Raw trajectory | X% | X% | X% | ? |
| Reflexion | X% | X% | X% | ? (existing) |
| Hybrid | X% | X% | X% | Predicted best |

**Metrics:**
- Success rate at each attempt number
- Improvement ratio: (success_attempt_2 - success_attempt_1) / success_attempt_1
- Context length used (efficiency)

**Hypotheses mapping:**
- H1: reflection and hybrid improve attempt-to-attempt vs raw-only
- H2: hybrid performs best (both grounding + wisdom)

---

## 6. Supervisor feedback — why ALFWorld is unsuitable for RQ2 and RQ3

### What your supervisor said (paraphrase)

"The text in ALFWorld is a very strong assumption. Your RQ2 and RQ3 won't work well with it."

This is an important and correct observation. Let me explain it precisely.

### The "strong text assumption" problem

In ALFWorld, when you do `look`, you receive:

```
You are in the kitchen. Looking quickly around you, you see a cabinet 10, a cabinet 9,
a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3,
a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1,
a diningtable 2, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1,
a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3,
a stoveburner 2, a stoveburner 1, and a toaster 1.
```

This is a complete, enumerated, perfectly formatted list of **every single object** in the room. There is:
- No ambiguity
- No partial observation
- No noise
- No uncertainty about what exists

This is what your supervisor means by "strong assumption" — in the real world, you never get this. In real life, you have to look around, objects might be hidden, you might misidentify things, you might not know where to search.

### Why this breaks RQ2

**RQ2 requires partial observability to be meaningful.**

Your Recovery Policy 1 (forced scan) is supposed to help the agent discover information it didn't have. But in ALFWorld:
- `look` always returns the complete object list
- The agent already knows everything in the room from the very first observation
- Scanning doesn't reveal new information
- Recovery by scanning is trivially pointless

Your Recovery Policy 2 (backtracking) requires that the agent can genuinely be stuck. But in ALFWorld:
- The agent always has perfect knowledge of available objects
- "Stuck" usually means the agent chose the wrong action from a known set
- With perfect text, the LLM can almost always reason its way out without backtracking

**Consequence**: Both stuck detectors and both recovery policies would show little-to-no effect, not because your ideas are wrong, but because the environment doesn't challenge them. The null result would not be informative.

### Why this breaks RQ3

**RQ3 requires meaningful information compression to be visible.**

The value of reflections over raw trajectory is: reflections compress long, noisy, hard-to-parse observations into short actionable lessons. But in ALFWorld:
- Observations are already short and perfectly formatted
- A raw trajectory of 5 steps is already easy to parse
- There is very little to "learn" from compression — the raw trajectory IS the lesson

**Consequence**: Reflexion, raw trajectory, and hybrid memory would all show similar performance in ALFWorld, not because your ideas are wrong, but because the environment doesn't discriminate between them. The experiment would be underpowered.

### In summary

| Research Question | ALFWorld suitable? | Why not? |
|-------------------|-------------------|----------|
| RQ1 (feedback shaping) | ✅ Yes | Binary reward gap is real, progress signals can be extracted |
| RQ2 (stuck + recovery) | ❌ No | Perfect observation = no real partial obs = recovery adds nothing |
| RQ3 (memory representation) | ❌ No | Clean text = raw trajectory ≈ reflection = can't discriminate |

---

## 7. Alternative environments — deep analysis

### 7.1 AI2-THOR — Best for RQ2

**What it is:** AI2-THOR (Allen Institute for AI, 2017) is a photorealistic 3D household simulation. An agent navigates through rooms (kitchen, living room, bedroom, bathroom) to complete household tasks like picking up objects, placing them in containers, or heating food in the microwave. There are 200+ richly decorated scenes.

**The key property:** The agent observes the world through an **egocentric RGB camera** (first-person view). Objects behind walls, around corners, or in other rooms are physically invisible until the agent moves to see them. This is architectural partial observability — not a software flag.

**Why this fits RQ2 perfectly:**

1. **Genuine visual partial observability**: the agent's camera can only see what is directly in front of it. Moving to a new room or rotating to a new angle genuinely reveals new objects. Recovery Policy 1 (scan/rotate) has real informational value — unlike ALFWorld where `look` returns everything.

2. **Realistic stuck states**: an agent searching for a "mug" may loop through the same kitchen area repeatedly if it hasn't looked inside the cabinet. Stuck-B detector (no progress for N steps) fires in exactly these realistic failure modes.

3. **Backtracking has clear semantics**: "return to the hallway and choose a different room" maps directly to physical navigation. Backtracking reveals new paths and rooms, with genuine reward for choosing differently.

4. **Visual modality aligns with EMAC+**: your EMMA model already takes RGB images as input. AI2-THOR gives you real egocentric images, making this the most natural extension of the existing framework.

5. **Household tasks parallel ALFWorld**: task structure is very similar to ALFWorld (pick, place, heat, cool, clean), so you can compare RQ2 results against your RQ1 ALFWorld baseline directly.

**How to install:**
```bash
pip install ai2thor

# Test it works (requires a display or virtual framebuffer)
python << 'EOF'
from ai2thor.controller import Controller

controller = Controller(scene="FloorPlan1")
event = controller.reset()
print("Scene loaded OK")
print("Frame shape:", event.frame.shape)  # (300, 300, 3) RGB image

event = controller.step(action="RotateRight")
print("After RotateRight:", event.metadata["agent"]["rotation"])
EOF
```

**On a headless server (Gadi HPC) — use virtual display:**
```bash
# Install Xvfb (virtual framebuffer)
sudo apt-get install xvfb  # Linux
# Run with virtual display:
Xvfb :1 -screen 0 1024x768x24 &
DISPLAY=:1 python your_script.py
# Or use ai2thor's built-in headless mode:
controller = Controller(scene="FloorPlan1", headless=True)
```

**How it works:**
```python
from ai2thor.controller import Controller

# Start a scene
controller = Controller(scene="FloorPlan1", gridSize=0.25)
event = controller.reset()

# Egocentric RGB image (this is what your EMMA model receives)
image = event.frame  # numpy array, shape (300, 300, 3)

# Available navigation actions
event = controller.step(action="MoveAhead")
event = controller.step(action="RotateRight")
event = controller.step(action="RotateLeft")
event = controller.step(action="LookUp")
event = controller.step(action="LookDown")

# Object interaction actions
event = controller.step(action="PickupObject", objectId="Mug|+00.00|+00.00|+00.00")
event = controller.step(action="PutObject", objectId="Sink|...", receptacleObjectId="Mug|...")

# What objects are currently visible (partial — only what the camera sees):
visible_objects = [obj for obj in event.metadata["objects"] if obj["visible"]]
print("Visible objects:", [o["objectType"] for o in visible_objects])
# Example: ["Mug", "Cup", "Sink"] — NOT every object in the scene
```

**What you would change in EMAC+ for RQ2 with AI2-THOR:**
- Replace the ALFWorld Flask client with a direct AI2-THOR controller loop
- Keep `dagger_server.py` structure; replace environment step/reset calls
- Feed `event.frame` (RGB image) directly to EMMA — it already expects a PIL image
- Add stuck detectors (Stuck-A and Stuck-B) based on repeated actions or position
- Measure: stuck frequency, recovery success rate, task success rate

**Suitable tasks for RQ2:**
- `FloorPlan1–30` (kitchen scenes) — pick, place, clean, heat objects
- `FloorPlan201–230` (living rooms) — navigate, find objects in partially visible rooms
- `FloorPlan301–330` (bedrooms) — more rooms = more stuck scenarios

**HPC setup:** Use `platform=CloudRendering` (not `headless=True`) on GPU nodes. `CloudRendering` uses NVIDIA EGL directly and bypasses the display stack. Run AI2-THOR only on a GPU node (`qsub -I -l select=1:ncpus=4:ngpus=1:mem=16gb`).

---

### 7.2 ScienceWorld — Best for RQ3

**What it is:** ScienceWorld (Wang et al., 2022) is a text-based science simulation. An agent must complete elementary school science experiments like:

- "Boil water using the pot on the stove"
- "Measure the volume of sand using the graduated cylinder"
- "Electrolyze water to produce hydrogen and oxygen"

**Why this fits RQ3 perfectly:**

1. **Long, complex tasks**: tasks require 30-60 steps. Raw trajectories are genuinely long and hard to parse. The benefit of compression (reflections) is significant.

2. **Dense, non-trivial observations**: observations describe chemical states, temperatures, measurements, reactions. These require domain understanding to interpret correctly. Reflections that say "The water must reach 100°C before it boils, not just be on the stove" capture insights raw trajectories can't.

3. **Cross-attempt improvement is measurable**: the CLIN paper (Majumder et al., 2023) used ScienceWorld specifically to show memory helps across attempts. You can directly compare with their baseline.

4. **Multiple attempts per task are natural**: the environment resets cleanly, and many tasks benefit greatly from knowing what you learned last time.

**How to install:**
```bash
pip install scienceworld
# Note: requires Java 8+ (used for the Java-based simulator)
sudo apt-get install default-jre  # or brew install java on Mac
```

**How it works:**
```python
from scienceworld import ScienceWorldEnv

env = ScienceWorldEnv("")
task_names = env.get_task_names()
# Tasks: boil, freeze, melt, mix, react, etc.

env.load("boil", 0, simplificationStr="")
obs, info = env.reset()
# obs = "You are in a kitchen. You see a stove with a pot on it. ..."

action = "look around"
obs, reward, done, info = env.step(action)
```

**What you would change in EMAC+ for RQ3 with ScienceWorld:**
- Replace ALFWorld with ScienceWorld environment
- Implement three memory conditions (raw, reflexion, hybrid) as described in Section 5
- Run each task up to M=3 attempts with memory accumulated across attempts
- Measure success improvement from attempt 1 → 2 → 3 per memory condition

**Why the CLIN baseline matters:**
CLIN (Majumder et al., 2023) is a paper that tested continual learning with memory on ScienceWorld. They showed that storing causal abstractions (like mini-rules: "water boils at 100°C") helps. You can:
1. Reproduce their baseline
2. Compare your hybrid memory (raw + reflexion) against their approach
3. This gives you a strong published comparison point

---

### 7.3 TextWorld Custom — Alternative for RQ2

**What it is:** TextWorld is a framework for generating custom text adventure games. It's the underlying engine ALFWorld uses. You can design your own game with specific properties.

**Why it could fit RQ2:**
- You can set `openContainers=False` so objects inside containers are not revealed until opened
- This creates genuine "I don't know what's in the box" partial observability
- You have full control over how much information is revealed per step

**How to use:**
```bash
pip install textworld
```

```python
import textworld
import textworld.gym

# Create a custom game where objects are hidden in containers
options = textworld.GameOptions()
options.nb_rooms = 3
options.nb_objects = 10
options.open_containers = False  # Hidden objects!

game = textworld.generator.make_game(options)
env = textworld.gym.make(game)
```

**Pros:** Already installed (ALFWorld depends on it), highly configurable, familiar environment structure
**Cons:** Requires game generation code, less standard/reproducible than using a fixed benchmark

---

### 7.4 Comparison table

| Environment | RQ2 Fit | RQ3 Fit | Install Difficulty | Baseline Papers |
|-------------|---------|---------|-------------------|-----------------|
| **ALFWorld** (current) | ❌ Poor | ❌ Poor | Already installed | EMAC+, ReAct, Reflexion |
| **AI2-THOR** ⭐ RQ2 | ✅ Excellent | 🔶 Medium | Medium (GPU + CloudRendering) | ALFRED, ManipulaTHOR |
| **ScienceWorld** ⭐ RQ3 | 🔶 Medium | ✅ Excellent | Medium (Java via conda) | CLIN, ScienceWorld paper |
| **TextWorld custom** | ✅ Good | 🔶 Medium | Medium (game design) | TextWorld paper |
| **DiscoveryWorld** | 🔶 Medium | ✅ Good | Hard (new, complex) | NeurIPS 2024 paper |

---

## 8. Summary: your contribution map

### What EMAC+ already has (baseline)

| Component | Status |
|-----------|--------|
| Binary feedback (done/not done) | ✅ Implemented |
| VLM + LLM collaborative planning | ✅ Implemented |
| DAgger online training | ✅ Implemented |
| DPO loss | ✅ Implemented |
| Reflexion memory (simple) | ✅ Implemented |
| Anti-repeat detection (unused) | ⚠️ Exists but dead code |
| Stuck detection | ❌ Not implemented |
| Recovery policies | ❌ Not implemented |
| Feedback enrichment | ❌ Not implemented |
| Memory comparison experiments | ❌ Not implemented |

### Your contributions

| RQ | What you add | Environment | Key files to modify |
|----|-------------|-------------|---------------------|
| **RQ1** | Progress feedback, failure-type feedback, anti-repeat penalty | ALFWorld ✅ | `dagger_server.py` |
| **RQ2** | Stuck-A/B detection + Policy 1/2 recovery | AI2-THOR | `dagger_server.py` + new env wrapper |
| **RQ3** | Raw vs Reflexion vs Hybrid memory comparison | ScienceWorld | `dagger_server.py`, `generate_reflections.py` |

### The story of your thesis

Your thesis makes a unified argument:

> EMAC+ is a strong baseline for embodied multimodal collaborative planning. However, its feedback loop is information-poor, its replanning is reactive rather than proactive, and its memory has not been systematically optimized. We address each gap:
>
> - **RQ1**: Enriching the feedback signal from binary to shaped makes training more efficient and reduces loops.
> - **RQ2**: Explicit stuck detection and recovery policies reduce failure-loop frequency in partially observable environments.
> - **RQ3**: Hybrid memory (raw grounding + verbal strategy) yields the strongest cross-attempt improvement.

### Practical next steps

1. **This week**: implement RQ1 in ALFWorld (it's the easiest — pure `dagger_server.py` changes). Run baseline and 3 treatment conditions. Record results.

2. **Next 2 weeks**: adapt the DAgger loop to AI2-THOR (replace ALFWorld Flask client with AI2-THOR controller using `platform=CloudRendering` on GPU nodes). Implement Stuck-A/B detectors and recovery policies. Run the 7-condition experiment from Section 4.

3. **Following 2 weeks**: set up ScienceWorld. Implement the three memory conditions. Run M=3 attempt experiments. Compare with CLIN baseline if possible.

4. **Write-up**: frame as: "EMAC+ as baseline → three improvements → three research questions → results".

---

## Appendix: How to read EMAC+ results

When you look at checkpoint files or logs, here is what each number means:

| Log field | Meaning |
|-----------|---------|
| `ROUND` | Which DAgger training round (0-11) |
| `SUCCESS` | Total envs solved so far (including previous rounds) |
| `ADDITIONAL SUCCESS` | New envs solved in this round |
| `ACCURACY` | success / 134 = task success rate |
| `W_reward` | DPO: how much the model prefers the chosen action (higher = better) |
| `L_reward` | DPO: how much the model assigns to rejected action (should be lower) |
| `NLL` | Negative log-likelihood of chosen action (BC component of DPO loss) |
| `Loss` | Total training loss (lower = better fit to expert) |

A healthy training run shows:
- SUCCESS increasing round-over-round
- W_reward > L_reward (and gap widening)
- Loss decreasing over training steps
- Reflexion memories helping — success in round 2 > round 1 for the same envs
