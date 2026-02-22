# Guide.md — How to Do Each Task in Your Thesis

> This is a practical how-to guide. Each section tells you exactly how to do a specific type of task. Reference this whenever you are stuck on HOW to do something (not just WHAT to do — that is in Plan.md).

---

## Table of Contents

1. [How to read a research paper](#1-how-to-read-a-research-paper)
2. [How to take paper notes that are actually useful](#2-how-to-take-paper-notes-that-are-actually-useful)
3. [How to set up your coding environment](#3-how-to-set-up-your-coding-environment)
4. [How to install each environment](#4-how-to-install-each-environment)
5. [How to run a baseline experiment and record results](#5-how-to-run-a-baseline-experiment-and-record-results)
6. [How to implement a code modification safely](#6-how-to-implement-a-code-modification-safely)
7. [How to design an experiment](#7-how-to-design-an-experiment)
8. [How to do statistical analysis on your results](#8-how-to-do-statistical-analysis-on-your-results)
9. [How to write your Thesis A report in Overleaf](#9-how-to-write-your-thesis-a-report-in-overleaf)
10. [How to write your main thesis report](#10-how-to-write-your-main-thesis-report)
11. [How to create good figures and tables](#11-how-to-create-good-figures-and-tables)
12. [How to prepare a presentation](#12-how-to-prepare-a-presentation)
13. [How to use the Gadi HPC cluster](#13-how-to-use-the-gadi-hpc-cluster)
14. [How to debug when things go wrong](#14-how-to-debug-when-things-go-wrong)

---

## 1. How to read a research paper

Reading a paper efficiently takes practice. The goal is NOT to read every word — it is to understand the key contribution and how it relates to your work.

### Step 1: First pass — 10 minutes
Read these parts ONLY on the first pass:
1. **Title and abstract** — what problem, what method, what result?
2. **Introduction** — last 2 paragraphs (usually state the contribution clearly)
3. **Figures and tables** — scan them all; they often tell the whole story visually
4. **Conclusion** — what did they find?

After 10 minutes you should know: what problem, what they tried, whether it worked.

### Step 2: Second pass — 30-60 minutes
Now read the full paper but skip the math proofs and implementation details on first read:
1. **Related Work** — which prior papers does this build on? Which ones does it compare against?
2. **Method** — how does it actually work? Draw a diagram for yourself.
3. **Experiments** — what did they measure? What was the baseline? How much improvement?
4. **Limitations** — what did the authors themselves say doesn't work well?

### Step 3: Deep dive — only for the 5-6 most important papers
Go through equations, implementation details, appendix. You will do this for:
- The EMAC+ paper (your base)
- ReAct and Reflexion (core framework)
- The environment papers for each RQ

### Questions to ask while reading
- What is the **main claim**?
- What is the **baseline** they compare against?
- What is the **gap** they identified in prior work?
- Does this relate to **RQ1, RQ2, or RQ3**?
- What could **I** do differently or better?

### Finding papers
- **Google Scholar** — search for the paper title
- **Semantic Scholar** — good for finding related papers
- **arXiv** — free preprints
- **Papers With Code** — papers + code + results
- **Connected Papers** — visual map of related papers

---

## 2. How to take paper notes that are actually useful

Don't just highlight text in the PDF — you won't remember it. Use this template for each paper.

### Note template (create one file per paper)

```markdown
# Paper: [Full title]
**Authors:** [Names]
**Year:** [Year]
**Venue:** [NeurIPS / ICML / arXiv / etc.]

## What problem does it solve?
[1-2 sentences]

## What is the key method?
[3-5 sentences. Focus on the mechanism, not the math.]

## What are the key results?
[Bullet points of numbers that matter to you]
- Task X: 75% success rate
- vs baseline: +12%
- Environment: ALFWorld, 134 envs

## What is the baseline they compare against?
[Explain what baseline means here]

## What are the limitations (from the paper)?
[Authors always list these in conclusion/discussion]

## How does this relate to my RQs?
- RQ1: [does it? how?]
- RQ2: [does it? how?]
- RQ3: [does it? how?]

## Quotes to use in my report
- "[exact quote]" (page X)

## My opinion / questions
[What confused me? What would I do differently?]
```

Store all notes in a folder like `thesis_notes/paper_notes/`.

---

## 3. How to set up your coding environment

### Full setup from scratch

```bash
# Step 1: Install conda (if not already installed)
# Download Miniconda for Linux: https://docs.conda.io/en/latest/miniconda.html
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Step 2: Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Step 3: Create environment from environment.yml
cd /path/to/EMAC-repo
conda env create -f environment.yml
conda activate emac

# Step 4: Install LAVIS in editable mode (skip deps — already in environment.yml)
pip install -e . --no-deps

# Step 5: Fix libstdc++ and Vulkan — required on HPC clusters
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/hpc-fixes.sh << 'EOF'
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export XDG_RUNTIME_DIR=/tmp/$USER-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
EOF

# Step 6: Move large cache to scratch (saves home quota)
mv ~/.ai2thor /srv/scratch/$USER/.ai2thor
echo 'export AI2THOR_HOME=/srv/scratch/$USER/.ai2thor' >> ~/.bashrc
```

> **Why conda before pip?** HPC clusters have old system C++ libraries. `environment.yml` installs spacy, opencv etc. via conda-forge which bundles its own compatible libraries, avoiding errors like `GLIBCXX_3.4.29 not found`.

> **Why `--no-deps`?** The LAVIS `pyproject.toml` requires `opencv-python-headless==4.5.5.64` but we use conda's opencv instead. `--no-deps` installs only LAVIS itself without replacing our conda packages.

---

### Downloading Vicuna-7B weights

```bash
python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lmsys/vicuna-7b-v1.1",
    local_dir="/srv/scratch/$USER/models/vicuna-7b-v1.1",
    ignore_patterns=["*.msgpack", "*.h5"]
)
EOF
```

This downloads ~13GB. Always store in `/srv/scratch` to avoid filling your home quota (15GB limit).

---

### Setting your OpenAI API key

**Option A — Edit utils.py directly (not recommended for shared machines):**
```python
# In utils.py, lines 41-42:
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
```

**Option B — Set as shell environment variable (recommended):**
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
echo 'export OPENAI_API_BASE="https://api.openai.com/v1"' >> ~/.bashrc
source ~/.bashrc
conda activate emac  # re-activate after sourcing .bashrc
```

---

### How to confirm the LLM API works

```bash
python << 'EOF'
import openai
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=5
)
print(response["choices"][0]["message"]["content"])
EOF
```

> **Note:** `text-davinci-003` is deprecated. Use `gpt-3.5-turbo` with `ChatCompletion.create` instead.

---

## 4. How to install each environment

### 4.1 ALFWorld (for RQ1)

Already installed via `environment.yml`. Just download the game data:

```bash
alfworld-download

# Test it works
python -c "from alfworld.agents.environment import get_environment; print('ALFWorld OK')"
```

**Common error:** If you get `FileNotFoundError` for game data, re-run `alfworld-download`.

---

### 4.2 AI2-THOR (for RQ2)

AI2-THOR is a photorealistic 3D household simulation with an egocentric camera. It provides genuine visual partial observability — the agent cannot see behind walls or into rooms it hasn't entered.

Already installed via `environment.yml`.

**⚠️ AI2-THOR must run on a GPU node — not the login node.**

```bash
# Step 1: Submit an interactive GPU job
qsub -I -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=2:00:00

# Step 2: Activate environment (the activate.d script sets everything automatically)
conda activate emac

# Step 3: Run AI2-THOR with xvfb-run (required for virtual display)
xvfb-run -a python << 'EOF'
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

controller = Controller(
    platform=CloudRendering,   # required on Katana — headless=True does NOT work
    scene="FloorPlan1",
    gridSize=0.25,
    width=300,
    height=300,
    gpu_device=0,
)
event = controller.reset()
print("Frame shape:", event.frame.shape)  # (300, 300, 3)

visible = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
print("Currently visible:", visible)

event = controller.step(action="RotateRight")
visible_after = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
print("After rotating right:", visible_after)

controller.stop()
print("AI2-THOR OK")
EOF
```

> **Why `platform=CloudRendering`?** `headless=True` still requires Vulkan/display system config. `CloudRendering` uses NVIDIA EGL directly, bypassing the display stack.

> **Why `xvfb-run -a`?** It creates a virtual framebuffer so the NVIDIA Vulkan ICD can initialize properly on headless nodes.

> **Why `VK_ICD_FILENAMES`?** Katana GPU nodes have the NVIDIA Vulkan ICD file at `/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` but it's not loaded by default. Setting this variable (done automatically by `hpc-fixes.sh`) forces Vulkan to use NVIDIA instead of the software renderer.

**What egocentric partial observation looks like:**
```python
# Agent is facing the kitchen counter
visible = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
# ['Mug', 'Cup', 'Sink', 'Faucet', 'SoapBottle']  ← only what's in camera view

event = controller.step(action="RotateLeft")  # turn to face the fridge
visible = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
# ['Fridge', 'Egg', 'Apple']  ← completely different objects now visible
```

This is what makes recovery meaningful: rotating or moving into a new area reveals objects the agent couldn't see before. Unlike ALFWorld's `look` which returns all objects in the room.

**Recommended scenes for RQ2:**
- `FloorPlan1–30` — kitchen scenes (pick, place, heat objects)
- `FloorPlan201–230` — living rooms (navigate, find objects across rooms)

**Key actions:**
| Action | Effect |
|--------|--------|
| `MoveAhead` | Move forward 0.25m |
| `RotateRight` / `RotateLeft` | Rotate 90° |
| `LookUp` / `LookDown` | Tilt camera |
| `PickupObject` | Pick up an object |
| `PutObject` | Place in receptacle |
| `OpenObject` | Open cabinet/fridge/drawer |

---

### 4.3 ScienceWorld (for RQ3)

Already installed via `environment.yml`. Java is pre-installed on Katana.

```bash
# Test it works
python << 'EOF'
from scienceworld import ScienceWorldEnv

env = ScienceWorldEnv("")
print("Available tasks:")
for t in env.get_task_names():
    print(" -", t)

env.load("boil", 0, simplificationStr="")
obs, info = env.reset()
print("\nInitial observation:")
print(obs[:500])
EOF
```

**Available task types:**
- `boil` — boil water in a pot
- `freeze` — freeze a substance
- `melt` — melt a solid
- `mix-paint` — mix paint colors
- `react-sodium` — chemical reaction
- `measure-melting-point` — science measurement

For RQ3, use tasks that take 30+ steps (boil, react, measure tasks).

**Common error:** `Java not found` — install via conda: `conda install -c conda-forge openjdk -y`

---

### 4.4 Verify all environments

```bash
# These can run on the login node
python -c "from lavis.models import load_model_and_preprocess; print('LAVIS OK')"
python -c "from alfworld.agents.environment import get_environment; print('ALFWorld OK')"
python -c "from scienceworld import ScienceWorldEnv; print('ScienceWorld OK')"

# AI2-THOR must run on a GPU node with xvfb-run
xvfb-run -a python << 'EOF'
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
controller = Controller(platform=CloudRendering, scene="FloorPlan1", gridSize=0.25, width=300, height=300, gpu_device=0)
event = controller.reset()
print("AI2-THOR OK — Frame shape:", event.frame.shape)
controller.stop()
EOF
```

---

## 5. How to run a baseline experiment and record results

### Before you run anything, create a results tracking file

Create a file called `experiment_log.md` in your project. Every time you run an experiment, add an entry:

```markdown
## Experiment: RQ1 Baseline
**Date:** 2026-02-22
**Config:** num_envs=10, num_rounds=3, enable_dpo=True, all feedback disabled
**Command:** python dagger_server.py (with modifications noted)
**GPU:** Gadi H100
**Runtime:** ~2 hours

### Results
| Round | Success | Accuracy | Avg Steps |
|-------|---------|----------|-----------|
| 0     | 3/10    | 30%      | 15.2      |
| 1     | 5/10    | 50%      | 12.8      |
| 2     | 6/10    | 60%      | 11.1      |

### Notes
- Environment 7 consistently fails (object not found bug?)
- Training loss: 0.45 → 0.31 over 3 rounds (decreasing = good)

### Errors / Issues
- None
```

### How to run dagger_server.py step by step

**Terminal 1 — Start the training server:**
```bash
conda activate emac
cd /path/to/EMAC-repo

# Edit these settings at the top of dagger_server.py before running:
# num_rounds = 3        # start small
# num_envs = 10         # start small
# run_training = True
# output_dir = "/path/to/your/outputs/"
# Also fix the LLM API key in utils.py

python dagger_server.py
# Should print: "Server running on IP: 0.0.0.0 PORT: 7860"
# Then it will BLOCK (wait for the ALFWorld client to connect)
```

**Terminal 2 — Start ALFWorld client:**
```bash
# The ALFWorld client sends observations to port 7860
# (ask your supervisor or check the project documentation for the client script)
# It typically looks like:
python alfworld_client.py  # or whatever the client script is called
```

**Watch the logs:**
```bash
# In a third terminal, watch the log file
tail -f /path/to/output_dir/running_nb01.log
```

You should see lines like:
```
Task: [0], Iter: [3], Obs: On the countertop, you see ..., VLM Action: take apple 1 from countertop 1, LLM Action: take apple 1 from countertop 1
```

### How to read the world.log output

After each round, `world.log` has:
```
-----
ROUND: 0
SUCCESS: 3
ADDITIONAL SUCCESS: 3
FAIL: 7
TOTAL: 10
ACCURACY: 0.30
TIME: 1823.45 s
-----
```

Copy these numbers into your experiment_log.md.

---

## 6. How to implement a code modification safely

### Rule 1: Never modify original files directly

Before you change any file, **create a copy**:

```bash
# Example: before modifying dagger_server.py for RQ1
cp dagger_server.py dagger_server_ORIGINAL.py
# Now edit dagger_server.py freely
# If you break it: cp dagger_server_ORIGINAL.py dagger_server.py
```

Even better — use git branches:

```bash
git checkout -b rq1-feedback-shaping
# Make all your RQ1 changes on this branch
git add .
git commit -m "RQ1: add progress feedback and failure-type classification"

# Switch back to original:
git checkout main
```

### Rule 2: Test every function in isolation first

Before integrating into dagger_server.py, test your new function separately:

```python
# test_rq1_functions.py
from dagger_server import extract_progress_signal, classify_failure

# Test progress signal
prev_obs = "You are in the kitchen."
curr_obs = "You pick up the apple 1 from the countertop 1."
print(extract_progress_signal(prev_obs, curr_obs))
# Expected: "Progress: you acquired an object."

# Test failure classification
obs = "Nothing happens."
print(classify_failure(obs))
# Expected: "FAILURE: Invalid action or precondition not met."
```

Run this before touching the main loop:
```bash
python test_rq1_functions.py
```

### Rule 3: Add logging, not print statements

For debugging during runs, use the logger:
```python
# In dagger_server.py, you can use:
logger.info(f"Progress signal: {progress_signal}")
logger.info(f"Failure type: {failure_type}")
logger.info(f"Stuck detected: {is_stuck}")
```

The logs go to the output file and you can search them later.

### Rule 4: Run on 2-3 envs first, not all 134

Before a full experiment run:
```python
# Temporarily set:
num_envs = 3
num_rounds = 2
```

Confirm it completes without crashing. Then restore full numbers.

---

## 7. How to design an experiment

A well-designed experiment has:
1. **One variable at a time** (or known combinations)
2. **A clear baseline** to compare against
3. **Fixed random seed** so results are reproducible
4. **Multiple runs** if possible (to show variance, not just one number)

### Experiment design template

```
Experiment name: RQ1 Condition A - Progress Feedback
Hypothesis: Adding progress feedback will increase success rate vs baseline
Independent variable: progress feedback (on / off)
Dependent variables: task success rate, avg steps to completion
Control variables: everything else stays the same (same seed, same num_envs, same model)

Conditions:
  Baseline: progress_feedback=False, all other changes=False
  Treatment: progress_feedback=True, all other changes=False

How I will run it:
  1. Set seed=42, num_envs=10, num_rounds=3
  2. Run Baseline → record success rate per round
  3. Enable progress_feedback → run again → record success rate per round
  4. Compare

How I will know if it worked:
  Success rate improvement of ≥5% is meaningful
  Avg steps decrease is a bonus indicator
```

### What "statistical significance" means for you

With small sample sizes (10-30 environments), you cannot do standard significance testing. Instead:
- Report the numbers clearly in a table
- Show results across multiple rounds (not just final round)
- If you have 3+ runs with different seeds, report mean ± standard deviation
- Be honest: "With n=10 environments, these results are preliminary"

---

## 8. How to do statistical analysis on your results

### Basic analysis (minimum required)

For each condition, create a table:

| Condition | Round 1 | Round 2 | Round 3 | Final Acc. |
|-----------|---------|---------|---------|------------|
| Baseline  | 30%     | 45%     | 55%     | 55%        |
| Condition A | 35%  | 52%     | 63%     | 63%        |
| Condition B | 28%  | 46%     | 58%     | 58%        |

Then compute:
- **Absolute improvement** vs baseline: 63% - 55% = +8%
- **Relative improvement**: 8/55 = +14.5%

### If you have multiple runs (different seeds)

```python
import numpy as np

# Results from 3 runs with seeds 42, 123, 456
condition_A_results = [0.63, 0.61, 0.65]  # final accuracy per run
baseline_results = [0.55, 0.53, 0.57]

print(f"Baseline: {np.mean(baseline_results):.2f} ± {np.std(baseline_results):.2f}")
print(f"Condition A: {np.mean(condition_A_results):.2f} ± {np.std(condition_A_results):.2f}")
print(f"Improvement: {np.mean(condition_A_results) - np.mean(baseline_results):.2f}")
```

### Plotting results (Python + matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

rounds = [0, 1, 2]
baseline = [0.30, 0.45, 0.55]
condition_a = [0.35, 0.52, 0.63]
condition_b = [0.28, 0.46, 0.58]

plt.figure(figsize=(8, 5))
plt.plot(rounds, baseline, 'k-o', label='Baseline')
plt.plot(rounds, condition_a, 'b-s', label='+ Progress Feedback')
plt.plot(rounds, condition_b, 'r-^', label='+ Failure-Type Feedback')
plt.xlabel('Training Round')
plt.ylabel('Task Success Rate')
plt.title('RQ1: Effect of Feedback Shaping on Success Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rq1_results.pdf', bbox_inches='tight')
plt.show()
```

Save figures as PDF for Overleaf (better quality than PNG).

---

## 9. How to write your Thesis A report in Overleaf

### Step 1: Get the faculty template

1. Go to Overleaf (overleaf.com) — sign up with your university email
2. Ask your supervisor for the faculty thesis template, or search the faculty website
3. Upload to Overleaf or use "New Project → Upload Project → ZIP file"

### Step 2: Understand the LaTeX structure

The template will have files like:
```
main.tex           ← the main file (compile this)
chapters/
  introduction.tex
  background.tex
  methodology.tex
  results.tex
  conclusion.tex
figures/
  diagram.pdf
references.bib     ← bibliography file
```

You edit the chapter files and compile `main.tex`.

### Step 3: Basic LaTeX you need to know

```latex
% Section headings
\section{Introduction}
\subsection{Background}
\subsubsection{Detailed topic}

% Bold and italic
\textbf{bold text}
\textit{italic text}

% Bullet list
\begin{itemize}
  \item First point
  \item Second point
\end{itemize}

% Numbered list
\begin{enumerate}
  \item First step
  \item Second step
\end{enumerate}

% Inline code
\texttt{python dagger\_server.py}

% Math (inline)
The loss function is $L = -\log\sigma(\beta \cdot \text{logits})$.

% Math (equation block)
\begin{equation}
  \mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)
\end{equation}

% Insert a figure
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/architecture.pdf}
  \caption{The EMAC+ system architecture.}
  \label{fig:architecture}
\end{figure}

% Reference a figure
As shown in Figure~\ref{fig:architecture}, the model consists of...

% Insert a table
\begin{table}[h]
  \centering
  \begin{tabular}{lcc}
    \hline
    Condition & Success Rate & Avg Steps \\
    \hline
    Baseline  & 55\%         & 12.3      \\
    + Progress & 63\%        & 10.8      \\
    \hline
  \end{tabular}
  \caption{RQ1 results across conditions.}
  \label{tab:rq1results}
\end{table}

% Citation
The ReAct framework~\cite{yao2022react} combines reasoning and acting.

% Reference format in references.bib:
@inproceedings{yao2022react,
  author={Shunyu Yao and others},
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  booktitle={ICLR},
  year={2023}
}
```

### Step 4: Structure of Thesis A report

A typical Thesis A report is 20-40 pages and covers:

```
1. Introduction (~3 pages)
   - What is the problem?
   - Why does it matter?
   - What will your thesis do?

2. Background / Literature Review (~15 pages)
   - Section 2.1: Embodied AI and task completion
   - Section 2.2: Language model agents (ReAct, Reflexion)
   - Section 2.3: VLM+LLM collaborative approaches (EMAC+)
   - Section 2.4: Feedback and reward shaping (for RQ1)
   - Section 2.5: Stuck detection and recovery (for RQ2)
   - Section 2.6: Memory representations in agents (for RQ3)
   - Section 2.7: Environments (ALFWorld, AI2-THOR, ScienceWorld)

3. Research Gaps (~2 pages)
   - Gap 1: Binary feedback limits training signal quality
   - Gap 2: No explicit stuck handling in current embodied agents
   - Gap 3: Memory representation not systematically compared

4. Research Questions (~2 pages)
   - RQ1 with hypotheses H1, H2
   - RQ2 with hypotheses H1, H2, H3
   - RQ3 with hypotheses H1, H2

5. Preliminary Experiments (~3 pages)
   - Describe your small-scale EMAC+ test
   - Show preliminary results table
   - What this tells you about feasibility

6. Research Timeline (~2 pages)
   - Phase A remaining
   - Phase B plan
   - Phase C plan
   - Risk mitigation (what if X goes wrong?)
```

---

## 10. How to write your main thesis report

The main thesis (written during B and C) is 60-100 pages. The key sections:

### Introduction (5-8 pages)
- Start broad (embodied AI in the real world)
- Narrow to your specific problem
- State your contributions (3 bullet points matching your 3 RQs)
- Outline the thesis structure

### Background (20-25 pages)
- Every paper you cite must be explained — don't just list names
- Organise by THEME, not by paper (group papers that solve the same type of problem)
- End each subsection with "however, [limitation that your work addresses]"

### System Design (10-15 pages)
- Full explanation of EMAC+ architecture (you can draw on your README.md)
- Your modifications for each RQ
- Diagrams for each component

### Experimental Setup (5-8 pages)
- Environments: describe ALFWorld (RQ1), AI2-THOR with CloudRendering (RQ2), ScienceWorld (RQ3)
- Metrics: what you measure and why
- Hyperparameters: list every important setting
- Hardware: GPU type, training time

### Results (15-20 pages)
- One section per RQ
- Tables of all conditions
- Figures showing learning curves
- Analysis connecting results back to hypotheses

### Discussion (5-8 pages)
- What do the results mean?
- Why did X work but Y didn't?
- Connect back to literature: does this agree with or contradict prior work?

### Conclusion (3-5 pages)
- Answer each RQ in 1-2 sentences
- State limitations honestly
- Future work (what would you do with 6 more months?)

### Writing tips
- Write EVERY DAY, even if just 200 words
- Write the methods section first — it is the most concrete
- Leave introduction and conclusion for last (you write them best when everything else is done)
- Never say "In this paper" — it's a thesis, say "In this thesis"
- Use past tense for experiments you did ("We ran..."), present tense for facts ("ALFWorld has 134 environments...")
- Every figure and table MUST be referenced from the text

---

## 11. How to create good figures and tables

### Architecture diagrams
Use **draw.io** (free, at diagrams.net):
- Draw your system architecture (boxes + arrows)
- Export as PDF
- Import into Overleaf

What to show:
- The EMAC+ model (ViT → Q-Former → Projection → LLM) — see README.md §5 for the ASCII version
- The DAgger training loop (cycle diagram)
- Your RQ1 modification (add a "Feedback Enricher" box)
- Your RQ2 modification (add a "Stuck Detector" and "Recovery Policy" box)
- Your RQ3 modification (show three memory types side by side)

### Result tables

Template for a results table in LaTeX:

```latex
\begin{table}[t]
  \centering
  \caption{Task success rates across RQ1 conditions.
           Bold = best result. Baseline = original EMAC+.}
  \label{tab:rq1}
  \begin{tabular}{lccc}
    \hline
    Condition              & Round 1 & Round 3 & Final  \\
    \hline
    Baseline               & 30\%    & 55\%    & 55\%   \\
    + Progress Feedback    & 35\%    & 63\%    & 63\%   \\
    + Failure-Type         & 32\%    & 58\%    & 58\%   \\
    + Anti-Repeat          & 31\%    & 57\%    & 57\%   \\
    \textbf{All Combined}  & \textbf{40\%} & \textbf{71\%} & \textbf{71\%} \\
    \hline
  \end{tabular}
\end{table}
```

### Learning curve plots

Show how performance improves round-by-round. Use:
- X axis = training round (or step)
- Y axis = success rate (0 to 1)
- One line per condition
- Shaded band = ± standard deviation if you have multiple runs

Save as PDF from matplotlib (as shown in Section 8).

---

## 12. How to prepare a presentation

### Thesis B presentation (~10-15 slides, ~15 min)

```
Slide 1: Title + your name
Slide 2: Problem statement (what is the gap?)
Slide 3: EMAC+ baseline (quick overview, 1 architecture diagram)
Slide 4: Your 3 RQs (table format works well)
Slide 5: RQ1 — what you implemented
Slide 6: RQ1 — preliminary results (table or bar chart)
Slide 7: RQ2 — what you plan / have set up
Slide 8: RQ3 — what you plan
Slide 9: Updated timeline (Gantt chart or simple table)
Slide 10: Questions / Discussion
```

### Thesis C final presentation (~20-25 slides, ~20 min)

```
Slide 1: Title
Slide 2: Motivation + problem
Slide 3: Background (3 key papers, brief)
Slide 4: EMAC+ overview
Slides 5-8: RQ1 — motivation, method, results
Slides 9-12: RQ2 — motivation, method, results
Slides 13-16: RQ3 — motivation, method, results
Slide 17: Overall discussion
Slide 18: Limitations
Slide 19: Conclusion — answer all 3 RQs in 1 sentence each
Slide 20: Future work
Slide 21: Thank you + questions
```

### Slide design tips
- **One message per slide** — if you need two sentences on a slide, maybe it should be two slides
- **Use visuals, not bullet points** — show a diagram instead of describing the architecture in text
- **Results tables should be simple** — only show the rows/columns the audience needs to understand the point
- **Practice out loud** 3 times before the actual presentation
- **Prepare for Q&A**: think "what could they ask about?"

---

## 13. How to use the Gadi HPC cluster

If you have access to Gadi (Gadi is NCI's HPC cluster, project CRUISE):

### Submit a job

The shell scripts in `run_scripts/instructblip/finetuning/` are SGE job scripts. To submit:

```bash
# Log into Gadi
ssh shuang@gadi.nci.org.au

# Navigate to your code
cd /srv/scratch/CRUISE/shuang/code/emac

# Submit a job
qsub run_scripts/instructblip/finetuning/ft_caption_alfworld.sh

# Check job status
qstat -u shuang

# View output log
tail -f /srv/scratch/CRUISE/shuang/results/<JOB_ID>_xrl_alignment.out
```

### Modify the job script for your experiments

Open `ft_caption_alfworld.sh` and find these lines:
```bash
#$ -l walltime=10:00:00   # How long the job can run (10 hours)
#$ -l mem=80G             # RAM requested
#$ -l ngpus=1             # Number of GPUs
#$ -l gpu_model=H100_NVL  # GPU type
```

For your experiments:
- Start with `ngpus=1` and `walltime=4:00:00` for small tests
- Increase to `ngpus=4` and `walltime=24:00:00` for full runs

### Key paths on Gadi (from the config files)
```
Code:    /srv/scratch/CRUISE/shuang/code/emac
Models:  /srv/scratch/CRUISE/shuang/huggingface/hub/vicuna-7b-v1.1
Data:    /srv/scratch/CRUISE/shuang/dataset/emac/
Output:  /srv/scratch/CRUISE/shuang/output/EMAC/
```

Ask your supervisor to confirm these paths and get your own scratch space.

---

## 14. How to debug when things go wrong

### Problem: ImportError when running the code

```
ModuleNotFoundError: No module named 'lavis'
```

**Fix:**
```bash
conda activate emac   # Make sure you are in the right environment
pip install -e .      # Reinstall in editable mode
```

### Problem: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Fix options:**
```python
# Reduce batch size
batch_size = 2  # from 4

# Reduce number of beam search beams
num_beams = 1  # from 5

# Enable gradient checkpointing in blip2_emac.yaml:
use_grad_checkpoint: True
```

### Problem: ALFWorld client cannot connect to server

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Fix:**
```bash
# Make sure dagger_server.py is running first and shows "Server running on IP: 0.0.0.0 PORT: 7860"
# Check port is not blocked by firewall
# If port 7860 is in use: run release_port() manually:
python -c "import subprocess, re, os; ..."  # or restart the machine
```

### Problem: OpenAI API error

```
openai.error.AuthenticationError: No API key provided
```

**Fix:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
# Then re-run the script
```

### Problem: LLM returning "9" constantly

The code converts content-filtered responses to "9". If you see this a lot:
- Your prompt may be triggering content filters
- Try a different prompt or a different model

### Problem: dagger_server.py hangs forever

This usually means the feedback_queue is empty — the ALFWorld client is not sending data. Check:
1. Is the ALFWorld client actually running?
2. Is it sending to the right port (7860)?
3. Is there a firewall blocking the connection?

### General debugging approach
1. Read the error message **carefully** — it tells you the file name and line number
2. Go to that line in the code
3. Add `print()` or `logger.info()` statements before and after to see values
4. Run the simplified test (2-3 envs) to reproduce quickly
5. Search the error message on Google/StackOverflow
6. Ask your supervisor if stuck for more than 1 day on the same error

### Saving yourself from bad experiments

Before any long experiment run:
```bash
# 1. Save your current dagger_server.py
git add dagger_server.py && git commit -m "Before RQ1 experiment run"

# 2. Save current config files
git add lavis/configs/ && git commit -m "Config state for RQ1 run"

# 3. Start a log file
echo "RQ1 Condition A - $(date)" >> experiment_log.md
```

If the experiment fails midway, your last checkpoint is saved in `output_dir/emma_checkpoint_{n}.pth`. You can resume from it by setting `resume_ckpt_path` in the YAML config.
