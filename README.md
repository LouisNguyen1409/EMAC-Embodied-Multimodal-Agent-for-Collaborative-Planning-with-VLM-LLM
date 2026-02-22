# EMAC+ — Embodied Multimodal Agent for Collaborative Planning with VLM+LLM

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [Core concepts explained simply](#2-core-concepts-explained-simply)
3. [Why VLM+LLM, not just VLM?](#3-why-vlmllm-not-just-vlm)
4. [Training vs Inference — what is the difference?](#4-training-vs-inference--what-is-the-difference)
5. [System architecture diagram](#5-system-architecture-diagram)
6. [File-by-file explanation](#6-file-by-file-explanation)
7. [Setup and installation](#7-setup-and-installation)
8. [How to run EMAC+](#8-how-to-run-emac)
9. [Glossary](#9-glossary)

---

## 1. What is this project?

EMAC+ is a research system that teaches an AI agent to complete **household tasks** in a simulated environment called **ALFWorld**. The tasks look like this:

```
Task: "Put a heated apple in the fridge."

The agent must:
1. Look around the room
2. Find the apple
3. Heat it in the microwave
4. Find the fridge
5. Put the apple in the fridge
```

The environment gives the agent:
- A **text description** of what it sees ("You are in the kitchen. You see a table, a microwave, and a fridge.")
- An **RGB image** (a photo of the current view)

The agent must respond with a text **action** ("go to microwave 1", "heat apple 1 with microwave 1", etc.).

**The key innovation** is that EMAC+ uses **two AI models together**:
- A **VLM** (Vision-Language Model, called EMMA) that looks at the image and history and proposes an action
- An **LLM** (Language-only Model, specifically GPT/text-davinci-003) that reads only the text history and proposes an action

During **training**, the LLM acts as the expert teacher. During **inference**, only the VLM is used — but it has been trained to behave like the expert LLM.

---

## 2. Core concepts explained simply

### 2.1 ALFWorld — the simulation environment

ALFWorld is a text+visual household simulation. Think of it as a simplified 3D house where an agent can navigate and interact with objects. It has **134 training environments** across **6 task types**:

| Task Type | Example |
|-----------|---------|
| `pick_and_place` | Put a cup on the table |
| `pick_clean_then_place` | Clean a fork and put it in a drawer |
| `pick_heat_then_place` | Heat a mug and put it in the coffeemachine |
| `pick_cool_then_place` | Cool a potato and put it in the fridge |
| `look_at_obj` | Look at a bowl under a desklamp |
| `pick_two_obj` | Put two apples on the counter |

At each step, ALFWorld sends the agent:
1. A text observation (what the agent currently "sees" in text)
2. An RGB image (224×224 pixels)
3. A flag `done=True/False` (has the task been completed?)

The agent must respond with one action string.

### 2.2 ReAct — the reasoning framework

EMAC+ uses the **ReAct** (Reason+Act) prompting pattern. Instead of just outputting actions, the agent alternates between:
- `think:` steps — the agent reasons about what to do next
- Action steps — the agent actually performs an action

Example ReAct trace:
```
> think: I need to find a heated apple. First I'll find the apple.
OK.
> go to countertop 1
On the countertop 1, you see an apple 1.
> take apple 1 from countertop 1
You pick up the apple 1.
> think: Now I have the apple. I need to heat it.
OK.
> go to microwave 1
The microwave is closed.
> heat apple 1 with microwave 1
You heat the apple 1 using the microwave 1.
```

### 2.3 Reflexion — learning from failure

After each failed episode, EMAC+ uses a technique called **Reflexion**. It asks an LLM to read the failed trace and generate a short "lesson learned" in bullet points:

```
New plan: I tried to heat the apple but went to the wrong microwave.
Next time, I should check all microwaves before giving up.
I should use 'examine microwave 1' to check if it is functional.
```

These reflections are stored in the agent's **memory** and prepended to the prompt in the next attempt, so the agent doesn't repeat the same mistake.

### 2.4 DAgger — online imitation learning

**DAgger** (Dataset Aggregation) is the training algorithm. Here is how it works:

```
Loop over training rounds:
  For each environment (134 total):
    1. Agent (VLM) observes the current state
    2. Expert (LLM) observes the same state
    3. Agent acts based on VLM output
    4. Expert provides what IT would have done (the "correct" action)
    5. Store (image, text_history, expert_action) in replay buffer
    After each episode:
    6. Train the VLM on stored examples to mimic the expert
```

The key idea: in standard supervised learning, you pre-collect expert data. In DAgger, **you collect data while the agent is running**, so the agent trains on states it actually visits (not just states an expert would visit). This fixes the distribution shift problem.

### 2.5 DPO — preference learning

**DPO** (Direct Preference Optimization) is an alternative training signal. Instead of just teaching the VLM to match the expert action, it teaches preferences:

- **Chosen (positive) response**: the LLM's expert action
- **Rejected (negative) response**: the VLM's own (potentially wrong) action

The loss function pushes the model to increase the probability of the chosen response and decrease the probability of the rejected response, relative to a reference model. This is more stable than simple imitation learning alone.

---

## 3. Why VLM+LLM, not just VLM?

This is a fundamental question. Let's think through it carefully.

### Why not just VLM (vision-language model)?

A VLM like InstructBLIP can see both the image and text. So why do we need a separate LLM?

**Problem 1 — VLMs are not good at long-horizon reasoning.**
Current VLMs are good at describing images or answering simple questions, but they struggle to reason over a long sequence of actions and observations (e.g., a 20-step household task). They tend to hallucinate, get confused, or repeat themselves.

**Problem 2 — VLMs are expensive to fine-tune.**
Large VLMs have billions of parameters. Fine-tuning all of them requires massive GPU memory. EMAC+ freezes most of the VLM (ViT encoder, QFormer, LLM backbone) and only trains the projection layer, making training feasible.

**Problem 3 — Vision is useful but not always required.**
In many steps, the agent navigates based on text memory ("I saw the apple on the countertop"). The image adds context but the critical reasoning is textual. A pure VLM may over-attend to the image and under-use the text history.

### Why not just LLM (language-only model)?

A pure LLM like GPT can reason beautifully over text. But:

**Problem 1 — The LLM cannot see.**
In a real embodied task, visual information matters. The agent might need to identify objects by color, position, or appearance. A pure LLM has no access to this.

**Problem 2 — The LLM must use an external API.**
text-davinci-003 requires a paid API call for every step. This is slow, expensive, and impossible to deploy without an internet connection.

**Problem 3 — The LLM cannot be trained on task-specific data.**
We want the agent to improve through experience. An LLM called via API cannot be fine-tuned in real-time.

### The EMAC+ solution: Teacher-Student

```
TRAINING TIME:
  LLM (text-davinci-003) = Expert Teacher
    - Has strong reasoning ability
    - Sees only text (history + observations)
    - Provides "correct" action labels

  VLM (EMMA/InstructBLIP) = Student
    - Sees both image and text
    - Starts with random/bad actions
    - Gets trained to copy the expert

INFERENCE TIME:
  Only VLM is used — no LLM API needed
  VLM has learned to reason like the LLM
  VLM also has access to visual information the LLM never saw
```

This is like a student who learns from a knowledgeable teacher, then takes an exam alone — but the student also has access to extra visual information the teacher didn't have.

---

## 4. Training vs Inference — what is the difference?

This confuses many people, so let's be very explicit.

### During Training (dagger_server.py)

```
Step 1: ALFWorld client sends observation to dagger_server.py via HTTP POST
Step 2: dagger_server.py receives: image + text_history + observation + done_flag

Step 3: VLM forward (EMMA model)
  - Input: current image + text history
  - Output: vlm_action (what the VLM thinks to do)

Step 4: LLM forward (text-davinci-003 API call)
  - Input: full text history (no image)
  - Output: llm_action (what the expert teacher thinks to do)

Step 5: Action decision (the "9" trick)
  - If LLM returns "9" (content filtered / error) → use VLM action
  - Otherwise → use LLM action (the expert action is always preferred)

Step 6: Send the action back to ALFWorld
  - The chosen action goes into action_queue → sent back to client

Step 7: Store training data
  - (image, text_history, llm_action=target, vlm_action=rejected) stored in buffer

Step 8: After each episode, train VLM
  - Sample from buffer
  - If DPO: train VLM to prefer llm_action over vlm_action
  - If BC (behaviour cloning): train VLM to predict llm_action exactly

Step 9: Reflexion
  - After each full round of 134 envs, failed episodes are analyzed
  - LLM generates a text lesson from the failure
  - Lesson stored in env_configs[env_i]['memory'] for next round
```

### During Inference (evaluate.py / alfworld_ft.yaml with evaluate=True)

```
Step 1: Load trained VLM checkpoint
Step 2: Load dataset (pre-collected images + histories)
Step 3: VLM forward ONLY — no LLM API call
Step 4: Compare VLM output to ground-truth label
Step 5: Compute metrics (accuracy, BLEU, etc.)
```

**Key distinction:**
- **Training**: both VLM and LLM are used; VLM learns from LLM
- **Inference**: only VLM is used; LLM is gone; VLM must perform on its own

### The two-process setup

Why does the code use HTTP POST requests between ALFWorld and dagger_server.py?

ALFWorld runs in its own process (or on a separate machine). It cannot directly call Python functions in dagger_server.py. So they communicate over HTTP:

```
ALFWorld Client Process                dagger_server.py Process
      |                                        |
      |-- POST /observation + image ---------> |
      |                                        | put to feedback_queue
      |                                        | main loop reads feedback_queue
      |                                        | LLM forward + VLM forward
      |                                        | decide action
      |                                        | put to action_queue
      |                                        | HTTP handler reads action_queue
      | <-- 200 OK + action string ----------- |
      |                                        |
   Execute action in sim                       |
   Get next observation                        |
      |                                        |
      |-- POST next observation -------------> |
```

The `action_queue` and `feedback_queue` are Python `queue.Queue` objects — thread-safe queues that synchronize the HTTP handler thread with the main training thread.

---

## 5. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        EMMA Model (VLM)                         │
│                                                                 │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐  │
│  │  ViT Encoder │    │    Q-Former    │    │  LLM Projection │  │
│  │ (EVA-CLIP-G) │ -> │ (BERT-based)  │ -> │    Layer (MLP)   │  │
│  │  FROZEN ❄️   │    │  FROZEN ❄️    │    │  TRAINABLE ✅    │  │
│  └──────────────┘    └────────────────┘    └────────┬────────┘  │
│        ↑                                             │           │
│  RGB Image                                          ↓           │
│  224×224                             Visual Tokens (32 tokens)  │
│                                                     │           │
│                                          ┌──────────▼──────────┐│
│                           Text History → │   Vicuna-7B LLM     ││
│                           Prompt      → │   (LLaMA backbone)  ││
│                                          │   FROZEN ❄️         ││
│                                          └──────────┬──────────┘│
│                                                     │           │
│                                              Action Text Output  │
└─────────────────────────────────────────────────────────────────┘

During Training:
  LLM Expert (text-davinci-003) ──────────────────────────────────┐
  [text history only, no image]                                   │
         ↓                                                         │
  Expert action label ──────────────────────────────────────────> │
                                                                  │
  DPO Loss = push VLM to choose expert action over its own action │
  BC  Loss = push VLM to predict expert action exactly            │
             ↑                                                     │
  Only the Projection Layer gets gradient updates ────────────────┘
```

### What is Q-Former?

Q-Former (Querying Transformer) is BLIP-2's key innovation. It acts as a **bridge** between the visual encoder and the language model:

- It has 32 learnable "query tokens" that can attend to image features
- These 32 tokens "compress" the rich visual information into a compact representation
- The 32 tokens are then projected into the LLM's embedding space
- The LLM sees them as if they were 32 extra text tokens at the start of the prompt

This is how vision and language are connected — the image is "translated" into 32 special tokens that the LLM understands.

---

## 6. File-by-file explanation

### 6.1 `dagger_server.py` — The main training loop (482 lines)

This is the most important file. It runs the DAgger online training procedure.

#### Lines 1–27 — Imports

```python
import hashlib          # Used to verify data integrity (MD5 checksums)
import http.server      # Python's built-in HTTP server
import socketserver     # TCP socket server for HTTP
import threading        # Run HTTP server in a separate thread
import queue            # Thread-safe FIFO queue for communication
import time             # Timing and sleep
import logging          # Write logs to file
import json             # Parse/serialize JSON data
import subprocess       # Run shell commands (kill old process)
import os               # File system operations
import re               # Regular expressions (parse text)
import gc               # Python garbage collector (free RAM)
import ast              # Parse Python literal strings safely
import torch            # PyTorch (the ML framework)
import numpy as np      # Numerical arrays
import random           # Random number generation
import torch.backends.cudnn as cudnn  # GPU settings
from PIL import Image   # Load and save images
from dataclasses import dataclass    # Simple data container
from typing import Dict, Any         # Type hints

from lavis.models import load_model_and_preprocess  # Load EMMA model
from lavis.common.optims import LinearWarmupCosineLRScheduler  # Learning rate schedule
from xrl_alignment.utils import *   # Imports everything from utils.py
from generate_reflections import update_memory  # Reflexion generation
from post_processing import action_postprocess, process_ob  # Clean up actions/observations
```

#### Lines 29–39 — Queues and FeedbackData

```python
action_queue = queue.Queue()    # Main loop puts action here → HTTP thread sends it
feedback_queue = queue.Queue()  # HTTP thread puts observation here → main loop reads it

@dataclass
class FeedbackData:
    history: str = ''           # Text history of actions/observations so far
    observation: str = ''       # Current text observation from ALFWorld
    task_type: str = ''         # E.g. "pick_and_place", "pick_heat_then_place"
    information: Dict[str, Any] = ''  # Extra metadata dict from the client
    done: bool = False          # True if the task is completed
    image: np.ndarray = None    # The current RGB image as a PIL Image
```

#### Lines 42–48 — Seed setting

```python
def set_seed(seed):
    random.seed(seed)           # Python random module
    np.random.seed(seed)        # NumPy random module
    torch.manual_seed(seed)     # PyTorch random module
    cudnn.benchmark = False     # Don't optimize based on input size
    cudnn.deterministic = True  # Always use same algorithm → reproducible results
```

#### Lines 51–101 — HTTP Server (ServerThread)

```python
class ServerThread(threading.Thread):
    def run(self):
        class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, ...): pass  # Suppress default HTTP logs

            def do_POST(self):
                # 1. Read request body
                content_length = int(self.headers['Content-Length'])
                data = self.rfile.read(content_length)

                # 2. Verify MD5 checksum (data integrity check)
                received_md5 = self.headers.get('MD5')
                calculated_md5 = hashlib.md5(data).hexdigest()
                if received_md5 != calculated_md5:
                    self.send_response(400)  # Bad request — data corrupted
                else:
                    # 3. Put observation into feedback_queue for main loop
                    feedback_queue.put(data.decode())

                    # 4. BLOCK: wait for main loop to put action into action_queue
                    action = action_queue.get()  # This BLOCKS until action is ready

                    # 5. Send action back to ALFWorld client
                    self.send_response(200)
                    self.wfile.write(str(action).encode())
```

**Why two queues?** Because the HTTP handler runs in a different thread than the training loop. Queues are thread-safe; you can't just share a global variable between threads safely.

**Why MD5?** The observation data (especially the image as JSON) can be large. MD5 verifies it wasn't corrupted during network transfer.

#### Lines 104–119 — release_port()

```python
def release_port():
    # Kill any process already listening on port 7860
    command = "ss -lptn 'sport = :7860'"  # ss = socket statistics tool
    output = subprocess.check_output(command, shell=True).decode()
    # Parse the PID from the output using regex
    pid_pattern = r',pid=(\d+),'
    match = re.search(pid_pattern, line)
    if match:
        pid = match.group(1)
        os.kill(int(pid), 9)  # Kill signal 9 = SIGKILL (force kill)
        time.sleep(5)         # Wait for port to be released
```

This prevents "port already in use" errors when restarting the server.

#### Lines 121–149 — process_feedback()

```python
def process_feedback(text):
    # Parse the raw HTTP body string into a FeedbackData object
    pattern = r'\[#OBSERVATION\](.*?)\[#HISTORY\](.*?)\[#INFORMATION\](.*?)\[#TYPE\](.*?)\[#DONE\](.*?)\[#IMAGE\](.*)'
    # Each field is wrapped in a custom tag: [#OBSERVATION], [#HISTORY], etc.
    mat = re.match(pattern, text, re.DOTALL)

    feedback_data.observation = mat.group(1)    # Text observation
    feedback_data.history = mat.group(2)        # Full text history (for VLM prompt)
    feedback_data.information = ast.literal_eval(mat.group(3))  # Dict metadata
    feedback_data.task_type = mat.group(4)      # Task type string
    feedback_data.done = mat.group(5) == "True" # Boolean done flag

    # Image is sent as a nested JSON array (list of lists of [R,G,B] values)
    py_data = json.loads(mat.group(6))
    feedback_data.image = Image.fromarray(np.asarray(py_data, dtype=np.uint8))
```

#### Lines 151–212 — Global setup

```python
# Start HTTP server immediately
release_port()
PORT = 7860
IP = "0.0.0.0"             # Listen on all network interfaces
server_thread = ServerThread(IP, PORT, action_queue, feedback_queue)
server_thread.start()       # HTTP server now running in background thread

set_seed(42)

# Experiment configuration
num_rounds = 12             # 12 DAgger rounds (each round = all 134 environments)
num_envs = 134              # 134 ALFWorld training environments
run_training = True         # Whether to train the VLM (set False for eval only)
save_ckpt = True            # Save checkpoint after each round
enable_dpo = True           # Use DPO loss (vs. simple BC/imitation loss)
enable_tc = False           # Use "thinking chains" — if False, strip think: steps

# PREFIXES maps task type to prompt file key
PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    ...
}

# Load few-shot prompts from prompts/alfworld_3prompts.json
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

# Initialize per-environment config (memory, success status, etc.)
env_configs[f'env_{i}'] = {
    'memory': [],           # List of Reflexion memory strings
    'is_success': False,    # Has this env been solved?
    'type': None,           # Task type
    'path_length': 0        # How many steps it took to solve
}

# Create replay buffer
if enable_dpo:
    buffer = DPOReplayBuffer(buffer_size=int(1e5), device=device)
else:
    buffer = ReplayBuffer(buffer_size=int(1e5), device=device)

# Setup AdamW optimizer (only projection layer params are trainable)
optimizer = torch.optim.AdamW(optim_params, lr=1e-5, betas=(0.9, 0.999))

# LR schedule: warm up from 1e-8 to 1e-5 over 300 steps, then cosine decay
lr_scheduler = LinearWarmupCosineLRScheduler(...)

# Mixed precision scaler (saves GPU memory, speeds training)
scaler = torch.cuda.amp.GradScaler()
```

#### Lines 254–342 — Main outer loop

```python
# Wait for first ALFWorld request
data = feedback_queue.get()
feedback_data = process_feedback(data)

while trial_idx < num_rounds:             # Loop over 12 rounds
    while cur_task < num_envs:            # Loop over 134 environments
        if env_configs[f'env_{cur_task}']["is_success"]:
            # Already solved: skip this env, send SKIP signal
            action_queue.put(["SKIP"])
        else:
            # Process incoming observation
            image = vis_processors["eval"](feedback_data.image).unsqueeze(0).to(device)
            # vis_processors resizes/normalizes the PIL Image to a tensor

            # Step 1: LLM forward (expert teacher)
            llm_action = llm_forward(str(env_history) + "> ", stop=['\n'],
                                     model="text-davinci-003").strip()
            # env_history is cast to string: it formats as the full ReAct prompt

            # Step 2: Skip thinking steps if enable_tc=False
            while llm_action.startswith("think:") and (not enable_tc):
                env_history.add("action", llm_action)
                env_history.add("observation", "OK.")
                llm_action = llm_forward(...)  # Ask again

            # Step 3: VLM forward (student)
            vlm_action = model.generate(
                {"image": image, "prompt": feedback_data.history},
                use_nucleus_sampling=False,  # Greedy decoding
                num_beams=5,                 # Beam search
                max_length=128,
                ...
            )[0]

            # Step 4: Action decision
            # If LLM outputs "9" → content filtered, use VLM action instead
            if llm_action.startswith("9"):
                llm_action = action_postprocess(vlm_action)
            else:
                llm_action = action_postprocess(llm_action)

            # Step 5: Update history with VLM's action
            env_history.add("action", vlm_action)

            # Step 6: Send VLM action to environment (agent acts in world)
            action_queue.put([vlm_action])

            # Step 7: Collect training data
            text_inputs.append(feedback_data.history)   # Prompt
            text_outputs_llm.append(llm_action)          # Expert label (positive)
            text_outputs_vlm.append(vlm_action)          # Student output (negative for DPO)
            image_inputs.append(image.squeeze(0).cpu().numpy())
```

#### Lines 356–413 — Training block (after each episode)

```python
# After episode ends (new task type detected), store data and train
for j in range(len(text_inputs)):
    if enable_dpo:
        buffer.add(
            visual_ob=image_inputs[j],
            text_input=text_inputs[j],
            w_text_output=text_outputs_llm[j],   # Chosen (expert = good)
            l_text_output=text_outputs_vlm[j]    # Rejected (student = bad)
        )
    else:
        buffer.add(visual_ob=..., text_input=..., text_output=text_outputs_llm[j])

model.train()
for e in range(train_iters_per_epoch * accum_grad_steps):
    samples = buffer.sample(batch_size)
    lr_scheduler.step(cur_epoch=trial_idx, cur_step=training_step)

    with torch.cuda.amp.autocast(enabled=True):  # FP16 mixed precision
        if enable_dpo:
            loss_metrics = model.dpo_forward(samples)
            loss = loss_metrics["loss"]
        else:
            loss_metrics = model.forward(samples)
            loss = loss_metrics["loss"]

    # Gradient accumulation: accumulate over accum_grad_steps before updating
    scaler.scale(loss / accum_grad_steps).backward()
    if (e + 1) % accum_grad_steps == 0:
        scaler.step(optimizer)     # Apply gradients
        scaler.update()            # Update scaler for next step
        optimizer.zero_grad()      # Reset gradients

model.eval()  # Switch back to eval mode after training
```

#### Lines 441–480 — End of round

```python
# Generate Reflexion memories for failed envs
env_configs = update_memory(trial_log_path, env_configs, model='text-davinci-003')

# Save checkpoint
save_checkpoint(model, optimizer, scaler, trial_idx, output_dir)

trial_idx += 1
```

---

### 6.2 `utils.py` — Utilities and buffers (518 lines)

#### LLM wrappers (lines 1–172)

```python
def get_completion(prompt, model='text-davinci-003', ...):
    # Calls OpenAI's text-davinci-003 (completion API)
    # Has token counting: if prompt + max_tokens > 4097, trim the prompt
    # Has retry logic: wait_random_exponential(min=1, max=60), up to 20 attempts
    response = openai.Completion.create(model=model, prompt=prompt, ...)
    text = response["choices"][0]["text"].split("\n")[0]  # Take first line only
    return text

def llm_forward(prompt, model, stop=["\n"]):
    # Wrapper with retry: if output < 5 chars, try again with higher temperature
    text = ""
    cur_try = 0
    while len(text.strip()) < 5 and cur_try < 6:
        text = get_completion(prompt=prompt, temperature=cur_try * 0.2, ...)
        cur_try += 1
    return text
```

The temperature escalation is important: if the LLM keeps returning very short responses, it may be stuck. Increasing temperature makes it more creative/random, which sometimes helps.

#### EnvironmentHistory (lines 175–223)

```python
class EnvironmentHistory:
    def __init__(self, base_query, start_info, memory, history):
        # base_query = few-shot prompt (2-3 example ReAct traces)
        # start_info = the initial observation of this specific env
        # memory = list of Reflexion strings from previous failed attempts
        # history = list of {label, value} dicts (actions + observations)
        self._cur_query = _get_base_query(base_query, start_info, memory)
        self._history = history
        self._last_action = ''
        self._is_exhausted = False  # True if same action repeated twice

    def add(self, label, value):
        # label must be 'action', 'observation', or 'human_edit'
        self._history.append({'label': label, 'value': value})
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True  # Detected repeated action!
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted  # NOTE: this is never called in dagger_server.py!

    def __str__(self):
        # Formats as a ReAct prompt string:
        # [few-shot examples]
        # [memory from previous attempts]
        # [initial observation]
        # > [action 1]
        # [observation 1]
        # > [action 2]
        # [observation 2]
        # ...
        s = self._cur_query + '\n'
        for item in self._history:
            if item['label'] == 'action':
                s += f'> {item["value"]}\n'
            elif item['label'] == 'observation':
                s += item['value'] + '\n'
        return s
```

**Important note:** `_is_exhausted` is set correctly (line 190) but `check_is_exhausted()` is never called anywhere in `dagger_server.py`. This means the anti-repeat detection exists in the data model but has no effect — it is dead code from the perspective of the running system.

#### ReplayBuffer (lines 334–405)

```python
class ReplayBuffer(BaseBuffer):
    # Stores (image, text_input, text_output) tuples for BC training
    visual_obs: np.ndarray  # Shape: (buffer_size, 3, 224, 224)
    text_inputs: List[str]  # Prompt strings
    text_outputs: List[str] # Expert action labels

    def add(self, visual_ob, text_input, text_output):
        self.visual_obs[self.pos] = visual_ob
        self.text_inputs[self.pos] = text_input
        self.text_outputs[self.pos] = text_output
        self.pos = (self.pos + 1) % self.buffer_size  # Circular buffer

    def _get_samples(self, batch_inds):
        return {
            "image": self.to_torch(self.visual_obs[batch_inds, :]),
            "text_input": [self.text_inputs[i] for i in batch_inds],
            "text_output": [self.text_outputs[i] for i in batch_inds],
        }
```

#### DPOReplayBuffer (lines 407–466)

```python
class DPOReplayBuffer(BaseBuffer):
    # Stores (image, text_input, positive_output, negative_output) for DPO
    w_text_outputs: List[str]  # Chosen (expert/LLM action)
    l_text_outputs: List[str]  # Rejected (VLM's own action)

    def _get_samples(self, batch_inds):
        return {
            "image": ...,
            "text_input": ...,
            "text_output_pos": [self.w_text_outputs[i] for i in batch_inds],
            "text_output_neg": [self.l_text_outputs[i] for i in batch_inds]
        }
```

---

### 6.3 `lavis/models/blip2_models/blip2_emac.py` — EMMA model (1054 lines)

This is the VLM (Visual Language Model) at the heart of EMAC+.

#### Architecture components (lines 102–233)

```python
class Blip2Emac(Blip2Base):
    def __init__(self, ...):
        # 1. Visual Encoder: EVA-CLIP-G (a large ViT trained on CLIP)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(...)
        # EVA-CLIP-G encodes 224×224 images into a sequence of patch features

        # FROZEN: visual encoder weights never change during EMAC+ training
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        # 2. Q-Former: a BERT-based transformer with 32 learnable query tokens
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token=32, ...)
        # Q-Former attends to both query tokens and image features
        # Output: 32 compact visual feature vectors

        # FROZEN: Q-Former also never changes
        for param in self.Qformer.parameters():
            param.requires_grad = False

        # 3. Projection Layer: linear layer mapping Q-Former output → LLM embedding space
        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size,  # 768
            self.llm_model.config.hidden_size  # 4096 (for LLaMA-7B)
        )
        # TRAINABLE: this is the only part that gets updated!
        for param in self.llm_proj.parameters():
            param.requires_grad = not freeze_proj_layer  # True = trainable

        # 4. LLM: LLaMA-7B (Vicuna fine-tune)
        self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16)

        # FROZEN: LLM weights never change
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 5. Reference models (for DPO)
        # ref_Qformer and ref_llm_proj are frozen copies of original weights
        # Used as the "before fine-tuning" baseline in DPO loss
        self.ref_Qformer = ...  # frozen copy
        self.ref_llm_proj = ...  # frozen copy
```

**Summary of what is frozen vs trainable:**
```
Component          | Frozen? | Why?
-------------------|---------|--------------------------------------------------
ViT (EVA-CLIP-G)  | YES ❄️  | Already excellent visual features; too expensive
Q-Former           | YES ❄️  | Bridging logic already learned from BLIP-2 pretraining
Projection Layer   | NO  ✅  | This is what we fine-tune — teaches visual→action mapping
LLM (Vicuna-7B)   | YES ❄️  | Backbone reasoning; too expensive; API-level changes not needed
ref_Qformer        | YES ❄️  | DPO reference model must stay fixed
ref_llm_proj       | YES ❄️  | DPO reference model must stay fixed
```

#### forward() — Behaviour Cloning training (lines 258–363)

```python
def forward(self, samples):
    image = samples["image"]

    # 1. Encode image through frozen ViT
    image_embeds = self.ln_vision(self.visual_encoder(image))  # [B, patches, 1408]

    # 2. Q-Former: compress 1408-dim patches into 32 visual query vectors
    query_output = self.Qformer.bert(query_embeds=query_tokens,
                                      encoder_hidden_states=image_embeds, ...)
    # query_output.last_hidden_state[:, :32, :] = 32 visual tokens

    # 3. Project to LLM embedding space
    inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :32, :])  # [B, 32, 4096]

    # 4. Tokenize text input (prompt/history) and text output (expert action)
    text_input_tokens = self.llm_tokenizer(samples['text_input'], ...)
    text_output_tokens = self.llm_tokenizer(samples["text_output"] + eos_token, ...)

    # 5. Concatenate: [visual_tokens | text_input | text_output]
    # Only compute loss on text_output positions (ignore input and visual)
    targets = ...  # -100 (ignore) for input/visual, real token IDs for output

    # 6. LLM forward pass — teacher forcing
    inputs_embeds = cat([inputs_llm, text_embeddings])  # prepend visual tokens
    outputs = self.llm_model(inputs_embeds=inputs_embeds, labels=targets)

    # 7. Loss = cross-entropy on predicted vs target action tokens
    return {"loss": outputs.loss}
```

#### dpo_forward() — DPO training (lines 365–519)

```python
def dpo_forward(self, samples):
    # Same image encoding as forward()

    # Process BOTH positive (LLM expert) and negative (VLM student) responses
    text_output_tokens_pos = tokenize(samples["text_output_pos"])  # chosen
    text_output_tokens_neg = tokenize(samples["text_output_neg"])  # rejected

    # Run LLM 4 times:
    pos_logits   = llm(image_tokens + text_input + text_output_pos)   # policy, chosen
    neg_logits   = llm(image_tokens + text_input + text_output_neg)   # policy, rejected
    ref_pos_logits = llm(ref_image_tokens + text_input + text_output_pos)  # reference, chosen
    ref_neg_logits = llm(ref_image_tokens + text_input + text_output_neg)  # reference, rejected

    # Compute log probabilities
    pos_logps     = _get_batch_logps(pos_logits, targets_pos)
    neg_logps     = _get_batch_logps(neg_logits, targets_neg)
    ref_pos_logps = _get_batch_logps(ref_pos_logits, targets_pos).detach()
    ref_neg_logps = _get_batch_logps(ref_neg_logits, targets_neg).detach()

    # DPO loss
    loss = dpo_loss(pos_logps, neg_logps, ref_pos_logps, ref_neg_logps, beta=0.1, ...)
```

#### DPO Loss function (lines 23–57)

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta, mix_ratio):

    # pi_logratios: how much MORE likely the policy assigns to chosen vs rejected
    pi_logratios = policy_chosen_logps - policy_rejected_logps

    # ref_logratios: same ratio for the reference model
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # logits: improvement over reference model
    logits = pi_logratios - ref_logratios

    # Core DPO loss: push logits to be positive (chosen > rejected)
    # -logsigmoid(beta * logits) → 0 when chosen is clearly better
    losses = -F.logsigmoid(beta * logits) - mix_ratio * policy_chosen_logps

    # The mix_ratio term adds a BC component: also maximize probability of chosen action
    # This prevents the model from diverging too far from useful behaviour

    return losses.mean(), chosen_rewards, rejected_rewards, nll_loss
```

#### generate() — Inference (lines 521–657)

```python
@torch.no_grad()  # No gradient computation during inference
def generate(self, samples, use_nucleus_sampling=False, num_beams=5, ...):
    image = samples["image"]
    prompt = samples["prompt"]   # The text history

    # Encode image
    image_embeds = self.ln_vision(self.visual_encoder(image))
    query_output = self.Qformer.bert(...)
    inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :32, :])

    # Tokenize prompt
    llm_tokens = self.llm_tokenizer(prompt, ...)

    # Concatenate visual tokens + prompt tokens
    inputs_embeds = cat([inputs_llm, text_embeddings])

    # Beam search generation
    outputs = self.llm_model.generate(
        inputs_embeds=inputs_embeds,
        num_beams=5,           # Keep top-5 candidates, pick best
        max_length=128,
        repetition_penalty=1.0,
        ...
    )

    # Decode token IDs to text
    output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_text  # List of action strings
```

---

### 6.4 `post_processing.py` — Action normalisation (97 lines)

ALFWorld expects very specific action strings. The LLM and VLM sometimes output slightly different formats ("put the apple in the fridge 1" vs "put apple 1 in/on fridge 1"). This file normalises them:

```python
def remove_the(sentence):
    # "take the apple 1" → "take apple 1"
    words = sentence.split(' ')
    words = [word for word in words if word.lower() != 'the']
    return ' '.join(words).strip()

def check_put(action):
    # Normalise "put X in Y" OR "put X on Y" → "put X in/on Y"
    # ALFWorld accepts both "in" and "on" via this format
    action = remove_the(action)
    match = re.match(r'put (.*) in (.*)', action)
    if match: return f"put {match.group(1)} in/on {match.group(2)}"
    ...

def check_goto(action):
    # "go to the cabinet 1 slowly" → "go to cabinet 1" (keep only 4 words)
    action = remove_the(action)
    words = action.split(" ")
    return " ".join(words[:4])

def action_postprocess(action):
    action = action.strip(string.punctuation).lower().strip()
    if action.startswith('>'):
        action = action.split(">")[1].strip()  # Remove leading ">"

    if action.startswith('put'):    return check_put(action)
    if action.startswith('go'):     return check_goto(action)
    if action.startswith('use'):    return check_use(action)
    if action.startswith('heat'):   return remove_the(action)
    # etc.
    return action
```

---

### 6.5 `generate_reflections.py` — Reflexion memory (54 lines)

```python
def _get_scenario(s):
    # Extract the most recent episode's actions from the trial log
    # The log is split by "\n\n", last second entry = current episode
    return s.split("\n\n")[-2].strip()

def _generate_reflection_query(log_str, memory):
    # Build a prompt for the LLM to generate a reflection:
    # "Here are past examples. Here is your failed episode.
    #  What should you do differently? Give your plan after 'Plan:'."
    scenario = _get_scenario(log_str)
    query = f"""You will be given the history of a past experience...
{FEW_SHOT_EXAMPLES}    ← 2 example reflections from reflexion_few_shot_examples.txt
{scenario}             ← the actual failed episode
Plans from past attempts: ...
New plan: """
    return query

def update_memory(trial_log_path, env_configs, model):
    # Read the trial log (all episodes in this round)
    # Split by "#####" separator
    # For each failed env, generate a reflection and add to env_configs[i]['memory']
    for i, env in env_configs.items():
        if not env['is_success']:
            reflection = get_completion(reflection_query, model)
            env_configs[i]['memory'].append(reflection)
    return env_configs
```

---

### 6.6 `train.py` — Offline training entry point (97 lines)

This file uses LAVIS's `RunnerBase` for **offline** training on a pre-collected dataset (as opposed to the online DAgger training in `dagger_server.py`):

```python
def main():
    cfg = Config(parse_args())   # Load alfworld_ft.yaml
    task = tasks.setup_task(cfg) # Set up "captioning" task
    datasets = task.build_datasets(cfg)   # Load images + annotations from disk
    model = task.build_model(cfg)         # Instantiate Blip2Emac
    runner = get_runner_class(cfg)(...)   # LAVIS training runner
    runner.train()                        # Standard epoch-based training loop
```

This is used when you have a pre-collected dataset of (image, history, expert_action) tuples and want to do supervised fine-tuning before running DAgger.

---

### 6.7 `evaluate.py` — Evaluation entry point (86 lines)

Similar to `train.py` but calls `runner.evaluate()` instead of `runner.train()`. Used to measure accuracy on a held-out dataset after training.

---

### 6.8 Config files

#### `lavis/configs/models/blip2/blip2_emac.yaml`
```yaml
model:
  arch: blip2_emac           # Which model class to use
  base_model: "https://...instruct_blip_vicuna7b_trimmed.pth"  # Pretrained weights

  freeze_vit: True            # Visual encoder: FROZEN
  freeze_qformer: True        # Q-Former: FROZEN
  freeze_proj_layer: False    # Projection layer: TRAINABLE

  llm_model: "/srv/.../vicuna-7b-v1.1"  # Path to LLaMA/Vicuna checkpoint

  avg_log_probs: True         # DPO: average log probs over tokens (vs sum)
  mix_ratio: 0.5              # DPO: weight of BC term (higher = more imitation)
```

#### `lavis/projects/instructblip/finetuning/alfworld_ft.yaml`
```yaml
datasets:
  alfworld_cap:
    data_type: images           # Load actual image files from disk
    build_info:
      annotations:
        train:
          url: /srv/.../train.json  # JSON file mapping image paths to expert actions
      images:
        storage: /srv/.../images/  # Directory of JPEG images
run:
  max_epoch: 10
  batch_size_train: 4
  world_size: 4                 # 4 GPUs for distributed training
```

---

### 6.9 Prompt files

#### `prompts/alfworld_3prompts.json`
Contains few-shot examples in ReAct format. For each task type (put, clean, heat, cool, examine, puttwo), there are 3 example episodes showing the expected dialogue pattern.

#### `prompts/reflexion_few_shot_examples.txt`
Contains 2 example failed episodes with their corresponding reflection summaries, used to teach the LLM how to generate good reflections.

---

## 7. Setup and installation

### Prerequisites
- Linux (Ubuntu 20.04+) or Gadi HPC
- Python 3.9
- CUDA 11.7+ with an NVIDIA GPU (A100 or H100 recommended — model is ~14GB)
- conda or miniconda

### Step 1: Clone and create environment

```bash
git clone <repo-url>
cd EMAC-Embodied-Multimodal-Agent-for-Collaborative-Planning-with-VLM-LLM

conda create -n emac python=3.9 -y
conda activate emac
```

### Step 2: Install dependencies
```bash
# Install core packages via conda first (avoids HPC library conflicts)
conda install -c conda-forge spacy opencv libtiff libjpeg-turbo libstdcxx-ng -y

# Install LAVIS and other dependencies via pip
pip install -e .
pip install -r requirements.txt
pip install "transformers==4.30.0"
pip install "numpy<2.0"
```

### Step 3: Fix libstdc++ (required on HPC clusters with old system libraries)
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6' > $CONDA_PREFIX/etc/conda/activate.d/libstdc++.sh
```

This runs automatically every time you `conda activate emac`, so you only need to do it once.

### Step 4: Verify installation
```bash
python -c "from lavis.models import load_model_and_preprocess; print('OK')"
# Should print: OK
```

### Step 5: Download model weights

You need two model checkpoints:

1. **InstructBLIP Vicuna-7B** (base VLM):
   - Download from: `https://storage.googleapis.com/.../instruct_blip_vicuna7b_trimmed.pth`
   - Or set `base_model` in `blip2_emac.yaml` to the URL (auto-downloaded)

2. **Vicuna-7B-v1.1** (the LLM backbone):
   - Download from Hugging Face: `lmsys/vicuna-7b-v1.1`
   - Set path in `blip2_emac.yaml`: `llm_model: "/path/to/vicuna-7b-v1.1"`

```bash
# Example: download Vicuna-7B
python -c "from huggingface_hub import snapshot_download; snapshot_download('lmsys/vicuna-7b-v1.1', local_dir='/path/to/vicuna-7b-v1.1')"
```

### Step 6: Configure OpenAI API (for LLM expert)

```bash
# Edit utils.py lines 41-42:
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"  # or your endpoint
```

### Step 7: Set up ALFWorld

```bash
pip install alfworld
# Download ALFWorld data
python -c "import alfworld; alfworld.agents.environment.download_data()"
```

---

## 8. How to run EMAC+

### Mode 1: Online DAgger Training (`dagger_server.py`)

This is the main training mode. It trains the VLM online as the agent interacts with ALFWorld.

**Terminal 1 — Start the training server:**
```bash
conda activate emac
cd /path/to/EMAC

# Edit output_dir and model paths in dagger_server.py first
python dagger_server.py
# You should see: "Server running on IP: 0.0.0.0 PORT: 7860"
# The server waits for ALFWorld client to connect
```

**Terminal 2 — Start the ALFWorld client:**
```bash
# The ALFWorld client sends observations to port 7860 and executes returned actions
# (The client code is separate from this repository)
# It sends HTTP POST requests to http://localhost:7860
```

**What happens:**
1. ALFWorld client starts, sends first observation
2. dagger_server.py processes observation, queries LLM and VLM, sends action back
3. Client executes action, sends next observation
4. This repeats for all 134 environments × 12 rounds
5. Every episode, the VLM is trained on collected data
6. Every round, Reflexion memories are generated for failed envs
7. Checkpoints saved as `emma_checkpoint_{round}.pth`

**Configuration in `dagger_server.py`:**
```python
num_rounds = 12        # How many rounds to train
num_envs = 134         # Number of environments
enable_dpo = True      # Use DPO loss
output_dir = "..."     # Where to save logs and checkpoints
```

### Mode 2: Offline Training (`train.py`)

Use this if you have pre-collected data and want to do supervised fine-tuning:

```bash
# On Gadi HPC (SGE scheduler):
qsub run_scripts/instructblip/finetuning/ft_caption_alfworld.sh

# Or directly:
python -m torch.distributed.run \
    --nproc_per_node=4 \        # Use 4 GPUs
    train.py \
    --cfg-path lavis/projects/instructblip/finetuning/alfworld_ft.yaml
```

You need to first edit `alfworld_ft.yaml` to point to your dataset:
```yaml
build_info:
  annotations:
    train:
      storage: /path/to/your/train.json
  images:
    storage: /path/to/your/images/
```

The `train.json` should be a list of entries like:
```json
[
  {
    "image": "env_0_step_3.jpg",
    "text_input": "[history prompt here]",
    "text_output": "heat apple 1 with microwave 1"
  },
  ...
]
```

### Mode 3: Evaluation (`evaluate.py`)

```bash
python evaluate.py \
    --cfg-path lavis/projects/instructblip/finetuning/alfworld_ft.yaml
```

This evaluates action prediction accuracy on a held-out validation set.

### Mode 4: ALFWorld evaluation (with trained model)

After training, you can evaluate the full agent (not just action prediction) on ALFWorld tasks:
1. Load a saved checkpoint into the model
2. Run `dagger_server.py` with `run_training = False`
3. Run the ALFWorld client
4. Check success rate from the world.log file

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **VLM** | Vision-Language Model — processes both images and text |
| **LLM** | Large Language Model — processes text only |
| **EMMA** | The specific VLM used here: BLIP-2 InstructBLIP with LLaMA/Vicuna-7B backbone |
| **BLIP-2** | A VLM architecture from Salesforce that uses a Q-Former bridge |
| **Q-Former** | Query Transformer — compresses image features into 32 tokens |
| **EVA-CLIP** | Visual encoder (ViT variant) pre-trained with CLIP objective |
| **Vicuna** | A fine-tuned version of LLaMA-7B; used as the language backbone |
| **DAgger** | Dataset Aggregation — online imitation learning algorithm |
| **DPO** | Direct Preference Optimization — trains model from preferred vs rejected pairs |
| **ReAct** | Reason+Act — prompting pattern alternating between thinking and acting |
| **Reflexion** | Generate text "lessons learned" from failed episodes; store as memory |
| **ALFWorld** | Household text+visual simulation with 134 environments, 6 task types |
| **Replay Buffer** | Memory that stores past (state, action) pairs for training |
| **Beam Search** | Generation strategy: keep top-k candidates at each step, return best |
| **Mixed Precision** | Use FP16 for forward pass (faster, less memory), FP32 for gradients |
| **Gradient Accumulation** | Simulate large batch by accumulating gradients over N steps |
| **BC** | Behaviour Cloning — imitation learning: predict expert action exactly |
| **LoRA / adaptor tuning** | Parameter-efficient fine-tuning methods (not used in main config) |
| **HPC / SGE** | High Performance Computing / Sun Grid Engine scheduler (Gadi cluster) |
