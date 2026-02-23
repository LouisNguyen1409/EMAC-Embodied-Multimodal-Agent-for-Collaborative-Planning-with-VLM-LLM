# Thesis Plan — EMAC+ Research

> **Your status:** Thesis A ✅ presentation done, ✅ 3 RQs defined. Today: 22 Feb 2026.
> **Working schedule:** 5 days/week including term breaks.
> **Total time:** 43 weeks (214 working days) until 18 Dec 2026.
>
> Tick tasks with `[x]` as you finish them. This is your single source of truth.

---

## Environment Decision (Final)

| RQ      | Environment  | Reason                                                                                                     |
| ------- | ------------ | ---------------------------------------------------------------------------------------------------------- |
| **RQ1** | ALFWorld     | Already set up. Fast. Get results in Term 1.                                                               |
| **RQ2** | **AI2-THOR** | Real egocentric camera = genuine partial observability. Much stronger than BabyAI-Text for stuck/recovery. |
| **RQ3** | ScienceWorld | 30-60 step tasks = memory compression matters. Has CLIN published baseline to compare against.             |

> ~~**Fallback rule for RQ2:** If AI2-THOR is not working by **21 May (Week 14)**, switch to BabyAI-Text.~~ **Resolved** — AI2-THOR confirmed working on Katana GPU nodes (Feb 2026). Use `platform=CloudRendering` + `xvfb-run -a`.

---

## 3 Research Questions

| RQ      | Question                                                                                | Hypotheses                                                                                                  |
| ------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **RQ1** | How does enriching the feedback signal affect training efficiency and task success?     | H1: Shaped feedback improves success vs binary. H2: Anti-repeat reduces loops.                              |
| **RQ2** | How does stuck-detection and recovery policy affect failure-loop frequency and success? | H1: Recovery reduces loops. H2: Backtracking better than scan. H3: Stagnation trigger reduces false alarms. |
| **RQ3** | How does memory representation affect improvement across re-attempts?                   | H1: Reflections improve attempt-to-attempt vs raw. H2: Hybrid performs best.                                |

---

## WEEK-BY-WEEK SCHEDULE

### TERM 1 — 22 Feb to 30 Apr (10 weeks)

> **Goal:** Thesis A report submitted + EMAC+ code running + RQ1 implemented + AI2-THOR setup begun.

---

**Week 1 — 22 Feb to 26 Feb**

- [ ] Read the **EMAC+ paper** fully — take structured notes (see Guide.md §1 and §2)
- [x] Install conda, create `emac` Python 3.9 environment
- [x] Run `conda env create -f environment.yml` + `pip install -e . --no-deps` — full environment set up
- [x] Create Overleaf project with faculty LaTeX thesis template

---

**Week 2 — 1 Mar to 5 Mar**

- [ ] Read **ReAct paper** (Yao et al., 2022) — take notes
- [ ] Read **Reflexion paper** (Shinn et al., 2023) — take notes
- [x] Download Vicuna-7B-v1.1 weights to HPC (start download, it takes hours)
- [x] Install ALFWorld and download its game data
- [x] Confirm ALFWorld launches without errors

---

**Week 3 — 8 Mar to 12 Mar**

- [ ] Read **DPO paper** (Rafailov et al., 2023) — take notes
- [ ] Read **InstructBLIP paper** (Dai et al., 2023) — take notes
- [x] Run first EMAC+ baseline test: `num_envs=10, num_rounds=2, run_training=False`
- [x] Confirm dagger_server.py and ALFWorld client communicate (2-terminal setup)
- [ ] Record preliminary success rate in experiment_log.md

---

**Week 4 — 15 Mar to 19 Mar**

- [ ] Read **ALFWorld paper** (Shridhar et al., 2021)
- [ ] Read **AI2-THOR / ALFRED paper** (Kolve et al., 2017 / Shridhar et al., 2020) — for RQ2 background
- [ ] Read **reward shaping paper** (Ng et al., 1999) — for RQ1 background
- [ ] Write **Introduction section** of Thesis A report in Overleaf (~3 pages)

---

**Week 5 — 22 Mar to 26 Mar**

- [ ] Read **CLIN paper** (Majumder et al., 2023) — for RQ3 background
- [ ] Read **ScienceWorld paper** (Wang et al., 2022) — for RQ3 environment
- [ ] Write **Literature Review section** Part 1: Embodied AI agents, ReAct, Reflexion (~5 pages)

---

**Week 6 — 29 Mar to 2 Apr**

- [ ] Read **Generative Agents** (Park et al., 2023) — agents with memory and planning
- [ ] Write **Literature Review section** Part 2: EMAC+, DPO, VLM+LLM approaches (~5 pages)
- [ ] Write **Literature Review section** Part 3: Environments (ALFWorld, AI2-THOR, ScienceWorld) (~3 pages)
- [x] Install `ai2thor` Python package + confirm working on Katana with `platform=CloudRendering` + `xvfb-run -a`

---

**Week 7 — 5 Apr to 9 Apr**

- [ ] Write **Research Gaps section** in Thesis A report (~2 pages — 3 clear gaps matching your 3 RQs)
- [ ] Write **Research Questions section** in Thesis A report (~2 pages — formal statement of RQ1, RQ2, RQ3 with hypotheses)
- [x] AI2-THOR basic environment test passed — FloorPlan1 loads, egocentric camera shows partial obs ✅

---

**Week 8 — 12 Apr to 16 Apr**

- [ ] Write **Preliminary Experiments section** in Thesis A report (describe your Week 3 baseline test + results)
- [ ] Write **Research Timeline section** in Thesis A report (use this Plan.md as reference)
- [ ] Begin **RQ1 implementation**: write `extract_progress_signal()` and `classify_failure()` functions in dagger_server.py
- [ ] Test these functions in isolation: `python test_rq1_functions.py`

---

**Week 9 — 19 Apr to 23 Apr**

- [ ] Write **Abstract** for Thesis A report
- [ ] Proofread and compile full Thesis A report in Overleaf
- [ ] Implement **anti-repeat penalty** using `check_is_exhausted()` in dagger_server.py
- [ ] Integrate all 3 RQ1 signals into the VLM prompt in dagger_server.py
- [ ] Run a 3-env test confirming RQ1 signals appear in logs

---

**Week 10 — 26 Apr to 30 Apr**

- [ ] **Submit Thesis A report** ✅
- [ ] Fix any remaining RQ1 code bugs from Week 9 testing
- [ ] Git commit all RQ1 code changes: `git checkout -b rq1-feedback-shaping && git commit`
- [ ] Start drafting RQ2 environment adapter (understand ALFRED action space)

---

### BREAK 1 — 1 May to 31 May (4 weeks)

> **Goal:** Run all RQ1 experiments. Get AI2-THOR working. Decide final RQ2 environment by Week 14.

---

**Week 11 — 3 May to 7 May**

- [ ] Run **RQ1 Condition 0 (Baseline)**: 10 envs, 3 rounds, no modifications — record results
- [ ] Run **RQ1 Condition A**: progress-shaped feedback only — record results
- [x] AI2-THOR environment loading and rendering confirmed working ✅ (completed Week 6-7, ahead of schedule)

---

**Week 12 — 10 May to 14 May**

- [ ] Run **RQ1 Condition B**: failure-type feedback only — record results
- [ ] Run **RQ1 Condition C**: anti-repeat penalty only — record results
- [ ] Deep dive AI2-THOR: understand the action space, observation format, HTTP/gym API

---

**Week 13 — 17 May to 21 May**

- [ ] Run **RQ1 Condition D**: all three signals combined — record results
- [ ] Record all RQ1 results in a table in experiment_log.md
- [ ] Plot RQ1 learning curves (success rate per round, all conditions on one graph)

---

**Week 14 — 24 May to 28 May** ⚠️ DECISION POINT

- [x] **AI2-THOR decision: confirmed working** — write adapter for dagger_server.py (AI2-THOR confirmed Feb 2026, weeks ahead of this deadline)
- [ ] Write the chosen RQ2 environment adapter (wrap gym API to match dagger_server.py HTTP format)
- [ ] Analyse RQ1 results: does shaped feedback help? Write 1-page analysis.

---

**Week 15 — 31 May to 4 Jun** _(week spans break + Term 2 start)_

- [ ] Test RQ2 adapter: confirm dagger_server.py receives observations from new environment
- [ ] Run 3-environment sanity check in RQ2 environment
- [ ] Write first draft of **Introduction** for main thesis report in Overleaf (reuse and expand Thesis A intro)

---

### TERM 2 — 1 Jun to 13 Aug (10 weeks)

> **Goal:** Complete RQ2 experiments. Set up ScienceWorld. Write Introduction + Background + System Design chapters. Thesis B presentation.

---

**Week 16 — 7 Jun to 11 Jun**

- [ ] Implement `detect_stuck_A()` — same action fails twice
- [ ] Implement `detect_stuck_B()` — N steps without progress (threshold=4)
- [ ] Implement `track_progress_checkpoints()` — record (step, obs) when progress occurs
- [ ] Unit test all three functions

---

**Week 17 — 14 Jun to 18 Jun**

- [ ] Implement `recovery_scan()` — inject `look` + context message into prompt
- [ ] Implement `recovery_backtrack()` — logical plan revert to last checkpoint
- [ ] Test on 3 RQ2 environments — confirm stuck events trigger and recovery fires

---

**Week 18 — 21 Jun to 25 Jun**

- [ ] Fix any bugs from Week 17 testing
- [ ] Git commit all RQ2 code: `git checkout -b rq2-stuck-recovery && git commit`
- [ ] Run **RQ2 Condition 0 (Baseline)**: no detector, no recovery — record results
- [ ] Run **RQ2 Condition A**: Stuck-A only, no recovery — record results

---

**Week 19 — 28 Jun to 2 Jul**

- [ ] Run **RQ2 Condition B**: Stuck-B only, no recovery — record results
- [ ] Run **RQ2 Condition C**: Stuck-A + Policy 1 (scan) — record results
- [ ] Run **RQ2 Condition D**: Stuck-A + Policy 2 (backtrack) — record results

---

**Week 20 — 5 Jul to 9 Jul**

- [ ] Run **RQ2 Condition E**: Stuck-B + Policy 1 (scan) — record results
- [ ] Run **RQ2 Condition F**: Stuck-B + Policy 2 (backtrack) — record results
- [ ] Record all RQ2 results in table + plot learning curves

---

**Week 21 — 12 Jul to 16 Jul**

- [ ] Analyse RQ2 results: which detector + policy combination is best? Write 1-page analysis.
- [ ] Write **Background / Literature Review** chapter for main thesis (expand from Thesis A, 15-20 pages)

---

**Week 22 — 19 Jul to 23 Jul**

- [ ] Write **System Design** chapter: EMAC+ architecture + your 3 modifications with diagrams (10-15 pages)
- [ ] Create architecture diagrams in draw.io (ViT→QFormer→LLM, DAgger loop, RQ1 modification)
- [x] Install ScienceWorld and Java — see Guide.md §4.3

---

**Week 23 — 26 Jul to 30 Jul**

- [ ] Explore ScienceWorld: run 3 task types (boil, freeze, react) — understand observation format
- [ ] Choose which ScienceWorld tasks to use for RQ3 — write 1-paragraph justification
- [ ] Write adapter to connect ScienceWorld to dagger_server.py loop

---

**Week 24 — 2 Aug to 6 Aug**

- [ ] Test ScienceWorld adapter on 3 environments — confirm end-to-end communication works
- [ ] Prepare **Thesis B presentation** slides (10-15 slides): RQ1 results, RQ2 status, RQ3 plan
- [ ] Write **Methodology** chapter section on RQ1 + RQ2 experimental design

---

**Week 25 — 9 Aug to 13 Aug**

- [ ] **Deliver Thesis B presentation** ✅
- [ ] Buffer week: catch up on anything delayed, polish presentation

---

### BREAK 2 — 14 Aug to 13 Sep (4 weeks)

> **Goal:** Complete all RQ3 experiments. Write RQ3 results. Keep writing thesis report.

---

**Week 26 — 16 Aug to 20 Aug**

- [ ] Implement `get_raw_trajectory_memory(env_history, last_k=5)` in dagger_server.py
- [ ] Implement `get_hybrid_memory(env_history, reflections, last_k=5)` in dagger_server.py
- [ ] Add `memory_type` flag that switches between: `none`, `raw`, `reflection`, `hybrid`

---

**Week 27 — 23 Aug to 27 Aug**

- [ ] Test all 3 memory types on 2 ScienceWorld environments — confirm memory appears correctly in prompts
- [ ] Git commit: `git checkout -b rq3-memory && git commit`
- [ ] Run **RQ3 Condition 0 (No memory)**: record success at attempt 1, 2, 3

---

**Week 28 — 30 Aug to 3 Sep**

- [ ] Run **RQ3 Condition A**: raw recent trajectory — record success per attempt
- [ ] Run **RQ3 Condition B**: Reflexion only (existing EMAC+ approach) — record success per attempt
- [ ] Run **RQ3 Condition C**: hybrid (raw + reflection) — record success per attempt

---

**Week 29 — 6 Sep to 10 Sep**

- [ ] Record all RQ3 results in table (rows = conditions, columns = attempt 1/2/3)
- [ ] Plot RQ3 results: bar chart of success per attempt per condition
- [ ] Analyse results: which memory type shows strongest improvement across attempts?

---

**Week 30 — 13 Sep to 17 Sep** _(spans Break 2 and Term 3 start)_

- [ ] Write **RQ1 Results** section for main thesis (tables + figures + analysis)
- [ ] Write **RQ2 Results** section for main thesis (tables + figures + analysis)

---

### TERM 3 — 14 Sep to 18 Dec (14 weeks)

> **Goal:** Finalise all results. Write complete thesis report. Final presentation. Submit.

---

**Week 31 — 20 Sep to 24 Sep**

- [ ] Write **RQ3 Results** section for main thesis (tables + figures + analysis)
- [ ] Re-run any experiments that had issues or need more data

---

**Week 32 — 27 Sep to 1 Oct**

- [ ] Write **Discussion** chapter: synthesise all 3 RQs — do results confirm hypotheses? (~5-8 pages)
- [ ] Create all remaining diagrams: DAgger loop, stuck-detection flowchart, memory diagram

---

**Week 33 — 4 Oct to 8 Oct**

- [ ] Write **Conclusion** chapter: answer each RQ in 2-3 sentences, state limitations, suggest future work (~3 pages)
- [ ] Write **Experimental Setup** chapter: environments, hyperparameters, hardware, metrics (~5 pages)

---

**Week 34 — 11 Oct to 15 Oct**

- [ ] Write **Abstract** for main thesis (~half page)
- [ ] Finalise **Introduction** chapter — write last after everything else is done
- [ ] First complete draft of all chapters assembled

---

**Week 35 — 18 Oct to 22 Oct**

- [ ] Read through full draft from start to end — mark all gaps and inconsistencies
- [ ] Fix all sections flagged in the read-through
- [ ] Send draft to supervisor for feedback

---

**Week 36 — 25 Oct to 29 Oct**

- [ ] Address supervisor feedback
- [ ] Polish all figures and tables (consistent style, captions complete, all referenced in text)
- [ ] Check all citations in references.bib are complete and formatted correctly

---

**Week 37 — 1 Nov to 5 Nov**

- [ ] Second full read-through — fix grammar, clarity, flow
- [ ] Ensure every figure and table is referenced from the text
- [ ] Compile PDF and check formatting (margins, font size, page numbers)

---

**Week 38 — 8 Nov to 12 Nov**

- [ ] Final proofread — word by word if possible
- [ ] Ask a friend/colleague to read one chapter for clarity
- [ ] Final PDF ready

---

**Week 39 — 15 Nov to 19 Nov**

- [ ] **Create final presentation** slides (20-25 slides): all 3 RQs + results + conclusion
- [ ] Create all presentation diagrams and result visualisations

---

**Week 40 — 22 Nov to 26 Nov**

- [ ] Practice presentation out loud 3 times
- [ ] Time yourself: aim for 20 min talk + 5 min Q&A prep
- [ ] Prepare answers for likely questions (What if results are negative? How does this compare to paper X?)

---

**Week 41 — 29 Nov to 3 Dec**

- [ ] **Deliver final presentation** ✅
- [ ] Final polish of thesis report based on presentation Q&A feedback

---

**Week 42 — 6 Dec to 10 Dec**

- [ ] **Submit final thesis report** ✅
- [ ] Buffer week

---

**Week 43 — 13 Dec to 17 Dec**

- [ ] Hard deadline: 18 Dec 2026
- [ ] Done. 🎓

---

## Visual Timeline (One-Page Overview)

```
TERM 1  │Feb22─────────────────────────────────────Apr30│
Wk 1-3  │ Read papers + Set up code + Run baseline      │
Wk 4-7  │ Write Thesis A report                         │
Wk 8-10 │ Implement RQ1 + Submit report                 │

BREAK 1 │May1──────────────────────────────────────May31│
Wk 11-13│ Run ALL RQ1 experiments + Analyse             │
Wk 14-15│ ⚠️ AI2-THOR decision + RQ2 adapter built      │

TERM 2  │Jun1──────────────────────────────────────Aug13│
Wk 16-17│ Implement RQ2 (stuck detectors + recovery)    │
Wk 18-20│ Run ALL RQ2 experiments                       │
Wk 21-22│ Analyse RQ2 + Write Background chapter        │
Wk 23-24│ ScienceWorld setup + Write System Design      │
Wk 25   │ Thesis B Presentation 🎤                      │

BREAK 2 │Aug14─────────────────────────────────────Sep13│
Wk 26-27│ Implement RQ3 memory types                    │
Wk 28-29│ Run ALL RQ3 experiments                       │
Wk 30   │ Write RQ1 + RQ2 results chapters              │

TERM 3  │Sep14─────────────────────────────────────Dec18│
Wk 31   │ Write RQ3 results chapter                     │
Wk 32-33│ Write Discussion + Conclusion + Exp Setup     │
Wk 34   │ First complete draft assembled                │
Wk 35-36│ Supervisor feedback → revise                  │
Wk 37-38│ Final proofread + polish                      │
Wk 39-40│ Final presentation prep + practice            │
Wk 41   │ Final Presentation 🎤 + Submit 🎓             │
Wk 42-43│ Buffer                                        │
```

---

## Progress Tracker

Update status as you go: ⬜ Not started → 🔄 In progress → ✅ Done

| Week  | Task                           | Status  | Notes                                           |
| ----- | ------------------------------ | ------- | ----------------------------------------------- |
| -     | Thesis A presentation          | ✅ Done |                                                 |
| -     | 3 RQs defined                  | ✅ Done |                                                 |
| 1     | Read EMAC+ paper               | ⬜      |                                                 |
| 1     | Set up conda + install deps    | ✅ Done | environment.yml + pip install -e . --no-deps    |
| 2     | Read ReAct + Reflexion         | ⬜      |                                                 |
| 2     | ALFWorld working               | ✅ Done |                                                 |
| 2     | Vicuna-7B downloaded           | ✅ Done | On /srv/scratch                                 |
| 6     | AI2-THOR installed + working   | ✅ Done | CloudRendering + xvfb-run -a on Katana GPU      |
| 3     | Run baseline EMAC+ (10 envs)   | ⬜      |                                                 |
| 4-9   | Write Thesis A report          | ⬜      |                                                 |
| 8-9   | RQ1 code implemented           | ⬜      |                                                 |
| 10    | Submit Thesis A report         | ⬜      |                                                 |
| 11-13 | All 5 RQ1 experiments run      | ⬜      |                                                 |
| 14    | AI2-THOR decision made         | ✅ Done | AI2-THOR confirmed working Feb 2026             |
| 16-17 | RQ2 stuck+recovery implemented | ⬜      |                                                 |
| 18-20 | All 7 RQ2 experiments run      | ⬜      |                                                 |
| 23    | ScienceWorld working           | ⬜      | Installed via environment.yml — run verify test |
| 25    | Thesis B presentation          | ⬜      |                                                 |
| 26-27 | RQ3 memory types implemented   | ⬜      |                                                 |
| 28-29 | All 4 RQ3 experiments run      | ⬜      |                                                 |
| 34    | First full thesis draft        | ⬜      |                                                 |
| 41    | Final presentation + submit    | ⬜      |                                                 |

---

## Paper Reading List

Mark as you go: [R] Read [N] Notes taken [W] Written about in report

### Week 1-3 (Essential — read first)

- [ ] [R][N][W] EMAC+ paper _(the paper this code is based on)_
- [ ] [R][N][W] ReAct — Yao et al., 2022
- [ ] [R][N][W] Reflexion — Shinn et al., 2023
- [ ] [R][N][W] InstructBLIP — Dai et al., 2023
- [ ] [R][N][W] DPO — Rafailov et al., 2023
- [ ] [R][N][W] ALFWorld — Shridhar et al., 2021

### Week 4-6 (RQ background)

- [ ] [R][N] Reward Shaping — Ng et al., 1999 _(RQ1)_
- [ ] [R][N] SayCan — Ahn et al., 2022 _(RQ1)_
- [ ] [R][N][W] AI2-THOR — Kolve et al., 2017 _(RQ2 environment paper)_
- [ ] [R][N][W] ALFRED — Shridhar et al., 2020 _(RQ2 task benchmark on AI2-THOR)_
- [ ] [R][N][W] CLIN — Majumder et al., 2023 _(RQ3)_
- [ ] [R][N][W] ScienceWorld — Wang et al., 2022 _(RQ3 environment)_
- [ ] [R][N] Generative Agents — Park et al., 2023 _(RQ3 background)_

### Bonus (read if time allows in Term 2)

- [ ] [R][N] MemoryBank — Zhong et al., 2023
- [ ] [R][N] VOYAGER — Wang et al., 2023 _(agent with skill library)_
- [ ] [R][N] AgentBench — Liu et al., 2023 _(benchmarking agents)_

---

## Experiment Log Template

Copy this block each time you run an experiment. Keep in `experiment_log.md`.

```
## Experiment: [RQ1/RQ2/RQ3] Condition [X]
Date:
Config: num_envs=, num_rounds=, seed=, feedback=, stuck=, memory=
Command:
GPU:
Runtime:

| Round | Success | Accuracy | Avg Steps | Notes |
|-------|---------|----------|-----------|-------|
| 0     |         |          |           |       |
| 1     |         |          |           |       |
| 2     |         |          |           |       |

Observations:
Issues:
Next step:
```
