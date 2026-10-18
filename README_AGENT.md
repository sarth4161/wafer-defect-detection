# Wafer Lot Analysis Agent (LLM + CNN + Grad-CAM)

The CNN you already built stays exactly as it is. We add an **LLM agent on top**:
you ask a question in plain English, the LLM decides which **tool** to call, the tool
runs your CNN, and the LLM writes the answer.

```
You: "Analyze lot 6 and tell me which wafers need review."
        |
        v
   LLM (Qwen)  --- "call summarize_lot(lot6)" --->  your code runs the CNN on all wafers of lot6
        ^                                                        |
        +------------- result (counts, low-confidence ids) ------+
        |
   LLM --- "call flag_for_review([116, 120, ...])" ---> your code writes outputs/review_queue.csv
        |
        v
   Final answer: "lot6: 12 FAIL, 13 PASS. 5 wafers have low confidence and were flagged."
```

**The LLM never looks at images.** The CNN does. The LLM only (1) understands the question,
(2) picks tools, (3) writes the answer. Every number comes from a tool result, so the LLM
cannot make up facts easily.

---

## 1. What is new

| File | What it does | Lines to learn |
|---|---|---|
| `wafer_tools.py` | The 6 tools (normal Python functions) + their descriptions for the LLM | ~270 |
| `agent.py` | The agent loop + terminal chat | ~120 |
| `agent_app.py` | Web UI (Gradio) | ~50 |
| `eval_agent.py` | Makes test questions automatically, measures accuracy | ~120 |
| `slurm_setup.sh`, `slurm_train.sh`, `slurm_agent.sh`, `slurm_eval.sh` | Batch jobs for the IITGN HPC (see `README_HPC.md`) | |
| `README_AGENT.md`, `README_HPC.md` | this file, and the step-by-step guide for the IITGN HPC | |

Small changes to old files: `utils.py`, `preprocess.py` and `make_fake_data.py` now also keep the
**lot name** and **wafer number** (the agent needs them). `requirements.txt` got `openai`.
Your CNN code (`model.py`, `train.py`, `gradcam.py`, ...) is **not changed**.

**The 6 tools**

| Tool | Question it answers |
|---|---|
| `list_lots` | "Which lots have the most wafers?" |
| `query_lot_data` | "What wafers are in lot 12?" |
| `classify_wafer` | "Is wafer 10 PASS or FAIL?" |
| `summarize_lot` | "Analyze lot 12" (CNN on all wafers of the lot) |
| `explain_wafer` | "Where did the model look on wafer 10?" (Grad-CAM) |
| `flag_for_review` | puts low-confidence wafers into `outputs/review_queue.csv` |

---

## 2. Where to run what

**On the IITGN HPC: follow `README_HPC.md`.** It uses only batch jobs (`sbatch`), because the HPC
guide does not allow running things interactively or on the login node.

| Job script | Does |
|---|---|
| `slurm_setup.sh` | once: Python libraries, Ollama, downloads the LLM |
| `slurm_train.sh` | preprocess + train the CNN + test the tools |
| `slurm_agent.sh` | LLM + web app (open it on your laptop through an ssh tunnel) |
| `slurm_eval.sh` | 50 test questions -> `outputs/agent_eval.json` |

**On your laptop:** the CNN is tiny and runs on CPU. The LLM is the heavy part. Use the free Groq API
(easiest) or Ollama with `qwen2.5:7b` / `qwen2.5:3b` if you have 16 GB RAM or a GPU.

**Which LLM?**
* **Ollama (local model, nothing leaves your machine).** Used by the HPC scripts. Default in `agent.py`.
* **Groq (free API).** Works from your laptop. Only the question text and tool results (class names,
  counts) are sent, never wafer images. Free tiers have rate limits.

---

## 3. Laptop steps (HPC users: use `README_HPC.md` instead)

1. Get `data/wafer_64.npz` and `outputs/best_model.pt` (train on the HPC with `slurm_train.sh`, then copy
   both files to your laptop with `scp`).
2. `python -m venv venv`, activate it, `pip install -r requirements.txt`
3. `python wafer_tools.py` (tests the CNN tools, no LLM)
4. Choose an LLM:
   * Groq: set `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY=your key`,
     `LLM_MODEL=llama-3.3-70b-versatile` (pick a current model that supports tool use).
   * Ollama: install it, `ollama pull qwen2.5:7b`. No variables needed (the defaults point to it).
5. `python agent.py` (terminal chat) or `python agent_app.py` (web UI at http://127.0.0.1:7860)
6. `python eval_agent.py --n 10` (50 questions; on a CPU laptop start with `--n 3`)

---

## 4. How to learn the code (in this order, about 2-3 hours)

1. **`wafer_tools.py`** - just Python + pandas + your CNN. Read `summarize_lot` and `explain_wafer`.
   Then read `SCHEMAS`: this is the text description of each tool that the LLM reads. The LLM chooses
   tools only from these descriptions, so clear descriptions = better tool choice.
2. **`agent.py`** - the heart. `run_agent()` is the whole "agent": ask the LLM, if it asks for a tool
   then run it and send the result back, repeat. Change `SYSTEM_PROMPT` and watch the behaviour change.
3. **`agent_app.py`** - a normal Gradio app around `run_agent()`.
4. **`eval_agent.py`** - how to test an LLM system with automatically generated questions.

Words to search when you want to go deeper: *LLM function calling / tool calling*, *OpenAI-compatible API*,
*ReAct agent loop*, *system prompt*, *Ollama*, *MCP (Model Context Protocol)*, *Pydantic*, *LangGraph*.
You do **not** need a framework for this project. Learn LangGraph / MCP afterwards if you want more.

---

## 5. Things you should know (be honest in interviews)

* **I could not run a real LLM while writing this.** I tested all code with a small scripted stand-in
  server that speaks the same API. So the loop, tools, UI and evaluation are tested, but how well
  **Qwen / Llama choose tools is not tested**. Your `eval_agent.py` run will tell you. Expect some
  mistakes, especially with the 3B model. Report your real numbers, not hoped-for numbers.
* **Lots are small in WM-811K.** Only some wafers are labeled, so most lots have few wafers
  (`list_lots` shows the biggest ones). Real lots have 25 wafers; here you only see the labeled ones.
* **The agent works on all kept wafers, and the CNN trained on most of them.** So the agent demo does
  **not** measure CNN quality. CNN quality comes from `evaluate.py` (test split). `eval_agent.py` measures
  only the agent: tool choice and faithful reporting.
* **"model_focus" (center / middle / edge)** is a rough text summary of Grad-CAM (average distance of the
  hottest 10% pixels from the wafer center). For local defects (Loc, Edge-Loc) and Donut the word can be
  imprecise. The heatmap picture is the real evidence.
* The review queue only writes a CSV file. There is no real fab system behind it.
* The agent keeps chat memory only in the terminal (`agent.py`). The web app answers one question at a time.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Connection error` / "Could not reach the LLM" | the LLM server is not running, or `LLM_BASE_URL` is wrong. For Ollama check `curl http://127.0.0.1:11434` |
| `model 'qwen2.5:7b' not found` | run `ollama pull qwen2.5:7b` with the same `OLLAMA_MODELS` folder |
| `data/wafer_64.npz has no lot information` | run preprocessing again (on the HPC: `sbatch slurm_train.sh`) |
| Agent never calls tools / writes made-up answers | model too small or does not support tools. Use `qwen2.5:7b` or a bigger one; check the printed tool list |
| Agent loops until "too many steps" | look at the tool calls printed; usually wrong arguments. Improve the tool descriptions in `SCHEMAS` or the rules in `SYSTEM_PROMPT` |
| Out of GPU memory | use `qwen2.5:3b`, or run the CNN on CPU (it is tiny) |
| Heatmap images do not show in the web app | start the app from the project folder (`cd wafer-defect-detection`) |
| `ollama: command not found` in the job | Ollama must be in `~/.local/bin` (done by `slurm_setup.sh`); check `PATH` / `LD_LIBRARY_PATH` lines in the job script |

## 7. Resume bullet (fill in YOUR real numbers only after you run the evaluation)

* *Built an LLM tool-calling agent (Qwen2.5-7B via Ollama) over a PyTorch wafer-defect CNN: classify,
  Grad-CAM explanation, lot-level summaries and a human-review queue; X% tool-selection accuracy and
  Y% answer correctness on N auto-generated evaluation questions.*
