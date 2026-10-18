"""
AGENT STEP 4 - Measure how good the agent is.

The questions are made automatically from the data, so the correct answers are known
(we get them by calling the tools directly, without the LLM).
For every question we check two things:
    1. tool selection : did the agent call the tool we expected?
    2. answer         : does the final answer contain the correct fact?

    python eval_agent.py              # 5 questions of each of 5 types = 25 questions
    python eval_agent.py --n 10       # 50 questions
Results are saved in outputs/agent_eval.json  (use these numbers in your resume / README).

Note: the answer check is simple text matching, so it can be a bit strict or a bit lenient.
Read the printed failures to see which one it is.
"""
import argparse
import json
import os
import re
import time

import numpy as np

import wafer_tools as wt
from wafer_tools import table, OUT_DIR
from agent import run_agent, MODEL

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=5, help="questions per type (5 types)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

# do not fill your real review queue with test flags
wt.REVIEW_FILE = os.path.join(OUT_DIR, "review_queue_eval.csv")
if os.path.exists(wt.REVIEW_FILE):
    os.remove(wt.REVIEW_FILE)

rng = np.random.default_rng(args.seed)
big_lots = wt.list_lots(min_wafers=5, limit=100000)["lots"]
lots = [x["lot"] for x in big_lots]
if not lots:
    raise SystemExit("No lot with at least 5 wafers found - cannot build questions.")


def pick_lots(k):
    return list(rng.choice(lots, size=min(k, len(lots)), replace=False))


def pick_wafers(k):
    return [int(i) for i in rng.choice(len(table), size=k, replace=False)]


# each item: question, expected tool, expected text (regex, case-insensitive)
items = []
for lot in pick_lots(args.n):                                           # type 1: PASS/FAIL count of a lot
    s = wt.summarize_lot(lot)
    items.append({"type": "lot_fail_count", "tool": "summarize_lot",
                  "question": f"Summarize {lot}. How many of its wafers are predicted FAIL?",
                  "expect": rf"\b{s['n_fail']}\b"})
for lot in pick_lots(args.n):                                           # type 2: low-confidence count
    s = wt.summarize_lot(lot)
    n = s["n_low_confidence"]
    items.append({"type": "low_conf_count", "tool": "summarize_lot",
                  "question": f"How many wafers in {lot} have low confidence predictions?",
                  "expect": rf"\b{n}\b" if n > 0 else r"\b0\b|\bno\b|\bnone\b|\bzero\b"})
for w in pick_wafers(args.n):                                           # type 3: one wafer
    c = wt.classify_wafer(w)
    items.append({"type": "wafer_class", "tool": "classify_wafer",
                  "question": f"What defect class does the CNN predict for wafer {w}?",
                  "expect": re.escape(c["predicted_class"]) if c["predicted_class"] != "none" else r"none|PASS"})
for w in pick_wafers(args.n):                                           # type 4: Grad-CAM
    e = wt.explain_wafer(w)
    items.append({"type": "gradcam_focus", "tool": "explain_wafer",
                  "question": f"Show me where the model looked on wafer {w}. Is it the center, middle or edge?",
                  "expect": e["model_focus"]})
for k in range(args.n):                                                 # type 5: find lots
    m = 5 + k                                                           # a different threshold each time
    first = wt.list_lots(min_wafers=m, limit=1)["lots"]
    if first:
        items.append({"type": "find_lots", "tool": "list_lots",
                      "question": f"Which lots have at least {m} wafers? List the first 3.",
                      "expect": rf"\b{first[0]['lot']}\b"})

print(f"Agent LLM: {MODEL} | {len(items)} questions\n")
rows, t0 = [], time.time()
for i, it in enumerate(items, 1):
    try:
        answer, trace, _, _ = run_agent(it["question"])
    except Exception as e:                       # e.g. LLM server not reachable
        answer, trace = f"ERROR: {e}", []
    tools_used = [t["tool"] for t in trace]
    tool_ok = it["tool"] in tools_used
    answer_ok = bool(re.search(it["expect"], answer or "", re.IGNORECASE))
    rows.append({**it, "answer": answer, "tools_used": tools_used,
                 "tool_ok": tool_ok, "answer_ok": answer_ok})
    print(f"[{i:2d}/{len(items)}] tool {'OK ' if tool_ok else 'BAD'} | answer {'OK ' if answer_ok else 'BAD'} | {it['type']}")

n = len(rows)
summary = {
    "llm": MODEL, "n_questions": n,
    "tool_selection_accuracy": sum(r["tool_ok"] for r in rows) / n,
    "answer_correctness": sum(r["answer_ok"] for r in rows) / n,
    "both_correct": sum(r["tool_ok"] and r["answer_ok"] for r in rows) / n,
    "avg_tool_calls_per_question": float(np.mean([len(r["tools_used"]) for r in rows])),
    "seconds_per_question": (time.time() - t0) / n,
    "per_type": {t: {"tool_ok": float(np.mean([r["tool_ok"] for r in rows if r["type"] == t])),
                     "answer_ok": float(np.mean([r["answer_ok"] for r in rows if r["type"] == t]))}
                 for t in sorted({r["type"] for r in rows})},
}
print("\n==== RESULT ====")
print(json.dumps(summary, indent=2))
print("\n==== FAILURES (read these, they teach you the most) ====")
for r in rows:
    if not (r["tool_ok"] and r["answer_ok"]):
        print(f"- [{r['type']}] {r['question']}\n    expected tool: {r['tool']} | used: {r['tools_used']}"
              f"\n    expected text: {r['expect']}\n    answer: {(r['answer'] or '')[:200]}")
with open(os.path.join(OUT_DIR, "agent_eval.json"), "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=2)
print("\nSaved outputs/agent_eval.json")
