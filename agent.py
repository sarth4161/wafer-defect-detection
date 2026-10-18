"""
AGENT STEP 2 - The agent loop (no framework, about 60 lines of real code).

    question -> LLM -> "call tool X with these arguments" -> we run X -> result goes back to LLM
                 ^                                                          |
                 +----------------------------------------------------------+
    ... repeat until the LLM gives a normal text answer (or MAX_STEPS is reached).

The LLM is reached through the "OpenAI-style" API. Ollama, vLLM and Groq all speak it,
so you switch between them only by changing 3 environment variables (see README_AGENT.md):

    LLM_BASE_URL   default http://localhost:11434/v1   (Ollama)
    LLM_MODEL      default qwen2.5:7b
    LLM_API_KEY    default "ollama"  (Ollama ignores it; Groq needs your real key)

Chat in the terminal:
    python agent.py
"""
import os
import json

from openai import OpenAI
from wafer_tools import SCHEMAS, run_tool

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MAX_STEPS = 8                                    # safety stop: at most 8 LLM calls per question

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a semiconductor wafer inspection assistant.
You can look at wafers only through your tools. A CNN classifies each wafer into one of
9 classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none.
'none' means no defect pattern = PASS. Any other class = FAIL.

Rules:
1. NEVER guess numbers, classes or lot names. Get them from a tool and copy them exactly.
2. If the user does not name a lot, call list_lots first.
3. For questions about a whole lot, call summarize_lot.
4. After summarize_lot, if there are low-confidence wafers, call flag_for_review with their ids.
5. To show where the model looked on a wafer, call explain_wafer and report model_focus.
6. Keep the final answer short: lead with the result, then 1-3 supporting facts.
7. Grad-CAM shows where the model looked. Never claim it proves the physical cause of a defect.
8. If a tool returns an error, read the message and try again with corrected arguments."""


def new_conversation():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_agent(question, messages=None):
    """
    Answer one question.
    Returns (answer_text, trace, image_paths, messages)
      trace  = list of {"tool", "args", "result"} -> what the agent did, step by step
      messages = whole conversation so far (pass it back in to continue the chat)
    """
    if messages is None:
        messages = new_conversation()
    messages.append({"role": "user", "content": question})
    trace, images = [], []

    for step in range(MAX_STEPS):
        reply = client.chat.completions.create(
            model=MODEL, messages=messages, tools=SCHEMAS, temperature=0
        ).choices[0].message

        # keep the LLM's message in the history (including its tool calls)
        entry = {"role": "assistant", "content": reply.content or ""}
        if reply.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in reply.tool_calls]
        messages.append(entry)

        if not reply.tool_calls:                       # no tool requested -> this is the final answer
            return reply.content, trace, images, messages

        for tc in reply.tool_calls:                    # the LLM asked us to run tool(s)
            raw = tc.function.arguments
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                result = run_tool(tc.function.name, args)
            except json.JSONDecodeError:
                args, result = {}, {"error": "arguments were not valid JSON"}
            trace.append({"tool": tc.function.name, "args": args, "result": result})
            if isinstance(result, dict) and result.get("heatmap_path"):
                images.append(result["heatmap_path"])
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)})   # give the result back to the LLM

    return "Stopped: the agent used too many steps without a final answer.", trace, images, messages


def format_trace(trace):
    """Readable text version of what the agent did."""
    lines = []
    for i, t in enumerate(trace, 1):
        short = json.dumps(t["result"])
        lines.append(f"{i}. {t['tool']}({json.dumps(t['args'])})\n   -> {short[:300]}{'...' if len(short) > 300 else ''}")
    return "\n".join(lines) if lines else "(no tools were called)"


if __name__ == "__main__":
    print(f"LLM: {MODEL} at {BASE_URL}")
    print("Ask about wafers and lots. Type 'quit' to stop. Example: Which lots have the most wafers?\n")
    history = None
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        answer, trace, images, history = run_agent(q, history)      # history keeps the chat memory
        print("\n--- what the agent did ---")
        print(format_trace(trace))
        print("\nAgent:", answer)
        if images:
            print("Heatmaps:", ", ".join(images))
        print()
