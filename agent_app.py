"""
AGENT STEP 3 - Web UI for the agent.   python agent_app.py   ->  open http://127.0.0.1:7860
Type a question about wafers / lots. You get the answer, the Grad-CAM pictures,
and the list of tools the agent called (so you can see HOW it got the answer).

On an HPC node, start it with:   GRADIO_SERVER_NAME=0.0.0.0 python agent_app.py
"""
import gradio as gr
from agent import run_agent, format_trace, MODEL
from wafer_tools import list_lots


def ask(question):
    if not question or not question.strip():
        return "Please type a question.", [], ""
    try:
        answer, trace, images, _ = run_agent(question)
    except Exception as e:      # most common reason: the LLM server is not running
        return f"Could not reach the LLM ({e}). Is the LLM server running? See README_AGENT.md.", [], ""
    return answer, images, format_trace(trace)


# example questions use real lot names from YOUR data (the biggest lots)
big = [x["lot"] for x in list_lots(min_wafers=5, limit=2)["lots"]] or ["lot1", "lot2"]
lot_a, lot_b = big[0], big[-1]
examples = [
    ["Which lots have the most wafers?"],
    [f"Analyze {lot_a} and tell me which wafers need review."],
    ["What does the CNN predict for wafer 10, and how confident is it?"],
    ["Show me where the model looked on wafer 25."],
    [f"Compare {lot_a} and {lot_b}. Which one has more FAIL wafers?"],
]

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, label="Ask about wafers or lots"),
    outputs=[gr.Markdown(label="Answer"),
             gr.Gallery(label="Grad-CAM (left: wafer, right: heatmap)"),
             gr.Textbox(label="Tools the agent called", lines=10)],
    examples=examples,
    cache_examples=False,
    title="Wafer Lot Analysis Agent (LLM + CNN + Grad-CAM)",
    description=f"The LLM ({MODEL}) decides which tools to call. The CNN does the actual wafer classification.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
