"""
PHASE 5 - Web app.   python app.py   ->  open http://127.0.0.1:7860
Upload a wafer map image -> PASS/FAIL, defect class, confidence, Grad-CAM heatmap.
"""
import gradio as gr
from PIL import Image
from predict import load_model, predict_image

model, device = load_model()


def run(img):
    if img is None:
        return "Please upload a wafer map image.", None, None, None
    r = predict_image(model, device, img)
    icon = "PASS" if r["status"] == "PASS" else "FAIL"
    text = (f"## Result: {icon}\n"
            f"**Predicted pattern:** {r['pred_class']}  \n"
            f"**Confidence:** {r['confidence']:.1%}  \n"
            f"**Probability of any defect:** {r['p_fail']:.1%}")
    if r["low_confidence"]:
        text += "\n\n⚠️ Low confidence - please check this wafer manually."
    big = lambda a, m: Image.fromarray(a).resize((384, 384), m)   # 64x64 is tiny on screen
    return text, r["probs"], big(r["wafer_rgb"], Image.NEAREST), big(r["overlay"], Image.BILINEAR)


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Wafer map image"),
    outputs=[gr.Markdown(label="Result"),
             gr.Label(num_top_classes=9, label="Class probabilities"),
             gr.Image(label="Wafer map (as the model sees it)"),
             gr.Image(label="Grad-CAM heatmap (red = most important)")],
    title="Wafer Defect Detection (CNN from scratch + Grad-CAM)",
    description="Upload a wafer map image. PASS = no defect pattern, FAIL = a defect pattern was found.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
