# Semiconductor Wafer Defect Detection with Explainability (CNN from Scratch, PyTorch)

A small, easy-to-read deep learning project that classifies **wafer map defect patterns**
on the public **WM-811K** dataset, explains each prediction with **Grad-CAM**, and tells you
whether a wafer **PASSES or FAILS**.

* The CNN is **built from scratch** in PyTorch - **no pretrained weights**, no `torchvision.models`.
* Grad-CAM is **implemented by hand** (about 30 lines, see `gradcam.py`).
* A simple web app lets you upload a wafer image and see the result + heatmap.

> **New:** an LLM agent on top of this CNN (ask questions about wafers and lots in plain English).
> See [README_AGENT.md](README_AGENT.md) and, for the IITGN HPC, [README_HPC.md](README_HPC.md).

## How PASS / FAIL is decided

WM-811K labels every wafer as one of 8 defect patterns or `none`.

| Model output | Result |
|---|---|
| `none` | **PASS** (no defect pattern) |
| `Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full` | **FAIL** |

The app also shows `P(defect) = 1 - P(none)` and warns when the confidence is below 60%.

> This is a **pattern-based** pass/fail for learning purposes, not a real fab yield decision.

## Project structure

```
wafer-defect-detection/
├── data/                 # put LSWMD.pkl here (not uploaded to GitHub)
├── outputs/              # trained model, plots, metrics are saved here
├── samples/              # example wafer PNGs for the app
├── utils.py              # paths, class names, loading, image <-> wafer conversion
├── model.py              # the CNN (Phase 2)
├── explore_data.py       # Phase 1 - inspect labels, imbalance, sizes, missing values
├── preprocess.py         # Phase 1 - resize to 64x64, stratified train/val/test split
├── train.py              # Phase 2/3 - train from scratch
├── baseline.py           # Phase 3 - majority-class + logistic regression baselines
├── evaluate.py           # Phase 3 - precision/recall/F1, confusion matrix, PASS/FAIL report
├── gradcam.py            # Phase 4 - Grad-CAM from scratch
├── predict.py            # Phase 5 - single-image inference (CLI)
├── app.py                # Phase 5 - web interface (Gradio)
├── export_samples.py     # save sample PNGs to try the app
├── make_fake_data.py     # optional: fake dataset to test the code without downloading
└── requirements.txt
```

## Step-by-step: how to run

### 1. Install

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
(For GPU, install PyTorch from https://pytorch.org first. CPU also works, just slower.)

### 2. Get the dataset

Download **WM-811K** from Kaggle ("WM-811K wafer map", file `LSWMD.pkl`, about 2 GB)
and put it here:

```
data/LSWMD.pkl
```

> No dataset yet? Run `python make_fake_data.py` to create a **fake** `data/LSWMD.pkl`
> and check that everything runs. Do not report results from fake data.

### 3. Phase 1 - understand the data

```bash
python explore_data.py
```
Prints the columns, labeled vs unlabeled wafers, class distribution, wafer sizes and
pixel values, and saves `outputs/class_distribution.png` and `outputs/sample_wafers.png`.

Key facts about WM-811K (confirm with the script output):
* Only a part of the wafers is labeled - unlabeled ones are dropped.
* Wafer maps have **different sizes**, so they are resized to 64x64.
* Pixel values: `0` = outside wafer, `1` = good die, `2` = defective die.
* **Strong class imbalance** - `none` is the large majority, `Near-full` and `Donut` are tiny.

```bash
python preprocess.py              # keeps max 20,000 "none" wafers -> faster training
# python preprocess.py --max_none 0   # use ALL none wafers (slower)
```
Each wafer becomes a 2-channel 64x64 input: *where dies exist* and *where dies are defective*
(area-averaged resizing, so thin scratches do not disappear).
A stratified 70/15/15 train/val/test split is created and saved in `data/wafer_64.npz`.

### 4. Phase 3 - baselines, then train

```bash
python baseline.py       # majority class + logistic regression
python train.py          # CNN from scratch (default 15 epochs)
```
Training uses Adam, cosine LR schedule, flip/rotation augmentation (these keep the defect type)
and **class-weighted cross-entropy** to handle the imbalance. The best model on **validation
macro-F1** is saved to `outputs/best_model.pt`.

### 5. Evaluate

```bash
python evaluate.py
```
Prints accuracy, macro-F1, per-class precision/recall/F1, a PASS/FAIL report, and saves
`outputs/confusion_matrix.png` and `outputs/test_results.json`.

### 6. Phase 4 - explainability

```bash
python gradcam.py
```
Saves `outputs/gradcam_examples.png`: for each class, the wafer next to its Grad-CAM overlay
(red = region that pushed the model to its prediction).

### 7. Phase 5 - app

```bash
python export_samples.py     # saves a few wafer PNGs into samples/
python app.py                # open http://127.0.0.1:7860
```
Upload an image -> you get **PASS/FAIL**, the defect class, confidence, class probabilities
and the heatmap overlay. Command-line version: `python predict.py samples/Center_0.png`.

**Image requirements:** the uploaded picture should be a clean wafer map without axes,
titles or colour bars. Supported colour styles: matplotlib *viridis* (purple / teal / yellow),
*jet*, or a gray image (black = outside, gray = good die, white = defective die).
Use `export_samples.py` to see the expected format.

## Results

Fill this table with **your** numbers after running the scripts (they depend on the preprocessing
options and your hardware):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | | |
| Logistic regression (16x16 pixels) | | |
| CNN from scratch (this project) | | |

PASS/FAIL (FAIL precision / recall): `__ / __`

Add `outputs/confusion_matrix.png` and `outputs/gradcam_examples.png` here:

```
![Confusion matrix](outputs/confusion_matrix.png)
![Grad-CAM](outputs/gradcam_examples.png)
```

## Design choices (short)

* **Macro-F1** is the main metric because accuracy is misleading when `none` dominates.
* **Class weights + capped `none`** deal with imbalance; the cap changes the test class mix,
  so compare results only between runs that use the same `--max_none`.
* **64x64, 2-channel input** keeps the network small (about 0.4M parameters) and fast.
* **Grad-CAM** uses the last conv layer (8x8 feature map), so heatmaps are coarse by design.

## Limitations

* Pass/fail comes from the predicted pattern, not from real electrical test results.
* Images drawn with other colour schemes, or screenshots with axes/margins, may be read wrongly.
* Split is random per wafer; wafers from the same lot may appear in both train and test.
* Grad-CAM shows where the model looked, which is not proof of the physical root cause.

## Troubleshooting

* `No module named 'pandas.indexes'` or another pickle error: `LSWMD.pkl` was saved with old pandas.
  The code already tries a fix. If it still fails, run it with Python 3.9-3.11 and `pip install "pandas<2"`,
  or on Google Colab / Kaggle.
* Out of memory in `preprocess.py`: the file needs several GB of RAM - use Colab/Kaggle or a bigger machine.
* Training too slow on CPU: lower `--epochs`, or keep the default `--max_none 20000`.

## Dataset credit

WM-811K (MIR Lab, National Taiwan University): Wu, Jang, Chen, "Wafer Map Failure Pattern Recognition
and Similarity Ranking for Large-Scale Data Sets", IEEE TSM, 2015.
