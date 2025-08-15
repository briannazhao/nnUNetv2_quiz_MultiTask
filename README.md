# Multi‑Task nnUNetv2 for FLARE23 Quiz (Pancreas + Subtype)

This repository contains a multi‑task extension of **nnUNetv2** that performs **3D pancreas/lesion segmentation** and **3‑class subtype classification** on abdominal CT scans. The code follows the MICCAI reproducibility checklist and includes training, inference, and evaluation pipelines. Experiments were run primarily in **Google Colab**.

---

## Environments and Requirements

- **OS**: Linux (tested in Google Colab)
- **CPU / RAM / GPU**: Colab GPU runtime (T4/L4/A100 class; session‑dependent)
- **CUDA**: session‑dependent (see commands below)
- **Python**: 3.10–3.11 recommended

To install requirements:

```bash
pip install -r requirements.txt
```

**Capture your environment:**

```bash
# OS
cat /etc/os-release
# Python
python -V
# Torch/CUDA
python - << 'PY'
import torch, platform
print("torch:", torch.__version__, " cuda:", torch.version.cuda, " is_available:", torch.cuda.is_available())
print("python:", platform.python_version())
PY
# GPU + driver
nvidia-smi
```

---

## Dataset Layout

The repository expects the following structure (train/validation by subtype; test = images only):

```
data
├── train
│   ├── subtype0
│   │   ├── quiz_0_041.nii.gz          # mask (0=background, 1=pancreas, 2=lesion)
│   │   └── quiz_0_041_0000.nii.gz     # image
│   ├── subtype1
│   └── subtype2
└── validation
    ├── subtype0
    │   ├── quiz_0_168.nii.gz
    │   └── quiz_0_168_0000.nii.gz
    ├── subtype1
    └── subtype2
# test/ contains only images: quiz_XXX_0000.nii.gz
```

For nnUNet‑style evaluation or training, you can also prepare `Dataset500_PancreasCancer/imagesTr|labelsTr|imagesVal|labelsVal` following the standard `_0000.nii.gz` image suffix and label name parity.

---

## Methods (brief)

### Inference
Each CT volume is **resampled to the target spacing** defined in `plans.json` (nnUNetv2 planning) and standardized with **per‑case z‑score**. We perform **sliding‑window 3D inference** with **Gaussian‑weighted blending** on the resampled grid and apply **argmax** over logits to obtain labels, then **restore the prediction to the original grid**. Classification uses the shared encoder’s bottleneck features with ROI‑aware pooling and a small MLP head; at test time we apply **logit adjustment** with the empirical class prior (τ = 1.0) to reduce class‑imbalance bias. The best classification head checkpoint (saved during training) is loaded when available.

### Evaluation
For segmentation we compute **Dice** for (i) pancreas (labels 1∪2) and (ii) lesion (label 2). For classification we report **accuracy**, **macro‑F1**, and the **confusion matrix**. Per‑case and summary CSVs are saved under `submission_outputs/eval_val/`.

---

## Training

- **Trainer**: `MultiTaskTrainer` (extends nnUNetTrainer)
- **Backbone**: `nnUNetResEncUNetMPlans` (3d_fullres)
- **Loss**: Dice (seg) + Cross‑Entropy (cls) with class weights; optional label smoothing
- **Multi‑task weighting**: segmentation + time‑ramped classification loss
- **Mixed precision**: AMP with warm‑up; gradient clipping

Typical command:

```bash
nnUNetv2_train 500 3d_fullres 0 -tr MultiTaskTrainer -p nnUNetResEncUNetMPlans
```

---

## Inference

Fast single‑script inference that writes segmentations and subtype CSV:

```bash
python predict_multitask.py \
  --input /path/to/imagesTs \
  --model /path/to/nnUNet_results/Dataset500_PancreasCancer/MultiTaskTrainer__nnUNetResEncUNetMPlans__3d_fullres \
  --output /submission_outputs \
  --device cuda
```

Notes:
- Uses target‑spacing resampling and Gaussian blending.
- Classification uses logit adjustment (τ=1.0) and loads the saved best cls head if present.
- A `--fast` flag (optional) can increase stride for ~10–20% runtime reduction at small potential cost in boundary quality.

---

## Evaluation

Evaluate on `data/validation/subtype{0,1,2}` or prepared `imagesVal/labelsVal`:

```bash
python evaluate_multitask.py \
  --val_root /path/to/data/validation \
  --model /path/to/model_dir \
  --out /submission_outputs/eval_val \
  --device cuda
```

Outputs:
- `validation_per_case.csv` – per‑case Dice + predicted/true subtype
- `validation_summary.csv` – mean Dice + accuracy/F1
- `cls_confusion_matrix.csv` – confusion matrix

---

## Results (Validation)

| Task           | Metric             | Value   |
|----------------|--------------------|---------|
| Segmentation   | Dice (pancreas)    | 0.4564  |
| Segmentation   | Dice (lesion)      | 0.1890  |
| Classification | Accuracy           | 0.3333  |
| Classification | Macro‑F1           | 0.1667  |

*Observed limitations include lesion under‑segmentation and subtype bias; see Discussion/Future Work.*

---

## Reproducibility Notes

- Random seeds fixed for PyTorch/Numpy where practical.
- All paths are parameterized via CLI; environment variables for nnUNetv2 are supported:
  - `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`.
- Colab‑ready notebook: `nnUNetv2_multitask_colab.ipynb` (training, inference, evaluation).

---

## Repository Structure (suggested)

```
.
├── multitask_trainer.py          # MultiTaskTrainer (seg + cls)
├── predict_multitask.py          # inference (resample + SW + priors)
├── evaluate_multitask.py         # val metrics + CSV export
├── utils/                        # io, spacing, metrics helpers
├── nnUNetv2_multitask_colab.ipynb
├── requirements.txt
└── README.md
```

---

## Submission Packaging (for quiz/test)

Predicted segmentations in `segmentations/` (`quiz_XXX.nii.gz`) and a CSV:

```
Names,Subtype
quiz_037.nii.gz,0
quiz_045.nii.gz,2
...
```

Zip as `your_name_results.zip` following the organizer’s template.

---

## Contact

Questions / issues: **Brianna Zhao** · brianna.zhao.m@gmail.com
