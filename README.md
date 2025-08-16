# Multi‑Task nnUNetv2 Quiz

This repository contains a multi‑task extension of **nnUNetv2** that performs **3D pancreas/lesion segmentation** and **3‑class subtype classification** on abdominal CT scans. The code follows the MICCAI reproducibility checklist and includes training, inference, and evaluation pipelines. Experiments were run primarily in **Google Colab**.

---

## Environments and Requirements

- **OS**: Linux (tested in Google Colab)
- **CPU / RAM / GPU**: Colab GPU runtime
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

## Repository Structure

```
.
├── data_prep.py
├── subtype_mapping.py
|── multitask_trainer.py          # MultiTaskTrainer (seg + cls)
├── predict_multitask.py          # inference (resample + SW + priors)
├── evaluate_multitask.py         # val metrics + CSV export
├── nnUNetv2_multitask_colab.ipynb
|── brianna_zhao_results.pdf
|── brianna_zhao_results.zip
├── requirements.txt
└── README.md
```

---

## Training

Train the multi-task model (segmentation + classification) using:

```bash
python -m nnunetv2_train 500 3d_fullres 0 \
  -tr MultiTaskTrainer \
  -p nnUNetResEncUNetMPlans
```

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

Run inference using the multi-task model to get both segmentations and subtype predictions:

```bash
python predict_multitask.py \
  --input /path/to/test/images \     # folder with *_0000.nii.gz images
  --model /path/to/trained/model \   # path to trained model folder
  --output /path/to/results \        # where to save predictions
  --device cuda \                    # cuda (default) or cpu
  --fast                            # optional: faster inference
```

Notes:

- Uses target‑spacing resampling and Gaussian blending.
- Classification uses logit adjustment (τ=1.0) and loads the saved best cls head if present.
- A `--fast` flag (optional) can increase stride for ~10–20% runtime reduction at small potential cost in boundary quality.

---

## Evaluation

Evaluate multi-task model on validation data:

```python
from evaluate_multitask import evaluate_multitask

results = evaluate_multitask(
    predictor=predictor,  # initialized nnUNetPredictor
    validation_root="/path/to/validation",  # contains subtype{0,1,2}/ folders
    output_dir="/path/to/outputs"  # where to save results
)
```

**Outputs** (saved to `{output_dir}/eval_val/`):

- `validation_per_case.csv`: Per-case Dice scores and subtypes
- `validation_summary.csv`: Mean Dice, accuracy, macro-F1
- `cls_confusion_matrix.csv`: 3x3 subtype confusion matrix

The validation folder should follow the structure:

```
validation/
├── subtype0/
│   ├── case1.nii.gz        # ground truth
│   └── case1_0000.nii.gz   # image
├── subtype1/
└── subtype2/
```

---

## Results (Validation)

| Task           | Metric          | Value  |
| -------------- | --------------- | ------ |
| Segmentation   | Dice (pancreas) | 0.4564 |
| Segmentation   | Dice (lesion)   | 0.1890 |
| Classification | Accuracy        | 0.3333 |
| Classification | Macro‑F1        | 0.1667 |

_Observed limitations include lesion under‑segmentation and subtype bias; see Discussion/Future Work._

---

## Reproducibility Notes

- Random seeds fixed for PyTorch/Numpy where practical.
- All paths are parameterized via CLI; environment variables for nnUNetv2 are supported:
  - `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`.
- Colab‑ready notebook: `nnUNetv2_multitask_colab.ipynb` (training, inference, evaluation).

---

## Contact

Questions / issues: **Brianna Zhao** · brianna.zhao.m@gmail.com
