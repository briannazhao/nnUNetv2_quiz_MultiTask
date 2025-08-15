# Multi-task nnUNetv2 for FLARE23 Quiz (Pancreas + Subtype)

This repository contains the code and documentation for our multi-task nnUNetv2 model trained on the FLARE23 Pancreas Cancer quiz dataset, performing both 3D segmentation (pancreas and lesion) and 3-class classification (subtype). The implementation follows the MICCAI reproducibility checklist.

## Dataset 

- [x] **Public dataset used:** de-identified pancreas CT scans provided
- [x] **Preprocessing details provided:** normalization, resampling, cropping handled via nnUNetv2 API
- [x] **Data splits:** 252 training, 72 test cases (`imagesTs`), 36 external validation cases

## Code

- [x] **All code required to reproduce key results is available**
- [x] **Training code:** Provided via `MultiTaskTrainer` in `nnunetv2.training.nnUNetTrainer`
- [x] **Inference code:** `predict_multitask.py` with accelerated inference
- [x] **Preprocessing code:** Uses nnUNetv2 built-in commands with slight adjustments for FLARE directory
- [x] **Postprocessing code:** Outputs `.nii.gz` segmentations and `subtype_results.csv`
- [x] **Evaluation code:** `evaluate_multitask.py` computes mean Dice, classification Accuracy, F1-score
- [x] **Dependencies and version info:** `environment.yml` or pip-based setup in progress
- [x] **Trained models:** Expected at `nnUNet_results/Dataset500_PancreasCancer/.../fold_0/`

## Training Details

- Trainer: `MultiTaskTrainer`
- Base architecture: `nnUNetResEncUNetMPlans`
- Configuration: `3d_fullres`
- Loss: Combined Dice (seg) + CE (cls), simple additive
- Optimizer: SGD (default nnUNetv2)
- Batch size: 2 (GPU-limited)
- Epochs: 140 (stopped when loss converges)
- Hardware: Google Colab Pro (T4 or L4)

## Evaluation Metrics

| Task             | Metric        | Value     |
|------------------|---------------|-----------|
| Segmentation     | Dice (pancreas) | 0.4564    |
| Segmentation     | Dice (lesion)   | 0.1890    |
| Classification   | Accuracy        | 0.3333    |
| Classification   | Macro-F1        | 0.1667    |

Additional per-case results and confusion matrix available in `/submission_outputs/eval_val`.

## Inference Details

```bash
python predict_multitask.py --input /path/to/imagesTs                             --model /path/to/nnUNet_results/...                             --output /submission_outputs                             --device cuda
```

- Outputs: `submission_outputs/segmentations/*.nii.gz` and `subtype_results.csv`
- Optional: `--faster` uses faster 3D patch size and tile step for ~10-20% speed gain


## Contact

For any questions or issues, please contact: [Brianna Zhao / brianna.zhao.m@gmail.com]
