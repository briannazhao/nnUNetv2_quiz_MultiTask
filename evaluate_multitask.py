import os
import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

def _clip_zscore(x, p_low=0.5, p_high=99.5, eps=1e-6):
    """Z-score normalize with outlier clipping"""
    lo, hi = np.percentile(x, [p_low, p_high])
    x = np.clip(x, lo, hi)
    m, s = x.mean(), x.std()
    return ((x - m) / (s + eps)).astype(np.float32)

def _pad_to_min_shape(arr, min_shape):
    """Pad array to minimum shape with zeros"""
    pads = []
    for a, m in zip(arr.shape, min_shape):
        add = max(0, m - a)
        pl = add // 2
        pr = add - pl
        pads.append((pl, pr))
    arr2 = np.pad(arr, pads, mode='constant', constant_values=0)
    slc = tuple(slice(p[0], p[0] + arr.shape[i]) for i, p in enumerate(pads))
    return arr2, slc

def _gen_tiles(shape, patch, stride):
    """Generate tile coordinates for sliding window"""
    D, H, W = shape
    Pd, Ph, Pw = patch
    Sd, Sh, Sw = stride
    z_starts = list(range(0, max(1, D - Pd + 1), Sd))
    y_starts = list(range(0, max(1, H - Ph + 1), Sh))
    x_starts = list(range(0, max(1, W - Pw + 1), Sw))
    if z_starts[-1] != max(0, D - Pd): z_starts.append(max(0, D - Pd))
    if y_starts[-1] != max(0, H - Ph): y_starts.append(max(0, H - Ph))
    if x_starts[-1] != max(0, W - Pw): x_starts.append(max(0, W - Pw))
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                yield z, y, x

def _gaussian_weight(patch):
    """Generate 3D Gaussian weight map"""
    zz = torch.linspace(-1, 1, steps=patch[0], dtype=torch.float32)[:, None, None]
    yy = torch.linspace(-1, 1, steps=patch[1], dtype=torch.float32)[None, :, None]
    xx = torch.linspace(-1, 1, steps=patch[2], dtype=torch.float32)[None, None, :]
    g = torch.exp(-0.5 * (zz**2 + yy**2 + xx**2))
    g /= g.max()
    return g

def _get_divisors_from_plans(model_dir_eff, fallback=(32, 32, 32)):
    """Extract network divisors from plans.json"""
    try:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        pm = PlansManager(os.path.join(model_dir_eff, "plans.json"))
        stg = pm.get_stage_from_scale_factor(1.0)
        props = pm.get_properties_of_stage(stg)
        if "num_pool_per_axis" in props:
            npp = props["num_pool_per_axis"]
            return tuple(int(2**int(v)) for v in npp)
        if "pool_op_kernel_sizes" in props:
            ks = props["pool_op_kernel_sizes"]
            acc = np.array([1, 1, 1], dtype=int)
            for s in ks:
                acc *= np.array(s, dtype=int)
            return tuple(int(v) for v in acc)
    except Exception:
        pass
    return fallback

def _pad_to_multiples_torch(vol_5d, multiples):
    """Pad 5D tensor to be divisible by multiples"""
    _, _, D, H, W = vol_5d.shape
    md, mh, mw = multiples
    pad_d = (md - (D % md)) % md
    pad_h = (mh - (H % mh)) % mh
    pad_w = (mw - (W % mw)) % mw
    pad = (0, pad_w, 0, pad_h, 0, pad_d)
    if any(pad):
        vol_5d = F.pad(vol_5d, pad, mode="constant", value=0.0)
    return vol_5d

def dice_bin(pred, gt):
    """Calculate binary Dice score"""
    i = (pred & gt).sum()
    u = pred.sum() + gt.sum()
    return (2.0 * i) / (u + 1e-8)

def compute_case_metrics(pred_lbl, gt_lbl):
    """Compute Dice scores for pancreas and lesion"""
    pred_pan = (pred_lbl > 0).astype(np.uint8)
    gt_pan = (gt_lbl > 0).astype(np.uint8)
    dice_pan = dice_bin(pred_pan, gt_pan)
    pred_les = (pred_lbl == 2).astype(np.uint8)
    gt_les = (gt_lbl == 2).astype(np.uint8)
    dice_les = dice_bin(pred_les, gt_les)
    return float(dice_pan), float(dice_les)

def macro_f1_from_cm(cm):
    """Calculate macro F1 score from confusion matrix"""
    K = cm.shape[0]
    f1s = []
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))

def evaluate_multitask(predictor, validation_root, output_dir):
    """
    Evaluate multi-task model on validation data
    
    Args:
        predictor: nnUNetPredictor instance
        validation_root: Path to validation data directory
        output_dir: Where to save evaluation results
    """
    device = predictor.device if hasattr(predictor, "device") else torch.device("cuda")
    net = predictor.network.to(device)
    net.eval()

    # Get patch size from plans or use fallback
    try:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        pl = PlansManager(os.path.join(predictor.model_dir, "plans.json"))
        stg = pl.get_stage_from_scale_factor(1.0)
        PATCH = tuple(pl.get_properties_of_stage(stg)['patch_size'])
    except Exception:
        PATCH = (80, 160, 160)
    STRIDE = tuple(max(1, int(p * 0.5)) for p in PATCH)
    gauss_w = _gaussian_weight(PATCH).to(device)

    # Collect validation cases
    pairs = []
    for subname, sublab in (("subtype0", 0), ("subtype1", 1), ("subtype2", 2)):
        subdir = os.path.join(validation_root, subname)
        if not os.path.isdir(subdir):
            continue
        for ip in sorted(glob.glob(os.path.join(subdir, "*_0000.nii.gz"))):
            stem = Path(ip).name.replace("_0000.nii.gz", "")
            gp = os.path.join(subdir, f"{stem}.nii.gz")
            if os.path.isfile(gp):
                pairs.append((stem, ip, gp, sublab))

    if not pairs:
        raise RuntimeError(f"No validation cases found in {validation_root}")

    # Run evaluation
    rows = []
    y_true = []
    y_pred = []

    for cid, img_path, gt_path, subtype_gt in tqdm(pairs, desc="Evaluating"):
        # Load data
        img = nib.load(img_path).get_fdata().astype(np.float32)
        gt = nib.load(gt_path).get_fdata().astype(np.uint8)

        # Predict segmentation
        img_n = _clip_zscore(img)
        img_pad, undo_pad = _pad_to_min_shape(img_n, PATCH)
        D, H, W = img_pad.shape

        # Sliding window inference
        logits_acc = torch.zeros((2, D, H, W), device=device, dtype=torch.float32)
        weight_acc = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)

        for z, y, x in _gen_tiles((D, H, W), PATCH, STRIDE):
            patch = torch.from_numpy(img_pad[z:z+PATCH[0], y:y+PATCH[1], x:x+PATCH[2]][None, None]).to(device)
            with torch.no_grad():
                out = net(patch)
                seg_logits = out[0] if isinstance(out, (list, tuple)) else out
                seg_logits = seg_logits.to(dtype=torch.float32)
            w = gauss_w
            logits_acc[:, z:z+PATCH[0], y:y+PATCH[1], x:x+PATCH[2]] += seg_logits[0] * w
            weight_acc[:, z:z+PATCH[0], y:y+PATCH[1], x:x+PATCH[2]] += w

        logits_acc = logits_acc / weight_acc.clamp_min(1e-6)
        zsl, ysl, xsl = undo_pad
        logits_acc = logits_acc[:, zsl, ysl, xsl]
        pred = torch.argmax(logits_acc, dim=0).cpu().numpy()

        # Predict subtype
        divisors = _get_divisors_from_plans(predictor.model_dir)
        vol = _clip_zscore(img).astype(np.float32)
        vol_t = torch.from_numpy(vol[None, None]).to(device=device, dtype=torch.float32)
        vol_t = _pad_to_multiples_torch(vol_t, divisors)

        with torch.no_grad():
            _, cls_out = net(vol_t, return_both=True)
            subtype_pred = int(torch.argmax(cls_out, dim=1).item())

        # Compute metrics
        dice_pan, dice_les = compute_case_metrics(pred, gt)

        rows.append({
            "case": cid,
            "dice_pancreas": dice_pan,
            "dice_lesion": dice_les,
            "subtype_pred": subtype_pred,
            "subtype_gt": subtype_gt,
        })
        y_true.append(int(subtype_gt))
        y_pred.append(int(subtype_pred))

    # Save results
    eval_dir = os.path.join(output_dir, "eval_val")
    os.makedirs(eval_dir, exist_ok=True)

    per_case_csv = os.path.join(eval_dir, "validation_per_case.csv")
    pd.DataFrame(rows).to_csv(per_case_csv, index=False)

    dice_pan_mean = float(np.mean([r["dice_pancreas"] for r in rows]))
    dice_les_mean = float(np.mean([r["dice_lesion"] for r in rows]))

    summary = {
        "dice_pancreas_mean": dice_pan_mean,
        "dice_lesion_mean": dice_les_mean,
    }

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    f1m = macro_f1_from_cm(cm.astype(np.float64))
    summary.update({
        "cls_accuracy": acc,
        "cls_macro_f1": f1m,
    })

    pd.DataFrame([summary]).to_csv(os.path.join(eval_dir, "validation_summary.csv"), index=False)

    cm_df = pd.DataFrame(cm, 
                        index=[f"gt_{k}" for k in [0, 1, 2]],
                        columns=[f"pred_{k}" for k in [0, 1, 2]])
    cm_df.to_csv(os.path.join(eval_dir, "cls_confusion_matrix.csv"))

    print(f"\n[Seg] Mean Dice — Pancreas: {dice_pan_mean:.4f} | Lesion: {dice_les_mean:.4f}")
    print(f"[Cls] Acc: {acc:.4f} | Macro-F1: {f1m:.4f}")
    print("[Eval] Per-case →", per_case_csv)
    print("[Eval] Summary  →", os.path.join(eval_dir, "validation_summary.csv"))
    print("[Eval] ConfMat  →", os.path.join(eval_dir, "cls_confusion_matrix.csv"))

    return summary