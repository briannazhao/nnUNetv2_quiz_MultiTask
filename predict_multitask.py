import os
import glob
from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

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

def predict_multitask(predictor, input_dir, output_dir, fast=False):
    """Run multi-task inference (segmentation + classification)
    
    Args:
        predictor: Initialized nnUNetPredictor
        input_dir: Path containing test images (*_0000.nii.gz)
        output_dir: Where to save results
        fast: Use larger stride for faster inference
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
    
    # Faster stride for quick inference
    if fast:
        STRIDE = tuple(max(1, int(p * 0.75)) for p in PATCH)
    else:
        STRIDE = tuple(max(1, int(p * 0.5)) for p in PATCH)

    gauss_w = _gaussian_weight(PATCH).to(device)
    
    # Setup output folders
    seg_dir = os.path.join(output_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)

    # Collect test files
    test_files = sorted(glob.glob(os.path.join(input_dir, "*_0000.nii.gz")))
    if not test_files:
        raise RuntimeError(f"No test files (*_0000.nii.gz) found in {input_dir}")

    # Run inference
    cls_rows = []

    for img_path in tqdm(test_files, desc="Predicting"):
        # Load & normalize
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata().astype(np.float32)
        
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
            subtype = int(torch.argmax(cls_out, dim=1).item())

        # Save outputs
        out_name = Path(img_path).name.replace("_0000.nii.gz", ".nii.gz")
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), img_nii.affine),
                 os.path.join(seg_dir, out_name))
        
        cls_rows.append({"Names": out_name, "Subtype": subtype})

    # Save classification results
    pd.DataFrame(cls_rows).to_csv(os.path.join(output_dir, "subtype_results.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input directory with test images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--fast', action='store_true', help='Use larger stride for faster inference')
    args = parser.parse_args()

    # Initialize predictor
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_gpu=True,
        device=torch.device(args.device),
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model,
        use_folds=None,
        checkpoint_name="checkpoint_best.pth",
    )

    # Run prediction
    predict_multitask(predictor, args.input, args.output, args.fast)

if __name__ == '__main__':
    main()