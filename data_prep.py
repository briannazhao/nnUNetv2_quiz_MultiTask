# Convert & rename ML-Quiz-3DMedImg data into nnU-Net format
import nibabel as nib
import numpy as np
import shutil
from glob import glob

source = '/content/drive/MyDrive/ML-Quiz-3DMedImg/data'  # USE YOUR OWN PATH
target = root  # from previous cell
counter = 0

def copy_case(img, lbl):
    global counter
    cid = f"pancreas_{counter:03d}"
    # Copy training image
    shutil.copy(img, os.path.join(target, 'imagesTr', f'{cid}_0000.nii.gz'))
    # Load, fix label, and save (from float to int)
    data = nib.load(lbl)
    arr = np.rint(data.get_fdata()).astype(np.int16)
    nib.save(nib.Nifti1Image(arr, data.affine, data.header),
             os.path.join(target, 'labelsTr', f'{cid}.nii.gz'))
    counter += 1

# Training data only (exclude validation folder)
for subtype in ['subtype0', 'subtype1', 'subtype2']:
    folder = f"{source}/train/{subtype}"
    for img in sorted(glob(f"{folder}/*_0000.nii.gz")):
        lbl = img.replace('_0000.nii.gz', '.nii.gz')
        if os.path.exists(lbl):
            copy_case(img, lbl)

# Test images
for img in sorted(glob(f"{source}/test/*.nii.gz")):
    shutil.copy(img, os.path.join(target, 'imagesTs', os.path.basename(img)))