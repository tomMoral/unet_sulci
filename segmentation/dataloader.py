import os
import numpy as np
import nibabel as nib


from .config import DATA_DIR

# Constants for data loading
T1W_FORMAT = os.path.join(DATA_DIR, "{}",
                          "T1w/T1w_acpc_dc_restore_brain.nii.gz")
LABELS_FORMAT = os.path.join(DATA_DIR, "{}", "T1w/aparc.a2009s+aseg.nii.gz")


def unet_pad(a):
    output = np.zeros((1, 272, 320, 272), dtype=np.float32)
    output[0, :260, :311, :260] = a
    return output

def load_brain(subject):
    """Load and preprocess the data from 1 subject
    
    PReprocess:
    Scale to (272, 320, 272)
    """
    # Load one subject
    t1w_im = unet_pad(nib.load(T1W_FORMAT.format(subject)).get_data())
    labels_im = unet_pad(nib.load(LABELS_FORMAT.format(subject)).get_data())

    return (t1w_im, labels_im)
