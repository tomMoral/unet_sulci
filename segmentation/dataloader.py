import re
import pathlib
from xml.etree import ElementTree
import numpy as np
import matplotlib as mpl
import nibabel as nib


from .config import DATA_DIR

# Constants for data loading
DATA_DIR_PATH = pathlib.Path(DATA_DIR)
T1W_PATH = pathlib.Path('T1w') / 'T1w_acpc_dc_restore_brain.nii.gz'
LABELS_PATH = pathlib.Path('T1w') / 'aparc.a2009s+aseg.nii.gz'
# T1W_FORMAT = os.path.join(DATA_DIR, "{}",
#                           "T1w/T1w_acpc_dc_restore_brain.nii.gz")
# LABELS_FORMAT = os.path.join(DATA_DIR, "{}", "T1w/aparc.a2009s+aseg.nii.gz")

_EXAMPLE_SUBJECT = 163432
GROUPED_LABEL_NAMES = {
    'background': 0, 'subcortical': 1, 'gyrus': 2, 'sulcus': 3}


def unet_pad(a):
    output = np.zeros((1, 272, 320, 272), dtype=np.float32)
    output[0, :260, :311, :260] = a
    return output


def load_brain(subject):
    """Load and preprocess the data from 1 subject

    """
    t1w_img = nib.load(str(DATA_DIR_PATH / str(subject) / T1W_PATH))
    labels_img = nib.load(str(DATA_DIR_PATH / str(subject) / LABELS_PATH))
    labels_info = _read_label_img_extension(labels_img)
    label_grouping = _group_destrieux_labels(labels_info['label_names'])
    labels_data = labels_img.get_data()
    new_labels = np.empty(labels_data.shape, dtype=int)
    for label, group in label_grouping.items():
        new_labels[labels_data == label] = group
    return t1w_img.get_data(), new_labels


def _read_label_img_extension(image):
    extension_header = ElementTree.fromstring(
        image.header.extensions[0].get_content())
    label_names = {
        int(l.get('Key')): l.text
        for l in extension_header.findall(".//Label")
    }
    label_colors = {
        int(l.get('Key')): tuple(
            float(l.get(c)) for c in ('Red', 'Green', 'Blue'))
        for l in extension_header.findall(".//Label")
    }
    destrieux_cm = mpl.colors.ListedColormap(
        np.array([
            label_colors.get(i, (0., 0., 0.))
            for i in range(max(label_colors) + 1)
        ]),
        name='Destrieux')
    return {
        'label_names': label_names,
        'label_colors': label_colors,
        'colormap': destrieux_cm
    }


def _group_destrieux_labels(label_names):
    grouping = {}
    for label, name in label_names.items():
        match = re.match(r'CTX_(R|L)H_(S|LAT_FIS|G|G_AND_S|POLE).*', name)
        if match is None:
            grouping[label] = GROUPED_LABEL_NAMES['subcortical']
        else:
            region_kind = match.group(2)
            if region_kind in ('S', 'LAT_FIS'):
                grouping[label] = GROUPED_LABEL_NAMES['sulcus']
            else:
                grouping[label] = GROUPED_LABEL_NAMES['gyrus']
    grouping[0] = 0
    assert len(grouping) == len(label_names)
    return grouping
