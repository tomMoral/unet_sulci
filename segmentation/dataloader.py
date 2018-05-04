import re
import os
import torch
import pathlib
import numpy as np
import nibabel as nib
import matplotlib as mpl
from xml.etree import ElementTree
import torch.multiprocessing as mp

from joblib import Memory


from . import config
from .utils import check_random_state


DATA_DIR = getattr(config, 'DATA_DIR', 'data')
CACHE_DIR = getattr(config, 'CACHE_DIR', '.')

# Constants for data loading
DATA_DIR_PATH = pathlib.Path(DATA_DIR)
T1W_PATH = pathlib.Path('T1w') / 'T1w_acpc_dc_restore_brain.nii.gz'
LABELS_PATH = pathlib.Path('T1w') / 'aparc.a2009s+aseg.nii.gz'

_EXAMPLE_SUBJECT = 163432
GROUPED_LABEL_NAMES = {
    'background': 0, 'subcortical': 1, 'gyrus': 2, 'sulcus': 3}


mem = Memory(cachedir=CACHE_DIR, verbose=0)


def unet_pad(a, shape=(272, 320, 272)):
    """Pad an array to a given shape"""

    if len(shape) == 4:
        w, h, z = a.shape
        output = np.zeros(shape, dtype=np.float32)
        output[0, :w, :h, :z] = a
    else:
        raise NotImplementedError("Should pad an array with dim=4")
    return output


def load_brain(subject):
    """Load and preprocess the data from 1 subject

    """
    t1w_img = nib.load(str(subject / T1W_PATH))
    labels_img = nib.load(str(subject / LABELS_PATH))
    labels_info = _read_label_img_extension(labels_img)
    labels_grouping = _group_destrieux_labels(labels_info['label_names'])
    labels_data = labels_img.get_data()
    return t1w_img.get_data(), _replace_label(labels_data, labels_grouping)


@mem.cache
def _replace_label(labels_data, labels_grouping):
    new_labels = np.empty(labels_data.shape, dtype=int)
    for label, group in labels_grouping.items():
        new_labels[labels_data == label] = group
    return new_labels


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


def load_patches(subject, random_state=None):
    """Load and preprocess the data from 1 subject

    PReprocess:
    Extract patches with size (64, 64, 64)
    """
    rng = check_random_state(random_state)

    # Load one subject
    t1w_im, labels_im = load_brain(subject)

    shape = np.asarray(t1w_im.shape)
    shape -= 64 + 1
    i0 = []
    for m in shape:
        i0 += [rng.randint(m)]
    w0, h0, z0 = i0

    X = torch.from_numpy(t1w_im[w0:w0 + 64,
                                h0:h0 + 64,
                                z0:z0 + 64].reshape((1, 1, 64, 64, 64)))

    y = torch.from_numpy(labels_im[w0:w0 + 64,
                                   h0:h0 + 64,
                                   z0:z0 + 64].reshape((1, 1, 64, 64, 64)))

    return (X, y)


def feeder(queue_feed, stop_event, batch_size=1, seed=None):
    """Batch feeder"""

    rng = np.random.RandomState(seed)
    list_subject = list(DATA_DIR_PATH.glob('[0-9]*/'))
    while not stop_event.is_set():
        subject = rng.choice(list_subject)
        X, y = load_patches(subject)
        X = torch.autograd.Variable(X)
        y = torch.autograd.Variable(y)

        queue_feed.put((X, y))
        print("Size of the queue:", queue_feed.qsize())


def get_queue_feeder(batch_size=1, maxsize_queue=10):
    if batch_size != 1:
        raise NotImplementedError()

    queue_feed = mp.Queue(maxsize=maxsize_queue)
    stop_event = mp.Event()

    batch_loader = mp.Process(
        target=feeder,
        args=(queue_feed, stop_event, batch_size))
    batch_loader.start()

    return queue_feed, stop_event, batch_loader
