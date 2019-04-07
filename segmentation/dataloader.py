import re
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


mem = Memory(location=CACHE_DIR, verbose=0)


def unet_pad(a, shape=(272, 320, 272)):
    """Pad an array to a given shape"""

    if len(shape) == 4:
        w, h, z = a.shape
        output = np.zeros(shape, dtype=np.float32)
        output[0, :w, :h, :z] = a
    else:
        raise NotImplementedError("Should pad an array with dim=4")
    return output


@mem.cache
def load_brain(subject):
    """Load and preprocess the data from 1 subject

    """
    t1_path = str(DATA_DIR_PATH / str(subject) / T1W_PATH)
    t1w_img = nib.load(t1_path)
    labels_path = str(DATA_DIR_PATH / str(subject) / LABELS_PATH)
    labels_img = nib.load(labels_path)
    labels_info = _read_label_img_extension(labels_img)
    labels_grouping = _group_destrieux_labels(labels_info['label_names'])
    labels_data = labels_img.get_data()
    return {
        'T1': t1w_img.get_data(),
        'labels': _replace_label(labels_data, labels_grouping),
        'T1_file': t1_path,
        'labels_file': labels_path,
        'subject_id': subject
    }


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


# def attention_weights(y_true, window_size=5, weight=10.):
def attention_weights(y_true, window_size=5, weight=7., gpu=False):
    padding = int(window_size / 2)
    mp = torch.nn.MaxPool3d((window_size, window_size, window_size),
                            stride=1, padding=padding)
    # background is 0 and subcortical is 1, see dataloader.GROUPED_LABEL_NAMES
    gray_matter = (y_true > 1).clone().to(dtype=torch.float32)
    if gpu:
        mp, gray_matter = mp.cuda(), gray_matter.cuda()
    opening = mp(gray_matter)
    return (opening - gray_matter) * float(weight) + 1.


def find_patch(img, min_nonzero, patch_size=32, random_state=None):
    rng = check_random_state(random_state)
    shape = np.asarray(img.shape)
    shape -= patch_size + 1
    for i in range(1000):
        i0 = [rng.randint(m) for m in shape]
        w0, h0, z0 = i0

        y = img[w0:w0 + patch_size,
                h0:h0 + patch_size,
                z0:z0 + patch_size].reshape(
                    (1, patch_size, patch_size, patch_size))
        if (y != 0).mean() > min_nonzero:
            return y, i0

    raise RuntimeError("couldn't find a patch without many zeros")


def load_patches(subject, min_nonzero=.2, patch_size=32,
                 random_state=None, gpu=False, attention_coef=7.):
    """Load and preprocess the data from 1 subject

    PReprocess:
    Extract patches with size (patch_size, patch_size, patch_size)
    """
    # Load one subject
    subject_info = load_brain(subject)
    t1w_im, labels_im = subject_info['T1'], subject_info['labels']
    t1w_im = np.asarray(t1w_im, dtype=np.float32)

    y, (w0, h0, z0) = find_patch(labels_im, min_nonzero, patch_size=patch_size,
                                 random_state=random_state)
    X = t1w_im[w0:w0 + patch_size,
               h0:h0 + patch_size,
               z0:z0 + patch_size].reshape(
                   (1, 1, patch_size, patch_size, patch_size))
    X /= np.max(t1w_im)

    y = torch.from_numpy(y)
    X = torch.from_numpy(X)
    if gpu:
        X, y = X.cuda(), y.cuda()

    subject_info['T1_patch'] = X
    subject_info['labels_patch'] = y
    subject_info['attention_weights'] = attention_weights(
        y, weight=attention_coef, gpu=gpu)
    subject_info.update({'patch_x0': w0, 'patch_y0': h0, 'patch_z0': z0})
    return subject_info


def cut_image(img, normalize=True, patch_size=32):
    img = np.array(img, dtype=np.float32)
    w, h, z = img.shape
    w_pad, h_pad, z_pad = map(int, (patch_size * np.ceil(d / patch_size)
                                    for d in img.shape))
    padded = np.zeros((w_pad, h_pad, z_pad))
    padded[:w, :h, :z] = img
    for i in range(0, w_pad, patch_size):
        for j in range(0, h_pad, patch_size):
            for k in range(0, z_pad, patch_size):
                patch = padded[i:i + patch_size,
                               j:j + patch_size,
                               k:k + patch_size][None]
                if normalize:
                    patch /= np.max(img)
                yield patch


def stitch_image(patches, img_shape, patch_size=32):
    if not hasattr(patches, '__next__'):
        patches = patches.__iter__()
    w_pad, h_pad, z_pad = map(int, (patch_size * np.ceil(d / patch_size)
                                    for d in img_shape))
    stitched = np.empty((w_pad, h_pad, z_pad))
    for i in range(0, w_pad, patch_size):
        for j in range(0, h_pad, patch_size):
            for k in range(0, z_pad, patch_size):
                try:
                    stitched[i:i + patch_size,
                             j:j + patch_size,
                             k:k + patch_size] = next(patches)
                except StopIteration:
                    raise ValueError('too few patches to complete image shape')
    w, h, z = img_shape
    return stitched[:w, :h, :z]


def list_subjects():
    return sorted([s.name for s in DATA_DIR_PATH.glob('[0-9]*/')])


def train_subjects(n_train=700):
    return list_subjects()[:n_train]


def test_subjects(n_train=700):
    return list_subjects()[n_train:]


def feeder_sync(subjects=None, seed=None, max_patches=None, patch_size=32,
                attention_coef=7., gpu=False, verbose=True):
    if subjects is None:
        subjects = list_subjects()
    rng = np.random.RandomState(seed)
    n_patches = 0
    while(True):
        subject = rng.choice(subjects)
        if verbose:
            print('subject: {}'.format(subject))
        for i in range(100):
            if n_patches == max_patches:
                return
            try:
                yield load_patches(
                    subject, gpu=gpu, attention_coef=attention_coef,
                    patch_size=patch_size, random_state=rng)
                n_patches += 1
            except Exception as e:
                import sys
                print(sys.exc_info())
                if verbose:
                    print('bad subject: {}\n{}'.format(subject, e))


def feeder(queue_feed, stop_event, batch_size=1, patch_size=32, seed=None):
    """Batch feeder"""

    rng = np.random.RandomState(seed)
    list_subject = DATA_DIR_PATH.glob('[0-9]*/')
    subject_nb = [subject_dir.name for subject_dir in list_subject]
    while not stop_event.is_set():
        subject = rng.choice(subject_nb)
        try:
            X, y = load_patches(subject, patch_size=patch_size)
            queue_feed.put((X, y))
            print("Size of the queue:", queue_feed.qsize())
        except FileNotFoundError:
            pass


def get_queue_feeder(batch_size=1, maxsize_queue=10, n_process=1):
    if batch_size != 1:
        raise NotImplementedError()

    if n_process < 1:
        raise ValueError("n_process should be positive. Got {}"
                         .format(n_process))

    queue_feed = mp.Queue(maxsize=maxsize_queue)
    stop_event = mp.Event()

    batch_loaders = []
    for _ in range(n_process):
        batch_loader = mp.Process(
            target=feeder,
            args=(queue_feed, stop_event, batch_size))
        batch_loader.start()
        batch_loaders.append(batch_loader)

    return queue_feed, stop_event, batch_loaders
