import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn import plotting as niplot
from nilearn import image

_DEFAULT_COLORS = ('white', 'gray', 'blue', 'red')

html = """
<!DOCTYPE html>
<html>
<head>
    <title>segmentation</title>
    <meta charset="UTF-8"/>
</head>
<body>
<div id="true-segmentation">
<h3>True</h3>
{true_map}
</div>
<div id="predicted-segmentation">
<h3>predicted</h3>
{predicted_map}
</div>
</body>
</html>

"""


def plot_segmentation(anat, y_true, y_pred, out_file='segmentation', **kwargs):

    small_anat = image.resample_img(anat, target_affine=np.eye(3) * 2)
    small_y_true = image.resample_to_img(
        y_true, small_anat, interpolation='nearest')
    small_y_pred = image.resample_to_img(
        y_pred, small_anat, interpolation='nearest')

    true_view = niplot.view_stat_map(
        small_y_true, bg_img=small_anat, cmap='tab20c_r')
    pred_view = niplot.view_stat_map(
        small_y_pred, bg_img=small_anat, cmap='tab20c_r')
    filled_html = html.format(true_map=true_view.get_iframe(),
                              predicted_map=pred_view.get_iframe())
    with open('{}.html'.format(out_file), 'wb') as f:
        f.write(filled_html.encode('utf-8'))

    fig, axes = plt.subplots(2, 1)

    niplot.plot_roi(
        roi_img=y_true,
        bg_img=anat,
        black_bg=False,
        vmin=0,
        vmax=5,
        axes=axes[0],
        cut_coords=[0, 0, 0],
        **kwargs)
    axes[0].set_title("Label")

    niplot.plot_roi(
        roi_img=y_pred,
        bg_img=anat,
        black_bg=False,
        vmin=0,
        vmax=5,
        axes=axes[1],
        cut_coords=[0, 0, 0],
        **kwargs)
    axes[1].set_title("Prediction")

    plt.savefig('{}.png'.format(out_file))


def check_cmap(cmap, dtype):
    if isinstance(cmap, str):
        return getattr(mpl.cm, cmap)
    if cmap is None:
        if issubclass(dtype.type, np.integer):
            return mpl.cm.Set1
        return mpl.cm.gray
    return cmap


def plot_patch(patch,
               z=0,
               ax=None,
               cmap=None,
               norm=None,
               cmap_bounds=None,
               cbar_ticks=None,
               vmin=None,
               vmax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if vmin is None:
        vmin = min(0, patch.min())
    if vmax is None:
        vmax = patch.max()
    ax.set_aspect(1)
    cmap = check_cmap(cmap, patch.dtype)
    cax = ax.imshow(
        patch[:, :, z],
        interpolation='nearest',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm)
    ax.figure.colorbar(
        cax, norm=norm, boundaries=cmap_bounds, ticks=cbar_ticks, ax=ax)


def plot_anat_patch(patch, z=0, ax=None, cmap=None):
    return plot_patch(patch, z, ax=ax, cmap=cmap, vmin=0, vmax=1)


def plot_segmentation_patch(patch, z=0, ax=None, colors=_DEFAULT_COLORS):
    cmap = mpl.colors.ListedColormap(colors)
    bounds = np.arange(len(colors) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return plot_patch(
        patch,
        z=z,
        cmap=cmap,
        norm=norm,
        cmap_bounds=bounds - .5,
        cbar_ticks=bounds,
        ax=ax,
        vmin=0,
        vmax=4)


def plot_anat_and_segmentation_patch(anat, segmentation, z=0):
    fig, axes = plt.subplots(2, 1)
    plot_anat_patch(anat, z=z, ax=axes[0])
    plot_segmentation_patch(segmentation, z=z, ax=axes[1])
    return fig


def plot_patch_prediction(anat, y_true, y_pred, z=0, patch_info={}):
    # TODO: use 'T1_file', 'patch_x0', 'patch_y0', 'patch_z0' in patch_info
    # to show patch position in slice. z gives the slice within the patch
    # patch shape is 64, 64, 64
    fig, axes = plt.subplots(1, 3, figsize=(8, 2))
    plot_anat_patch(anat, z=z, ax=axes[0])
    axes[0].set_title('Anat')
    plot_segmentation_patch(y_true, z=z, ax=axes[1])
    axes[1].set_title('True y')
    plot_segmentation_patch(y_pred, z=z, ax=axes[2])
    axes[2].set_title('Prediction')
    fig.tight_layout()
    return fig
