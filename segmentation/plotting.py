import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn import plotting as niplot

_DEFAULT_COLORS = ('white', 'gray', 'blue', 'red')


def plot_segmentation(anat, y_true, y_pred, out_file=None, **kwargs):

    fig, axes = plt.subplots(2, 1)

    niplot.plot_roi(
        roi_img=y_true,
        bg_img=anat,
        black_bg=False,
        vmin=0,
        vmax=5,
        axes=axes[0],
        **kwargs)
    axes[0].set_title("Label")

    niplot.plot_roi(
        roi_img=y_pred,
        bg_img=anat,
        black_bg=False,
        vmin=0,
        vmax=5,
        axes=axes[1],
        **kwargs)
    axes[1].set_title("Prediction")

    if out_file is not None:
        plt.savefig(out_file)


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
