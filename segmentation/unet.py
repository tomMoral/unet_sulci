import torch
import numpy as np
import torch.nn.functional as F

from nilearn import image

from . import dataloader, metrics


class Unet(torch.nn.Module):
    """Unet network for the sulcus segmentation
    """
    def __init__(self, n_outputs, name="U-NET"):

        super().__init__()

        self.conv_c1_1 = torch.nn.Conv3d(1, 64, (3, 3, 3), padding=1)
        self.conv_c1_2 = torch.nn.Conv3d(64, 64, (3, 3, 3), padding=1)

        self.conv_c2_1 = torch.nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.conv_c2_2 = torch.nn.Conv3d(128, 128, (3, 3, 3), padding=1)

        self.conv_c3_1 = torch.nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.conv_c3_2 = torch.nn.Conv3d(256, 256, (3, 3, 3), padding=1)

        self.upconv_d2 = torch.nn.ConvTranspose3d(256, 128, (2, 2, 2),
                                                  stride=(2, 2, 2))
        self.conv_d2_1 = torch.nn.Conv3d(256, 128, (3, 3, 3), padding=1)
        self.conv_d2_2 = torch.nn.Conv3d(128, 128, (3, 3, 3), padding=1)

        self.upconv_d1 = torch.nn.ConvTranspose3d(128, 64, (2, 2, 2),
                                                  stride=(2, 2, 2))
        self.conv_d1_1 = torch.nn.Conv3d(128, 64, (3, 3, 3), padding=1)
        self.conv_d1_2 = torch.nn.Conv3d(64, 64, (3, 3, 3), padding=1)
        self.conv_d1_3 = torch.nn.Conv3d(64, n_outputs, (1, 1, 1))

    def forward(self, x):
        """
        x : input with shape (batch_size, 1, 64, 64, 64)
        """
        # Declare max-pool operator
        m = torch.nn.MaxPool3d((2, 2, 2))

        x = F.relu(self.conv_c1_1(x))  # (?, 64, 64, 64, 64)
        x = x_unit1 = F.relu(self.conv_c1_2(x))  # (?, 64, 64, 64, 64)

        x = m(x)  # (?, 64, 32, 32, 32)

        x = F.relu(self.conv_c2_1(x))  # (?, 128, 32, 32, 32)
        x = x_unit2 = F.relu(self.conv_c2_2(x))  # (?, 128, 32, 32, 32)

        x = m(x)  # (?, 128, 16, 16, 16)

        x = F.relu(self.conv_c3_1(x))  # (?, 256, 16, 16, 16)
        # code final
        x = F.relu(self.conv_c3_2(x))  # (?, 256, 16, 16, 16)

        x_up = self.upconv_d2(x)  # (?, 128, 32, 32, 32)
        x = torch.cat([x_unit2, x_up], 1)  # (?, 256, 32, 32, 32)

        x = F.relu(self.conv_d2_1(x))  # (?, 128, 32, 32, 32)
        x = F.relu(self.conv_d2_2(x))  # (?, 128, 32, 32, 32)

        x_up = self.upconv_d1(x)  # (?, 64, 64, 64, 64)
        x = torch.cat([x_unit1, x_up], 1)  # (?, 128, 64, 64, 64)

        x = F.relu(self.conv_d1_1(x))  # (?, 64, 64, 64, 64)
        x = F.relu(self.conv_d1_2(x))  # (?, 64, 64, 64, 64)
        x = self.conv_d1_3(x)  # (?, n_output, 64, 64, 64)

        # In reall

        return x


def segmentation_loss(y_pred, y, attention_weights, gpu=False):
    # Rebalance the weights of the class, hard-coded from subject 414229
    # class_weights = torch.from_numpy(
    #     np.array([.014, .124, 0.282, 0.58], dtype=np.float32))
    # rebalance hard-coded from many sampled patches
    # see dataloader.GROUPED_LABEL_NAMES
    class_weights = torch.from_numpy(
        np.array([0.042, 0.135, 0.283, 0.54], dtype=np.float32))
    # class_weights = torch.from_numpy(
    #     np.array([1., 1., 1., 1.], dtype=np.float32))
    if gpu:
        class_weights = class_weights.cuda()
    losses = F.cross_entropy(y_pred, y, weight=class_weights, reduce=False)
    losses = losses * attention_weights
    return losses.mean()


def test_full_img(model, test_subject, gpu=False):
    test_subject = dataloader.load_brain(test_subject)
    test_img_shape = test_subject['T1'].shape
    test_batches = dataloader.cut_image(test_subject['T1'])
    test_pred = []
    for batch in test_batches:
        batch = torch.from_numpy(np.array([batch], dtype=np.float32))
        if gpu:
            batch = batch.cuda()
        with torch.no_grad():
            test_pred.append(np.argmax(np.array(model(batch).data), axis=1)[0])
    stitched = dataloader.stitch_image(test_pred, test_img_shape)
    pred_img = image.new_img_like(test_subject['labels_file'],
                                  stitched)
    true_img = image.new_img_like(test_subject['labels_file'],
                                  test_subject['labels'])
    anat_img = image.load_img(test_subject['T1_file'])
    iou = metrics.intersection_over_union(true_img.get_data() == 3,
                                          pred_img.get_data() == 3)

    return {'pred_img': pred_img, 'true_img': true_img, 'anat_img': anat_img,
            'iou': iou, 'subject': test_subject}
