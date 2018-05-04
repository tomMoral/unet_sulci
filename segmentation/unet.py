import torch
import numpy as np
import torch.nn.functional as F


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


def segmentation_loss(X, y):
    n_batch, n_outputs, w, h, z = X.size()
    X = X.transpose(1, 4)
    X = X.resize(n_batch * w * h * z, n_outputs)
    X = F.softmax(X)

    y = y.resize(n_batch * w * h * z)

    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(X, y)
