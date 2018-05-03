import torch
import numpy as np
import torch.nn.functional as F


class Unet(torch.nn.Module):
    """Unet network for the sulcus segmentation
    """
    def __init__(self, n_outputs, name="U-NET"):

        super().__init__()

        self.conv_c1_1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_c1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_c2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_c2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv_c3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_c3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv_c4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_c4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv_c5_1 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv_c5_2 = torch.nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.upconv_d4 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_d4_1 = torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_d4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv_d3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_d3_1 = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_d3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv_d2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_d2_1 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_d2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv_d1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_d1_1 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_d1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_d1_3 = torch.nn.Conv2d(64, n_outputs, (1, 1, 1), padding=1)

    def forward(self, x):
        """
        x : input with shape (batch_size, 1, 260, 320, 260)
        """
        # Declare max-pool operator
        m = torch.nn.MaxPool2d(2)

        x = F.relu(self.conv_c1_1(x))  # (?, 64, 272, 320, 272)
        x = x_unit1 = F.relu(self.conv_c1_2(x))  # (?, 64, 272, 320, 272)

        x = m(x)  # (?, 64, 136, 160, 136)
        print(x.size())

        x = F.relu(self.conv_c2_1(x))  # (?, 128, 136, 160, 136)
        x = x_unit2 = F.relu(self.conv_c2_2(x))  # (?, 128, 136, 160, 136)

        x = m(x)  # (?, 128, 68, 80, 68)
        print(x.size())

        x = F.relu(self.conv_c3_1(x))  # (?, 256, 68, 80, 68)
        x = x_unit3 = F.relu(self.conv_c3_2(x))  # (?, 256, 68, 80, 68)

        x = m(x)  # (?, 256, 34, 40, 34)
        print(x.size())

        x = F.relu(self.conv_c4_1(x))  # (?, 512, 34, 40, 34)
        x = x_unit4 = F.relu(self.conv_c4_2(x))  # (?, 512, 34, 40, 34)

        x = m(x)  # (?, 512, 17, 20, 17)
        print(x.size())

        x = F.relu(self.conv_c5_1(x))  # (?, 1024, 17, 20, 17)
        x = F.relu(self.conv_c5_2(x))  # (?, 1024, 17, 20, 17)

        x_up = self.upconv_d4(x)  # (?, 512, 34, 40, 34)
        x = torch.cat([x_unit4, x_up], dim=1)  # (?, 1024, 34, 40, 34)

        x = F.relu(self.conv_d4_1(x))  # (?, 512, 34, 20, 34)
        x = F.relu(self.conv_d4_2(x))  # (?, 512, 34, 40, 34)

        x_up = self.upconv_d3(x)  # (?, 256, 68, 80, 68)
        x = torch.cat([x_unit3, x_up], dim=1)  # (?, 512, 68, 80, 68)

        x = F.relu(self.conv_d3_1(x))  # (?, 256, 68, 80, 68)
        x = F.relu(self.conv_d3_2(x))  # (?, 256, 68, 80, 68)

        x_up = self.upconv_d2(x)  # (?, 128, 136, 160, 136)
        x = torch.cat([x_unit2, x_up], dim=1)  # (?, 256, 136, 160, 136)

        x = F.relu(self.conv_d2_1(x))  # (?, 128, 136, 160, 136)
        x = F.relu(self.conv_d2_2(x))  # (?, 128, 136, 160, 136)

        x_up = self.upconv_d1(x)  # (?, 64, 272, 320, 272)
        x = torch.cat([x_unit1, x_up], dim=1)  # (?, 128, 272, 320, 272)

        x = F.relu(self.conv_d1_1(x))  # (?, 64, 272, 320, 272)
        x = F.relu(self.conv_d1_2(x))  # (?, 64, 272, 320, 272)
        x = self.conv_d1_3(x)  # (?, n_outputs, 272, 320, 272)

        # In reall

        return x


class Mnet(torch.nn.Module):
    def __init__(self, n_outputs, name="M-NET"):

        super().__init__()

        self.unet_x = Unet(n_outputs, name="Unet-x")
        self.unet_y = Unet(n_outputs, name="Unet-y")
        self.unet_z = Unet(n_outputs, name="Unet-z")

    def forward(self, X):
        n_batch, n_outputs, w, h, z = X.size()
        X_width = X.split(1, dim=2)
        X_height = X.split(1, dim=3)
        X_depth = X.split(1, dim=4)


def segmentation_loss(X, y):
    n_batch, n_outputs, w, h, z = X.size()
    X = X.transpose(1, 4)
    X = X.resize(n_batch * w * h * z, n_outputs)
    X = F.softmax(X)

    y = y.reshape(n_batch * w * h * z, 1)

    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(X, y)
