
import torch
import numpy as np


from segmentation.unet import Unet, segmentation_loss
from segmentation.dataloader import load_brain


unet = Unet(n_outputs=10)

learning_rate = 1e-4
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)


X, y = load_brain(100206)
X, y = np.array([X]), np.array([y])
X = torch.autograd.Variable(torch.from_numpy(X))
y = torch.autograd.Variable(torch.from_numpy(y))

import IPython
IPython.embed()

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = unet(X)

    # Compute and print loss.
    loss = segmentation_loss(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
