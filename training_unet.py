
import torch
import numpy as np
import torch.multiprocessing as mp


from segmentation.unet import Unet, segmentation_loss
from segmentation.dataloader import load_brain, get_queue_feeder


if __name__ == "__main__":

    queue_feed, stop_event = get_queue_feeder(batch_size=1, maxsize_queue=10)

    unet = Unet(n_outputs=4)

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    try:

        for t in range(500):

            X, y = queue_feed.get()

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = unet(X)

            # Compute and print loss.
            loss = segmentation_loss(y_pred, y)
            print("[Iteration {}] cost function {:.3e}".format(t, float(loss.data)))

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
    finally:
        stop_event.set()
        for _ in range(10):
            X, y = queue_feed.get()
        batch_loader.join()
