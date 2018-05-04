
import torch
import numpy as np
import torch.multiprocessing as mp


from segmentation.unet import Unet, segmentation_loss
from segmentation.dataloader import load_brain, get_queue_feeder


if __name__ == "__main__":

    queue_feed, stop_event, batch_feeder = get_queue_feeder(batch_size=1,
                                                            maxsize_queue=10)

    unet = Unet(n_outputs=4)

    learning_rate = 1e-6
    optimizer = torch.optim.SGD(unet.parameters(), lr=learning_rate,
                                momentum=.8)

    try:
        cost = []
        for t in range(500):

            X, y = queue_feed.get()

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = unet(X)

            # Compute and print loss.
            loss = segmentation_loss(y_pred, y)
            cost.append(float(loss.data))
            print("[Iteration {}] cost function {:.3e}"
                  .format(t, cost[-1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Iteration {}] Finished.".format(t))

    finally:
        stop_event.set()
        try:
            # MAke some room in the queue if it is saturated
            queue_feed.get()
        except Exception:
            pass
        batch_feeder.join()
