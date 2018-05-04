
import torch
import numpy as np
import torch.multiprocessing as mp


from segmentation.unet import Unet, segmentation_loss
from segmentation.dataloader import load_brain, get_queue_feeder
from segmentation.dataloader import cut_image, stitch_image


def plot_segmentation(X_tst, y_pred, y_, iteration=0):
    from nilearn import plotting
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)

    plotting.plot_roi(
        roi_img=X_tst,
        bg_img=y_pred,
        black_bg=False,
        vmin=0, vmax=5,
        axes=axes[0]
    )
    axes[0].set_title("Prediction")

    plotting.plot_roi(
        roi_img=X_tst,
        bg_img=y_,
        black_bg=False,
        vmin=0, vmax=5,
        axes=axes[1]
    )
    axes[1].set_title("Label")

    plt.savefig("save_fig_{}.png".format(iteration))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiemnt')
    parser.add_argument('--gpu', action="store_true",
                        help='Use the GPU for training')

    args = parser.parse_args()

    from segmentation.config import DATA_DIR
    DATA_DIR_PATH = pathlib.Path(DATA_DIR)
    tst_subject = list(DATA_DIR_PATH.glob('[0-9]*/'))[0]

    queue_feed, stop_event, batch_feeder = get_queue_feeder(batch_size=1,
                                                            maxsize_queue=20,
                                                            n_process=10)

    unet = Unet(n_outputs=4)
    if args.gpu:
        torch.cuda.set_device(0)
        unet = unet.cuda()

    learning_rate = 1e-10
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    X_tst, y_tst = load_brain(tst_subject)
    img_shape = X_tst.shape
    batch_tst, labels_tst = cut_image(X_tst), cut_image(y_tst)
    batch_tst = np.array(list(batch_tst))
    labels_tst = np.array(list(labels_tst))
    batch_tst = torch.autograd.Variable(torch.from_numpy(batch_tst)).cuda()
    labels_tst = torch.autograd.Variable(torch.from_numpy(labels_tst)).cuda()

    try:
        cost = []
        for t in range(500):

            X, y = queue_feed.get()
            if args.gpu:
                X = X.cuda()
                y = y.cuda()

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = unet(X)

            # Compute and print loss.
            loss = segmentation_loss(y_pred, y, args.gpu)
            cost.append(float(loss.data))
            print("[Iteration {}] cost function {:.3e}"
                  .format(t, cost[-1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Iteration {}] Testing.".format(t))
            tst_pred = np.array(unet(X_tst).data)
            tst_pred = stitch_image(tst_pred, img_shape)
            plot_segmentation(X_tst, tst_pred, y_tst)
            print("[Iteration {}] Finished.".format(t))

    finally:
        stop_event.set()
        for p in batch_feeder:
            try:
                # MAke some room in the queue if it is saturated
                queue_feed.get()
            except Exception:
                pass
            p.join()
