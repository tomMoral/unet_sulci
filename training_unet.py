
import torch
import pathlib
import numpy as np
import torch.multiprocessing as mp


from segmentation.unet import Unet, segmentation_loss
from segmentation.dataloader import load_brain, get_queue_feeder, load_patches
from segmentation.dataloader import cut_image, stitch_image, _EXAMPLE_SUBJECT
from segmentation.plotting import plot_segmentation, plot_patch_prediction


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiment')
    parser.add_argument('--gpu', action="store_true",
                        help='Use the GPU for training')
    parser.add_argument('--preprocessors', type=int, default=5,
                        help='# of process to load the patches.')

    args = parser.parse_args()
    print(args.gpu)

    from segmentation.config import DATA_DIR
    # DATA_DIR_PATH = pathlib.Path(DATA_DIR)
    # tst_subject = list(DATA_DIR_PATH.glob('[0-9]*/'))[0]

    tst_subject = _EXAMPLE_SUBJECT
    # queue_feed, stop_event, batch_feeder = get_queue_feeder(
    #     batch_size=1, maxsize_queue=20, n_process=args.preprocessors)

    unet = Unet(n_outputs=4)
    if args.gpu:
        torch.cuda.set_device(0)
        unet = unet.cuda()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    X_tst, y_tst = load_brain(tst_subject)
    img_shape = X_tst.shape
    batch_tst, labels_tst = cut_image(X_tst), cut_image(y_tst)
    batch_tst = np.array(list(batch_tst), dtype=np.float32)
    labels_tst = np.array(list(labels_tst), dtype=np.float32)
    batch_tst = torch.from_numpy(batch_tst)
    labels_tst = torch.from_numpy(labels_tst)
    if args.gpu:
        batch_tst = batch_tst.cuda()
        labels_tst = labels_tst.cuda()

    try:
        cost = []
        for t in range(500):

            X, y = load_patches(_EXAMPLE_SUBJECT)
            # X, y = queue_feed.get()
            if args.gpu:
                X = X.cuda()
                y = y.cuda()

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = unet(X)

            # Compute and print loss.
            loss = segmentation_loss(y_pred, y, args.gpu)
            cost.append(loss.item())
            print("[Iteration {}] cost function {:.3e}"
                  .format(t, cost[-1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Iteration {}] Testing.".format(t))
            # tst_pred = np.array(unet(batch_tst).data)
            tst_pred = np.array(unet(X).data)
            tst_pred = np.argmax(tst_pred, axis=1)
            # tst_pred = stitch_image(tst_pred, img_shape)
            # fig = plot_segmentation(X_tst, tst_pred, y_tst)
            fig = plot_patch_prediction(
                np.array(X.data)[0, 0], np.array(y.data)[0], tst_pred[0], z=30)
            fig.savefig('prediction_iteration_{}.png'.format(t))
            print("[Iteration {}] Finished.".format(t))

    finally:
        print("Stopping the batche_feeders")
        # stop_event.set()
        # for p in batch_feeder:
        #     try:
        #         # MAke some room in the queue if it is saturated
        #         queue_feed.get()
        #     except Exception:
        #         pass
        # for p in batch_feeder:
        #     p.join()
