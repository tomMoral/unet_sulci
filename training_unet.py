import os
import datetime
import json

import torch
import pathlib
import numpy as np
# import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from nilearn import image

from segmentation.unet import Unet, segmentation_loss
from segmentation import dataloader
from segmentation.plotting import plot_segmentation, plot_patch_prediction
from segmentation.utils import get_commit_hash
import segmentation.config

RESULTS_DIR = pathlib.Path(getattr(segmentation.config, 'RESULTS_DIR', '.'))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def time_stamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Program to launch experiment')
    parser.add_argument('--gpu', action="store_true",
                        help='Use the GPU for training')
    parser.add_argument('--preprocessors', type=int, default=5,
                        help='# of process to load the patches.')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_iter', type=int, default=5000)
    parser.add_argument('--attention', type=float, default=7.)
    parser.add_argument('--lr_step_size', type=int, default=500)
    parser.add_argument('--lr_decay', type=float, default=.8)

    args = parser.parse_args()

    out_dir = RESULTS_DIR / 'training_unet_{}'.format(
        time_stamp())
    plots_dir = out_dir / 'figures'
    test_pred_dir = out_dir / 'test_predictions'
    plots_dir.mkdir(parents=True)
    test_pred_dir.mkdir(parents=True)

    # queue_feed, stop_event, batch_feeder = get_queue_feeder(
    #     batch_size=1, maxsize_queue=20, n_process=args.preprocessors)

    unet = Unet(n_outputs=4)
    if args.gpu:
        torch.cuda.set_device(0)
        unet = unet.cuda()

    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)

    train_subjects = dataloader.train_subjects()
    test_subject = dataloader.load_brain(dataloader.test_subjects()[0])
    test_img_shape = test_subject['T1'].shape
    batch_tst = dataloader.cut_image(test_subject['T1'])
    feeder = dataloader.feeder_sync(
        subjects=train_subjects, seed=0, gpu=args.gpu,
        attention_coef=args.attention)
    try:
        cost = []
        for t in range(args.n_iter):
            scheduler.step()

            patch = next(feeder)
            X, y = patch['T1_patch'], patch['labels_patch']
            attention = patch['attention_weights']

            # Forward pass: compute predicted y by passing x to the model.
            y_pred = unet(X)

            # Compute and print loss.
            loss = segmentation_loss(y_pred, y, attention, args.gpu)
            cost.append(loss.item())
            print("[Iteration {}] cost function {:.3e}"
                  .format(t, cost[-1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not t % 10:
                y_pred = y_pred.data
                y_pred = np.argmax(y_pred, axis=1)
                fig = plot_patch_prediction(
                    np.array(X.data)[0, 0], np.array(y.data)[0],
                    y_pred[0], z=30, patch_info=patch)
                fig.savefig(str(
                    plots_dir / 'prediction_iteration_{}.png'.format(t)))
                plt.close('all')
                print('learning rate: ', optimizer.param_groups[0]['lr'])
        # test on whole image:
        print('Testing ...')
        test_pred = []
        for batch in batch_tst:
            batch = torch.from_numpy(np.array([batch], dtype=np.float32))
            if args.gpu:
                batch = batch.cuda()
            test_pred.append(np.argmax(np.array(unet(batch).data), axis=1)[0])
        stitched = dataloader.stitch_image(test_pred, test_img_shape)
        pred_img = image.new_img_like(test_subject['labels_file'],
                                      stitched)
        true_img = image.new_img_like(test_subject['labels_file'],
                                      test_subject['labels'])
        anat_img = image.load_img(test_subject['T1_file'])
        pred_img.to_filename(
            str(test_pred_dir /
                'prediction_for_subject_{}_iter_{}.nii.gz'.format(
                    test_subject['subject_id'], t)))
        true_img.to_filename(
            str(test_pred_dir / 'true_labels_for_subject_{}.nii.gz'.format(
                test_subject['subject_id'])))
        anat_img.to_filename(
            str(test_pred_dir / 'T1_for_subject_{}.nii.gz'.format(
                test_subject['subject_id'])))
        plot_segmentation(
            test_subject['T1_file'],
            true_img,
            pred_img,
            out_file=str(
                plots_dir /
                'whole_image_segmentation_subject_{}_iter_{}.png'.format(
                    test_subject['subject_id'], t)))
        with open(str(out_dir / 'parameters.json'), 'w') as pf:
            pf.write(
                json.dumps(dict(args.__dict__, commit=get_commit_hash())))
    finally:
        print('Results saved in {}'.format(str(out_dir)))
        # print("Stopping the batch_feeders")
        # stop_event.set()
        # for p in batch_feeder:
        #     try:
        #         # MAke some room in the queue if it is saturated
        #         queue_feed.get()
        #     except Exception:
        #         pass
        # for p in batch_feeder:
        #     p.join()
