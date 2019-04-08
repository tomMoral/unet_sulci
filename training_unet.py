import os
import datetime
import json

import torch
import pathlib
import numpy as np
# import torch.multiprocessing as mp
from matplotlib import pyplot as plt

from segmentation.unet import Unet, segmentation_loss, test_full_img
from segmentation import dataloader
from segmentation.plotting import plot_segmentation, plot_patch_prediction
from segmentation.utils import get_commit_hash
import segmentation.config

RESULTS_DIR = pathlib.Path(getattr(segmentation.config, 'RESULTS_DIR', '.'))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


MANUAL_TEST_SUBJECT = [
    '101309',
    '108121',
    '102008',
    '107321',
    '102311',
    '121315',
    '108525'
]


def time_stamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Program to launch experiment')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Use the GPU for training')
    parser.add_argument('--preprocessors', type=int, default=5,
                        help='# of process to load the patches.')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_iter', type=int, default=5000)
    parser.add_argument('--attention', type=float, default=1.)
    parser.add_argument('--lr_step_size', type=int, default=1000)
    parser.add_argument('--lr_decay', type=float, default=.8)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')

    args = parser.parse_args()

    out_dir = RESULTS_DIR / 'training_unet_{}'.format(
        time_stamp())
    plots_dir = out_dir / 'figures'
    test_pred_dir = out_dir / 'test_predictions'
    plots_dir.mkdir(parents=True)
    test_pred_dir.mkdir(parents=True)

    # queue_feed, stop_event, batch_feeder = get_queue_feeder(
    #     batch_size=1, maxsize_queue=20, n_process=args.preprocessors)

    patch_size = 64

    unet = Unet(n_outputs=4)

    use_gpu = False
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        unet = unet.cuda()
        use_gpu = True

    learning_rate = args.learning_rate
    optimizer = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}[
        args.optimizer](unet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)

    train_subjects = set(dataloader.train_subjects())
    train_subjects.difference_update(MANUAL_TEST_SUBJECT)
    train_subjects = list(train_subjects)
    test_subjects = MANUAL_TEST_SUBJECT
    test_subjects += dataloader.test_subjects()
    test_subjects = test_subjects[:10]

    feeder = dataloader.feeder_sync(
        subjects=train_subjects, seed=0, use_gpu=use_gpu,
        attention_coef=args.attention, patch_size=patch_size)
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
            loss = segmentation_loss(y_pred, y, attention, use_gpu=use_gpu)
            cost.append(loss.item())
            print("[Iteration {}] cost function {:.3e}"
                  .format(t, cost[-1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not t % 10:
                y_pred = y_pred.data.cpu()
                y_pred = np.argmax(y_pred, axis=1)
                fig = plot_patch_prediction(
                    np.array(X.data.cpu())[0, 0], np.array(y.data.cpu())[0],
                    y_pred[0], z=4, patch_info=patch)
                fig.savefig(str(
                    plots_dir / 'prediction_iteration_{}.png'.format(t)))
                plt.close('all')
                print('learning rate: ', optimizer.param_groups[0]['lr'])

                torch.save(unet.state_dict(), str(out_dir / 'model_saved.pth'))

        # test on whole image:
        print('Testing ...')
        all_iou = {}
        for test_subject in test_subjects:
            try:
                print('testing on subject ', test_subject)
                test_res = test_full_img(unet, test_subject,
                                         patch_size=patch_size, use_gpu=use_gpu)
                test_res['pred_img'].to_filename(
                    str(test_pred_dir /
                        'prediction_for_subject_{}_iter_{}.nii.gz'.format(
                            test_subject, t)))
                test_res['true_img'].to_filename(
                    str(test_pred_dir /
                        'true_labels_for_subject_{}.nii.gz'
                        .format(test_subject)))
                print('intersection over union on test img: ', test_res['iou'])
                all_iou[test_subject] = test_res['iou']
                test_res['anat_img'].to_filename(
                    str(test_pred_dir / 'T1_for_subject_{}.nii.gz'
                        .format(test_subject)))
                plot_segmentation(
                    test_res['subject']['T1_file'],
                    test_res['true_img'],
                    test_res['pred_img'],
                    out_file=str(
                        plots_dir /
                        'whole_image_segmentation_subject_{}_iter_{}'
                        .format(test_subject, t)))
                with open(str(out_dir / 'parameters.json'), 'w') as pf:
                    pf.write(
                        json.dumps(
                            dict(args.__dict__,
                                 iou={k: float(v) for k, v in all_iou.items()},
                                 commit=get_commit_hash())))
            except Exception as e:
                print('testing on subject {} failed:\n{}'.format(
                    test_subject, e))
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
