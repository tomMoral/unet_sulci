import numpy as np

from segmentation import dataloader

volumes = np.array([0, 0, 0, 0])
for patch_nb, (X, y) in enumerate(
        dataloader.feeder_sync(max_patches=10000, verbose=False)):
    y = np.array(y)
    for i in range(len(volumes)):
        volumes[i] += (y == i).sum()
        print('{}: {}'.format(
            patch_nb, volumes / volumes.sum()), end='\r', flush=True)

print('\n\n')

inv_volumes = 1 / volumes
print('volumes: {}'.format(volumes))
print('normalized volumes: {}'.format(volumes / volumes.sum()))
print('inverse volumes: {}'.format(inv_volumes))
print('normalized inverse volumes: {}'.format(inv_volumes / inv_volumes.sum()))
