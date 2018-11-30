# 2018 Retreat P8: Automatic identification of Brain Sulci based on DNN


## Overall plan:

### Dataset
* HCP 900 for which we have
  * A cleaned AC/PC-aligned T1w image of each subject brain (not in MNI space)
  * The [Destrieux atlas](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2937159/) in voxel space

Data can be downloaded within the Inria network 
```bash
rsync -avzh --prune-empty-dirs --include="*/" --include="*/T1w/T1*brain.nii.gz" --include="*/T1w/*a2009*.nii.gz" --exclude="*" -e ssh dragostore:/data/data/HCP900/* .
```


### Current Status
* Implementation based on PyTorch
* [Overleaf account of where we are](https://v1.overleaf.com/21500988bsgxdrgdjwnc)
* Will upload a [Boto](http://boto.cloudhackers.com/en/latest/)-based script to download the files

## Old plan

### Base ourselves on (but this is open to discussion):
*  Keras (Install with `conda install -c conda-forge keras`)
*  An [implementation of the U-net](https://github.com/jocicmarko/ultrasound-nerve-segmentation.git)

### Then:
1. Generalise it to 3D
2. Develop a framework to test performance
3. Tune the network architecture
4. Check it’s the resulting network’s capability to detect sulci that 3) Check it’s the resulting network’s capability to detect sulci that have never been labeled. For this we might want to exclude some sulci from the training set (and not just some subjects)


## References
1. [Automatic recognition of cortical sulci of the human brain using a congregation of neural networks.](https://www.ncbi.nlm.nih.gov/pubmed/12044997)
  * Old style and might try to name the sulci which we don't want to.
2. [DeepNAT: Deep Convolutional Neural Network for Segmenting Neuroanatomy
](https://arxiv.org/pdf/1702.08192.pdf)
  * Brain segmentation in grey matter, white matter, and subcortical structures