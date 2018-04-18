# 2018 Retreat P8: Automatic identification of Brain Sulci based on DNN


## Overall plan:

### Dataset
* HCP 900 for which we have
  * A cleaned AC/PC-aligned T1w image of each subject brain (not in MNI space)
  * The [Destrieux atlas](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2937159/) in voxel space

Data can be downloaded with 
```bash
rsync -avzh --prune-empty-dirs --include="*/" --include="*/T1w/T1*brain.nii.gz" --include="*/T1w/*a2009*.nii.gz" --exclude="*" -e ssh dragostore:/data/data/HCP900/* .
```

### Base ourselves on (but this is open to discussion):
*  Keras (Install with `conda install -c conda-forge keras`)
*  An [implementation of the U-net](https://github.com/jocicmarko/ultrasound-nerve-segmentation.git)

### Then:
1. Generalise it to 3D
2. Develop a framework to test performance
4. Tune the network architecture
4. Check it’s the resulting network’s capability to detect sulci that 3) Check it’s the resulting network’s capability to detect sulci that have never been labeled. For this we might want to exclude some sulci from the training set (and not just some subjects)