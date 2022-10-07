# LookupTablePN
## Environment Configuration
An available docker can be found [here][1] with anaconda virtual environments named py37 and py27 already configured, where py37 is the python3 environment and py27 is the python2 environment.
 
  [1]: https://hub.docker.com/repository/docker/chidianlizi/pointnet
If you want to configure your environment locally, we also recommend using [Anaconda3][https://www.anaconda.com/]. After installing Anaconda, do the following steps one by one.
```bash
conda create -n py3 python=3.9.12
conda activate py3
pip install -r requirements.txt
conda deactivate py3
conda create -n py27 python=2.7.18
conda activate py27
conda install --channel https://conda.anaconda.org/marta-sd tensorflow-gpu=1.2.0
pip install -r requirements_py27.txt
conda deactivate py27
```


## Usage
Before starting, please place the files in the following directory format:
```
LookupTablePN
├── data
│   ├── train
│   │   ├── models
│   │   │   ├── componentname
│   │   │   │   ├── componentname.obj
│   │   │   │   ├── componentname.mtl
│   │   │   │   ├── componentname.xml
│   │   │   ├── ...
```
### Step 1. Clustering analysis of parts based on geometry & generation of labels
First enter the python3 environment.
```bash
cd ./single_spot_table
python obj_geo_based_classification.py
```
Follow the prompts on the command line step by step.
After running, two folders will be created in the 'data/train' directory:

 - parts_classification: result of geometry classification
 - split: single meshes from assembly

### Step 2. Point cloud sampling & semantic labeling of point clouds using the above categories
```bash
cd ./single_spot_table
python obj_sample_and_autolabel.py
```
After running, two folders will be created in the 'data/train' directory:

 - labled_pc: in .pcd format
 - unlabeled_pc: in .xyz format

### Step 3. Slicing & Making lookup table
```bash
cd ./single_spot_table
python slice.py
```
Complete point cloud slicing, feature dictionary generation, de-duplication and other steps according to command line prompts.

After running, three folders will be created in the 'data/train' directory:

 - welding_zone: point cloud slices
 - welding_zone_comp: point cloud slices after de-duplication
 - lookup_table: welding information corresponding to the slices

one folder will be created in the 'data' directory:

- ss_lookup_table: feature dictionaries
    - dict: feature dictionaries
    - dict_comp: feature dictionaries after de-duplication
    - comp.txt: filename of slices after de-duplication
    - norm_fd.pkl: Normal index for lookup


### Step 4. Making training data set

Use the point cloud slices from the *data/train/welding_zone_comp* folder to create a dataset for semantic segmentation. Switch to python2 environment.
```bash
source activate py27 #if use docker 
cd ./single_spot_table
python seg_makeh5.py
```

After running, the following files will be created in the 'data/train' directory:

 - train.txt: slices for training
 - test.txt: slices for testing
 - dataset: h5 format dataset

### Step 5. Training
```bash
source activate py27 #if use docker 
cd ./single_spot_table
python seg_train.py
```
If you use the default configuration, the semantic segmentation network will be saved in the directory *data/seg_model*.

### Step 6. Testing
Inference on unlabeled point cloud slices using semantic segmentation network, obtain semantic labels, generate feature dictionaries and match in lookup table, return torch pose of templates and write the pose into .xml file.

Before this step, make point cloud slices of the test components and remove the pose from the xml files for testing. If not specified, the python3 environment is used.

```bash
cd ./single_spot_table
python test_preprocessing.py.py
```

The folder *welding_zone_test* will be placed in the directory *./data/test*.

Then, run scripts for inference and matching:

```bash
source activate py27 #if use docker 
cd ./single_spot_table
python seg_infer.py --test_input='../data/test/welding_zone_test'
```

The folder *results* will be placed in the directory *./data/test*.
The folder has separate xml files for each welding spot pose, the *foldername+'.xml'* in each subfolder is a simple merge of all welding spots for evaluation, and does not follow the format of the source file for merging.

### Step 7. Evaluation
Run
```bash
cd ./single_spot_table
python evaluation.py
```
During runtime, a couple of pop-up windows will be displayed to take a screenshot to save the slices comparison, and in the subfolder of each component of the results folder, there will be:

 - eval.txt: Compare with the ground truth and save the evaluation results
 - figs: Used to store slices for comparison
    -  correct
    -  collided
    -  safe
    -  The comparison in the above three folders, three images as a group, are the predicted pose, the ground truth, the template slice matched in the lookup table and the template pose.



