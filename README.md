# LookupTablePN


## Prerequisites
The following libraries have no specific version requirements unless otherwise stated, I just list the versions I use.

 - python 3.9.12
 - numpy 1.21.5
 - open3d 0.14.1 
 - trimesh 3.12.6
 - scikit-learn 1.0.2

## Usage
Before starting, please place the files in the following directory format:
```
LookupTablePN++
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




