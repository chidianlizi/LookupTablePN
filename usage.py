import os
import sys
CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0,CURRENT_PATH)
from lut import LookupTable
from train import TrainPointNet2

# preprocessing for lookup table 
lut = LookupTable(path_data='./data', label='PDL', hfd_path_classes=None, pcl_density=40, crop_size=400, num_points=2048)
lut.make()


# change to py2 env
# train pn++
tr = TrainPointNet2(path_data='./data')
# make dataset
tr.make_dataset(crop_size=400, num_points=2048)
# training
tr.train(log_dir='./data/seg_model', gpu=0, num_point=2048, max_epoch=100, batch_size=16, learning_rate=0.001)

