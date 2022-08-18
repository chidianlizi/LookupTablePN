from genericpath import exists
import open3d as o3d
import numpy as np
import os
import sys
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'utils'))

from foundation import load_pcd_data, points2pcd, draw, fps
from xml_parser_ds2 import parse_frame_dump, parse_xml_to_array, list2array
from math_util import rotate_mat, rotation_matrix_from_vectors
import pickle
from xml.dom.minidom import Document
import copy
import time

# load label dictionary
f = open('../data/train/parts_classification/label_dict.pkl', 'rb')
lable_list = pickle.load(f)
label_dict_r = dict([val, key] for key, val in lable_list.items())   


class WeldScene:
    """
    segment the labeled point cloud and create a dataset for semantic segmentation
    """
    def __init__(self, pc_path):
        self.pc = o3d.geometry.PointCloud()
        xyzl = load_pcd_data(pc_path)
        print (xyzl.shape)
        self.xyz = xyzl[:,0:3]
        self.l = xyzl[:,3]
        self.pc.points = o3d.utility.Vector3dVector(self.xyz)

    def crop(self, weld_info, bbox=1, num_points=2048, vis=False):
        """
        bbox: number of cut bounding box, 1 = use one normal info, 2 = use two normals, for the rotated slice bbox must be 1
        the default point cloud contains a minimum of 2048 points, if not enough then copy and fill
        """
        pc = copy.copy(self.pc)
        if bbox == 1:
            # tow surface normals at the welding spot
            norm1 = np.around(weld_info[4:7], decimals=6)
            norm2 = np.around(weld_info[7:10], decimals=6)
            EXTENT = 390
            CROP_EXTENT = np.array([EXTENT, EXTENT, EXTENT])
            weld_spot = weld_info[1:4]
            # move the coordinate center to the welding spot
            pc.translate(-weld_spot)
            # rotation at this welding spot
            rot = weld_info[10:13]*np.pi/180
            rotation = rotate_mat(axis=[1,0,0], radian=rot[0])
            # torch pose
            pose = np.zeros((3,3))
            pose[0:3, 0] = weld_info[14:17]
            pose[0:3, 1] = weld_info[17:20]
            pose[0:3, 2] = weld_info[20:23]
            # cauculate the new pose after rotation
            pose_new = np.matmul(rotation, pose)         
            tf = np.zeros((4,4))
            tf[3,3] = 1.0
            tf[0:3,0:3] = rotation
            pc.transform(tf)
            # new normals
            norm1_r = np.matmul(rotation, norm1.T)
            norm2_r = np.matmul(rotation, norm2.T)

            weld_info[4:7] = norm1_r
            weld_info[7:10] = norm2_r
            weld_info[14:17] = pose_new[0:3, 0]
            weld_info[17:20] = pose_new[0:3, 1]
            weld_info[20:23] = pose_new[0:3, 2]

            coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=[0,0,0])
            norm_ori = np.array([0, 0, 1])
            # bounding box of cutting area
            R = rotation_matrix_from_vectors(norm_ori, norm_ori) 
            bbox = o3d.geometry.OrientedBoundingBox(center=(EXTENT/2-60)*norm1_r+(EXTENT/2-60)*norm2_r , R=R, extent=CROP_EXTENT)

            cropped_pc = pc.crop(bbox)
            idx_crop = bbox.get_point_indices_within_bounding_box(pc.points)
            xyz_crop = self.xyz[idx_crop]
            xyz_crop -= weld_spot
            xyz_crop_new = np.matmul(rotation, xyz_crop.T).T
            l_crop = self.l[idx_crop]
            xyzl_crop = np.c_[xyz_crop_new, l_crop]
            xyzl_crop = np.unique(xyzl_crop, axis=0)
            while xyzl_crop.shape[0] < num_points:
                xyzl_crop = np.vstack((xyzl_crop, xyzl_crop))
            xyzl_crop = fps(xyzl_crop, num_points)
            if vis:
                o3d.visualization.draw_geometries([cropped_pc, coor1, bbox])
                
        if bbox == 2:
            norm1 = np.around(weld_info[4:7], decimals=6)
            norm2 = np.around(weld_info[7:10], decimals=6)
            # print (norm1, norm2)
            EXTENT_X = 390
            EXTENT_Y = 390
            EXTENT_Z = 250
            CROP_EXTENT = np.array([EXTENT_X, EXTENT_Y, EXTENT_Z])
            weld_spot = weld_info[1:4]
            pc.translate(-weld_spot)
            # rotation at this welding spot
            rot = weld_info[10:13]*np.pi/180
            rotation = rotate_mat(axis=[1,0,0], radian=rot[0])
            # torch pose
            pose = np.zeros((3,3))
            pose[0:3, 0] = weld_info[14:17]
            pose[0:3, 1] = weld_info[17:20]
            pose[0:3, 2] = weld_info[20:23]
            # cauculate the new pose after rotation
            pose_new = np.matmul(rotation, pose)     
            tf = np.zeros((4,4))
            tf[3,3] = 1.0
            tf[0:3,0:3] = rotation
            pc.transform(tf)
            # new normals
            norm1_r = np.matmul(rotation, norm1.T)
            norm2_r = np.matmul(rotation, norm2.T)

            weld_info[4:7] = norm1_r
            weld_info[7:10] = norm2_r
            weld_info[14:17] = pose_new[0:3, 0]
            weld_info[17:20] = pose_new[0:3, 1]
            weld_info[20:23] = pose_new[0:3, 2]
            norm_ori = np.array([0, 0, 1])
            R1 = rotation_matrix_from_vectors(norm_ori, norm1_r)
            R2 = rotation_matrix_from_vectors(norm_ori, norm2_r)
            bbox1 = o3d.geometry.OrientedBoundingBox(center=(EXTENT_Z/2-60)*norm1_r+(EXTENT_Y/2-60)*norm2_r , R=R1, extent=CROP_EXTENT)
            bbox2 = o3d.geometry.OrientedBoundingBox(center=(EXTENT_Z/2-60)*norm2_r+(EXTENT_Y/2-60)*norm1_r, R=R2, extent=CROP_EXTENT)
            coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0,0,0])

            pc1 = pc.crop(bbox1)
            idx_crop_1 = bbox1.get_point_indices_within_bounding_box(pc.points)
            xyz_crop_1 = self.xyz[idx_crop_1]
            xyz_crop_1 -= weld_spot
            xyz_crop_1_new = np.matmul(rotation, xyz_crop_1.T).T
            l_crop_1 = self.l[idx_crop_1]
            pc2 = pc.crop(bbox2)
            idx_crop_2 = bbox2.get_point_indices_within_bounding_box(pc.points)
            xyz_crop_2 = self.xyz[idx_crop_2]
            xyz_crop_2 -= weld_spot
            xyz_crop_2_new = np.matmul(rotation, xyz_crop_2.T).T
            l_crop_2 = self.l[idx_crop_2]

            cropped_pc = pc1 + pc2

            xyz_crop = np.vstack((xyz_crop_1_new, xyz_crop_2_new))
            l_crop = np.hstack((l_crop_1, l_crop_2))
            xyzl_crop = np.c_[xyz_crop, l_crop]
            xyzl_crop = np.unique(xyzl_crop, axis=0)
            while xyzl_crop.shape[0] < num_points:
                xyzl_crop = np.vstack((xyzl_crop, xyzl_crop))
            xyzl_crop = fps(xyzl_crop, num_points)
            if vis:
                o3d.visualization.draw_geometries([cropped_pc, coor, bbox1, bbox2])

        return xyzl_crop, cropped_pc, weld_info


def slice_one(pc_path, xml_path, name):
    # create welding scene
    ws = WeldScene(pc_path)
    frames = list2array(parse_frame_dump(xml_path))
    d = {}
    minpoints = 1000000
    for i in range(frames.shape[0]):
        weld_info = frames[i,3:].astype(float)
        cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info, bbox=1)
        # draw(cxyzl[:,0:3], cxyzl[:,3])
        points2pcd('../data/train/welding_zone/'+name+'_'+str(i)+'.pcd', cxyzl)
        d[name+'_'+str(i)] = new_weld_info
    print ('num of welding spots: ',len(d))      
    with open(os.path.join('../data/train/lookup_table',name+'.pkl'), 'wb') as tf:
        pickle.dump(d,tf,protocol=2)

def merge_lookup_table():
    dict_all = {}
    path = '../data/train/lookup_table'
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path, file), 'rb') as f:
            fd = pickle.load(f)
            dict_all.update(fd)
    with open('../data/train/lookup_table/lookup_table.pkl', 'wb') as tf:
        pickle.dump(dict_all,tf,protocol=2)

def get_feature_dict():
    '''Save a feature dictionary for each slice in a .pkl file, the dict contains
    the nummber of each class and the coordinates of bounding box of each cluster of class'''
    
    path = '../data/train/welding_zone'
    if not os.path.exists(os.path.join(ROOT, 'data/ss_lookup_table/dict')):
        os.makedirs(os.path.join(ROOT, 'data/ss_lookup_table/dict'))
    with open('../data/train/lookup_table/lookup_table.pkl', 'rb') as f:
        dict_all = pickle.load(f)
    files = os.listdir(path)

    for i, file in enumerate(files):
        print (str(i)+'/'+str(len(files)), file)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  
        if os.path.splitext(file)[1] == '.pcd':
            name = os.path.splitext(file)[0]
            print ('get feature dict --> ', name)
            # load slice
            xyzl = load_pcd_data(os.path.join(path,file))
            # draw(xyzl[:,0:3],xyzl[:,3])
            xyz = xyzl[:,0:3]
            l = xyzl[:,3]
            label = np.squeeze(l)
            feature_dict = {}
            normals = dict_all[name][4:7]+dict_all[name][7:10]
            feature_dict['normals'] = normals
            torch = dict_all[name][0]
            feature_dict['torch'] = torch
            elements = []
            min_edge = np.min(np.max(xyz, axis=0)-np.min(xyz, axis=0))

            for i in range(len(label_dict_r)):
                idx = np.argwhere(label==i)
                idx = np.squeeze(idx)
                l_i = label[idx]
                xyz_i = xyz[idx]
                if xyz_i.shape[0]>10:
                    eps = min_edge / 2
                    c = DBSCAN(eps=eps, min_samples=10).fit(xyz_i)
                    number = c.labels_.max()+1
                    # print ('The number of '+label_dict_r[i]+':'+str(number))
                    # draw(xyz_i, c.labels_)
                    feature_info = np.zeros(shape = (number, 8, 3), dtype = float)
                    for _ in range(number):
                        idx_f = np.argwhere(c.labels_==_)
                        idx_f = np.squeeze(idx_f)
                        xyz_f = xyz_i[idx_f] # each cluster of each class
                        l_f = l_i[idx_f]
                        # print xyz_f.shape
                        geometry = o3d.geometry.PointCloud()
                        geometry.points = o3d.utility.Vector3dVector(xyz_f)
                        # geometry.paint_uniform_color([0.53, 0.81, 1.0])

                        bbox = geometry.get_axis_aligned_bounding_box()

                        feature_info[_,:,:] = np.asarray(bbox.get_box_points())
                    feature_dict[label_dict_r[i]] = feature_info
                else:
                    feature_dict[label_dict_r[i]] = None
            # o3d.visualization.draw_geometries(elements)
            with open(os.path.join(ROOT,  'data/ss_lookup_table/dict', os.path.splitext(file)[0]+'.pkl'), 'wb') as tf:
                pickle.dump(feature_dict,tf,protocol=2)

def similarity(feature_dict1, feature_dict2):
    loss_amount = 0
    loss_geo = 0
    loss_norm = 0
    norm1 = feature_dict1['normals']
    norm2 = feature_dict2['normals']
    loss_norm = np.sum((norm1-norm2)**2)
    torch1 = feature_dict1['torch']
    torch2 = feature_dict2['torch']
    loss_torch = (torch1-torch2)**2
    for i in range(len(label_dict_r)):
        # print feature_dict1[labeldict[i]]
        # get the number of each class
        if type(feature_dict1[label_dict_r[i]]) == type(None):
            class_num_1_cur = 0
        else:
            class_num_1_cur = feature_dict1[label_dict_r[i]].shape[0]
            
        if type(feature_dict2[label_dict_r[i]]) == type(None):
            class_num_2_cur = 0
        else:
            class_num_2_cur = feature_dict2[label_dict_r[i]].shape[0]
        
        if class_num_1_cur == class_num_2_cur and class_num_1_cur != 0:
            f1 = feature_dict1[label_dict_r[i]]
            f1.sort(axis=0)
            f2 = feature_dict2[label_dict_r[i]]
            f2.sort(axis=0)
            for _ in range(class_num_1_cur):
                box1_all_points = f1[_] #shape(8, 3)
                box2_all_points = f2[_] #shape(8, 3)
                loss_geo += np.sum((box1_all_points-box2_all_points)**2)
        loss_amount += abs(class_num_1_cur-class_num_2_cur)
    return 10*loss_norm + 10*loss_torch + loss_amount + loss_geo/100000


def decrease_lib():
    '''
    removal of redundant slices
    '''
    path = os.path.join(ROOT,  'data/ss_lookup_table/dict')
    if not os.path.exists('../data/train/welding_zone_comp'):
        os.makedirs('../data/train/welding_zone_comp')
    files = os.listdir(path)
    used = np.zeros(len(files))
    new_lib = []
    for i in range(len(files)):
    # for i, file in enumerate(files):
        print (i, files[i])
        if used[i] == 0:
            used[i] = 1
            new_lib.append(files[i])
            with open(os.path.join(path, files[i]), 'rb') as f:
                fd1 = pickle.load(f)
            # for j, _ in enumerate(files):
            for j in range(i+1, len(files)):
                if used[j] == 1:
                    continue
                with open(os.path.join(path, files[j]), 'rb') as g:
                    fd2 = pickle.load(g)
                if similarity(fd1, fd2)<1e-5:
                    used[j] = 1
                    print ('remove redundant slices: ', str(i)+' and '+ str(j) +' are repeated')
    
    print ('remove redundant slices --> before: ', len(files))
    print ('remove redundant slices --> after: ', len(new_lib))

    fileObject = open('../data/ss_lookup_table/comp.txt', 'w')  
    for ip in new_lib:  
        fileObject.write(str(ip))  
        fileObject.write('\n') 
    fileObject.close()  
    for file in new_lib:
        name = os.path.splitext(file)[0]
        src = '../data/train/welding_zone/'+name+'.pcd'
        # src2 = './data/welding_zone/'+name+'.xml'
        os.system('cp %s ../data/train/welding_zone_comp' % (src))
        # os.system('cp %s ./data/welding_zone_comp' % (src2))

def move_files():
    '''
    move the feature dicts to the compact lib
    '''
    path_dict = '../data/ss_lookup_table/dict_comp'
    if not os.path.exists(path_dict):
        os.makedirs(path_dict)
    path = '../data/train/welding_zone_comp'
    files = os.listdir(path)
    for file in files:
        if os.path.splitext(file)[1] == '.pcd':
            name = os.path.splitext(file)[0]
            # print (name)
            os.system('cp %s ../data/ss_lookup_table/dict_comp' % ('../data/ss_lookup_table/dict/'+name+'.pkl'))

def norm_index():
    path = '../data/ss_lookup_table/dict_comp'
    d = {}
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path, file), 'rb') as f:
            fd = pickle.load(f)
        # print (type(fd['normals']))
        norm = np.round(fd['normals'],2).astype(float)
        # print(norm)
        s = ''
        for i in norm:
            if i == 0:
                s += str(0.0)
            else:
                s += str(i)
        # print (s)
        if s not in d:
            l = []
            l.append(os.path.join(path, file))
            d[s] = l
        else:
            l = d[s]
            l.append(os.path.join(path, file))
            d[s] = l
    with open(os.path.join('../data/ss_lookup_table', 'norm_fd.pkl'), 'wb') as tf:
        pickle.dump(d,tf,protocol=2)


if __name__ == '__main__':
    path_welding_zone = '../data/train/welding_zone'
    path_lookup_table = '../data/train/lookup_table'
    if not os.path.exists(path_welding_zone):
        os.makedirs(path_welding_zone)
    if not os.path.exists(path_lookup_table):
        os.makedirs(path_lookup_table)
    files = os.listdir('../data/train/models')
    print ('Generate one point cloud slice per welding spot')
    i = 1
    for file in files:
            print (str(i)+'/'+str(len(files)), file)
            i += 1
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))      
            pc_path = '../data/train/labeled_pc/'+file+'.pcd'
            xml_path = '../data/train/models/'+file+'/'+file+'.xml'
            name = file
            slice_one(pc_path, xml_path, name)
     
    merge_lookup_table()
    print ('Extract feature dictionary from point cloud slices\n')
    input('(Press Enter)')
    get_feature_dict()
    print ('Removing duplicate point cloud slices\n')
    input('(Press Enter)')
    decrease_lib()
    move_files()
    print ('Use the normal information to generate an index for easy searching\n')
    input('(Press Enter)')
    norm_index()
    print('FINISHED')