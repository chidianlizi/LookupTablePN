import os
import sys
import open3d as o3d
import numpy as np
import time
import pickle
CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH) # dir /utils
ROOT = os.path.dirname(BASE) # dir /lookup_table
sys.path.insert(0,os.path.join(ROOT,'utils'))
from math_util import get_projections_of_triangle, get_angle
from foundation import points2pcd

PATH_COMP = '../data/train/models'
PATH_XYZ = '../data/train/unlabeled_pc'
PATH_PCD = '../data/train/labeled_pc' 

def sample_and_label(path, label_list):
    '''Convert mesh to pointcloud
    two pc will be generated, one is .pcd format with labels, one is .xyz format withou labels
    '''
    namestr = os.path.split(path)[-1]
    files = os.listdir(path)
    # label_list = {}
    label_count = 0

    allpoints = np.zeros(shape=(1,4))
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            # print (np.asarray(mesh.triangles).shape)
            if np.asarray(mesh.triangles).shape[0] > 1:
                key = os.path.abspath(os.path.join(path, file))
                label = label_list[classdict[key]]
                # mesh.compute_vertex_normals()
                # coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin = mesh.get_center())
                # o3d.visualization.draw_geometries([mesh, coor])
                    
                number_points = int(mesh.get_surface_area()/40) # get number of points according to surface area
                pc = mesh.sample_points_poisson_disk(number_points, init_factor=5) # poisson disk sampling
                xyz = np.asarray(pc.points)
                l = label * np.ones(xyz.shape[0])
                xyzl = np.c_[xyz, l]
                print (file, 'sampled point cloud: ', xyzl.shape)
                allpoints = np.concatenate((allpoints, xyzl), axis=0)
    points2pcd(os.path.join(PATH_PCD, namestr+'.pcd'), allpoints[1:])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(allpoints[1:,0:3])
    o3d.io.write_point_cloud(os.path.join(PATH_XYZ, namestr+'.xyz'),pc)

############################################################################
def sample_test_pc(path):
    '''Convert mesh to pointcloud without labels
    
    '''
    namestr = os.path.split(path)[-1]
    files = os.listdir(path)
    for file in files:
        if file == namestr+'.obj':
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            # print (np.asarray(mesh.triangles).shape)

                # mesh.compute_vertex_normals()
                # coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin = mesh.get_center())
                # o3d.visualization.draw_geometries([mesh, coor])
                    
            number_points = int(mesh.get_surface_area()/40) # get number of points according to surface area
            pc = mesh.sample_points_poisson_disk(number_points, init_factor=5) # poisson disk sampling
            xyz = np.asarray(pc.points)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(xyz)
            o3d.io.write_point_cloud(os.path.join(PATH_XYZ, namestr+'.xyz'),pc)
############################################################################


if __name__ == '__main__':
    f = open('../data/train/parts_classification/class_dict.pkl', 'rb')
    classdict = pickle.load(f)
    label_list = {}
    i = 0
    for v in classdict.values():
        if v not in label_list:
            label_list[v] = i
            i += 1
    with open(os.path.join('../data/train/parts_classification/label_dict.pkl'), 'wb') as tf:
        pickle.dump(label_list,tf,protocol=2)

    path = '../data/train/split'
    path_xyz = '../data/train/unlabeled_pc'
    path_pcd = '../data/train/labeled_pc'    
    if not os.path.exists(path_xyz):
        os.makedirs(path_xyz)
    if not os.path.exists(path_pcd):
        os.makedirs(path_pcd)
    folders = os.listdir(path)
    count = 0
    total = len(folders)
    for folder in folders:
        if os.path.isdir(os.path.join(path, folder)):
            count += 1
            print ('sampling... ...', folder)
            print (str(count)+'/'+str(total-2))
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # print (os.path.join(path, folder))
            sample_and_label(os.path.join(path, folder), label_list)
    

    # =======================================================
    # for test data
    # path = '../data/test/models'
    # path_xyz = '../data/test/pc'
    # if not os.path.exists(path_xyz):
    #     os.makedirs(path_xyz)
    # folders = os.listdir(path)
    # count = 0
    # total = len(folders)
    # for folder in folders:
    #     count += 1
    #     print ('sampling... ...', folder)
    #     print (str(count)+'/'+str(total))
    #     print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    #     sample_test_pc(os.path.join(path, folder))
