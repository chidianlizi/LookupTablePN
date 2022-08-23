#################################################################
# After splitting the obj model, features are extracted         #
# from each obj part, and then clustering after downscaling     #
# and normalization, here SpectralClustering is used,           #
# so the optimal number of clusters needs to be given by        #
# evaluation, and the results may be different for multiple     #
# runs of the script. To get better results may need to be a    #
# little manually adjusted.                                     #
#################################################################

from importlib.resources import path
import os
from re import L
import sys
import numpy as np
import trimesh
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score
import pickle
import matplotlib.pyplot as plt
from yaml import ValueToken

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH) # dir /utils
ROOT = os.path.dirname(BASE) # dir /lookup_table
sys.path.insert(0,os.path.join(ROOT,'utils'))
from math_util import get_projections_of_triangle, get_angle
from foundation import show_obj

def split(path_file):
    '''
    take the assembly apart
    '''
    i=0
    idx_start = 0
    with open(path_file,'r') as f:
        print (path_file)
        count = 0
        namestr = os.path.splitext(os.path.split(path_file)[-1])[0]
        os.makedirs(ROOT+'/data/train/split/'+namestr)
        outstr = os.path.join(ROOT, 'data/train/split', namestr) + '/' + namestr + '_'
        for line in f.readlines():
            if line[0:1] == 'o':
                idx_start = count
                path_out = outstr + str(i) + '.obj'
                i += 1
                g = open(path_out , 'w')
                g.write('mtllib ' + namestr + '.mtl'+'\n')
                g.write(line)
            elif line[0:1] == 'v':
                if i>0:
                    g = open(outstr + str(i-1) + '.obj', 'a')
                    g.write(line)   
                count +=1
            elif line[0:1] == 'f':
                new_line = 'f '
                new_line += str(int(line.split()[1])-idx_start) + ' '
                new_line += str(int(line.split()[2])-idx_start) + ' '
                new_line += str(int(line.split()[3])-idx_start) + '\n'
                if i>0:
                    g = open(outstr + str(i-1) + '.obj', 'a')
                    g.write(new_line)            
            else:
                if i>0:
                    g = open(outstr + str(i-1) + '.obj', 'a')
                    g.write(line)
        os.system('cp %s %s'%(ROOT+'/data/train/models/'+namestr+'/'+namestr+'.mtl', os.path.join(ROOT, 'data/train/split', namestr)))

def label_special_cases(features, files):
    '''
    if there is obvious part in the plates, then label it first manually
    usually can be ignored
    '''
    index = np.where((features[:,4]<0.6) & (features[:,0]>8e6))[0]
    labels_dict = {}
    list = []   
    for i in range(len(index)):
        # print(files[index[i]])
        list.append(files[index[i]])
        labels_dict[files[index[i]].strip()] = -1
    features_del = np.delete(features, index, axis=0)
    os.system('mkdir -p ../data/train/label_temp_folder/special_cases')   
    for case in list:
        files.remove(case)
        case = case.rstrip()
        os.system('cp %s %s'%(case, '../data/train/label_temp_folder/special_cases'))
    # print (features_del.shape)
    # print (labels_dict)
    return features_del, files, labels_dict

def extract_facet_normals(mesh):
    '''
    histogram of normal distribution, use only one normal from coplanar meshes
    the voting interval is 30 degrees
    '''
    x_bins = np.zeros((6,))
    y_bins = np.zeros((6,))
    z_bins = np.zeros((6,))
    for facet in mesh.facets:
        face = facet[0]
        face_normal = mesh.face_normals[face]
        _, angle_x = get_angle(face_normal, [1,0,0])
        _, angle_y = get_angle(face_normal, [0,1,0])
        _, angle_z = get_angle(face_normal, [0,0,1])
        if angle_x == 180:
            angle_x = 179
        if angle_y == 180:
            angle_y = 179
        if angle_z == 180:
            angle_z = 179
        x_bins[int(angle_x//30)] += 1          
        y_bins[int(angle_y//30)] += 1
        z_bins[int(angle_z//30)] += 1
    return x_bins, y_bins, z_bins  

def extract_face_normals(mesh):
    '''
    histogram of normal distribution
    the voting interval is 30 degrees
    '''
    x_bins = np.zeros((6,))
    y_bins = np.zeros((6,))
    z_bins = np.zeros((6,))
    for i in range(len(mesh.faces)):
        face_normal = mesh.face_normals[i]
        _, angle_x = get_angle(face_normal, [1,0,0])
        _, angle_y = get_angle(face_normal, [0,1,0])
        _, angle_z = get_angle(face_normal, [0,0,1])
        if angle_x == 180:
            angle_x = 179
        if angle_y == 180:
            angle_y = 179
        if angle_z == 180:
            angle_z = 179
        x_bins[int(angle_x//30)] += 1          
        y_bins[int(angle_y//30)] += 1
        z_bins[int(angle_z//30)] += 1
    return x_bins, y_bins, z_bins  

def extract_feature_from_mesh(path_to_mesh:str):
    '''Extract feature from a mesh
    
    Args:
        path_to_mesh (str): The path of the mesh to extract
    Returns:
        feature(np.ndarray): A feature descriptor
    '''
    feature = np.zeros(shape=(26,), dtype=np.float64)
    mesh = trimesh.load_mesh(path_to_mesh)
    mesh.remove_duplicate_faces()
    # move the coordinate system to the inertia principal axis
    mesh.apply_transform(mesh.principal_inertia_transform)
    # mesh_o3d = mesh.as_open3d
    # mesh_o3d.compute_vertex_normals()
    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=[0,0,0])
    # o3d.visualization.draw_geometries([mesh_o3d, coor])
    vert_max = mesh.vertices.max(axis=0)
    vert_min = mesh.vertices.min(axis=0)
    # 0: area
    feature[0] = mesh.area
    # 1: volume
    feature[1] = mesh.volume
    # 2: euler_number
    feature[2] = mesh.euler_number
    # 3: side length of a cube ratio, 1.0 for cube
    feature[3] = mesh.identifier[2]
    # 4: compare the volume of the mesh with the volume of its convex hull
    feature[4] = np.divide(mesh.volume, mesh.convex_hull.volume)  
    # 5. the sum of the projections of each mesh on the three coordinate planes
    xoy = 0.0
    yoz = 0.0
    xoz = 0.0    
    for i in range(len(mesh.faces)):
        p1 = mesh.vertices[mesh.faces[i]][0]
        p2 = mesh.vertices[mesh.faces[i]][1]
        p3 = mesh.vertices[mesh.faces[i]][2]
        proj_xoy, proj_yoz, proj_xoz = get_projections_of_triangle(p1, p2, p3)
        xoy += proj_xoy
        yoz += proj_yoz
        xoz += proj_xoz
    feature[5:8] = [xoy, yoz, xoz]

    # 6. histogram of normal distribution
    x_bins, y_bins, z_bins = extract_facet_normals(mesh)
    feature[8:14] = x_bins
    feature[14:20] = y_bins
    feature[20:26] = z_bins
    
    return feature
    
    
def write_all_parts():
    '''Get the filename of all the single plate
    Eliminate non-shaped meshes
    
    Args:
        None
    Returns:
        None
    '''
    path_to_all_components = os.path.join(ROOT, 'data/train/split')
    all_components = os.listdir(path_to_all_components)
    with open(os.path.join(path_to_all_components, 'all_parts.txt'), 'w') as f:
        for component in all_components:
            files = os.listdir(os.path.join(path_to_all_components, component))
            for file in files:
                if os.path.splitext(file)[1] == '.obj':                
                    content = open(os.path.join(path_to_all_components, component, file), 'r')
                    lines = content.readlines()
                    if len(lines) > 10:
                        f.writelines(os.path.join(path_to_all_components, component, file)+'\n')
                            
def label():
    '''Automatic labeling
    Cluster analysis, each cluster is a class
    Args:
        None
    Returns:
        None
    '''
    features = []
    labels_dict = {}
    files = open(ROOT+'/data/train/split/all_parts.txt', 'r').readlines()
    
    # extract features
    for file in files:
        feature = extract_feature_from_mesh(file.strip())
        features.append(feature)
    features = np.asarray(features)
    # np.save(ROOT+'/data/train/split/features.npy',features)
    features = np.load(ROOT+'/data/train/split/features.npy')
    features, files, labels_dict = label_special_cases(features, files)
    
    # reduce the dimension of the normal features
    dim_norm = 3
    norm_info = features[:,5:26]
    transformer = PCA(n_components=dim_norm)
    norm_transformed = transformer.fit_transform(norm_info)
    
    new_features = np.zeros(shape=(features.shape[0], 5+dim_norm))
    new_features[:,0:5] = features[:,0:5]
    new_features[:,5:(5+dim_norm)] = norm_transformed
    # normalization
    minMax = MinMaxScaler((0,1))
    features_norma= minMax.fit_transform(new_features)
    
    # select the suitable parameters
    gamma_finl = 0
    n_finl = 0
    score_finl = 0
    for _, gamma in enumerate((0.001, 0.005, 0.1, 0.5, 1)):
        for n in range(3,16):
            y_pred = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(features_norma)
            score = calinski_harabasz_score(features_norma, y_pred)
            if score > score_finl:
                score_finl = score
                gamma_finl = gamma
                n_finl = n
                # ds_finl = ds
                # dn_finl = dn
            # print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", n, "score:", score)
    print ('best score: ', score_finl, 'best gamma: ', gamma_finl, 'best n_clusters: ', n_finl)
    # print('ds: ', ds, 'dn: ', dn)
    # =========================================================================
    print ('Next you will see the classification results, the category names on the window are from 0 to n.')
    print ('If you want to adjust them later, please remember the category corresponding to each visualization.')
    input("(Press Enter)")
    n_clusters = n_finl
    gamma = gamma_finl
    clusterer = SpectralClustering(n_clusters=n_clusters, gamma=gamma)
    y_pred = clusterer.fit_predict(features_norma)
    # print("Calinski-Harabasz Score", calinski_harabasz_score(features_norma, y_pred))
    # print(y_pred.shape)
    for i in range(n_clusters):
        os.system('mkdir -p ../data/train/label_temp_folder/%s'%str(i))        
        idx = np.where(y_pred==i)
        for j in range(len(idx[0])):
            labels_dict[files[idx[0][j]].strip()] = i
            os.system('cp %s %s'%(files[idx[0][j]].strip(), '../data/train/label_temp_folder/%s'%str(i)))
    
    # save the automatically generated labels
    # print (labels_dict)
    with open('../data/train/label_temp_folder/labels_dict.pkl', 'wb') as tf:
        pickle.dump(labels_dict,tf,protocol=2)
          
    # display  
    for i in range(n_clusters):
        show_obj('../data/train/label_temp_folder/'+str(i))
    # show_obj('../data/train/label_temp_folder/special_cases')

    # =========================================================================
    
def relabel(n):
    '''Redo the automatic labeling
    Cluster analysis, each cluster is a class
    Args:
        n(int): the number of classes you want
    Returns:
        None
    '''
    features = []
    labels_dict = {}
    files = open(ROOT+'/data/train/split/all_parts.txt', 'r').readlines()
    
    # extract features
    for file in files:
        feature = extract_feature_from_mesh(file.strip())
        features.append(feature)
    features = np.asarray(features)
    # np.save(ROOT+'/data/train/split/features.npy',features)
    features = np.load(ROOT+'/data/train/split/features.npy')
    features, files, labels_dict = label_special_cases(features, files)
    
    # reduce the dimension of the normal features
    dim_norm = 3
    norm_info = features[:,5:26]
    transformer = PCA(n_components=dim_norm)
    norm_transformed = transformer.fit_transform(norm_info)
    
    new_features = np.zeros(shape=(features.shape[0], 5+dim_norm))
    new_features[:,0:5] = features[:,0:5]
    new_features[:,5:(5+dim_norm)] = norm_transformed
    # normalization
    minMax = MinMaxScaler((0,1))
    features_norma= minMax.fit_transform(new_features)
    
    # select the suitable parameters
    gamma_finl = 0
    score_finl = 0
    for _, gamma in enumerate((0.001, 0.005, 0.1, 0.5, 1)):
        y_pred = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(features_norma)
        score = calinski_harabasz_score(features_norma, y_pred)
        if score > score_finl:
            score_finl = score
            gamma_finl = gamma
    print ('best score: ', score_finl, 'best gamma: ', gamma_finl)
    # print('ds: ', ds, 'dn: ', dn)
    # =========================================================================

    input("(Press Enter)")
    n_clusters = n
    gamma = gamma_finl
    clusterer = SpectralClustering(n_clusters=n_clusters, gamma=gamma)
    y_pred = clusterer.fit_predict(features_norma)
    # print("Calinski-Harabasz Score", calinski_harabasz_score(features_norma, y_pred))
    # print(y_pred.shape)
    for i in range(n_clusters):
        os.system('mkdir -p ../data/train/label_temp_folder/%s'%str(i))        
        idx = np.where(y_pred==i)
        for j in range(len(idx[0])):
            labels_dict[files[idx[0][j]].strip()] = i
            os.system('cp %s %s'%(files[idx[0][j]].strip(), '../data/train/label_temp_folder/%s'%str(i)))
    
    # save the automatically generated labels
    # print (labels_dict)
    with open('../data/train/label_temp_folder/labels_dict.pkl', 'wb') as tf:
        pickle.dump(labels_dict,tf,protocol=2)
          
    # display  
    for i in range(n_clusters):
        show_obj('../data/train/label_temp_folder/'+str(i))
    # show_obj('../data/train/label_temp_folder/special_cases')

if __name__ == '__main__':
    # 1. take the assembly apart
    print ('Ensure that the components to be split are placed in the required directory format')
    input('(Press Enter)')
    # print ('Step1. Split the assembly')
    # path_to_components = os.path.join(ROOT, 'data', 'train', 'models')
    # components = os.listdir(path_to_components)
    # for comp in components:
    #     path_to_comp = os.path.join(path_to_components, comp)
    #     files = os.listdir(path_to_comp)
    #     for file in files:
    #         if os.path.splitext(file)[1] == '.obj':
    #             split(os.path.join(path_to_comp, file))
    # write_all_parts()
    # input('(Press Enter)')
    # 2. label these parts
    print ('Step2. Label the split parts')
    print ('The following parameters are automatically selected by the algorithm, please adjust later if you are not satisfied')
    label()
    # 3. relabel
    print ('For the classification results, it is recommended that the initial classification be more detailed, and the subsequent operations can combine different categories')
    print ('In other words, if some similar parts are classified in different categories, then there is no need to reclassify them and they can be merged in subsequent operations.')
    input('(Press Enter)')
    satisfaction = True
    satisfaction = True if input('Are you satisfied with this classification?(y/n)\n')=='y' else False
    while not satisfaction:
        os.system('rm -rf ../data/train/label_temp_folder')
        n = int(input('How many categories do you want to divide into?\n'))
        relabel(n)
        satisfaction = True if input('Are you satisfied with this classification?(y/n)\n')=='y' else False 
    # 4. merge
    print ('Current categories are')
    f = open('../data/train/label_temp_folder/labels_dict.pkl', 'rb')
    labeldict = pickle.load(f)
    current_classes = []
    for v in labeldict.values():
        if v not in current_classes:
            current_classes.append(v)
    print (current_classes)
    is_merge = input ('Do you want to merge some classes?(y/n)\n')
    if is_merge == 'y':
        used = []
        new_classes = []
        end = False
        i = 0
        while not end:
            print ('Which classes do you want to merge into the new class?')
            input_str = input ('(i.e. [0<Space>1<Space>2<Enter>])\n')
            to_be_merged = list(map(int, input_str.split()))
            used += to_be_merged
            new_classes.append(i)
            for c in to_be_merged:
                current_classes.remove(c)
                for key in labeldict:
                    if labeldict[key] == c:
                        labeldict[key] = 100+i
            if input ('Are there any more classes you want to merge?(y/n)\n') == 'y':
                end = False
            else:
                end = True
            i += 1
    values = []
    for key in labeldict:
        if labeldict[key] not in values:
            values.append(labeldict[key])
    class_dict = {}
    for i in range(len(values)):
        class_name = 'class_'+str(i)
        for key in labeldict:
            if labeldict[key] == values[i]:
                class_dict[key] = class_name
                
    os.makedirs('../data/train/parts_classification/')
    with open(os.path.join('../data/train/parts_classification/class_dict.pkl'), 'wb') as tf:
        pickle.dump(class_dict,tf,protocol=2)
    for key in class_dict:
        class_dir = os.path.join('../data/train/parts_classification', class_dict[key])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        os.system('cp %s %s'%(key, class_dir))
    
    for i in range(len(values)):
        show_obj('../data/train/parts_classification/'+'class_'+str(i))
    print ('FINISHED!')

    
