import numpy as np
from scipy.spatial import cKDTree

def cal_sparsity(ft_data,height_map):
    '''
    Calculate the sparsity of the heighty map
    :param ft_data: the footprint numpy array, 0 for no footprint, 255 for footprint
    :param height_map: the height map numpy array, 0 for empty, 255 for occupied
    '''
    # calculate the sparsity of the height map
    sparsity = np.logical_and(ft_data, height_map)*255 #return boolean array
    sparsity_rate = 1 - np.sum(sparsity)/np.sum(ft_data)
    return sparsity_rate

def cal_chamfer_hausdorff_distance(ft_data,height_map):
    '''
    Calculate the chamfer distance and hausdorff distance between the footprint and the height map, normalized by the size of the footprint image
    :param ft_data: the footprint numpy array, 0 for no footprint, 255 for footprint
    :param height_map: the height map numpy array, 0 for empty, 255 for occupied
    '''
    # calculate the chamfer distance between the footprint and the height map
    points1 = np.argwhere(ft_data==255)
    points2 = np.argwhere(height_map==255)

    if len(points1) == 0 or len(points2) == 0:
        return np.inf, np.inf
    
    # distances_matrix = cdist(points1, points2, 'euclidean')
    # min_distances = np.min(distances_matrix, axis=1) 
    # chamfer_distances= np.mean(min_distances)
    # normalized_chamfer_distance = chamfer_distances/(ft_data.shape[0])
    
    # hausdorff_distance= np.max(min_distances)
    # normalized_hausdorff_distance = hausdorff_distance/(ft_data.shape[0])

    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    min_distances1, _ = tree2.query(points1) # for each point in points1, find the closest point in points2
    chamfer_distance= np.mean(min_distances1)
    normalized_chamfer_distance = chamfer_distance/(ft_data.shape[0]) # normalized by the size of the footprint image

    hausdorff_distance = np.max(min_distances1)
    normalized_hausdorff_distance = hausdorff_distance/(ft_data.shape[0])

    return normalized_chamfer_distance, normalized_hausdorff_distance, chamfer_distance, hausdorff_distance

