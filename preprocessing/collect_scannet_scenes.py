import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
sys.path.append(".")
from scannet_util import g_label_names, g_raw2scannet
from lib.pc_util import read_ply_xyzrgbnormal
from lib.utils import get_eta
from lib.config import CONF
from prepare_data.gen_indoor3d_h5 import Get_h5

CLASS_NAMES = g_label_names
RAW2SCANNET = g_raw2scannet
NUM_MAX_PTS = 100000

def collect_one_scene_data_label(scene_name, out_filename):
    # Over-segmented segments: maps from segment to vertex/point IDs
    if(CONF.DATA_TYPE == "train"):
        data_folder = os.path.join(CONF.SCANNET_DIR, scene_name)
        mesh_seg_filename = os.path.join(data_folder, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
        #print mesh_seg_filename
        with open(mesh_seg_filename) as jsondata:
            d = json.load(jsondata)
            seg = d['segIndices']
            #print len(seg)
        segid_to_pointid = {}
        for i in range(len(seg)):
            if seg[i] not in segid_to_pointid:
                segid_to_pointid[seg[i]] = []
            segid_to_pointid[seg[i]].append(i)

        # Raw points in XYZRGBA
        ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
        points = read_ply_xyzrgbnormal(ply_filename)
        
        # Instances over-segmented segment IDs: annotation on segments
        instance_segids = []
        labels = []
        # annotation_filename = os.path.join(data_folder, '%s.aggregation.json'%(scene_name))
        annotation_filename = os.path.join(data_folder, '%s_vh_clean.aggregation.json'%(scene_name))
        #print annotation_filename
        with open(annotation_filename) as jsondata:
            d = json.load(jsondata)
            for x in d['segGroups']:
                instance_segids.append(x['segments'])
                labels.append(x['label'])


        # Each instance's points
        instance_points_list = []
        instance_labels_list = []
        semantic_labels_list = []
        for i in range(len(instance_segids)):
            segids = instance_segids[i]
            pointids = []
            for segid in segids:
                pointids += segid_to_pointid[segid]
            instance_points = points[np.array(pointids),:]
            instance_points_list.append(instance_points)
            instance_labels_list.append(np.ones((instance_points.shape[0], 1))*i)   
            label = RAW2SCANNET[labels[i]]
            label = CLASS_NAMES.index(label)
            semantic_labels_list.append(np.ones((instance_points.shape[0], 1))*label)

        # Refactor data format

        scene_points = np.concatenate(instance_points_list, 0)
        scene_points = scene_points[:,0:9] # XYZ+RGB+NORMAL
        instance_labels = np.concatenate(instance_labels_list, 0) 
        semantic_labels = np.concatenate(semantic_labels_list, 0)
        data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)

        if data.shape[0] > NUM_MAX_PTS:
            choices = np.random.choice(data.shape[0], NUM_MAX_PTS, replace=False)
            data = data[choices]

        #print("shape of subsampled scene data: {}".format(data.shape))

        np.save(out_filename, data)
    else:
        # Raw points in XYZRGBA
        data_folder = os.path.join(CONF.SCANNET_DIR, scene_name)
        ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
        points = read_ply_xyzrgbnormal(ply_filename)
        points = points[:,0:9]
        np.save(out_filename, points)

if __name__=='__main__':
    os.makedirs(CONF.PREP_SCANS, exist_ok=True)
    #all_data_label_file = open("./prepare_data/meta/all_data_label.txt","w")
    print("Type is: ",CONF.DATA_TYPE)
    print("Block size is: ",CONF.BLOCK_SIZE)
    print("Stride size is: ",CONF.STRIDE_SIZE)
    print("Begin to collect sacnnet scenes...")
    for i, scene_name in enumerate(tqdm(CONF.SCENE_NAMES)):
        try:
            start = time.time()
            out_filename = scene_name+'.npy' # scene0000_00.npy
            #all_data_label_file.write(out_filename+"\n")
            if os.path.exists(os.path.join(CONF.PREP_SCANS, out_filename)):
                continue
            collect_one_scene_data_label(scene_name, os.path.join(CONF.PREP_SCANS, out_filename))
            
            # report
            num_left = len(CONF.SCENE_NAMES) - i - 1
            eta = get_eta(start, time.time(), 0, num_left)

        except Exception as e:
            assert(0)
            print(scene_name+'ERROR!!')

    #all_data_label_file.close()
    print("Sacnnet scenes collection finished!!\n")
    if(CONF.DATA_TYPE=="train"):
        Get_h5()
    else:
        Get_h5()
    print("\nall done!")