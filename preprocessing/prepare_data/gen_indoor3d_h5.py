import os
import numpy as np
import sys
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import data_prep_util
import indoor3d_util
from tqdm import tqdm
sys.path.append(".")
from lib.config import CONF

# Constants
data_dir = CONF.PREP_SCANS
indoor3d_data_dir = data_dir
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'
# Set paths
if(CONF.DATA_TYPE == "train"):
    filelist = os.path.join(BASE_DIR, 'meta/scannetv2_train_val.txt')
else:
    filelist = os.path.join(BASE_DIR, 'meta/scannetv2_test.txt')
    
# print(len(data_label_files))
# assert 0
output_dir = CONF.SCANNET_H5_DIR

os.makedirs(CONF.PREP, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, CONF.DATA_TYPE + '_h5_file'), exist_ok=True)

output_filename_prefix = os.path.join(output_dir,CONF.DATA_TYPE+'_h5_file', 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
output_all_file = os.path.join(output_dir, 'all_files.txt')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save


def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype) 
        #print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        #print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return




def Get_h5():
    print("Begin to generate room block(h5 file)...")
    data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]

    fout_room = open(output_room_filelist, 'w')
    all_file = open(output_all_file, 'w')

    sample_cnt = 0
    for i, data_label_filename in enumerate(tqdm(data_label_files)):
        #print(data_label_filename)
        data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=CONF.BLOCK_SIZE, stride=CONF.STRIDE_SIZE,
                                                     random_sample=False, sample_num=None)
        #print('{0}, {1}'.format(data.shape, label.shape))
        for _ in range(data.shape[0]):
            fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

        sample_cnt += data.shape[0]
        insert_batch(data, label, i == len(data_label_files)-1)

    fout_room.close()
    print("Total samples: {0}".format(sample_cnt))

    for i in range(h5_index):
        all_file.write(os.path.join(CONF.DATA_TYPE+'_h5_file', 'ply_data_all_') + str(i) +'.h5\n')

    all_file.close()
    print("generate room block(h5 file) finished!!")

if __name__ == "__main__":
    Get_h5()