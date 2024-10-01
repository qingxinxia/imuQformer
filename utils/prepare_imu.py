import os
import glob
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

rootpath = r'D:\code\IMUqformer'
imu_path = os.path.join(rootpath, 'examples', 'imu', 'subset_0000')
label_path = os.path.join(rootpath, 'examples', 'text', 'subset_0000')
target_body_part = ['left_wrist_v2.npz', 'right_wrist_v2.npz']
window_size = 180

# Get all folders in the specified directory
data_folders = [f for f in os.listdir(imu_path) if os.path.isdir(os.path.join(imu_path, f))]
label_files = [f for f in os.listdir(label_path)]

# # Print or store the folder names
# for folder in data_folders:
#     print(folder)

# load imu data
min_threshold = 500   # if data length less than threshold, ignore
max_threshold = 2600  # if data length longer than threshold, cut
datalen = []
data_dict = {}
for data_f in data_folders:
    imus = []
    for body_p in target_body_part:
        npz_data = np.load(os.path.join(imu_path, data_f, body_p))
        imu = npz_data['linear_acceleration']
        if len(imu) < min_threshold:
            continue
        imus.append(imu)
        datalen.append(len(imu))
    if len(imus) > 0:
        tmp = np.concatenate(imus, axis=1)[:max_threshold,:]
        data_dict[data_f] = np.pad(tmp, ((0, max_threshold-len(tmp)), (0, 0)), 'minimum')  # pad with minimum value, media,maximum,mean
# plt.hist(datalen, density=True, bins=60)
# plt.show()

# load text (activity) description of imu data
label_dict = {}
# for txtf in label_files:
for txtf in data_dict.keys():
    with open(os.path.join(label_path, txtf + '.txt'), 'r') as f:
        tfile = f.read()
        label_dict[txtf] = tfile

##########################################################################

# def sliding_window(elements, window_size):
#     step = int(window_size/2)
#     new_elements = []
#     for i in range(0, len(elements) - window_size + 1, step):
#         new_elements.append(elements[i:i + window_size])
#         # new_label.append(label[i:i + window_size])
#     return new_elements
#
# data_seg, labels = [], []
# for data_f in data_folders:
#     imu = data_dict[data_f]
#     if len(imu) <= window_size:
#         continue
#     imu_seg = sliding_window(imu, window_size)
#     data_seg.append(imu_seg)
#     labels.append([data_f] * len(imu_seg))
# data_seg = np.concatenate(data_seg)
# labels = np.concatenate(labels)

##########################################################################

# generate dataloader
class IMU_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        return

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # sample = {'data': self.data[idx, :].astype('float'),
        sample = {'data': self.data[idx, :],
                  'label': self.labels[idx]}
        return sample


data = np.concatenate([list(data_dict.values())], axis=0)  # batch162, len2600, dim6
labels = np.concatenate([list(label_dict.values())], axis=0)  # batch162,

imu_dataset = IMU_Dataset(data, labels)

##########################################################################

# save data
savepath = os.path.join(rootpath, 'dataloader.pkl')
with open(savepath, 'wb') as f:
    pickle.dump(imu_dataset, f)

def load_data():
    with open(savepath, 'rb') as f:
        imu_dataset = pickle.load(f)
    return imu_dataset