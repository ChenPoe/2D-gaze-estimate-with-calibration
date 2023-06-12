import random
import h5py
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, h5py_name):
        self.data_file = h5py.File(h5py_name, 'r')
        self.name_list = list(self.data_file.keys())

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        frame_name = self.name_list[idx]

        frame_data = self.data_file[frame_name]
        face_img = frame_data["faceImg"][:]
        leftEye_img = frame_data["leftEyeImg"][:]
        rightEye_img = frame_data["rightEyeImg"][:]
        rect = frame_data["rects"][:]
        target = frame_data["targets"][:]

        return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                "rects": torch.from_numpy(rect).type(torch.FloatTensor),
                "targets": torch.from_numpy(target).type(torch.FloatTensor)
                }


if __name__ == '__main__':
    dataset = TrainDataset('train_data/data_1.hdf5')
    support_ratio = 0.3
    support_num = random.randint(1, min(1 + int(dataset.__len__() * support_ratio), dataset.__len__(), 50))
    # 将训练数据按support_num分割为训练集和支持集
    indices = random.sample(range(dataset.__len__()), support_num)
    support_dataset = [dataset[i] for i in indices]
    train_dataset = [dataset[i] for i in range(dataset.__len__()) if i not in indices]
    print(support_dataset.__len__(), train_dataset.__len__(), dataset.__len__())
