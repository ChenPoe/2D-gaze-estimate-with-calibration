import torch
from feature_extractor import feature_extractor
from sfo_model import SFOModel
from dataset import TrainDataset
from datetime import datetime
import math
from torch.utils.data import DataLoader
import csv


class test_calibration:

    def __init__(self, calibrate_data_dir, test_data_dir, checkpoint_dir):

        if calibrate_data_dir is None:
            calibrate_data_dir = 'output/data/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.hdf5'
        self.calibrate_data_dir = calibrate_data_dir

        if test_data_dir is None:
            test_data_dir = 'output/data/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.hdf5'
        self.test_data_dir = test_data_dir 
        self.lr = 0.001
        self.epochs = 1 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor()
        self.generator = SFOModel()
        if torch.cuda.device_count() > 1:
            self.generator = torch.nn.parallel.DataParallel(self.generator)
        self.generator.load_state_dict(torch.load(checkpoint_dir), strict=False)
        self.generator.to(self.device)
        self.feature_extractor.to(self.device)
        self.calibration_features = []
        self.y_cs = []
        
    def prepare_calibration_features(self, calibration_images):    

        train_dataset = TrainDataset(calibration_images)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
         
        for ii, data in enumerate(train_dataloader):
            
            self.feature_extractor.zero_grad()
            data["faceImg"] = data["faceImg"].to(self.device)  # .unsqueeze(0)
            data["leftEyeImg"] = data["leftEyeImg"].to(self.device)
            data['rightEyeImg'] = data['rightEyeImg'].to(self.device)
            data['rects'] = data['rects'].to(self.device)
            targets = data['targets'] 
            
            feature = self.feature_extractor(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])  
            self.calibration_features.append(feature)
            self.y_cs.append(targets)   

        self.y_cs = torch.stack(self.y_cs, dim=1)         

    def prepare_data(self):
        self.prepare_calibration_features(self.calibrate_data_dir)

    def test(self, result_file='outputs/SFOlogs/result_'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S') +'.csv'): 
        train_loader = DataLoader(TrainDataset(self.test_data_dir),
                                  batch_size=1,
                                  num_workers=0,
                                  shuffle=False,
                                  drop_last=True,
                                  pin_memory=False)
        self.y_cs = self.y_cs.to(self.device)
        torch.autograd.set_detect_anomaly(True)
        count = 0
        total_dis = 0

        with open(result_file, mode='w', newline='') as result_csv:
            result_writer = csv.writer(result_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['x', 'y', 'distance'])

            for epoch in range(1):
                for ii, data in enumerate(train_loader):
                    self.generator.eval()

                    data["faceImg"] = data["faceImg"].to(self.device)   
                    data["leftEyeImg"] = data["leftEyeImg"].to(self.device)
                    data['rightEyeImg'] = data['rightEyeImg'].to(self.device)
                    data['rects'] = data['rects'].to(self.device)
                    targets = data['targets'] 
                    
                    y_q, direction_probs = self.generator(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'], self.calibration_features, self.y_cs)  
                    K = len(self.calibration_features)
                    y_q= y_q.detach().cpu().numpy().tolist()[0]
                    targets =  targets.numpy().tolist()[0]
                    print(y_q, targets)
                    distance = scr_dis(y_q, targets)   
                    result_writer.writerow([targets[0], targets[1], distance])
                    total_dis += distance
                    count += 1
                    if ii % 10 == 0:
                        print('测试样本数：',ii, distance, K)
                print('测试进度：' + str(epoch + 1) + '/' + str(self.epochs))
                print('Distance={0:.4f}.'.format(distance))


def scr_dis(p1, p2):
    scr_w = 1920
    scr_h = 1080

    cam_w = 34.416
    cam_h = 19.359
    return math.sqrt((cam_w/scr_w)**2*(p1[0]-p2[0])*(p1[0]-p2[0]) + (cam_h/scr_h)**2*(p1[1]-p2[1])*(p1[1]-p2[1]))              

if __name__ == '__main__':
    
    tc = test_calibration(calibrate_data_dir= 'test_data/calibrate_20.h5py',
                          test_data_dir='test_data/Test_760.h5py',
                          checkpoint_dir='checkpoint/calibrate_model.pt')
    tc.prepare_data()
    tc.test(result_file='output/log/result.csv')
