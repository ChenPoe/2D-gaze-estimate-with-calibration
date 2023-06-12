import torch
from feature_extractor import feature_extractor
from sfo_model import DirectLoss, SFOModel
from dataset import TrainDataset
import random
import os
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


class train_calibration:

    def __init__(self, train_data_dir, lr=0.01, epochs=50000):

        self.train_data_dir = train_data_dir
        self.train_data_files = [os.path.join(self.train_data_dir, f) for f in os.listdir(self.train_data_dir) if
                                 os.path.isfile(os.path.join(self.train_data_dir, f))]
        print(self.train_data_files)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor()
        self.generator = SFOModel()
        if torch.cuda.device_count() > 1:
            self.generator = torch.nn.parallel.DataParallel(self.generator)
        self.generator.load_state_dict(torch.load('checkpoint/calibrate_model.pt'), strict=False)
        self.generator.to(self.device)
        self.feature_extractor.to(self.device)
        self.calibration_features = []
        self.y_cs = []

    def prepare_calibration_features(self, calibrate_dataset):

        self.calibration_features = []
        self.y_cs = []

        calibrate_dataloader = DataLoader(calibrate_dataset, batch_size=1, shuffle=False)

        for ii, data in enumerate(calibrate_dataloader):
            self.feature_extractor.zero_grad()
            data["faceImg"] = data["faceImg"].to(self.device)  # .unsqueeze(0)
            data["leftEyeImg"] = data["leftEyeImg"].to(self.device)
            data['rightEyeImg'] = data['rightEyeImg'].to(self.device)
            data['rects'] = data['rects'].to(self.device)
            targets = data['targets']

            feature = self.feature_extractor(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])
            self.calibration_features.append(feature)
            self.y_cs.append(targets)
            # print(len(self.calibration_features), feature.size(), targets.size())   

        self.y_cs = torch.stack(self.y_cs, dim=1)

    def train(self):
        log_dir = "outputs/SFOlogs/" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-logs'
        writer = SummaryWriter(log_dir=log_dir)
        loss_op = DirectLoss(self.device).to(self.device)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=self.lr)

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.epochs):
            for train_data_file in self.train_data_files:
                # print('正在训练：',train_data_file)
                dataset = TrainDataset(train_data_file)

                support_ratio = 0.3
                support_num = random.randint(2, min(2 + int(dataset.__len__() * support_ratio), dataset.__len__(), 50))
                # 将训练数据按support_num分割为训练集和支持集
                indices = random.sample(range(dataset.__len__()), support_num)
                support_dataset = [dataset[i] for i in indices]
                train_dataset = [dataset[i] for i in range(dataset.__len__()) if i not in indices]

                self.prepare_calibration_features(support_dataset)
                self.y_cs = self.y_cs.to(self.device)
                batch_size = train_dataset.__len__()  # 视训练集大小而定
                train_loader = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          shuffle=False,
                                          drop_last=True,
                                          pin_memory=False)

                for ii, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    self.generator.train()

                    data["faceImg"] = data["faceImg"].to(self.device)
                    data["leftEyeImg"] = data["leftEyeImg"].to(self.device)
                    data['rightEyeImg'] = data['rightEyeImg'].to(self.device)
                    data['rects'] = data['rects'].to(self.device)
                    targets = data['targets'].to(self.device)

                    y_q, direction_probs = self.generator(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'],
                                                          data['rects'], self.calibration_features, self.y_cs)
                    print(y_q, targets)
                    loss = loss_op(y_q, targets, self.y_cs, direction_probs)
                    self.generator.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # if ii % 100 == 0:
                    #     print('训练样本数：',ii * batch_size, loss)

            print('训练进度：' + str(epoch + 1) + '/' + str(self.epochs))
            writer.add_scalar('train_loss_epoch', loss, global_step=epoch)
            print('Loss={0:.4f}.'.format(torch.mean(loss).detach().cpu().numpy().tolist()))

            if epoch % 50 == 0:
                self.save_state_dict('output/checkpoint/train_calibrate_temp_model.pt')
        print('训练完成！')
        self.save_state_dict('output/checkpoint/train_calibrate_model.pt')
        writer.close()

    def save_state_dict(self, save_dir=None):
        if save_dir is None:
            save_dir = 'outputs/checkpoint/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pt'
        torch.save(self.generator.state_dict(), save_dir)


if __name__ == '__main__':
    tc = train_calibration('train_data', epochs=50000, lr=0.0005)
    tc.train()
