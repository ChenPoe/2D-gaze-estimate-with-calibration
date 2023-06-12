import torch
from feature_extractor import feature_extractor
import torch.nn.functional as F
import time
from torch import nn


class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        support_set_y = torch.transpose(torch.stack(support_set_y), 0, 1)

        softmax = nn.Softmax(dim=0)

        K = len(similarities)

        similarities = torch.stack(similarities)
        softmax_similarities = softmax(similarities)
        softmax_similarities = torch.transpose(softmax_similarities, 0, 1)

        preds = torch.sum(torch.mul(softmax_similarities, support_set_y), dim=1, keepdim=True)
        return preds.squeeze(1)


class DirectLoss(nn.Module):
    def __init__(self, device):
        super(DirectLoss, self).__init__()
        self.device = device
        self.mse = nn.MSELoss().to(self.device)
        self.crossentropy = nn.CrossEntropyLoss().to(self.device)

    def forward(self, y_q, y_target, y_c, direction_probs):

        direction_targets = []
        for i in range(y_c.size()[1]):
            y_ci = y_c[:, i]
            # print(i, y_ci, y_q, y_target)
            if (y_target[0, 0] - y_ci[0, 0]) > 0:
                if (y_target[0, 1] - y_ci[0, 1]) > 0:
                    direction_target = [1, 0, 0, 0]
                else:
                    direction_target = [0, 0, 1, 0]
            else:
                if (y_target[0, 1] - y_ci[0, 1]) > 0:
                    direction_target = [0, 1, 0, 0]
                else:
                    direction_target = [0, 0, 0, 1]

            direction_targets.append(direction_target)

        direction_targets = torch.tensor(direction_targets, dtype=torch.float32).to(self.device)
        direction_targets = direction_targets.repeat(direction_probs.size()[0], 1, 1)

        mse_loss = self.mse(y_q, y_target)

        direct_loss = self.crossentropy(direction_probs, direction_targets)
        total_loss = mse_loss + direct_loss
        return total_loss


class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()
        self.similarity_net = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(512 + 4, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self.direction_net = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 6),
        )

    def forward(self, support_set, input_image, y_c):
        similarities = []
        y_s = []
        direction_probs = []
        for j, support_image in enumerate(support_set):
            y_ci = y_c[0, j]
            support_image = support_image.repeat(input_image.size(0), 1)
            embedding = torch.cat([support_image, input_image], dim=1)

            direction_embedding = embedding.clone()
            direction_vector = self.direction_net(direction_embedding)

            delta_pos = direction_vector[:, :2]
            # print(delta_pos, y_ci)
            direction_prob = direction_vector[:, -4:]
            direction_prob = F.softmax(direction_prob, dim=1)

            similarity = self.similarity_net(torch.cat([embedding, direction_prob], dim=1))
            similarities.append(similarity)
            direction_probs.append(direction_prob)
            y_si = y_ci + delta_pos
            y_s.append(y_si)
        direction_probs = torch.stack(direction_probs)
        direction_probs = torch.transpose(direction_probs, 0, 1)
        return similarities, y_s, direction_probs


class SFOModel(nn.Module):
    def __init__(self):
        super(SFOModel, self).__init__()
        self.feature_extractor = feature_extractor()
        self.attention_net = AttentionalClassify()
        self.distance_net = DistanceNetwork()

    def forward(self, leftEyeImg, rightEyeImg, faceImg, rects, calibration_features, y_c):

        X_q = self.feature_extractor(leftEyeImg, rightEyeImg, faceImg, rects)
        X_c = calibration_features

        similarities, y_s, direction_probs = self.distance_net(X_c, X_q, y_c)

        y_q = self.attention_net(similarities, y_s)

        return y_q, direction_probs

