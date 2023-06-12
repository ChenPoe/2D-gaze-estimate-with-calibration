import os.path
import time
from datetime import datetime
import math

import cv2
import dlib
import h5py
import numpy as np
import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen
import sys

from torch.utils.data import DataLoader

from seg_utils import seg_eyes, seg_align, get_points
from dataset import TrainDataset
from sfo_model import SFOModel

win_size = (1900, 1000)
widget_width = 1800
widget_height = 900
rect1 = (200, 100, 500, 400)
rect2 = (1800 - 500 - 200, 100, 500, 400)
batch_size = 1
calibrate_points = [2, 2]  # 手动设置校准点数
calibrate_data_dir = None


class PointWidget(QWidget):
    def __init__(self, parent=None):
        super(PointWidget, self).__init__(parent)
        self.target_point = None
        self.pred_point = None

        self.color_left_rect = False
        self.color_right_rect = False

        self.win_width = widget_width
        self.win_height = widget_height
        self.win = QRectF(0, 0, self.win_width, self.win_height)
        self.rect1 = QRectF(rect1[0], rect1[1], rect1[2], rect1[3])
        self.rect2 = QRectF(rect2[0], rect2[1], rect2[2], rect2[3])

        # self.setFixedSize(widget_width, widget_height)
        self.resize(widget_width, widget_height)

    def paintEvent(self, event):
        qp = QPainter(self)
        pen = QPen(Qt.SolidLine)
        pen.setWidth(2)
        qp.setPen(pen)

        if self.target_point is not None:
            qp.setBrush(QColor(Qt.red))
            qp.drawEllipse(self.target_point.x() - self.x(), self.target_point.y() - self.y(), 10, 10)  # 绘制点

        if self.pred_point is not None:
            qp.setBrush(QColor(Qt.blue))
            qp.drawEllipse(self.pred_point.x() - self.x(), self.pred_point.y() - self.y(), 10, 10)  # 绘制点

    def set_target_point(self, point):
        self.target_point = point
        self.update()

    def set_pred_point(self, point):
        self.pred_point = point
        self.update()

    def clear_point(self):
        self.target_point = None
        self.pred_point = None
        self.update()


class CalibrationThread(QThread):

    def __init__(self, parent=None, h5f_dir='data.hdf5'):
        super(CalibrationThread, self).__init__(parent)
        self.h5f_dir = h5f_dir
        self.calibration_list = get_points(self.parent().point_widget.win_width, self.parent().point_widget.win_height,
                                           calibrate_points[0], calibrate_points[1])
        self.list_length = len(self.calibration_list)

    def run(self):
        self.parent().point_widget.clear_point()
        for ii in range(self.list_length):

            self.parent().target_signal.emit(QPoint(self.calibration_list[ii][0], self.calibration_list[ii][1]))
            is_valid = False
            while not is_valid:
                self.parent().log('请用鼠标左键点击红点，同时保持对屏幕内光标的注视。')
                self.parent().click_valid = True
                while self.parent().click_valid:
                    time.sleep(1)
                target = self.parent().target
                # pos = self.parent().mousePos
                img = self.parent().predict.captrue()
                # self.parent().click_valid = False
                self.parent().log('摄像头捕捉完成。')
                is_valid, H, W, L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = \
                    self.parent().predict.align(img)

                if not is_valid:
                    self.parent().log('摄像头未捕捉到头像！请再次点击。')

            self.parent().log('摄像头捕捉到头像！')
            data = self.parent().predict.segment(img, H, W,
                                                 L_face, R_face, B_face, T_face,
                                                 L_eye1, R_eye1, B_eye1, T_eye1,
                                                 L_eye2, R_eye2, B_eye2, T_eye2,
                                                 save_h5f=True, target=target)
            self.parent().log('第 ' + str(ii + 1) + '/' + str(self.list_length) + ' 组校准数据准备完成！')
        self.parent().point_widget.clear_point()
        self.parent().log('所有校准数据准备完成！')

        self.parent().calibration_button.setEnabled(True)
        self.parent().test_button.setEnabled(True)
        self.parent().calibrate_flag = True


class TestThread(QThread):

    def __init__(self, parent=None):
        super(TestThread, self).__init__(parent)

    def run(self):
        self.parent().point_widget.clear_point()
        self.parent().predict.prepare_supporting_data()
        is_valid = False
        while not is_valid:
            self.parent().log('请用在窗口空白处点击鼠标左键，同时保持对屏幕内点击位置的注视。')
            self.parent().click_valid = True
            while self.parent().click_valid:
                time.sleep(1)
            target = self.parent().target
            pos = self.parent().mousePos
            img = self.parent().predict.captrue()

            # self.parent().click_valid = False
            self.parent().log('摄像头捕捉完成。')

            is_valid, H, W, L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = \
                self.parent().predict.align(img)

            if not is_valid:
                self.parent().log('摄像头未捕捉到头像！请再次点击。')

        self.parent().log('摄像头捕捉到头像！')
        self.parent().target_signal.emit(QPoint(pos[0], pos[1]))
        data = self.parent().predict.segment(img, H, W,
                                             L_face, R_face, B_face, T_face,
                                             L_eye1, R_eye1, B_eye1, T_eye1,
                                             L_eye2, R_eye2, B_eye2, T_eye2,
                                             save_h5f=False, target=target)

        predict = self.parent().predict.prediction(data)[0]
        gaze = [predict[0] - self.parent().x(), predict[1] - self.parent().y()]
        self.parent().log('预测完成！预测坐标：[{0:.4f},{1:.4f}]'.format(predict[0], predict[1]))
        if self.parent().point_widget.win.contains(QPoint(gaze[0], gaze[1])):
            self.parent().pred_signal.emit(QPoint(gaze[0], gaze[1]))
        dis = self.parent().predict.distance(predict, target)
        self.parent().log('预测误差={:.4f}cm.'.format(dis))
        self.parent().log('预测误差={:.4f}cm.'.format(dis))
        self.parent().calibration_button.setEnabled(True)
        self.parent().test_button.setEnabled(True)


class PointDisplay(QWidget):
    pred_signal = pyqtSignal(QPoint)
    target_signal = pyqtSignal(QPoint)

    def __init__(self, parent=None):

        super(PointDisplay, self).__init__(parent)
        self.setWindowTitle('注视点估计')
        self.resize(win_size[0], win_size[1])
        self.center()

        self.point_widget = PointWidget(self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFixedHeight(100)

        self.calibration_button = QPushButton('校准', self)
        self.test_button = QPushButton('测试', self)
        self.quit_button = QPushButton('退出', self)

        self.test_thread = TestThread(self)
        self.calibration_thread = CalibrationThread(self)

        self.target_signal.connect(self.point_widget.set_target_point)
        self.pred_signal.connect(self.point_widget.set_pred_point)

        self.calibration_button.clicked.connect(self.calibration)
        self.test_button.clicked.connect(self.test)
        self.quit_button.clicked.connect(self.quit)
        self.target = None

        vbox = QVBoxLayout()
        vbox.addWidget(self.point_widget)
        vbox.addWidget(self.text_edit)

        hbox = QHBoxLayout()
        hbox.addWidget(self.calibration_button)
        hbox.addWidget(self.test_button)
        hbox.addWidget(self.quit_button)

        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.calibration_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.log('开始加载模型：')
        self.predict = pog_pred(calibrate_data_dir)

        self.log('准备完毕！请点击下方按键开始校准或测试。')
        self.calibration_button.setEnabled(True)
        self.test_button.setEnabled(True)
        self.mousePos = None
        self.click_valid = False
        self.calibrate_flag = False
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.showMaximized()

    def log(self, text):
        self.text_edit.append(text)

    def log_accept(self, event):
        if self.log_signal is not None:
            self.text_edit.append(self.log_signal)

    def calibration(self):
        self.log('校准开始：')
        self.calibration_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.calibration_thread.start()
        pass

    def test(self):
        self.log('测试开始：')

        if self.calibrate_flag is not True:
            self.log('未收集校准数据，请先校准！')
        else:
            self.calibration_button.setEnabled(False)
            self.test_button.setEnabled(False)
            self.log('检测到校准数据，开始准备支持集特征嵌入！')
            self.test_thread.start()

    def quit(self):
        self.calibration_thread.quit()
        self.test_thread.quit()
        self.close()

    def mousePressEvent(self, event):
        if self.click_valid and event.button() == Qt.LeftButton:
            self.mousePos = [event.pos().x(), event.pos().y()]
            self.target = [event.pos().x() + self.x(), event.pos().y() + self.y()]
            self.log('鼠标坐标：' + str(self.target))
            self.click_valid = False


class pog_pred:

    def __init__(self, calibrate_data_dir=None):
        # super(pog_pred, self).__init__(parent)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = SFOModel()  # AFFNet.fine_tuned_model()
        if torch.cuda.device_count() > 1:
            self.generator = torch.nn.parallel.DataParallel(self.generator)
        self.generator.load_state_dict(torch.load('checkpoint/calibrate_model.pt'), strict=False)
        self.generator.to(self.device)

        predictor_model = 'model/shape_predictor_68_face_landmarks.dat'
        self.seg_detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
        self.seg_predictor = dlib.shape_predictor(predictor_model)

        # import face_alignment
        # self.seg_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
        #                                                flip_input=False)  # , device='cpu')

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if calibrate_data_dir is None:
            if not os.path.isdir('output'):
                os.mkdir('output')
                os.mkdir('output/data')
            calibrate_data_dir = 'output/data/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.hdf5'
        self.calibrate_data_dir = calibrate_data_dir
        self.calibration_features = []
        self.y_cs = []

    def captrue(self):
        reg, frame = self.cap.read()
        camera_frame = cv2.flip(frame, 1)  # 图片左右调换
        # img = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        return camera_frame

    def align(self, img, offset=1, ratio=1.7):
        img_size = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rects = self.seg_detector(img_gray, 0)

        is_valid = True
        if len(rects) == 0:
            is_valid = False

        if is_valid:
            preds = [[p.x, p.y] for p in self.seg_predictor(img, rects[0]).parts()]

            face_preds = preds[:]
            eye1_preds = preds[36: 42]
            eye2_preds = preds[42: 48]

            L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = seg_eyes(eye1_preds, eye2_preds, img_size
                                                                                      , offset=offset, ratio=ratio)
            L_face, R_face, B_face, T_face = seg_align(face_preds, img_size, offset=offset)  # , ratio=ratio)

        else:
            L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        return is_valid, img_size[0], img_size[
            1], L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2

    def segment(self, img, H, W, L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2,
                T_eye2, save_h5f=False, target=None):

        if target is None:
            target = [0., 0.]

        rect = [(R_face - L_face) / W, (T_face - B_face) / H, L_face / W, B_face / H,
                (R_eye2 - L_eye2) / W, (T_eye2 - B_eye2) / H, L_eye2 / W, B_eye2 / H,
                (R_eye1 - L_eye1) / W, (T_eye1 - B_eye1) / H, L_eye1 / W, B_eye1 / H]
        rect = np.array(rect)
        rect = rect.astype(np.float16)

        face_img = img[B_face: T_face, L_face: R_face]
        leftEye_img = img[B_eye2: T_eye2, L_eye2: R_eye2]
        rightEye_img = img[B_eye1: T_eye1, L_eye1: R_eye1]

        face_img = cv2.resize(face_img, (224, 224))

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255
        face_img = face_img.transpose(2, 0, 1)

        leftEye_img = cv2.resize(leftEye_img, (112, 112))
        leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        rightEye_img = cv2.resize(rightEye_img, (112, 112))
        rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
        rightEye_img = cv2.flip(rightEye_img, 1)
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        if save_h5f is True:
            self.save(face_img, leftEye_img, rightEye_img, rect, target)
        else:
            return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                    "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                    "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                    "rects": torch.from_numpy(rect).type(torch.FloatTensor),
                    }

    def save(self, face_img, leftEye_img, rightEye_img, rect, target):
        with h5py.File(self.calibrate_data_dir, 'a') as h5f:
            if list(h5f.keys()) is None:
                frame_name = str(1)
            else:
                frame_name = str(len(list(h5f.keys())) + 1)
            grp = h5f.create_group(frame_name)
            grp["faceImg"] = face_img.astype(np.float16)
            grp["leftEyeImg"] = leftEye_img.astype(np.float16)
            grp["rightEyeImg"] = rightEye_img.astype(np.float16)
            grp["rects"] = np.array(rect).astype(np.float16)
            grp["targets"] = np.array(target).astype(np.float16)

    def prediction(self, data):
        with torch.no_grad():
            data["faceImg"] = data["faceImg"].unsqueeze(0).to(self.device)
            data["leftEyeImg"] = data["leftEyeImg"].unsqueeze(0).to(self.device)
            data['rightEyeImg'] = data['rightEyeImg'].unsqueeze(0).to(self.device)
            data['rects'] = data['rects'].unsqueeze(0).to(self.device)
            # data['targets'] = data['targets'].unsqueeze(0).to(self.device)
            gazes = self.generator(data["leftEyeImg"], data["rightEyeImg"],
                                   data['faceImg'], data['rects'],
                                   self.calibration_features, self.y_cs)
            gaze = gazes[0].detach().cpu()
            return gaze.numpy().tolist()

    def prepare_supporting_data(self):

        calibrate_dataset = TrainDataset(self.calibrate_data_dir)
        calibrate_dataloader = DataLoader(calibrate_dataset, batch_size=1, shuffle=False)
        self.calibration_features = []
        self.y_cs = []

        for ii, data in enumerate(calibrate_dataloader):
            self.generator.zero_grad()
            data["faceImg"] = data["faceImg"].to(self.device)
            data["leftEyeImg"] = data["leftEyeImg"].to(self.device)
            data['rightEyeImg'] = data['rightEyeImg'].to(self.device)
            data['rects'] = data['rects'].to(self.device)
            targets = data['targets']

            feature = self.generator.feature_extractor(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'],
                                                       data['rects'])
            self.calibration_features.append(feature)
            self.y_cs.append(targets)

        self.y_cs = torch.stack(self.y_cs, dim=1).to(self.device)

    def distance(self, p1, p2):
        # 预先输入设备屏幕分辨率和屏幕尺寸参数

        scr_w = 1920
        scr_h = 1080

        cam_w = 34.416
        cam_h = 19.359

        return math.sqrt(
            (cam_w / scr_w) ** 2 * (p1[0] - p2[0]) * (p1[0] - p2[0]) + (cam_h / scr_h) ** 2 * (p1[1] - p2[1]) * (
                    p1[1] - p2[1]))


def main():
    app = QApplication(sys.argv)
    ex = PointDisplay()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
