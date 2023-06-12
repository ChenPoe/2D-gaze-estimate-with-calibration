import json
import os
import shutil
import time
from skimage import io
import cv2
import numpy as np
import h5py
import dlib
from seg_utils import seg_align, seg_eyes



def video_segment(img_dir, detector, predictor, offset=1, ratio=1.7):
    print(img_dir)
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)  # , device='cpu')
    input_img = io.imread(img_dir)
    img_size = input_img.shape

    img = cv2.imread(img_dir)
    img_size = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)

    is_valid = True
    if len(rects) == 0:
        is_valid = False

    if is_valid:
        preds = [[p.x, p.y] for p in predictor(img, rects[0]).parts()]

        face_preds = preds[:]
        eye1_preds = preds[36: 42]  # np.concatenate((preds[17: 22], preds[36: 42]), axis=0)
        eye2_preds = preds[42: 48]  # np.concatenate((preds[22: 27], preds[42: 48]), axis=0)

        L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = seg_eyes(eye1_preds, eye2_preds, img_size
                                                                                  , offset=offset, ratio=ratio)
        L_face, R_face, B_face, T_face = seg_align(face_preds, img_size, offset=offset)  # , ratio=ratio)

    else:
        L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    return is_valid, img_size[0], img_size[
        1], L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2


class prepare_h5py:

    def __init__(self, data_path, out_path='../temp', data_name='dataset.hdf5', if_train=False, json_path=None):
        self.data_path = data_path
        out_name = os.path.join(out_path, data_name)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        self.out_path = out_path
        self.data_names = os.listdir(self.data_path)
        '''self.dir_list = []
        for subject in subjects:
            self.dir_list.append(os.path.join(data_path, subject))'''
        self.begin_time = time.time()
        self.h5f = h5py.File(out_name, 'a')
        self.if_train = if_train
        self.json_path = json_path
        predictor_model = 'src/dlib_model/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
        self.predictor = dlib.shape_predictor(predictor_model)

    def prepare(self, data_path, frame_name, label_path=None):

        frame_dir = os.path.join(data_path, frame_name)

        is_valid, H, W, L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 \
            = video_segment(frame_dir, self.detector, self.predictor, offset=1, ratio=1.7)

        if is_valid:
            rect = [(R_face - L_face) / W, (T_face - B_face) / H, L_face / W, B_face / H,
                    (R_eye2 - L_eye2) / W, (T_eye2 - B_eye2) / H, L_eye2 / W, B_eye2 / H,
                    (R_eye1 - L_eye1) / W, (T_eye1 - B_eye1) / H, L_eye1 / W, B_eye1 / H]

            grp = self.h5f.create_group(frame_name)

            img = cv2.imread(frame_dir)
            img = np.array(img)

            # H = img.shape[0]
            # W = img.shape[1]
            # print(H, img.shape[0], W, img.shape[1])

            face_img = img[B_face: T_face, L_face: R_face]
            leftEye_img = img[B_eye2: T_eye2, L_eye2: R_eye2]
            rightEye_img = img[B_eye1: T_eye1, L_eye1: R_eye1]

            # cv2.imwrite('examples/pm/'+ frame_name, leftEye_img)

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

            grp["faceImg"] = face_img.astype(np.float16)
            grp["leftEyeImg"] = leftEye_img.astype(np.float16)
            grp["rightEyeImg"] = rightEye_img.astype(np.float16)
            grp["rects"] = np.array(rect).astype(np.float16)
            with open(self.json_path, 'r') as f:
                labels = json.load(f)
            target = np.array(labels[frame_name[:-4]]['Target'])
            grp["targets"] = target.astype(np.float16)

            finish_time = time.time()
            print(frame_name + ':' + 'Preparation completed! Time used = {:.2f} s'.format(
                finish_time - self.begin_time))

    def run(self):

        for frame_name in self.data_names:
            self.prepare(frame_name=frame_name, data_path=self.data_path)

        finish_time = time.time()
        print('All of preparations completed! Total time used = {:.2f} s'.format(
            finish_time - self.begin_time))

    def close(self):
        finish_time = time.time()
        print('All of predictions completed! Total time used = {:.2f} s'.format(
            finish_time - self.begin_time))
        shutil.rmtree(self.out_path)


if __name__ == "__main__":
    for i in range(1, 30):

        data_file = 'calibrate_dataset'
        data_path = data_file + '/' + str(i)
        temp_path = 'train_data'
        if not os.path.isdir(temp_path):
            os.mkdir(temp_path)
        data_name = 'data_' + str(i) + '.hdf5'
        json_read_path = data_file + '/labels.json'

        ph = prepare_h5py(data_path, temp_path, data_name, if_train=False, json_path=json_read_path)
        ph.run()
