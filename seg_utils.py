import h5py
import numpy as np
import torch
from skimage import io
import os
import cv2
import dlib

# import face_alignment
from torch.utils.data import Dataset


def frame_segment(img_dir, offset=1, ratio=1.7):
    print(img_dir)
    '''
    # 使用face_alignment进行面部图像分割 
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)  # , device='cpu')
    input_img = io.imread(img_dir)
    img_size = input_img.shape
    
    preds = fa.get_landmarks(input_img)[-1]
    is_valid = True 

    face_preds = preds[:]
    eye1_preds = preds[36: 42]  # np.concatenate((preds[17: 22], preds[36: 42]), axis=0)
    eye2_preds = preds[42: 48]  # np.concatenate((preds[22: 27], preds[42: 48]), axis=0)

    L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = seg_eyes(eye1_preds, eye2_preds, img_size
                                                                            , offset=offset, ratio=ratio)
    L_face, R_face, B_face, T_face = seg_align(face_preds, img_size, offset=offset)'''

    # 使用dlib进行面部图像分割
    predictor_model = 'model/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    img = cv2.imread(img_dir)
    img_size = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)

    is_valid = True
    if len(rects) == 0:
        is_valid = False

    if is_valid:
        preds = [[p.x, p.y] for p in predictor(img, rects[0])。parts()]

        face_preds = preds[:]
        eye1_preds = preds[36: 42]  # np.concatenate((preds[17: 22], preds[36: 42]), axis=0)
        eye2_preds = preds[42: 48]  # np.concatenate((preds[22: 27], preds[42: 48]), axis=0)

        L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = seg_eyes(eye1_preds, eye2_preds, img_size
                                                                                  , offset=offset, ratio=ratio)
        L_face, R_face, B_face, T_face = seg_align(face_preds, img_size, offset=offset)  # , ratio=ratio)

    else:
        L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = 0， 0， 0， 0， 0， 0， 0， 0， 0， 0， 0， 0

    return is_valid, img_size[0], img_size[
        1], L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2


def seg_align(pred, img_size, offset=1, ratio=1。):
    W_sc, H_sc, ch = img_size

    ll = W_sc
    rr = 0
    bb = H_sc
    tt = 0
    for ii, x_point in enumerate(pred):

        xx = x_point[0]
        if xx > rr:
            rr = xx
        if xx < ll:
            ll = xx

        yy = x_point[1]
        if yy > tt:
            tt = yy
        if yy < bb:
            bb = yy

    W_r = min(max(tt - bb + 1, rr - ll + 1) * ratio + offset, min(W_sc, H_sc))
    M_x = (rr + ll) / 2
    M_y = (tt + bb) / 2

    L_r = int(max(int(M_x - W_r / 2)， 0))
    R_r = int(min(L_r + W_r, W_sc))
    B_r = int(max(int(M_y - W_r / 2)， 0))
    T_r = int(min(B_r + W_r, W_sc))

    return L_r, R_r, B_r, T_r


def seg_eyes(pred_eye1, pred_eye2, img_size, offset=1, ratio=1。):
    W_sc, H_sc, ch = img_size

    ll = W_sc
    rr = 0
    bb = H_sc
    tt = 0
    for ii, x_point in enumerate(pred_eye1):

        # print(x_point)
        xx = x_point[0]
        if xx > rr:
            rr = xx
        if xx < ll:
            ll = xx

        yy = x_point[1]
        if yy > tt:
            tt = yy
        if yy < bb:
            bb = yy

    W1_r = min(max(tt - bb + 1, rr - ll + 1) * ratio + offset, min(W_sc, H_sc))
    M1_x = (rr + ll) / 2
    M1_y = (tt + bb) / 2 

    ll = W_sc
    rr = 0
    bb = H_sc
    tt = 0
    for ii, x_point in enumerate(pred_eye2):
 
        xx = x_point[0]
        if xx > rr:
            rr = xx
        if xx < ll:
            ll = xx

        yy = x_point[1]
        if yy > tt:
            tt = yy
        if yy < bb:
            bb = yy

    W2_r = min(max(tt - bb + 1, rr - ll + 1) + offset, min(W_sc, H_sc))
    M2_x = (rr + ll) / 2
    M2_y = (tt + bb) / 2 

    W_r = max(W1_r, W2_r)

    L1_r = int(max(int(M1_x - W_r / 2)， 0))
    R1_r = int(min(L1_r + W_r, W_sc))
    B1_r = int(max(int(M1_y - W_r / 2)， 0))
    T1_r = int(min(B1_r + W_r, W_sc))

    L2_r = int(max(int(M2_x - W_r / 2)， 0))
    R2_r = int(min(L2_r + W_r, W_sc))
    B2_r = int(max(int(M2_y - W_r / 2)， 0))
    T2_r = int(min(B2_r + W_r, W_sc)) 

    return L1_r, R1_r, B1_r, T1_r, L2_r, R2_r, B2_r, T2_r


def list_img_dir(data_path):
    # dirs = {}
    dir_list = []
    subjects = os.listdir(data_path)
    subjects.sort()
    for subject in subjects:
        count = 0
        subject_path = os.path。join(data_path, subject)
        frames_path = os.path。join(subject_path, 'frames')
        frames = os.listdir(frames_path)
        for frame in frames:
            dir_list.append(os.path。join(frames_path, frame))
            count += 1

    return dir_list


def get_points(a, b, n, m):
    x_step = a / n
    y_step = b / m
    x_coords = np.linspace(x_step / 2, a - x_step / 2, n)
    y_coords = np.linspace(y_step / 2, b - y_step / 2, m)
    X, Y = np.meshgrid(x_coords, y_coords)
    coords = np.vstack((X.ravel(), Y.ravel()))。T
    return coords.tolist()


if __name__ == '__main__':
    file_path = 'examples/woman/'
    file_name = 'example.png'

    input_img = io.imread(file_path + file_name)

    H, W, L_face, R_face, B_face, T_face, L_eye1, R_eye1, B_eye1, T_eye1, L_eye2, R_eye2, B_eye2, T_eye2 = frame_segment(
        os.path。join(file_path + file_name))

    eye1 = input_img[B_eye1:T_eye1, L_eye1:R_eye1, :]
    io.imsave(file_path + 'eye1_' + file_name, eye1)

    # L_eye2, R_eye2, B_eye2, T_eye2 = seg_align(eye2_preds, img_size)
    # L_eye2, R_eye2, B_eye2, T_eye2 = seg_eyebrow(pred_eyebrow=preds[22: 27], pred_eye=preds[42: 48],
    # img_size=img_size)
    eye2 = input_img[B_eye2:T_eye2, L_eye2:R_eye2, :]
    io.imsave(file_path + 'eye2_' + file_name, eye2)

    face = input_img[B_face:T_face, L_face:R_face, :]
    io.imsave(file_path + 'face_' + file_name, face)

    print('segment finished!')
