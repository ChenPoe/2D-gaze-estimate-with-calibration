# 2D-gaze-estimate-with-calibration


基于AFF-Net和少样本学习实现的注视点估计。

AFF-Net [1] 被用作从面部图像中特征提取的主干网络，删去最后的全连接层，对预处理好的图像数据推理得到256维的特征嵌入。

参考有监督少样本个性化方法[2]，校准网络则将待推理的查询图像特征嵌入和用于校准的支持图像特征嵌入及其对应的校准点坐标进行计算，得到支持集中每张图像对应的注视点坐标预测和权重。

最终，通过对预测的注视点坐标进行加权求和，得到查询图像对应的注视点估计坐标。`main_window.py` 提供了一个用于收集数据和测试算法的图形化用户界面，运行前请参照`model/shape_predictor.txt`下载人脸关键点检测模型。

#### 参考文献

[1] Bao, Yiwei, et al. "Adaptive feature fusion network for gaze tracking in mobile tablets." *2020 25th International Conference on Pattern Recognition (ICPR)*. IEEE, 2021.

[2] He, Junfeng, et al. "On-device few-shot personalization for real-time gaze estimation." *Proceedings of the IEEE/CVF international conference on computer vision workshops*. 2019.
