import os
import shutil

import numpy as np
import cv2


files_dir = r'D:\PythonProjects\Deformable-DETR-main\output\semantic_kakuten_hatsutameshi\val_visualize'
files = os.listdir(files_dir)

R = 0.
G = 0.
B = 0.
R_2 = 0.
G_2 = 0.
B_2 = 0.
N = 0

for file in files:
    img = cv2.imread(os.path.join(files_dir, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) / 255
    h, w, c = img.shape
    N += h*w

    R_t = img[:, :, 0]
    R += np.sum(R_t)
    R_2 += np.sum(np.power(R_t, 2.0))

    G_t = img[:, :, 1]
    G += np.sum(G_t)
    G_2 += np.sum(np.power(G_t, 2.0))

    B_t = img[:, :, 2]
    B += np.sum(B_t)
    B_2 += np.sum(np.power(B_t, 2.0))

R_mean = R/N
G_mean = G/N
B_mean = B/N

R_std = np.sqrt(R_2/N - R_mean*R_mean)
G_std = np.sqrt(G_2/N - G_mean*G_mean)
B_std = np.sqrt(B_2/N - B_mean*B_mean)

print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean, G_mean, B_mean))
print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))

