import os
import cv2
import numpy as np

data_dir = '/home/symao/data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb'
out_dir = '/home/symao/data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb_undistort'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

K = np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]])
D = np.array([0.2624,-0.9531,-0.0054,0.0026,1.1633])
K2 = np.array([[525.0,0,319.5],[0,525.0,239.5],[0,0,1]])

for f in os.listdir(data_dir):
    img = cv2.imread(os.path.join(data_dir, f))
    img2 = cv2.undistort(img, K, D, K2)
    cv2.imwrite(os.path.join(out_dir,f), img2)
    # cv2.imshow('img', np.hstack((img,img2)))
    # key = cv2.waitKey(0)
    # if key == 27:
    #     break

w,h = 640,480
fx = K2[0,0]/w
fy = K2[1,1]/h
cx = K2[0,2]/w
cy = K2[1,2]/h
open(os.path.join(out_dir,'../calib_dso.txt'),'w').write(
    '%f %f %f %f 0\n%d %d\n%f %f %f %f 0\n%d %d\n'%(fx,fy,cx,cy,w,h,fx,fy,cx,cy,w,h)
)