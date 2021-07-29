import cv2
import os

table1 = [x.strip().split(' ') for x in open('/home/symao/open_ws/dso/build/image_list.txt','r').readlines()]
table2 = [x.strip().split(' ') for x in open('/home/symao/open_ws/dso/build/save_info.txt','r').readlines()]

tmap = {}
for a,b in table1:
    tmap[int(a)] = b

for a,b in table2:
    a = int(a)
    fread = tmap[a]
    fwrite = b
    cv2.imwrite(fwrite, cv2.resize(cv2.imread(fread),(640,480)))