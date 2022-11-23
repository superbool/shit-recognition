import os
import time

path = '/data'
folder = f'{path}/{time.strftime("%Y%m%d")}'

if not os.path.exists(folder):
    os.mkdir(folder)

# 拍照
os.system(
    f'fswebcam -r 1920*1080 --delay 3 --skip 10 {folder}/{time.strftime("%Y%m%d%H%M%S")}.jpg'
)


#* * * * * python /data/camera.py