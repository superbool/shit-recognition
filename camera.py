import os
import time
from subprocess import call

path = '/data'
folder = path + '/' + time.strftime("%Y%m%d")

if not os.path.exists(folder):
    os.mkdir(folder)
# 拍照
# f'fswebcam -r 1920*1080 --delay 3 --skip 10 {folder}/{time.strftime("%Y%m%d%H%M%S")}.jpg'

def capture():
    dtime = time.strftime("%Y%m%d%H%M%S")
    call(["fswebcam", "-d", "/dev/video0", "-r", "1280*720", "--no-banner", "--delay", "3", "--skip", "10",
          "%s/%s.jpg" % (folder, dtime)])

capture()

# * * * * * python /data/camera.py
