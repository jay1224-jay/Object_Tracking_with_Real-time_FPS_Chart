import threading
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import sys

from itertools import count  # 引入 count 函數

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

fps = 0

def getFPS():
    global fps
    return fps


# Set up tracker.
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[1]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

# Read video
video_path = "output.mp4"
video = cv2.VideoCapture(0)  # use camera as video source

if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()
 
bbox = (287, 23, 86, 320)

print("Select ROI and press ENTER")
bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)

Stop = False

def tracking():
    global fps, Stop
    if Stop:
        return
    ok, frame = video.read()
    timer = cv2.getTickCount()
    ok, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : Stop = True
    return fps


fig, ax = plt.subplots()
x_data, y_data = [], []
ln, = plt.plot([], [], 'r-', animated=True)

currentMax = 30
offset = 100

def init():
    ax.set_xlim(0, offset)
    ax.set_ylim(0, 60)
    return ln,

def update(frame):

    global currentMax

    currentFPS = 0
    if not Stop:
        currentFPS = tracking()
        x_data.append(frame)
        print("FPS:", currentFPS)
        y_data.append(currentFPS)

        ln.set_data(x_data, y_data)

        if frame >= offset:
            ax.set_xlim(frame - offset, frame)

    return ln,

print("start tracking...")



ani = FuncAnimation(fig, update, frames=count(), init_func=init, interval=10, blit=True)
plt.title('Real-time FPS Chart')
plt.xlabel('Time')
plt.ylabel('FPS')
plt.show()
