import pyrealsense2.pyrealsense2 as rs
from multiprocessing import Process, Queue, Value
import numpy as np
import cv2
import time
import imutils
import os

from flask import Response
from flask import Flask
from flask import render_template
from flask import request

def save_img(img1, img2, img3):
  path = 'out/'
  t = time.time()
  cv2.imwrite('{}{}_i.png'.format(path, t), img1)
  cv2.imwrite('{}{}_c.png'.format(path, t), img2)
  cv2.imwrite('{}{}_d.png'.format(path, t), img3)
  print('Saved {}'.format(t))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  global save
  if request.method == 'POST':
    save.value = 1
  return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
  global img_q, save
  frame = None
  while True:
    if not img_q.empty():
      frame = img_q.get_nowait()
    if frame is None: continue
    (flag, encodedImage) = cv2.imencode(".jpg", frame)
    if not flag:
      continue   
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
          bytearray(encodedImage) + b'\r\n')

def capture():
  global img_q, save
  w, h = 640, 480

  # Configure depth and color streams
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)

  # Start streaming
  pipeline.start(config)
  pc = rs.pointcloud()

  decimate = rs.decimation_filter()
  decimate.set_option(rs.option.filter_magnitude, 2 ** 1)
  prepare_pc = 10

  thr = 150
  l = 10
  while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
        
    if not depth_frame or not color_frame:
      continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    dd = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_SUMMER)
    dd = cv2.normalize(dd, None, 0, 255, cv2.NORM_MINMAX)
    
    if prepare_pc > 0:
      prepare_pc -= 1
      points = pc.calculate(depth_frame)
      pc.map_to(color_frame)

    out = np.zeros((h, w, 4),np.uint8)

    depth = cv2.cvtColor(dd, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(depth, thr, 255, cv2.THRESH_BINARY_INV)
    ret, mask2 = cv2.threshold(depth, thr - l, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, mask2)

    out[:,:,:3] = cv2.bitwise_and(color_image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    out[:,:,3] = mask

    images = np.vstack((np.hstack((color_image, dd)),
                        np.hstack((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), out[:,:,:3]))))
    
    if not img_q.full():
      img_q.put_nowait(images)

    if save.value == 1:
      save_img(color_image, dd, depth_image)
      save.value = 0

img_q = Queue(1)
save = Value('i', 0)
p = Process(target=capture, args=())
p.start()
while img_q.empty():
  time.sleep(0.001)

app.run(host="0.0.0.0", port=8000, debug=True,
        threaded=True, use_reloader=False)
