import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
import imutils
import os

path = 'out/'
w, h = 1280, 720
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

accurate = False
prepare_pc = 10

tt = time.time()

try:
    while True:
        print(1 / (time.time() - tt))
        tt = time.time()
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if not accurate:
            dd = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_SUMMER)
            dd = cv2.normalize(dd, None, 0, 255, cv2.NORM_MINMAX)
        else:
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.holes_fill, 3)
            filtered_depth = spatial.process(depth_frame)
            dd = np.asanyarray(filtered_depth.get_data())
            dd = cv2.applyColorMap(cv2.convertScaleAbs(dd, alpha=0.03), cv2.COLORMAP_SUMMER)
            dd = cv2.normalize(dd, None, 0, 255, cv2.NORM_MINMAX)
            depth_frame = filtered_depth
        if prepare_pc > 0:
            prepare_pc -= 1
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)

        #depth_frame = decimate.process(depth_frame)
         
        
        images = np.vstack((color_image, dd))
        images = imutils.resize(images, height=900)

        cv2.imshow('RealSense', images)
        #cv2.imshow('RealSense2', dd)
        key = cv2.waitKey(1)
        if key == ord("s"):
            t = time.time()
            cv2.imwrite('{}{}_i.png'.format(path, t), color_image)
            cv2.imwrite('{}{}_c.png'.format(path, t), di)
            cv2.imwrite('{}{}_d.png'.format(path, t), depth_image)
            print('Saved {}'.format(t))
        if key == ord("p"):
            t = len(os.listdir('ply/')) + 1
            points = pc.calculate(depth_frame)
            pc.map_to(color_frame)
            points.export_to_ply('ply/{}.ply'.format(t), color_frame)
            print('Saved {}'.format(t))
        elif key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
