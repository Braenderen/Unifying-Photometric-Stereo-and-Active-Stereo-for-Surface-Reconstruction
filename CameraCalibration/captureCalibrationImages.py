import cv2
import pyrealsense2 as rs
import numpy as np
from rotpy.system import SpinSystem
from rotpy.camera import CameraList
import rtde_control
import rtde_receive
import csv
import os
import copy


def capture(robotIP = None, path = None, cameras = None, rsPipeline = None, pgCamera = None):
    # Setup robot coomunication
    if path is None:
        raise ValueError("No path provided")
    
    if cameras is None:
        raise ValueError("No cameras provided")

    if robotIP is None:
        raise ValueError("No robot IP provided")
    else:
        rtde_r = rtde_receive.RTDEReceiveInterface(robotIP)
        rtde_c = rtde_control.RTDEControlInterface(robotIP)
        
        # Enable freedrive
        rtde_c.teachMode()
        print("Freedrive enabled")

        num = 0
        robotPoseC = []
        robotPoseJ = []
        imageCaptured = False

    if True:
        ## Setup point gray
        PgSystem = SpinSystem()
        PgCameras = CameraList.create_from_system(PgSystem, update_cams=True, update_interfaces=True)

        pgCamera = PgCameras.create_camera_by_index(0)

        pgCamera.init_cam()
        # print('Node is readable:', pgCamera.camera_nodes.AcquisitionFrameRate.is_readable())
        # print('Node is writable:', pgCamera.camera_nodes.AcquisitionFrameRate.is_writable())
        # print("PG framerate: ", pgCamera.camera_nodes.AcquisitionFrameRate.get_node_value())
        # pgCamera.camera_nodes.AcquisitionFrameRateEnable.set_node_value(0)
        # print('Node is writable:', pgCamera.camera_nodes.AcquisitionFrameRate.is_writable())
        # pgCamera.camera_nodes.AcquisitionFrameRate.set_node_value(15)
        # print(pgCamera.camera_nodes.AcquisitionFrameRate.get_node_value())
        pgCamera.begin_acquisition()

   # Create directories
    if not os.path.exists(path):
        os.makedirs(path) 
    if 'rsIR1' in cameras:
        if not os.path.exists(path+'rsIR1/'):
            os.makedirs(path+'rsIR1/')
    if 'rsIR2' in cameras:
        if not os.path.exists(path+'rsIR2/'):
            os.makedirs(path+'rsIR2/')
    if 'rsColor' in cameras:
        if not os.path.exists(path+'rsColor/'):
            os.makedirs(path+'rsColor/')
    if 'pg' in cameras:
        if not os.path.exists(path+'pg/'):
            os.makedirs(path+'pg/')

    try:
        while True:
            # Get frames from RealSense
            if rsPipeline is not None: 
                frames = rsPipeline.wait_for_frames()

            if pgCamera is not None:
                try:
                    image_cam = pgCamera.get_next_image(timeout=5)

                    pgImg = image_cam.deep_copy_image(image_cam)

                    image_cam.release()
                    pgTemp = pgImg.get_image_data()
                    pgTemp = np.frombuffer(pgTemp, dtype=np.uint8)
                    pgTemp = pgTemp.reshape((pgImg.get_height(), pgImg.get_width()))
                    # resize image
                    pgSmall = copy.deepcopy(pgTemp)
                    pgSmall = cv2.resize(pgSmall, (640, 480))
                    cv2.imshow('pg image', pgSmall)
                except:
                    print("Error capturing image from PointGrey camera")
            
            if 'rsIR1' in cameras:
                left_frame = frames.get_infrared_frame(1)
                left_image = np.asanyarray(left_frame.get_data())
                leftTemp = cv2.resize(left_image, (640, 480))  
                cv2.imshow('Left IR', leftTemp)
            
            if 'rsIR2' in cameras:
                right_frame = frames.get_infrared_frame(2)
                right_image = np.asanyarray(right_frame.get_data())
                rightTemp = cv2.resize(right_image, (640, 480))  
                cv2.imshow('Right IR', rightTemp)
            
            if 'rsColor' in cameras:
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                colorTemp = cv2.resize(color_image, (640, 480))  
                cv2.imshow('RS Color', colorTemp)

            # Wait for key press, if no key is pressed for 50ms, continue
            k = cv2.waitKey(50)

            if k == 27 or num == 99: # wait for 'esc' to exit
                if num == 99:
                    print("Maximum number of images reached!")
                if imageCaptured:
                    file = open(path+'robotPoseC.csv', 'w+', newline ='')
                    with file:     
                        write = csv.writer(file)
                        write.writerows(robotPoseC)
                    file = open(path+'robotPoseJ.csv', 'w+', newline ='')
                    with file:     
                        write = csv.writer(file)
                        write.writerows(robotPoseJ)
                break
            elif k == ord('s'): # wait for 's' key to save
                imageCaptured = True
                print("Image " + str(num).zfill(2) + " saved!")
                # Save images
                if 'rsIR1' in cameras:
                    cv2.imwrite(path+'rsIR1/' + str(num).zfill(2) + '.png', left_image)
                if 'rsIR2' in cameras:
                    cv2.imwrite(path+'rsIR2/' + str(num).zfill(2) + '.png', right_image)
                if 'rsColor' in cameras:
                    cv2.imwrite(path+'rsColor/' + str(num).zfill(2) + '.png', color_image)
                if 'pg' in cameras:
                    cv2.imwrite(path+'pg/' + str(num).zfill(2) + '.png', pgTemp)

                num += 1

                robotPoseC.append(rtde_r.getActualTCPPose()) # (x,y,z,rx,ry,rz)
                robotPoseJ.append(rtde_r.getActualQ())  #q1 q2 q3 q4 q5 q6

    finally:
        if rsPipeline is not None:
            rsPipeline.stop()
        if pgCamera is not None:
            pgCamera.end_acquisition()
            pgCamera.deinit_cam()
            pgCamera.release()
        
        # Disable freedrive
        #rtde_c.endTeachMode()
        print("Teach mode stopped!")


def setupRS(lIR_res = None, lIR_fps = None, color_res = None, color_fps = None, rIR_res = None, rIR_fps = None):
    # RealSense camera configuration
    config = rs.config()

    # Left IR camera
    if lIR_res is not None:    
        # Check if resolution is valid
            # List of settigns
            # Y8 (Left and right imager IR)
            # 1280 x720 6, 15, 30
            # 848 x 480 6, 15, 30, 60, 90
            # 640 x 480 6, 15, 30, 60, 90
            # 640 x 360 6, 15, 30, 60, 90
            # 480 x 270 6, 15, 30, 60, 90
            # 424 x 240 6, 15, 30, 60, 90
        if lIR_res[0] not in [1280, 848, 640, 480, 424]:
            raise ValueError("Invalid resolution for left IR camera")
        if lIR_res[1] not in [720, 480, 360, 270, 240]:
            raise ValueError("Invalid resolution for left IR camera")
        
        # Check if fps is valid
        if lIR_fps not in [6, 15, 30, 60, 90]:
            raise ValueError("Invalid fps for left IR camera")
        
        # Check fps at 1280x720
        if lIR_res[0] == 1280 and lIR_res[1] == 720 and lIR_fps not in [6, 15, 30]:
            raise ValueError("Invalid fps for 1280x720 resolution")
        
        config.enable_stream(rs.stream.infrared, 1, lIR_res[0], lIR_res[1], rs.format.y8, lIR_fps)

    # Right IR camera
    if rIR_res is not None:
        # Check if resolution is valid
            # (Same settings as left IR camera)
        if rIR_res[0] not in [1280, 848, 640, 480, 424]:
            raise ValueError("Invalid resolution for right IR camera")
        if rIR_res[1] not in [720, 480, 360, 270, 240]:
            raise ValueError("Invalid resolution for right IR camera")
        
        # Check if fps is valid
        if rIR_fps not in [6, 15, 30, 60, 90]:
            raise ValueError("Invalid fps for right IR camera")
        
        # Check fps at 1280x720
        if rIR_res[0] == 1280 and rIR_res[1] == 720 and rIR_fps not in [6, 15, 30]:
            raise ValueError("Invalid fps for 1280x720 resolution")
        
        config.enable_stream(rs.stream.infrared, 2, rIR_res[0], rIR_res[1], rs.format.y8, rIR_fps)

    # RealSense Color camera
    if color_res is not None:
        # Check if resolution is valid
            # Color (ex rgb8)
            # 1920x1080 6, 15, 30
            # 1280x 720 6, 15, 30
            # 960 x 540 6, 15, 30, 60
            # 848 x 480 6, 15, 30, 60
            # 640 x 480 6, 15, 30, 60
            # 640 x 360 6, 15, 30, 60
            # 424 x 240 6, 15, 30, 60
            # 320 x 240 6, 30, 60
            # 320 x 180 6, 30, 60
        if color_res[0] not in [1920, 1280, 960, 848, 640, 424, 320]:
            raise ValueError("Invalid resolution for color camera")
        if color_res[1] not in [1080, 720, 540, 480, 360, 240, 180]:
            raise ValueError("Invalid resolution for color camera")
        
        # Check if fps is valid
        if color_fps not in [6, 15, 30, 60]:
            raise ValueError("Invalid fps for color camera")
        
        # Check specail case fps
        if color_res[0] == 320 and color_res[1] == 240 and color_fps not in [6, 30, 60]:
            raise ValueError("Invalid fps for 320x240 resolution")
        if color_res[0] == 320 and color_res[1] == 180 and color_fps not in [6, 30, 60]:
            raise ValueError("Invalid fps for 320x180 resolution")
        if color_res[0] == 1280 and color_res[1] == 720 and color_fps not in [6, 15, 30]:
            raise ValueError("Invalid fps for 1280x720 resolution")
        if color_res[0] == 1920 and color_res[1] == 1080 and color_fps not in [6, 15, 30]:
            raise ValueError("Invalid fps for 1920x1080 resolution")

        # setup RGB camera
        config.enable_stream(rs.stream.color, color_res[0], color_res[1], rs.format.bgr8, color_fps)
    
    return config

def setupPg():
    PgSystem = SpinSystem()
    PgCameras = CameraList.create_from_system(PgSystem, update_cams=True, update_interfaces=True)

    PgCamera = PgCameras.create_camera_by_index(0)

    PgCamera.init_cam()
    # print('Node is readable:', PgCamera.camera_nodes.AcquisitionFrameRate.is_readable())
    # print('Node is writable:', PgCamera.camera_nodes.AcquisitionFrameRate.is_writable())
    # print("PG framerate: ", PgCamera.camera_nodes.AcquisitionFrameRate.get_node_value())
    # PgCamera.camera_nodes.AcquisitionFrameRateEnable.set_node_value(0)
    # print('Node is writable:', PgCamera.camera_nodes.AcquisitionFrameRate.is_writable())
    # PgCamera.camera_nodes.AcquisitionFrameRate.set_node_value(15)
    # print(PgCamera.camera_nodes.AcquisitionFrameRate.get_node_value())
    PgCamera.begin_acquisition()

    return  

def rsDisableEmitter(profile, printInfo = False):
    # Disable IR emmiter
    device = profile.get_device()
    depth_sensor = device.query_sensors()[0]
    laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    laser_range = depth_sensor.get_option_range(rs.option.laser_power)
    set_laser = 0
    depth_sensor.set_option(rs.option.laser_power, set_laser)
    laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    if printInfo:
        print("Current laser power = ", laser_pwr)
        print("laser power range = " , laser_range.min , "~", laser_range.max)
        print("Laser power set to ", set_laser)

def rsSetExposure(profile, exposure, printInfo = False):
    raise ValueError("This function is not working properly")
    # Disable auto exposure
    device = profile.get_device()
    depth_sensor = device.query_sensors()[0]
    auto_exposure = depth_sensor.set_option(rs.option.enable_auto_exposure, False)
    if printInfo:
        print("Auto exposure: ", depth_sensor.get_option(rs.option.enable_auto_exposure))

        # Set exposure
        exposure = depth_sensor.get_option(rs.option.exposure)
        print("Current exposure = ", exposure)
        exposure_range = depth_sensor.get_option_range(rs.option.exposure)
        print("Exposure range = " , exposure_range.min , "~", exposure_range.max)
        set_exposure = 2143
        depth_sensor.set_option(rs.option.exposure, set_exposure)
        exposure = depth_sensor.get_option(rs.option.exposure)
        print("New exposure = ", exposure)

# pgPath='calibImgFeb13/pg/'

####################################################
# Main
####################################################
robotIP = "192.168.1.30"
path = './calibMay22/'
#raise(ValueError("Please provide a path"))
#cameras = ['rsIR1','pg']#['rsIR1', 'rsIR2', 'rsColor', 'pg']
cameras = ['pg']

# RealSense camera configuration
if 'rsIR1' in cameras:
    config = setupRS(lIR_res = (1280, 720), lIR_fps = 15, color_res = (1920, 1080), color_fps = 15)
    #
    ## Start realSense pipeline
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    #
    ## Disable IR emmiter
    rsDisableEmitter(profile, printInfo = True)

    # Set exposure
    # rsSetExposure(profile, 2143, printInfo = True)
else:
    pipeline = None


# Capture images
capture(robotIP, path, cameras, pipeline, None)

