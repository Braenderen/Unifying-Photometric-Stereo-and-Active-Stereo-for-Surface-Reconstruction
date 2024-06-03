# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import imageio
import glob
import rtde_receive
import rtde_control
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
import open3d as o3d
from rotpy.system import SpinSystem
from rotpy.camera import CameraList
from scipy.spatial.transform import Rotation as R
import copy
import math
import time
import json
from scipy.spatial.transform import Rotation as R
import os
import csv
#import shutil

def hom2axan(hom):
    # calculate the angle axis from the homogenous matrix
    rotm = hom[0:3, 0:3]
    axan = R.from_matrix(rotm).as_rotvec()
    return axan

def upscale_image_to_size(image, new_width, new_height, placeholder=0):
    # Get the original image dimensions
    original_height, original_width = image.shape

    # Create an empty NumPy array with NaN values
    upscaled_image = np.full((new_height, new_width), placeholder)

    # Calculate the scaling factors for both dimensions
    x_scale = new_width / original_width
    y_scale = new_height / original_height

    # Create indices for resampling
    x_indices = np.arange(0, new_width)
    y_indices = np.arange(0, new_height)

    # Resample the original image to the new size
    for y in y_indices:
        for x in x_indices:
            source_x = int(x / x_scale)
            source_y = int(y / y_scale)
            upscaled_image[y, x] = image[source_y, source_x]

    return upscaled_image

def calculate_angle_axis(pos_obj, pos_tcp, rotation): # rot is in deg
    # take the first 3 elements of pos_obj and pos_tcp
    pos_obj = pos_obj[:3]
    pos_tcp = pos_tcp[:3]
    dir_vec = []
    for i in range(3):
        tmp = pos_tcp[i] - pos_obj[i]
        if tmp == 0:
            dir_vec.append(0.0000001)
        else:
            dir_vec.append(tmp)

    tmp_dist = math.sqrt(dir_vec[0]**2 + dir_vec[1]**2 + dir_vec[2]**2)
    rot_v = math.acos(dir_vec[2] / tmp_dist)
    rot_v = math.radians(90) + ((math.pi / 2) - rot_v)

    rot_h = math.atan2(dir_vec[1], dir_vec[0])

    rx = 0
    ry = -rot_v
    rz = rot_h

    rot_to_allign = [rz, ry, rx]
    rotm_to_allign = R.from_euler('ZYX', rot_to_allign, degrees=False)

    rot_about_z = [math.radians(rotation), 0, 0]
    rotm_about_z = R.from_euler('ZYX', rot_about_z, degrees=False)

    rotm_combined = rotm_to_allign * rotm_about_z
    angle_axis = hom2axan(rotm_combined.as_matrix())

    return angle_axis

def sample_points_on_circle(center_x, center_y, height,radius, num_points):
    # Calculate the angle between each sampled point
    angle_step = 0.7*2 * np.pi / num_points
    
    # Initialize an empty list to store the sampled points
    points = []
    
    # Sample points on the circle using the parametric equation
    for i in range(num_points):
        angle = i * angle_step-0.9*np.pi
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((x, y,height))
    
    return points


################################
# Main
################################
robotIP = "192.168.1.30"
savePath = "./data/real_objects/BuddhaDome/"
# Define path for loading calibration data
calibrationPath = "./CameraCalibration/calibMay11/"
moveSpeed = 0.5
moveAcceleration = 0.3

# Filename
filename = "testNewPose"


#########
# Setup
#########
rtde_r = rtde_receive.RTDEReceiveInterface(robotIP)
rtde_c = rtde_control.RTDEControlInterface(robotIP)
print("Connected to robot")

if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath + "images/"):
    os.makedirs(savePath + "images/")

# Start teach mode
# rtde_c.teachMode()
# pose = rtde_r.getActualTCPPose()
# print("Current pose: ", pose)
# exit()


#startConfiguration =[2.522757053375244, -1.5525153318988245, 1.5787968635559082, -1.6806319395648401, -1.6663396994220179, 0]
#print(startConfiguration)
#rtde_c.moveJ(startConfiguration, moveSpeed, moveAcceleration)

counter = 0
fileCounter = 0
image_data = []

# radius of the circle

#! delete?

# Define the object position (found by moving the robot to the desired position and reading the TCP pose)
objectPosition = [0.6063561094517514, -0.5384153582483283, -0.5724699597120233]  #Cross

# Get current pose and extract the light position
actual_position = rtde_r.getActualTCPPose()
LightPosition = actual_position[0:3]


#############
# Realsense
#############
# Create a RealSense pipeline
if False:
    pipeline = rs.pipeline()
    config = rs.config()

    # # Configure the pipeline to stream color and depth frames
    rgb_width = 1920
    rgb_height = 1080
    #config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, 30)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start the pipeline
    jsonObj = json.load(open("./highAccuracyCustomConfig.json"))
    json_string= str(jsonObj).replace("'", '\"')
    pipeline_profile = pipeline.start(config)

    #load in json file
    dev = pipeline_profile.get_device()
    advnc_mode = rs.rs400_advanced_mode(dev)
    advnc_mode.load_json(json_string)

    # Get sensors of realsense
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]
    color_sensor = device.query_sensors()[1]

    # Create a temporal filter
    temp_filter = rs.temporal_filter()

    time.sleep(1)


    print("Realsense up and running")
    #Turn on projector
    if not depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    filtered = temp_filter.process(depth_frame)
    depth_image = np.asanyarray(filtered.get_data())

    cv2.imwrite(savePath + 'depth.png', depth_image)
    np.save(savePath + 'depth.npy', depth_image)
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)


light_file = open(filename+"light.txt", "w")
lightPositions = []
baseFrameLightPositions = []

TransformationPoses = []

#load in transformation matrix
cam2base = np.load(calibrationPath+"/pg/T_cam2base.npy")

# Create base to camera transformation matrix
R_inv = cam2base[0:3, 0:3].T
t = -R_inv @ cam2base[0:3, 3]
t = t.reshape(3,1)
base2cam = np.zeros((4,4))
base2cam[0:3, 0:3] = R_inv
base2cam[0:3, 3] = t.reshape(3,)
base2cam[3, 3] = 1



################################
# Setup point gray camera
################################
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

counter = 0
fileCounter = 0
image_data = []
num = 0
robotPoseC = []
robotPoseJ = []

# startConfiguration = [2.522757053375244, -1.5525153318988245, 1.5787968635559082, -1.6806319395648401, -1.6663396994220179, 0]
# rtde_c.moveJ(startConfiguration, moveSpeed, moveAcceleration)

print("Starting image acquisition")

rtde_c.teachMode()
try:
    while True:
        try:
            if pgCamera is not None:
                # try:
                # image_cam = pgCamera.get_next_image(timeout=5)

                # pgImg = image_cam.deep_copy_image(image_cam)

                # image_cam.release()
                # pgTemp = pgImg.get_image_data()
                # pgTemp = np.frombuffer(pgTemp, dtype=np.uint8)
                image_cam = pgCamera.get_next_image(timeout=5)
                pgImg = image_cam.deep_copy_image(image_cam)

                image_cam.release()
                pgTemp = pgImg.get_image_data()
                dt = np.dtype(np.uint16)
                dt = dt.newbyteorder('>')
                pgTemp = np.frombuffer(pgTemp, dtype=dt)
                pgTemp = pgTemp.reshape((pgImg.get_height(), pgImg.get_width()))

                # resize image
                pgTemp2 = copy.deepcopy(pgTemp)
                pgTemp2 = (pgTemp2/256).astype(np.uint8)
                pgTemp2 = cv2.resize(pgTemp2, (640, 480))
                cv2.imshow('pg image', pgTemp2)
                # except:
                #     print("Error capturing image from PointGrey camera")
        except:
            print("Error capturing image from PointGrey camera")
            continue
        
        # Wait for key press, if no key is pressed for 50ms, continue
        k = cv2.waitKey(50)

        if k == 27 or num == 99: # wait for 'esc' to exit
            if num == 99:
                print("Maximum number of images reached!")
            if imageCaptured:
                file = open(savePath+'robotPoseC.csv', 'w+', newline ='')
                with file:     
                    write = csv.writer(file)
                    write.writerows(robotPoseC)
                file = open(savePath+'robotPoseJ.csv', 'w+', newline ='')
                with file:     
                    write = csv.writer(file)
                    write.writerows(robotPoseJ)
            break
        elif k == ord('s'): # wait for 's' key to save
            # os.system('spd-say "Moving robot to new position"')
            # currentPose = rtde_r.getActualTCPPose()

            # newAxis = calculate_angle_axis(objectPosition, currentPose, 0)
            # newAxis = newAxis.tolist()
            # configuration =  rtde_c.getInverseKinematics([currentPose[0], currentPose[1], currentPose[2], newAxis[0], newAxis[1], newAxis[2]])
            # print("Configuration: ", configuration)
            # configuration[5] = 0
            # rtde_c.endTeachMode()
            # rtde_c.moveJ(configuration, moveSpeed, moveAcceleration)
            # rtde_c.teachMode()

            # #input("Press Enter to continue...")
            # print("Press Enter to continue...")
            # cv2.waitKey(0)
            os.system('spd-say "Capturing images"')

            while True:
                # Wait for the next frame
                try:
                    if True:
                        image_cam = pgCamera.get_next_image(timeout=5)
                        pgImg = image_cam.deep_copy_image(image_cam)

                        image_cam.release()
                        pgTemp = pgImg.get_image_data()
                        dt = np.dtype(np.uint16)
                        dt = dt.newbyteorder('>')
                        pgTemp = np.frombuffer(pgTemp, dtype=dt)
                        pgTemp = pgTemp.reshape((pgImg.get_height(), pgImg.get_width()))

                        # image_cam = PgCamera.get_next_image(timeout=5)
                        # PgImage = image_cam.deep_copy_image(image_cam)
                        # image_cam.release()

                        # pgTemp = PgImage.get_image_data()
                        # pgTemp = np.frombuffer(pgTemp, dtype=np.uint8)
                        # pgTemp = pgTemp.reshape((PgImage.get_height(), PgImage.get_width()))
                        
                        image_data.append(pgTemp)  
                    counter += 1
                except:
                    
                    print("No image")
                    
                if counter > 10:
                    counter = 0
                    fileCounter += 1
                    #filename = "test" + str(fileCounter)
                    actual_pose = rtde_r.getActualTCPPose()

                    actual_orientation = actual_pose[3:7]
                    #LightPosition = actual_pose[0:3]

                    r = R.from_rotvec(actual_orientation)

                    rotationMatrix = r.as_matrix()
                    Tcp2Base = np.zeros((4,4))
                    Tcp2Base[0:3, 0:3] = rotationMatrix
                    Tcp2Base[0:3, 3] = np.array(actual_pose[0:3])
                    Tcp2Base[3, 3] = 1
                    Tcp2Cam = base2cam @ Tcp2Base

                    TransformationPoses.append(Tcp2Cam)

                    LightPosition = Tcp2Cam[0:3,3]

                    #convert object position to numpy 
                    objectPositionArray = np.array(objectPosition)

                    # Convert the light position from base frame to camera frame
                    LightPositionArray = np.array(LightPosition)
                    LightPositionCam = Tcp2Cam[0:3, 3]
                
                    #convert object position to camera frame
                    objectPositionCam = base2cam @ np.append(objectPositionArray, 1)

                    differenceVector = [(LightPositionCam[0] - objectPositionCam[0]), (LightPositionCam[1] - objectPositionCam[1]), (LightPositionCam[2] - objectPositionCam[2])]
                    lightNormalized = np.array(differenceVector) / np.linalg.norm(differenceVector)
                    lightPositions.append(lightNormalized)
            
                    # Average of the images
                    equal_fraction = 1.0 / (len(image_data))       
                    output = np.zeros_like(image_data[0]) 
                            
                    for img in image_data:
                        output = output +  img * equal_fraction              
                    #output = output.astype(np.uint16)  

                    image_data = []
                    
                    #cv2.imwrite(savePath +"images/" + str(fileCounter).zfill(2) + '.png', output)
                    plt.imsave(savePath + "images/" + str(fileCounter).zfill(2) + '.png', output, cmap='gray')
                    np.save(savePath + "images/" + str(fileCounter).zfill(2) + '.npy', output)
                    os.system('spd-say "Free drive enabled"')
                    break

            #HEEEEERRREEEE
            imageCaptured = True
            num += 1
            print("Image " + str(num).zfill(2) + " saved!")
            # Save images

            robotPoseC.append(rtde_r.getActualTCPPose()) # (x,y,z,rx,ry,rz)
            robotPoseJ.append(rtde_r.getActualQ())  #q1 q2 q3 q4 q5 q6

finally:
    if pgCamera is not None:
        pgCamera.end_acquisition()
        pgCamera.deinit_cam()
        pgCamera.release()
    
    # Disable freedrive
    rtde_c.endTeachMode()
    print("Teach mode stopped!")

    np.save(savePath + "lightPositions.npy", lightPositions)

    np.save(savePath + "TCP2CamPoses.npy", TransformationPoses)


    pgCamera.end_acquisition()
    pgCamera.deinit_cam()
    pgCamera.release()

    print("Done")



                    
            