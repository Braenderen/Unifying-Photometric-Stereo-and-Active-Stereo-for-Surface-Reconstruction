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

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.30")
rtde_c = rtde_control.RTDEControlInterface("192.168.1.30")

savePath = "./data/real_objects/newBalance/"
if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath + "images/"):
    os.makedirs(savePath + "images/")

print("Connected to robot")

# Define path for loading calibration data
calibrationPath = "./CameraCalibration/calibMay8/"

moveSpeed = 0.5
moveAcceleration = 0.3

startConfiguration =[2.522757053375244, -1.5525153318988245, 1.5787968635559082, -1.6806319395648401, -1.6663396994220179, 0]
#print(startConfiguration)
#rtde_c.moveJ(startConfiguration, moveSpeed, moveAcceleration)

counter = 0
fileCounter = 0
image_data = []

# radius of the circle
circle_r = 0.35

#! delete?
#first combination
#objectPosition = [0.35942440113561963, -0.18236104481758086, -0.524959060230167]

# r = 0.30
#z = 0.15
#0.7

# z = -0.15
# r = 0.35
#0.7

# Define the object position (found by moving the robot to the desired position and reading the TCP pose)
objectPosition = [0.35942440113561963, -0.18236104481758086, -0.524959060230167]

#objectPosition = [0.4757333297116558, -0.2546516578893908, -0.5647929056551847] # shoe position

# Get current pose and extract the light position
actual_position = rtde_r.getActualTCPPose()
LightPosition = actual_position[0:3]

# Define the circle position
circle_x = objectPosition[0]
circle_y = objectPosition[1]


# Define the light positions sampled on the circle
sampling_circles = []

# sample more points on another circle
circle_points = sample_points_on_circle(circle_x, circle_y,-0.15, 0.25, num_points=10)
sampling_circles.append(circle_points)

# sample more points on another circle
circle_points = sample_points_on_circle(circle_x, circle_y,0.35, 0.30, num_points=10)
sampling_circles.append(circle_points)

# Sample point on a circle
circle_points = sample_points_on_circle(circle_x, circle_y,0.2, 0.30, num_points=10)
sampling_circles.append(circle_points)

# sample more points on another circle
circle_points = sample_points_on_circle(circle_x, circle_y,0.05, 0.30, num_points=10)
sampling_circles.append(circle_points)

#! delete?
#print("Sampling circles shape: ", np.array(sampling_circles).shape)
#print(circle_points)


# Filename
filename = "testNewPose"

# Create a RealSense pipeline
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

align = False
if align:
    #Align the depth frame to color frame
    align_to = rs.stream.color
    align = rs.align(align_to)


time.sleep(1)


light_file = open(filename+"light.txt", "w")
lightPositions = []
baseFrameLightPositions = []

TransformationPoses = []



print("Realsense up and running")
#Turn on projector
if not depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1)


frames = pipeline.wait_for_frames()

if align:
    aligned_depth_frame = align.process(frames).get_depth_frame()
else:
    aligned_depth_frame = frames.get_depth_frame()

filtered = temp_filter.process(aligned_depth_frame)

depth_image = np.asanyarray(filtered.get_data())

fillAndUpscale = False
if fillAndUpscale:
    mask = depth_image == 0
    mask = mask.astype(np.uint8)
    plt.imshow(depth_image)
    plt.show()
    filled_depth_image = cv2.inpaint(depth_image,mask,3,cv2.INPAINT_TELEA)

    #resize depth image to rgb image size


    filled_depth_image = cv2.resize(filled_depth_image, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
    plt.imshow(filled_depth_image)
    plt.show()
    depth_image =filled_depth_image


cv2.imwrite(savePath + 'images/depth.png', depth_image)
np.save(savePath + 'depth.npy', depth_image)

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

if False:
    base2cam = np.linalg.inv(cam2base)

    np.save(savePath + "real_lightPositions.npy", lightPositions)
    depth_image_original = copy.deepcopy(depth_image)

    depth_image = o3d.geometry.Image(depth_image)

    realsenseIntrinsic = np.load("./realsenseIntrinsics.npy")
    #realsenseDistortion = np.load("./realsenseDistortion.npy")

    #realsenseIntrinsic = o3d.core.Tensor(realsenseIntrinsic)

    realsenseCameraMatrix = o3d.camera.PinholeCameraIntrinsic()
    realsenseCameraMatrix.intrinsic_matrix = realsenseIntrinsic
    realsenseCameraMatrix.width = 1280
    realsenseCameraMatrix.height = 720

    pcd = o3d.geometry.create_point_cloud_from_depth_image(depth_image,realsenseCameraMatrix,depth_scale=1)

    #read in camera matrix

    cameraMatrix = np.load("./mtx.npy")
    distCoeffs = np.load("./dist.npy")
    rotation = np.load("./R.npy")
    translation = np.load("./T.npy")

    intrinsic = o3d.core.Tensor(cameraMatrix)

    rotationMatrix = np.transpose(rotation)

    #make transformation matrix

    extrinsic = np.zeros((4,4))

    extrinsic[0:3, 0:3] = rotationMatrix

    #flatten translation vector

    translation = translation.flatten()

    translation = -rotationMatrix @ translation

    extrinsic[0:3, 3] = translation

    extrinsic[3, 3] = 1

    extrinsic = o3d.core.Tensor(extrinsic)

    pcd = pcd.transform(extrinsic)


    points = np.asarray(pcd.points)

    imagePoints = cv2.projectPoints(points, [0,0,0],[0,0,0],cameraMatrix, distCoeffs)

    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

################################
# Setup point gray camera
################################
# PgSystem = SpinSystem()
# PgCameras = CameraList.create_from_system(PgSystem, update_cams=True, update_interfaces=True)

# PgCamera = PgCameras.create_camera_by_index(0)

# PgCamera.init_cam()

# PgCamera.begin_acquisition()
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

print("Starting image acquisition")
for circle in sampling_circles:
    startConfiguration =[2.522757053375244, -1.5525153318988245, 1.5787968635559082, -1.6806319395648401, -1.6663396994220179, 0]
    rtde_c.moveJ(startConfiguration, moveSpeed, moveAcceleration)

    for point in circle:
        newAxis = calculate_angle_axis(objectPosition, point, 0)
        newAxis = newAxis.tolist()
        configuration =  rtde_c.getInverseKinematics([point[0], point[1], point[2], newAxis[0], newAxis[1], newAxis[2]])
        print("Configuration: ", configuration)
        configuration[5] = 0
        rtde_c.moveJ(configuration, moveSpeed, moveAcceleration)

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

                if False:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())

                    image_data.append(color_image)

                counter += 1

                # print("image received")
                # print(counter)

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
                    break
                        
              


np.save(savePath + "lightPositions.npy", lightPositions)

np.save(savePath + "TCP2CamPoses.npy", TransformationPoses)

print("Done")
pgCamera.end_acquisition()
pgCamera.deinit_cam()
pgCamera.release()
pipeline.stop()

startConfiguration =[2.522757053375244, -1.5525153318988245, 1.5787968635559082, -1.6806319395648401, -1.6663396994220179, 0]

rtde_c.moveJ(startConfiguration, moveSpeed, moveAcceleration)
