from scipy.sparse import spdiags, csr_matrix, vstack
from scipy.sparse.linalg import cg
import numpy as np
from tqdm.auto import tqdm
import time
import pyvista as pv
#import psutil
import copy
import matplotlib.pyplot as plt
import random
import open3d as o3d
import cv2


estimated_depth = np.load('./data/NettoGuy/depth_0_True_10000.npy')

# plt.imshow(estimated_depth)
# plt.show()

# exit()



depth = np.load('./data/NettoGuy/depth.npy')/1000
mask = plt.imread('./data/NettoGuy/mask.png')

R =  np.load('./data/NettoGuy/R.npy')
T =  np.load('./data/NettoGuy/T.npy').flatten()



RS_intrinsics = np.load('./data/NettoGuy/rs_mtx.npy')
pg_intrinsics = np.load('./data/NettoGuy/pg_mtx.npy')
pg_dist = np.load('./data/NettoGuy/pg_dist.npy')

R_inv = R.T
t_inv = -R_inv @ T

rsToPg = np.eye(4)
rsToPg[:3, :3] = R_inv
rsToPg[:3, 3] = t_inv

Z_image = o3d.geometry.Image(depth.astype(np.float32))


pcd = o3d.geometry.PointCloud.create_from_depth_image(Z_image, o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], RS_intrinsics[0,0], RS_intrinsics[1,1], RS_intrinsics[0,2], RS_intrinsics[1,2]),depth_scale = 1.0)

pcd.transform(rsToPg)

#np.save('./data/NettoGuy/pcd.xyz', pcd.points)



object_points = np.asarray(pcd.points)



image_coordinates = cv2.projectPoints(object_points, np.zeros((3,1)), np.zeros((3,1)), pg_intrinsics, np.zeros((1,5)))[0]

empty = np.zeros((1200,1920))
for i in range(image_coordinates.shape[0]):
    x = int(image_coordinates[i,0,0])
    y = int(image_coordinates[i,0,1])
    if x >= 0 and x < 1920 and y >= 0 and y < 1200:
        empty[y,x] = object_points[i,2]


#set values above 1 to 0

eroded_mask = cv2.erode(mask, np.ones((9,9)), iterations = 1)

# plt.imshow(eroded_mask)
# plt.show()

empty[empty > 1] = 0

#set all 0 to nan

# plt.imshow(empty*mask)
# plt.show()

# plt.imshow(empty*eroded_mask)
# plt.show()

# exit()


empty[empty == 0] = np.nan

Z_image = o3d.geometry.Image(empty.astype(np.float32)*eroded_mask)


pcd = o3d.geometry.PointCloud.create_from_depth_image(Z_image, o3d.camera.PinholeCameraIntrinsic(empty.shape[1], empty.shape[0], pg_intrinsics[0,0], pg_intrinsics[1,1], pg_intrinsics[0,2], pg_intrinsics[1,2]),depth_scale = 1.0)


pcd_est = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(estimated_depth.astype(np.float32)), o3d.camera.PinholeCameraIntrinsic(estimated_depth.shape[1], estimated_depth.shape[0], pg_intrinsics[0,0], pg_intrinsics[1,1], pg_intrinsics[0,2], pg_intrinsics[1,2]),depth_scale = 1.0)
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

o3d.io.write_point_cloud("./data/NettoGuy/pointcloud.xyz", pcd)
o3d.io.write_point_cloud("./data/NettoGuy/pointcloud_est.xyz", pcd_est)



np.save('./data/NettoGuy/depth_pg.npy', empty*eroded_mask)
