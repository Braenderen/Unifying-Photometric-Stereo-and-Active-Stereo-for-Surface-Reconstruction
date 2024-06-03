import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import time
from generateImages import ImageGenerator
from segmentImages import ImageSegmenter
from depth_normal_fusion.dnfusion import dnFusion
from os.path import exists
from rps import RPS
import psutil
import copy
import random

#sys.path.insert(0, '../Segmentation/')
#sys.path.insert(0, '../sensorPaper/')
#sys.path.insert(0, '../images/')
 



scene = "./data/bunny_diffuse/Scene.xml"
savePath = "./data/bunny_diffuse/"


GENERATE_IMAGES = False
USE_META_SEGMENT = True
USE_SYNTHETIC_DATA = True

GENERATE_ONLY = False




class UnifiedPS:

    def __init__(self, scene, dataPath):
        
        self.dataPath = dataPath
        self.imageGenerator = ImageGenerator(scene, dataPath)
        self.imageSegmenter = ImageSegmenter(dataPath)
        self.rps = RPS()
        self.DNF = dnFusion()
        self.correctDepthImage = True

        # self.rps.load_mask(filename=dataPath+"/mask.png")
        # self.rps.load_lightnpy(filename=dataPath+"/lightPositions.npy")
        # self.rps.load_npyimages(foldername=dataPath+"/images/")
        self.mask = None
        self.sampled_mask = None
        self.rawMesh = None

        #self.METHOD = RPS.L2_SOLVER    # Least-squares
        #self.METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
        #self.METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
        #self.METHOD = RPS.RPCA_SOLVER    # Robust PCA
        self.METHOD = RPS.L2_NEAR_SOLVER    # L2 residual minimization with near light
        self.lambda_p = 0.1
        self.lambda_n = 0.9
        self.lambda_s = 0.1

        # Camera parameters
        self.vfov = 40.6105
        self.img_width = 1980
        self.img_height = 1200
        #self.mode = 'Adaptive'
        self.mode = 'CentralWithFB'
        self.sigma = 0.2

        self.DNF.setWeights(self.lambda_p, self.lambda_n, self.lambda_s)

        self.DNF.sigmaGauss = self.sigma

        self.DNF.setCameraParamsMitsuba(self.vfov, self.img_width, self.img_height)



    def solveDepth(self,Z_0,N_0):

        self.DNF.setCameraParamsMitsuba(self.vfov, self.img_width, self.img_height)

       
        #mask = self.DNF.erodedMask(Z_0)

        Z_original = copy.deepcopy(Z_0)

        # x, y, w, h = cv2.boundingRect(self.mask.astype(np.uint8))
        # borderpx = 10
        # x_temp = x - borderpx
        # y_temp = y - borderpx
        # w_temp = w + 2*borderpx
        # h_temp = h + 2*borderpx

        # if(x_temp < 0 or y_temp+h_temp > Z_0.shape[0] or x_temp+w_temp > Z_0.shape[1] or y_temp < 0):
        #     pass
        # else:
        #     x = x_temp
        #     y = y_temp
        #     w = w_temp
        #     h = h_temp

        RGB_0 = np.zeros(N_0.shape,dtype=np.uint8)

        #self.mask = self.erodedMask(self.mask)
        #self.mask = self.erodedMask(self.mask)
        #self.mask = self.erodedMask(self.mask)

        stacked_mask = np.stack((self.mask,self.mask,self.mask),axis=-1)


        if USE_SYNTHETIC_DATA:
            Z_0 = self.DNF.magnitudeToDepth(Z_0)
    
        Z_0 = Z_0*self.mask

        #displat depth map

        # plt.imshow(Z_0)
        # plt.title('Depth map')
        # plt.show()


        N_0 = N_0*stacked_mask

        #display normal map

        #psutil.disp_normalmap(normal=N_0, height=N_0.shape[0], width=N_0.shape[1])


        if USE_SYNTHETIC_DATA:
            #Z_0_sparse = cv2.resize(Z_0, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
            Z_0_sparse = Z_0
            

        #display downsampled depth map

        plt.imshow(Z_0)
        plt.title('Depth map')
        plt.show()



        #!TODO: Fix så man kan croppe billedet for hurtigere beregninger

        # Z_0 = Z_0[y:y+h, x:x+w]
        # N_0 = N_0[y:y+h, x:x+w, :]
        # self.mask = self.mask[y:y+h, x:x+w]

        # self.bb_x = x
        # self.bb_y = y
        # self.bb_w = w
        # self.bb_h = h
       
      
        self.stacked_mask = stacked_mask


        #!TODO: Fix så man kan sætte sparse depth map

        #show images

        # plt.imshow(Z_0)
        # plt.title('Depth map')
        # plt.show()

        # plt.imshow(N_0)
        # plt.title('Normal map')
        # plt.show()


        self.DNF.setNormalMap(N_0)
        self.DNF.setSparseDepthMap(Z_0)

        self.DNF.setDenseDepthMap(Z_0)
        self.DNF.setWeights(self.lambda_p, self.lambda_n, self.lambda_s)
        self.DNF.setMask(self.mask)
        
        self.DNF.sigmaGauss = self.sigma

        # Construct matrices
        timeit = True
        self.DNF.constructMu(timeit=timeit)

        crop = True
        if crop:
            # Crop the images to the mask size and add a border
            x, y, w, h = cv2.boundingRect(self.mask.astype(np.uint8))
            borderpx = 10
            x = x - borderpx
            y = y - borderpx
            w = w + 2*borderpx
            h = h + 2*borderpx

            # if(x_temp < 0 or y_temp+h_temp > Z_0.shape[0] or x_temp+w_temp > Z_0.shape[1] or y_temp < 0):
            #     pass

            self.DNF.crop(x, y, w, h)
        else:
            x = 0
            y = 0



        self.DNF.constructA(self.mode,self.sigma,timeit=timeit)

        # Construct b vector
        self.DNF.constructb(timeit=timeit)

        # Remove zero rows
        self.DNF.removeZeroRows()

        # Solve the least squares problem
        Z, istop, itn, r1norm = self.DNF.solve(timeit=timeit)

        #create numpy array with 0 of original Z size

        # Z_return = np.zeros(Z_original.shape)

        # #fill in the Z values

        # Z_return[y:y+h, x:x+w] = Z

        return Z, N_0, RGB_0, x, y


    def setMask(self):
        self.rps.load_mask(filename=self.dataPath + "mask.png")

        mask = cv2.imread(self.dataPath + "mask.png",0)
        mask[mask > 0] = 1

        #to numpy array

        mask = np.array(mask)

        self.mask = mask

    def solvePS(self):
        self.rps.solve(method=self.METHOD)
        self.rps.save_normalmap(filename=self.dataPath + "est_normal")    # Save the estimated normal map


    def setGroundTruth(self, normalPath):
        self.N_gt = psutil.load_normalmap_from_npy(filename=normalPath)
        self.N_gt = np.reshape(self.N_gt, (self.rps.height*self.rps.width, 3)) 
        #self.N_gt = self.N_gt*self.N_mask

    def loadData(self):

        if self.METHOD == RPS.L2_NEAR_SOLVER:
            #self.rps.load_baselight_npy(self.dataPath + "baseFrameLightPositions.npy")
            self.rps.load_light_transform_npy(self.dataPath + "TCP2CamPoses.npy")
            self.rps.load_depth_npy(self.dataPath + "depth.npy")

            #!TODO: check where cameraMatrix file is
            self.rps.load_camera_matrix("./mtx.npy")
        
        self.rps.load_lightnpy(self.dataPath + "lightPositions.npy")



        self.rps.load_npyimages(self.dataPath + "images/")

    def computeAngularError(self):

        if self.rps.NearNormals is not None:
            angular_err_near = psutil.evaluate_angular_error(self.N_gt, self.rps.NearNormals, self.rps.background_ind)
            angular_err = psutil.evaluate_angular_error(self.N_gt, self.rps.N, self.rps.background_ind)

            #binarize unconstrained mask

            uc_mask = self.rps.unconstrained_mask

            zero_mask = np.zeros(uc_mask.shape)

            zero_mask[uc_mask > 2] = 1

            #display mask


            plt.imshow(zero_mask)
            plt.title('Mask')
            plt.show()

            

            error_image = angular_err.reshape((self.rps.height,self.rps.width))*zero_mask

            error_image_near = angular_err_near.reshape((self.rps.height,self.rps.width))*zero_mask


            angular_err = error_image.flatten()
            angular_err_near = error_image_near.flatten()         

            angular_error_non_zero = angular_err[angular_err != 0]
            angular_error_near_non_zero = angular_err_near[angular_err_near != 0]

            mean_error = np.mean(angular_error_non_zero)
            mean_error_near = np.mean(angular_error_near_non_zero)

            print('Mean angular error original: ', mean_error)
            print('Mean angular error near: ', mean_error_near)


            #display error image for both normal maps

          
            #display error image

            fig, axs = plt.subplots(2)
            fig.suptitle('Error Images plot')


            axs[0].imshow(error_image)
            axs[1].imshow(error_image_near)
            
            plt.show()




            return angular_err, angular_err_near
        else:
            angular_err = psutil.evaluate_angular_error(self.N_gt, self.rps.N, self.rps.background_ind)
            
            angular_error_non_zero = angular_err[angular_err != 0]

            mean_error = np.mean(angular_error_non_zero)
            print('Angular error: ', mean_error)

            return angular_err

    def displayNormalMap(self):
        psutil.disp_normalmap(normal=self.rps.N, height=self.rps.height, width=self.rps.width)


    def loadDepth(self, depthPath):
        self.Z_0 = np.load(depthPath)

        #set all 0 values to 10

        #self.Z_0[self.Z_0 == 0] = 10


    def generateImages(self):
        self.imageGenerator.generate_images()


    def generateMask(self):

        self.mask = self.imageSegmenter.segment_image()
        cv2.imwrite(self.dataPath + "meta_mask.png", self.mask)
        #self.rps.load_mask(self.dataPath + "mask.png")

    def erodedMask(self,mask):
         # Creating kernel 
        kernel = np.ones((7, 7), np.uint8) 

        #erode mask
        erodedMask = cv2.erode(mask, kernel)

        return erodedMask
        
    def cloudError(self, GT,pcd):
        # Calculates RMS of NN between pcd and ground truth pcd

        if GT is None:
            raise ValueError('GT is empty')

        if pcd is None:
            raise ValueError('data is empty')
        
        dist = GT.compute_point_cloud_distance(pcd)
        print(dist)
        print(dist.shape)

        MSE = np.square(dist)
        RMSE = np.sqrt(MSE)

        return MSE, RMSE
    

    
    
    def meshFromDepth(self,depth,mask,filename,sample_factor=0.25):

        #create list of coordinates of mask and sample 1000 randomly


        if sample_factor < 1:
            if USE_SYNTHETIC_DATA:

                indicies = np.argwhere(mask > 0)

                #print first 10 indicies

                number_of_points = len(indicies)*sample_factor
                number_of_points = int(number_of_points)

                print(number_of_points)
                random_points = random.choices(indicies, k=number_of_points)

                empty_mask = np.zeros(mask.shape)

                for point in random_points:

                    empty_mask[point[0],point[1]] = 1

        else:
            empty_mask = mask


      
        

        # Create a point cloud and mesh from depth image

        print("Creating mesh")

        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth.astype(np.float32)*empty_mask.astype(np.float32)), o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        pcd.orient_normals_consistent_tangent_plane(100)

        #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            

      
        
        #o3d.visualization.draw_geometries([pcd])


        
        densities = np.asarray(densities)
        # density_colors = plt.get_cmap('plasma')(
        #     (densities - densities.min()) / (densities.max() - densities.min()))
        # density_colors = density_colors[:, :3]
        # density_mesh = o3d.geometry.TriangleMesh()
        # density_mesh.vertices = mesh.vertices
        # density_mesh.triangles = mesh.triangles
        # density_mesh.triangle_normals = mesh.triangle_normals
        # density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        # o3d.visualization.draw_geometries([density_mesh])


        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        #o3d.visualization.draw_geometries([mesh])
        
        
        o3d.io.write_triangle_mesh(savePath + filename + "_mesh.ply", mesh)

        #write raw pointcloud

        o3d.io.write_point_cloud(savePath + filename +  "_pointcloud.ply", pcd)

        print("Done creating mesh")

        return empty_mask

    def meshFrompcd(self,pcd,filename):
       
        pcd.orient_normals_consistent_tangent_plane(100)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
   
        densities = np.asarray(densities)
   

        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        #o3d.visualization.draw_geometries([mesh])
        
        
        o3d.io.write_triangle_mesh(savePath + filename + "_mesh.ply", mesh)

        #write raw pointcloud

        o3d.io.write_point_cloud(savePath + filename +  "_pointcloud.ply", pcd)

        print("Done creating mesh from depth normal")




if __name__ == "__main__":

  
    ps = UnifiedPS(scene, savePath)

    if GENERATE_IMAGES:
        ps.generateImages()

    if GENERATE_ONLY:
        exit()
        


    

    
    if USE_META_SEGMENT:

        print("Before segmenting")

        if not exists(savePath + "meta_mask.png"):
            ps.generateMask()

        print("After segmenting")
            
        ps.rps.load_mask(filename=savePath + "meta_mask.png")

        tempMask = cv2.imread(savePath + "meta_mask.png",0)

        print(tempMask.shape)

        #to numpy array

        tempMask = np.array(tempMask)

        #binary mask

        tempMask[tempMask > 0] = 1

        ps.mask = tempMask

    else:
        ps.setMask()

    ps.loadData()
    ps.loadDepth(savePath + "depth.npy")

 ##start of shit
    if USE_SYNTHETIC_DATA:
        Z_0 = ps.DNF.magnitudeToDepth(ps.Z_0)
    
    ps.mask = ps.erodedMask(ps.mask)
    ps.mask = ps.erodedMask(ps.mask)
    ps.mask = ps.erodedMask(ps.mask)

    sampled_mask = ps.meshFromDepth(Z_0,ps.mask,"raw_bunny", sample_factor=0.25)

    ps.sampled_mask = sampled_mask

    #write raw mesh to file




    ps.setGroundTruth(savePath + "trueNormals.npy")
    ps.solvePS()


    #display normal map

    #ps.displayNormalMap()

   

    #display ground truth normal map

    #psutil.disp_normalmap(normal=ps.N_gt, height=ps.rps.height, width=ps.rps.width)


    print("Angular error")

    #print mean angular error
    # angular_error = ps.computeAngularError()

    # angular_error_copy = copy.deepcopy(angular_error)
   
    # #take part which is not 0

    # #reshape error vector to image

    # error, error_near  = angular_error_copy

    # error_image = error.reshape((ps.rps.height,ps.rps.width))
    # error_image_near = error_near.reshape((ps.rps.height,ps.rps.width))

    # #display error image

    # plt.imshow(error_image)
    # plt.title('Angular error')
    # plt.show()

    

    N_image = ps.rps.NearNormals.reshape((ps.rps.height,ps.rps.width,3))


    ps.Z_0 = ps.Z_0*ps.sampled_mask
    Z, N_0, RGB_0, bb_x, bb_y  = ps.solveDepth(ps.Z_0,N_image)



    #ps.setGroundTruth(savePath + "trueNormals.npy")

  
    #errode mask again
    # for i in range(2):
    #     ps.mask = ps.DNF.erodedMask(Z)
    #     Z = ps.mask*Z

    stacked_mask = np.stack((ps.mask,ps.mask,ps.mask),axis=-1)

    #make empty numpy array with same size as original image size

    Z_image = np.zeros((ps.img_height,ps.img_width))

    #make vector of normals corresponding to non zero z values


    print("Shapes")
    non_zero = np.where(Z > 0)
   
    N_where_Z = N_0[non_zero]

    N_z = np.asarray(N_where_Z)

    print(N_z.shape)



    #reshape to 1D array



    #fill in the Z values

    Z_image[bb_y:bb_y+Z.shape[0], bb_x:bb_x+Z.shape[1]] = Z

    Z_image = Z_image*ps.mask

   
    Z_0_image = np.zeros((ps.img_height,ps.img_width))

    Z_0_image[bb_y:bb_y+Z.shape[0], bb_x:bb_x+Z.shape[1]] = ps.DNF.Z_0Dense

    plt.imshow(Z_image)
    plt.title('Reconstructed Depth map')
    plt.show()
    

    #Z_0_image = o3d.geometry.Image(ps.DNF.Z_0Dense.astype(np.float32)*Z_0_mask)

    Z0_image = o3d.geometry.Image(Z_0_image.astype(np.float32))

    

    #color_image = o3d.geometry.Image(RGB_0.astype(np.float32)*stacked_mask_N0.astype(np.float32))

    #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,Z0_image)

    #create point cloud from rgbd image

    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(Z0_image, o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)

    #change color of point cloud
    pcd.paint_uniform_color([125/255,125/255,125/255])

    #rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,o3d.geometry.Image(Z_image.astype(np.float32)))
    #pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)
   
    pcd2 = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(Z_image.astype(np.float32)), o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)
    pcd2.paint_uniform_color([124/255,255/255,0])

    camera_matrix = np.array([[ps.DNF.f_x, 0, ps.DNF.c_x], [0, ps.DNF.f_y, ps.DNF.c_y], [0, 0, 1]])
    
    image_coordinates = cv2.projectPoints(pcd.points, np.zeros((3,1)), np.zeros((3,1)), camera_matrix, np.zeros((1,5)))[0]


    pcd2.normals = o3d.utility.Vector3dVector(N_z)
    pcd.normals = o3d.utility.Vector3dVector(N_z)

    o3d.visualization.draw_geometries([pcd,pcd2])

    #write point cloud to file

    
    ps.meshFrompcd(pcd,"reconstructed_bunny")
    ps.meshFromDepth(Z_0_image,ps.sampled_mask,"original_bunny_z0", sample_factor=1)
    #ps.meshFrompcd(pcd2,"original_bunny")

    # MSE, RMSE = ps.cloudError(pcd,pcd2)
    # print('MSE: ', MSE)
    # print('RMSE: ', RMSE)

    print("Done")


    
