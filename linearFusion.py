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
import shutil
import pyvista as pv

###########################
# Log start time
###########################
start_time = time.time()


###########################
#! Load config file
###########################
# -*- coding: utf-8 -*-
import yaml

# Define config path
if len(sys.argv) > 1:
    configPath = sys.argv[1]
else:
    configPath = "./test/configs/xfusion_difuse_15lights.yaml"

print(configPath)

with open(configPath, 'r') as stream:
    config = yaml.safe_load(stream)

if config is None:
    raise ValueError('Config loaded is empty')

# Load parameters based on config
SCENE_PATH = config['scene']
dataSetPath = config['dataPath']
resultsPath = config['resultsPath']

GENERATE_IMAGES = config['generateImages']
USE_META_SEGMENT = config['useMetaSegment']
USE_SYNTHETIC_DATA = config['useSyntheticData']
NUN_LIGHTS = int(config['numLights'])
USE_CACHED_DATA = config['useCachedData']
ADD_DEPTH_NOISE = config['addDepthNoise']
NOISE_SIGMA = config['noiseSigma']

# Copy the config file to the results folder
# create results folder if it does not exist
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)
shutil.copy(configPath, resultsPath)


#############################
#! END of config file
#############################

#############################
#! UnifiedPS CLASS
#############################
class UnifiedPS:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.imageGenerator = None
        self.imageSegmenter = None
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
        self.lambda_p = float(config['lambda_p'])
        self.lambda_n = float(config['lambda_n'])
        self.lambda_s = float(config['lambda_s'])
        self.lambda_b = float(config['lambda_b'])
        # self.lambda_p = 0.1
        # self.lambda_n = 0.9
        # self.lambda_s = 0.1



        # Camera parameters
        self.vfov = 40.6105
        self.img_width = 1980
        self.img_height = 1200

        # Set different parameters
        self.DNF.setCameraParamsMitsuba(self.vfov, self.img_width, self.img_height)
        #self.mode = 'Adaptive'
        self.mode = config['dnfmode']
        self.sigma = config['sigma']
        self.DNF.sigmaGauss = self.sigma
        self.DNF.setWeights(self.lambda_p, self.lambda_n, self.lambda_s)
        


    def solveDepth(self,Z_0,N_0,Z0dense=None):
        # Make copy of Z_0
        Z_0_copy = copy.deepcopy(Z_0)
        Z_0 = Z_0_copy

        self.DNF.setMask(self.mask)
        self.DNF.setNormalMap(N_0)
        self.DNF.setSparseDepthMap(Z_0)
        if Z0dense is not None:
            self.DNF.setDenseDepthMap(Z0dense)
        else:
            self.DNF.setDenseDepthMap(Z_0)
        self.DNF.setWeights(self.lambda_p, self.lambda_n, self.lambda_s)
        print('lambda_p: ', self.lambda_p)
        print('lambda_n: ', self.lambda_n)
        print('lambda_s: ', self.lambda_s)
        self.DNF.sigmaGauss = self.sigma
        timeit = True

        # Construct Mu
        self.DNF.constructMu(timeit=timeit)

        # Mask the different images
        self.DNF.applyMask(voidDepth=0)

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

        #stacked_mask = np.stack((self.mask,self.mask,self.mask),axis=-1)
        #Z_0 = Z_0*self.mask # Remove depth points out of mask

        #displat depth map
        # plt.imshow(Z_0)
        # plt.title('Depth map for depth normal fusion')
        # plt.show()

        #N_0 = N_0*stacked_mask # Remove normals out of mask

        #display normal map
        # psutil.disp_normalmap(normal=N_0, height=N_0.shape[0], width=N_0.shape[1])


        if USE_SYNTHETIC_DATA:
            #Z_0_sparse = cv2.resize(Z_0, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
            Z_0_sparse = Z_0
            

        # Construct A matrix
        self.DNF.constructA(self.mode,self.sigma,timeit=timeit)

        # Construct b vector
        self.DNF.constructb(timeit=timeit)

        # Remove points out of mask
        #self.DNF.maskA()

        # Remove zero rows
        self.DNF.removeZeroRows()

        print('solving ...')
        # Solve the least squares problem
        Z, istop, itn, r1norm = self.DNF.solve(timeit=timeit)

        # Uncrop the image
        if crop:
            Z_uncropped = np.zeros((self.img_height, self.img_width))
            Z_uncropped[y:y+h, x:x+w] = Z
            Z = Z_uncropped
            # N_0_uncropped = np.zeros((self.img_height, self.img_width, 3))
            # N_0_uncropped[y:y+h, x:x+w] = N_0
            # N_0 = N_0_uncropped

        return Z


    def setMask(self):
        raise NotImplementedError('Method not implemented')
        self.rps.load_mask(filename=self.dataPath + "mask.png")

        mask = cv2.imread(self.dataPath + "mask.png",0)
        if mask is None:
            raise ValueError('Mask is empty')

        mask[mask > 0] = 1

        #to numpy array
        mask = np.array(mask)

        # set mask
        self.mask = mask

    def solvePS(self):
        self.rps.solve(method=self.METHOD)
    
    def saveNormalmap(self, path, normalMap=None):
        self.rps.save_normalmap(filename=path + "est_normal", normal=normalMap)    # Save the estimated normal map

    def setGroundTruth(self, normalPath):
        self.N_gt = psutil.load_normalmap_from_npy(filename=normalPath)
        self.N_gt = np.reshape(self.N_gt, (self.rps.height*self.rps.width, 3)) 
        #self.N_gt = self.N_gt*self.N_mask

    def loadData(self):
        if self.METHOD == RPS.L2_NEAR_SOLVER:
            #self.rps.load_baselight_npy(self.dataPath + "baseFrameLightPositions.npy")
            self.rps.load_light_transform_npy(self.dataPath + "TCP2CamPoses.npy")
            self.rps.load_depth_npy(self.dataPath + "depth.npy")
            self.rps.meshFilePath = resultsPath + "coarse_reconstruction_mesh.ply"
            if USE_SYNTHETIC_DATA:
                self.rps.load_camera_matrix(self.dataPath + "Mitsuba_mtx.npy")
            else:
                self.rps.load_camera_matrix(self.dataPath + "mtx.npy")

            #self.rps.load_camera_matrix(self.dataPath + "_mtx.npy")
        self.rps.load_lightnpy(self.dataPath + "lightPositions.npy")
        self.rps.load_npyimages(self.dataPath + "images/") #!TODO: speed up

    def computeAngularError(self):
        if self.rps.NearNormals is not None:
            angular_err_near = psutil.evaluate_angular_error(self.N_gt, self.rps.NearNormals, self.rps.background_ind)
            now = time.time()
            self.rps._solve_l2()
            print("Time to solve SBL: ", time.time()-now)
            angular_err = psutil.evaluate_angular_error(self.N_gt, self.rps.N, self.rps.background_ind)

            #binarize unconstrained mask

            uc_mask = self.rps.unconstrained_mask

            zero_mask = np.zeros(uc_mask.shape)

            zero_mask[uc_mask > 2] = 1

            #display mask


            # plt.imshow(zero_mask)
            # plt.title('Mask of underconstrained points')
            # plt.show()

            

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
            raise ValueError('Not implemented')
            angular_err = psutil.evaluate_angular_error(self.N_gt, self.rps.N, self.rps.background_ind)
            
            angular_error_non_zero = angular_err[angular_err != 0]

            mean_error = np.mean(angular_error_non_zero)
            print('Angular error: ', mean_error)

            return angular_err

    def displayNormalMap(self, NormalMap):
        #psutil.disp_normalmap(normal=self.rps.N, height=self.rps.height, width=self.rps.width)
        psutil.disp_normalmap(normal=NormalMap, height=self.rps.height, width=self.rps.width)


    def loadDepth(self, depthPath):
        self.Z_0 = np.load(depthPath)

        #set all 0 values to 10

        #self.Z_0[self.Z_0 == 0] = 10


    def generateImages(self,sceneFile, savePath, numLights):
        self.imageGenerator = ImageGenerator(sceneFile, savePath, numLights)
        self.imageGenerator.generate_images()


    def generateMask(self):
        self.imageSegmenter = ImageSegmenter(self.dataPath)
        self.mask = self.imageSegmenter.segment_image()
        cv2.imwrite(self.dataPath + "meta_mask.png", self.mask)

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

                print("Number of depth points used for reconstruction: ", number_of_points)
                random_points = random.choices(indicies, k=number_of_points)

                empty_mask = np.zeros(mask.shape)

                for point in random_points:

                    empty_mask[point[0],point[1]] = 1
            else:
                empty_mask = mask
        else:
            empty_mask = mask

        # Create a point cloud and mesh from depth image
        print("Creating mesh")

        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth.astype(np.float32)*empty_mask.astype(np.float32)), o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        #print("Before orienting normals")
        #o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        #pcd.orient_normals_consistent_tangent_plane(100)
        pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

        #print("After orienting normals")
        #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        
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
        
        
        o3d.io.write_triangle_mesh(resultsPath + filename + "_mesh.ply", mesh)

        #write raw pointcloud

        o3d.io.write_point_cloud(resultsPath + filename +  "_pointcloud.ply", pcd)

        np.save(resultsPath + filename + "_depthMask.npy",empty_mask)

        print("Done creating mesh")

    def meshFrompcd(self,pcd,filename):

        #pcd.orient_normals_consistent_orient_normals_consistent_tangent_planetangent_plane(100)
        pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
   
        densities = np.asarray(densities)
   

        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        #o3d.visualization.draw_geometries([mesh])
        
        
        o3d.io.write_triangle_mesh(resultsPath + filename + "_mesh.ply", mesh)

        #write raw pointcloud

        o3d.io.write_point_cloud(resultsPath + filename +  "_pointcloud.ply", pcd)

        print("Done creating mesh from depth normal")

    def evaluateFusion(self, Z, Z_0, Z_GT=None):
        show = False
        if Z_GT is not None:
            # Save the results
            np.save(resultsPath + 'Zreconstructed', Z)
            np.save(resultsPath + 'Zreconstructed_masked', Z*self.mask)
            np.save(resultsPath + 'Z_0', Z_0)
            np.save(resultsPath + 'Z_GT', Z_GT)
            np.save(resultsPath + 'maskForEval', self.mask)

            # Calculate mean absolute error
            mae = np.mean(np.abs(Z_GT - Z))
            print("Mean absolute error: ", mae)
            np.save(resultsPath + 'mae', mae)

            # Calculate mean squared error
            mse = np.mean(np.square(Z_GT - Z))
            print("Mean squared error: ", mse)
            np.save(resultsPath + 'mse', mse)

            # Calculate root mean squared error
            rmse = np.sqrt(mse)
            print("Root mean squared error: ", rmse)
            np.save(resultsPath + 'rmse', rmse)

            # Calculate mean absolute error
            maeMasked = np.mean(np.abs(Z_GT*ps.mask - Z*ps.mask))
            print("Mean absolute error Masked: ", maeMasked)
            np.save(resultsPath + 'maeMasked', maeMasked)

            # Calculate mean squared error
            mseMasked = np.mean(np.square(Z_GT*ps.mask - Z*ps.mask))
            print("Mean squared error Masked: ", mseMasked)
            np.save(resultsPath + 'mseMasked', mseMasked)

            # Calculate root mean squared error
            rmseMasked = np.sqrt(mseMasked)
            print("Root mean squared error Masked: ", rmseMasked)
            np.save(resultsPath + 'rmseMasked', rmseMasked)


            # Print statistics
            print("Z_GT min: ", np.min(Z_GT))
            print("Z_GT max: ", np.max(Z_GT))
            print("Z_GT mean: ", np.mean(Z_GT))

            print("Z min: ", np.min(Z))
            print("Z max: ", np.max(Z))
            print("Z mean: ", np.mean(Z))

            if show:
                plt.subplot(1,3,1)
                plt.imshow(Z_GT*self.mask)
                plt.title('Ground truth depth map')
                plt.subplot(1,3,2)
                plt.imshow(Z)
                plt.title('Reconstructed depth map')
                plt.subplot(1,3,3)
                plt.imshow(Z*self.mask)
                plt.title('Reconstructed depth map masked')
                plt.show()

            # Show and save the difference between Z_0 and zImg
            np.save(resultsPath + 'Zdiff', (Z_GT - Z))
            plt.imsave(resultsPath + 'Zdiff.png', (Z_GT - Z))
            if show:
                plt.imshow((Z_GT - Z))
                plt.title('Zdiff')
                plt.show()

            # Show and save the difference between Z_0 and Z masked
            np.save(resultsPath + 'Zdiff_masked', (Z_GT - Z)*self.mask)
            plt.imsave(resultsPath + 'Zdiff_masked.png', (Z_GT - Z)*self.mask)
            if show:
                plt.imshow((Z_GT - Z)*self.mask)
                plt.title('Zdiff_masked')
                plt.show()
        else:
            print("\n\n\n\nWARNING: No ground truth depth map provided\n\n\n\n")

#############################
#! END OF UnifiedPS CLASS
#############################

#############################
#! MAIN
#############################
if __name__ == "__main__":
    print(round(time.time() - start_time,2),"Hello World!")

    ############################
    #! Initialize the pipeline
    ############################
    
    ps = UnifiedPS(dataSetPath)
    ps.rps.setSavePath(resultsPath)
    print(round(time.time() - start_time,2), "Pipeline initialized.")

    ############################
    #! Generate images
    ############################
    #if GENERATE_IMAGES:
    if False:
        ps.generateImages(SCENE_PATH, dataSetPath, NUN_LIGHTS)    
        print(round(time.time() - start_time,2), "Images generated.")
    else:
        print(round(time.time() - start_time,2), "*Skipped* image generation.")

    ############################
    #! Depth correction 
    #!  (Only synthetic data)
    ############################
    if USE_SYNTHETIC_DATA:
        ignoreCash_or_No_file = not USE_CACHED_DATA or not exists(dataSetPath + "depth.npy")
        if ignoreCash_or_No_file:
            tempDepth = np.load(dataSetPath + "trueDepth.npy")
            Z_0_corrected = ps.DNF.magnitudeToDepth(tempDepth)
            ps.Z_0 = Z_0_corrected
            np.save(dataSetPath + "depth.npy",Z_0_corrected)
            print(round(time.time() - start_time,2), "Depth converted.")
        else:
            print(round(time.time() - start_time,2), "*Skipped* depth correction.")


    ############################
    #! Load/generat mask
    ############################
    if USE_META_SEGMENT:
        if not USE_CACHED_DATA:
            ps.generateMask()
            print(round(time.time() - start_time,2), "Mask generated.")
        elif not exists(dataSetPath + "meta_mask.png"):
            ps.generateMask()
            print(round(time.time() - start_time,2), "Mask generated.")
        else:
            print(round(time.time() - start_time,2), "*Skipped* mask generation.")

        # Use rps method for loading mask, it binarizes the mask it self    
        ps.rps.load_mask(filename=dataSetPath + "meta_mask.png")

        # Load mask for binarysation
        tempMask = cv2.imread(dataSetPath + "meta_mask.png",0)

        # to numpy array
        tempMask = np.array(tempMask)

        # binary mask
        tempMask[tempMask > 0] = 1

        # set mask
        ps.mask = tempMask
    else:
        ps.setMask()
    if USE_SYNTHETIC_DATA:
        ps.mask = ps.erodedMask(ps.mask) # Erode mask to remove edge effects

    print(round(time.time() - start_time,2), "Mask loaded.")
    
    # Save mask to results
    cv2.imwrite(resultsPath + "mask.png", ps.mask)


    ############################
    #! Load data
    ############################
    ps.loadData()
    ps.loadDepth(dataSetPath + "depth.npy")
    print(round(time.time() - start_time,2), "Data loaded.")

    ############################
    #! Add depth noise
    ############################
    Z_0GT = copy.deepcopy(ps.Z_0)
    if ADD_DEPTH_NOISE:
        ps.Z_0 = ps.Z_0 + np.random.normal(0, NOISE_SIGMA, ps.Z_0.shape)*ps.mask


    # ############################
    # #! Generate coarse mesh
    # ############################
    ignoreCash_or_No_file = not USE_CACHED_DATA or not exists(resultsPath + "coarse_reconstruction_mesh.ply")
    if ignoreCash_or_No_file:
        ps.meshFromDepth(ps.Z_0,ps.mask,"coarse_reconstruction", sample_factor=0.25)
        print(round(time.time() - start_time,2), "Coarse reconstruction done.")
    else:
        print(round(time.time() - start_time,2), "*Skipped* Coarse reconstruction.")

    ps.sampled_mask = np.load(resultsPath + "coarse_reconstruction_depthMask.npy")


    ############################
    #! Solve Photometric Stereo
    ############################
    if False:
        methods = [RPS.L2_SOLVER,RPS.RPCA_SOLVER,RPS.L2_NEAR_SOLVER, RPS.RPCA_NEAR_SOLVER]
        for method in methods:
            ps.METHOD = method  
            if ps.METHOD == RPS.L2_NEAR_SOLVER:
                if not exists(resultsPath + "Near_est_normal.npy") or not USE_CACHED_DATA:
                    ps.solvePS()
                    ps.saveNormalmap(resultsPath+"Near_", ps.rps.NearNormals)
                    np.save(resultsPath + "unconstrainedMask.npy", ps.rps.unconstrained_mask)
                    print(round(time.time() - start_time,2), "Photometric stereo solved with method: Near")
                else:
                    print(round(time.time() - start_time,2), "*Skipped* Photometric stereo with Near.")
            elif ps.METHOD == RPS.RPCA_NEAR_SOLVER:
                if not exists(resultsPath + "NearRPCA_est_normal.npy") or not USE_CACHED_DATA:
                    ps.solvePS()
                    ps.saveNormalmap(resultsPath+"NearRPCA_", ps.rps.NearNormals)
                    np.save(resultsPath + "unconstrainedMaskRPCA.npy", ps.rps.unconstrained_mask)
                    print(round(time.time() - start_time,2), "Photometric stereo solved with method: Near RPCA")
                else:
                    print(round(time.time() - start_time,2), "*Skipped* Photometric stereo with Near.")
            elif ps.METHOD == RPS.L2_SOLVER:
                if not exists(resultsPath + "LS_est_normal.npy") or not USE_CACHED_DATA:
                    ps.solvePS()
                    ps.saveNormalmap(resultsPath+"LS_", ps.rps.N)
                    print(round(time.time() - start_time,2), "Photometric stereo solved with method: LS")
                else:
                    print(round(time.time() - start_time,2), "*Skipped* Photometric stereo with LS.")
            elif ps.METHOD == RPS.RPCA_SOLVER:
                if not exists(resultsPath + "RPCA_est_normal.npy") or not USE_CACHED_DATA:
                    ps.solvePS()
                    ps.saveNormalmap(resultsPath+"RPCA_", ps.rps.N)
                    print(round(time.time() - start_time,2), "Photometric stereo solved with method: RPCA")
                else:
                    print(round(time.time() - start_time,2), "*Skipped* Photometric stereo with RPCS.")
            elif ps.METHOD == RPS.SBL_SOLVER_MULTICORE:
                if not exists(resultsPath + "SBL_est_normal.npy") or not USE_CACHED_DATA:
                    ps.solvePS()
                    ps.saveNormalmap(resultsPath+"SBL_", ps.rps.N)
                    print(round(time.time() - start_time,2), "Photometric stereo solved with method: SBL")
                else:
                    print(round(time.time() - start_time,2), "*Skipped* Photometric stereo with SBL.")
            elif ps.METHOD == RPS.L1_SOLVER_MULTICORE:
                if not exists(resultsPath + "L1_est_normal.npy") or not USE_CACHED_DATA:
                    ps.solvePS()
                    ps.saveNormalmap(resultsPath+"L1_", ps.rps.N)
                    print(round(time.time() - start_time,2), "Photometric stereo solved with method: L1")
                else:
                    print(round(time.time() - start_time,2), "*Skipped* Photometric stereo with L1.")
            else:
                raise ValueError('Method not implemented')
        
        print(round(time.time() - start_time,2), "Photometric stereo solved.")
        os.system('spd-say "your program has finished"')

    # ps.rps.unconstrained_mask = np.load(resultsPath + "unconstrainedMask.npy")
    # N_image = np.load(resultsPath + "Near_est_normal.npy")
    # ps.rps.NearNormals = N_image.reshape((ps.rps.height*ps.rps.width, 3))


    ############################
    #! Solve Depth and Normal Fusion
    ############################
    #N_image = np.load(resultsPath + "Near_est_normal.npy")
    N_image = np.load(dataSetPath+ "trueNormals.npy")

    # # flip normals
    N_image[:,:,0] = -N_image[:,:,0]
    N_image[:,:,2] = -N_image[:,:,2]

    N_image[:,:,0] = -N_image[:,:,0]
    N_image[:,:,1] = -N_image[:,:,1]


    #ps.displayNormalMap(N_image)
    
    #N_image = ps.rps.NearNormals.reshape((ps.rps.height,ps.rps.width,3))
    #Z_0sparse = ps.Z_0
    Z_0sparse = ps.Z_0*ps.sampled_mask
    
    print(round(time.time() - start_time,2), "Solving depth and normal fusion.")
    #Z  = ps.solveDepth(Z_0sparse,N_image,Z_0GT)
    Z  = ps.solveDepth(Z_0sparse,N_image)
    print(round(time.time() - start_time,2), "Depth and normal fusion solved.")


    depth_mask_bilateral = ps.DNF.depthMask(Z_0sparse,ps.mask)

    Z_bilateral, mesh_bilateral = ps.dnf.bilateral_normal_integration(N_image, ps.mask, ps.DNF.it_limit,depth_mask = depth_mask_bilateral,depth_map = Z_0sparse,K = ps.rps.CameraMatrix,lambda1 = ps.lambda_b)

    #!TODO: Save results to results folder
    print("!\n!\n!\nSAVE RESULTS TO CACHE")


    ############################
    #! Errode mask
    ############################
    if False:
        # plt.subplot(1,2,1)
        # plt.imshow(ps.mask)
        # plt.title("Before erroding further")
        # test = copy.deepcopy(ps.mask)
        for i in range(6):
            ps.mask = ps.erodedMask(ps.mask)
            
        # plt.subplot(1,2,2)
        # plt.imshow(test + ps.mask)
        # plt.title("After erroding further")
        # plt.show()
    

    ############################
    #! Solve Depth and Normal Fusion
    ############################
    ps.DNF.evaluateErrorTermsAtEnd()
    ps.evaluateFusion(Z, Z_0sparse, Z_0GT)



    #make empty numpy array with same size as original image size

    Z_image = np.zeros((ps.img_height,ps.img_width))

    #make vector of normals corresponding to non zero z values

    non_zero = np.where(Z > 0)
   
    N_where_Z = N_image[non_zero]

    N_z = np.asarray(N_where_Z)

    #fill in the Z values
    #Z_image[bb_y:bb_y+Z.shape[0], bb_x:bb_x+Z.shape[1]] = Z

    #Z_image = Z_image*ps.mask
    Z_image = Z*ps.mask
    plt.imshow(Z_image)
    plt.title('Reconstructed Depth map after fusion')
    plt.show()

    plt.imshow(Z_bilateral*ps.mask)
    plt.title('Reconstructed Depth map after bilateral normal integration')
    plt.show()

    np.save(resultsPath + 'Z_bilateral', Z_bilateral*ps.mask)
    np.save(resultsPath + 'bilateral_mesk', mesh_bilateral)
    
    Z_GT = Z_0GT*ps.mask
   
    #Z_0_image = np.zeros((ps.img_height,ps.img_width))
    #Z_0_image[bb_y:bb_y+Z.shape[0], bb_x:bb_x+Z.shape[1]] = ps.DNF.Z_0Dense
    #Z_0_image = o3d.geometry.Image(ps.DNF.Z_0Dense.astype(np.float32)*Z_0_mask)
    Z_GT_image = o3d.geometry.Image(Z_GT.astype(np.float32))

    #stacked_mask = np.stack((ps.mask,ps.mask,ps.mask),axis=-1)
    #color_image = o3d.geometry.Image(RGB_0.astype(np.float32)*stacked_mask_N0.astype(np.float32))
    #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,Z0_image)
    
    #create point cloud from rgbd image
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)
    pcd_GT = o3d.geometry.PointCloud.create_from_depth_image(Z_GT_image, o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)

    #change color of point cloud
    pcd_GT.paint_uniform_color([125/255,125/255,125/255])

    #rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,o3d.geometry.Image(Z_image.astype(np.float32)))
    #pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)
   
    pcd_reconstructed = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(Z_image.astype(np.float32)), o3d.camera.PinholeCameraIntrinsic(ps.img_width, ps.img_height, ps.DNF.f_x, ps.DNF.f_y, ps.DNF.c_x, ps.DNF.c_y),depth_scale = 1.0)
    pcd_reconstructed.paint_uniform_color([124/255,255/255,0])

    camera_matrix = np.array([[ps.DNF.f_x, 0, ps.DNF.c_x], [0, ps.DNF.f_y, ps.DNF.c_y], [0, 0, 1]])
    
    #image_coordinates = cv2.projectPoints(np.asarray(pcd.points), np.zeros((3,1)), np.zeros((3,1)), camera_matrix, np.zeros((1,5)))[0]


    pcd_reconstructed.normals = o3d.utility.Vector3dVector(N_z)
    pcd_GT.normals = o3d.utility.Vector3dVector(N_z)
    print("pcd has normals: ", pcd_GT.has_normals())
    o3d.visualization.draw_geometries([pcd_GT,pcd_reconstructed])

  


