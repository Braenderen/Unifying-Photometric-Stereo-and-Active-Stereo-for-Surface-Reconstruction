#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Photometric Stereo in Python
"""
__author__ = "Yasuyuki Matsushita <yasumat@ist.osaka-u.ac.jp>"
__version__ = "0.1.0"
__date__ = "11 May 2018"

import psutil
import rpsnumerics
import numpy as np
from sklearn.preprocessing import normalize
import time
import sys
sys.path.append("./build/")
from MyLib import NearSolver
from scipy.sparse import csc_array
import scipy.sparse
import matplotlib.pyplot as plt
import copy
import cv2 as cv
from os.path import exists

class RPS(object):
    """
    Robust Photometric Stereo class
    """
    # Choice of solution methods
    L2_SOLVER = 0   # Conventional least-squares
    L1_SOLVER = 1   # L1 residual minimization
    L1_SOLVER_MULTICORE = 2 # L1 residual minimization (multicore)
    SBL_SOLVER = 3  # Sparse Bayesian Learning
    SBL_SOLVER_MULTICORE = 4    # Sparse Bayesian Learning (multicore)
    RPCA_SOLVER = 5    # Robust PCA
    L2_NEAR_SOLVER = 6 # L2 residual minimization for near light source
    RPCA_NEAR_SOLVER=7 # Robust PCA for near light source

    def __init__(self):
        self.M = None   # measurement matrix in numpy array
        self.L = None   # light matrix in numpy array
        self.BL = None  # base frame light matrix in numpy array
        self.N = None   # surface normal matrix in numpy array
        self.NearNormals = None
        self.CameraMatrix = None # Camera matrix
        self.height = None  # image height
        self.width = None   # image width
        self.foreground_ind = None    # mask (indices of active pixel locations (rows of M))
        self.background_ind = None    # mask (indices of inactive pixel locations (rows of M))
        self.rawMesh = None
        self.meshFilePath = None
        self.savePath = None
    
    def setSavePath(self,savePath):
        if savePath is None:
            raise ValueError("savePath is None")
        self.savePath = savePath

    def load_lighttxt(self, filename=None):
        """
        Load light file specified by filename.
        The format of lights.txt should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.txt
        """
        self.L = psutil.load_lighttxt(filename)


    def load_light_transform_npy(self, filename=None):
        self.Light2CameraTransforms = psutil.load_light_transform_npy(filename)

    def load_lightnpy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.L = psutil.load_lightnpy(filename)

    def load_baselight_npy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.BL = psutil.load_baselight_npy(filename)

    def load_depth_npy(self, filename=None):
        """
        Load depth numpy array file specified by filename.
        The format of depth.npy should be
            depth1 depth2 depth3 ... depthf

        :param filename: filename of depth.npy
        """
        self.Depth = psutil.load_depth_npy(filename)

    def load_camera_matrix(self, filename=None):
        """
        Load camera matrix from a file specified by filename.
        The format of the camera matrix file should be
            f_x 0 c_x
            0 f_y c_y
            0 0 1

        :param filename: filename of the camera matrix
        """
        self.CameraMatrix = psutil.load_camera_matrix(filename)

    def load_cam2base_matrix(self, filename=None):
        """
        Load camera to base transformation matrix from a file specified by filename.
        The format of the camera to base transformation matrix file should be
            R11 R12 R13 T1
            R21 R22 R23 T2
            R31 R32 R33 T3
            0   0   0   1

        :param filename: filename of cam2base_matrix.npy
        :return: camera to base transformation matrix (4 \times 4)
        """
        self.Cam2BaseTransform = psutil.load_cam2base_matrix(filename)

    def load_images(self, foldername=None, ext=None):
        """
        Load images in the folder specified by the "foldername" that have extension "ext"
        :param foldername: foldername
        :param ext: file extension
        """
        self.M, self.height, self.width = psutil.load_images(foldername, ext)

    def load_npyimages(self, foldername=None):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        self.M, self.height, self.width = psutil.load_npyimages(foldername)

    def load_mask(self, filename=None):
        """
        Load mask image and set the mask indices
        In the mask image, pixels with zero intensity will be ignored.
        :param filename: filename of the mask image
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        mask = psutil.load_image(filename=filename)
        mask = mask.reshape((-1, 1))
        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        psutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None, normal=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        if normal is None:
            raise ValueError("normal is None")
        psutil.save_normalmap_as_npy(filename=filename, normal=normal, height=self.height, width=self.width)


   
    def solve(self, method=L2_SOLVER):
        if self.M is None:
            raise ValueError("Measurement M is None")
        # if self.L is None:
        #     raise ValueError("Light L is None")
        # if self.M.shape[1] != self.L.shape[1]:
        #     raise ValueError("Inconsistent dimensionality between M and L")

        if method == RPS.L2_SOLVER:
            self._solve_l2()
        elif method == RPS.L1_SOLVER:
            self._solve_l1()
        elif method == RPS.L1_SOLVER_MULTICORE:
            self._solve_l1_multicore()
        elif method == RPS.SBL_SOLVER:
            self._solve_sbl()
        elif method == RPS.SBL_SOLVER_MULTICORE:
            self._solve_sbl_multicore()
        elif method == RPS.RPCA_SOLVER:
            self._solve_rpca()
        elif method == RPS.L2_NEAR_SOLVER:

            # if self.BL is None:
            #     raise ValueError("Base light BL is None")
            if self.CameraMatrix is None:
                raise ValueError("Camera matrix is None")
            if self.Depth is None:
                raise ValueError("Depth is None")
            
          
            self._solve_near_l2()
        elif method == RPS.RPCA_NEAR_SOLVER:

            # if self.BL is None:
            #     raise ValueError("Base light BL is None")
            if self.CameraMatrix is None:
                raise ValueError("Camera matrix is None")
            if self.Depth is None:
                raise ValueError("Depth is None")
            
          
            self._solve_near_RPCA()
        else:
            raise ValueError("Undefined solver")
    

    def removeShadowOtsu(self):
        if "metal" in self.savePath:
            isMetal = True
        else:
            isMetal = False

        M = copy.deepcopy(self.M)

        #print("M shape: ", M.shape)


        #print("M dtype: ", M.dtype)

        for i in range(M.shape[1]):

            image = M[:,i].reshape(self.height,self.width)

            image_copy = copy.deepcopy(image)

           
            image = np.array(image).astype(np.float64)

            #normalize image to range 0-255

            image *= 65535.0/image.max() 

            img = image.astype(np.uint16)

            if isMetal:
                aboveMask = np.zeros_like(img)
                aboveMask[img > 6000] = 1
                img[img > 6000] = 0

            # plt.subplot(1,2,1)
            # plt.imshow(img)
            # plt.title("Original image")
            # plt.subplot(1,2,2)
            # plt.hist(img.flatten(), bins=1000, range=(0.001,img.max()))
            # plt.title("Histogram of original image")
            # plt.show()
            
            # Otsu's thresholding after Gaussian filtering
            blur = cv.GaussianBlur(img,(5,5),0)
            ret3,th3 = cv.threshold(blur,0,65535,cv.THRESH_BINARY+cv.THRESH_OTSU)
            
            mask = copy.deepcopy(th3)
            #binarize the mask

            mask[mask > 0] = 1
            if isMetal:
                combinedmask = mask + aboveMask
                combinedmask[combinedmask > 1] = 1
                mask = combinedmask
                # plt.subplot(1,3,1)
                # plt.imshow(mask)
                # plt.title("Otsu mask")
                # plt.subplot(1,3,2)
                # plt.imshow(combinedmask)
                # plt.title("Combined mask")
                # plt.subplot(1,3,3)
                # plt.imshow(image_copy)
                # plt.title("Original image")
                # plt.show()


            #Invert the mask
            #mask = 1 - mask

            mask = mask.astype(np.float32)

            masked_image = copy.deepcopy(mask * image_copy)

            #plot masked image
            # plt.subplot(1,4,1)
            # plt.imshow(masked_image)
            # plt.title("Otsu masked image")
            # plt.subplot(1,4,2)
            # plt.hist(masked_image.flatten(), bins=1000, range=(0.001,masked_image.max()))
            # plt.title("Histogram of masked image")
            # plt.subplot(1,4,3)
            # plt.hist(image_copy.flatten(), bins=1000, range=(0.001,image_copy.max()))
            # plt.title("Histogram of original image")
            # plt.subplot(1,4,4)
            # plt.imshow(image_copy)
            # plt.title("Image with shadows")
            # plt.show()

            M[:,i] = masked_image.flatten()

        return M


    def _solve_near_RPCA(self):
        """
        Lambertian Photometric stereo based on least-squares
        Woodham 1980, modified for near light source
        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        
        M_copy = copy.deepcopy(self.M)
        M_orignial = copy.deepcopy(self.M)

         #check if meshFilePath exists
        # if not exists(self.meshFilePath):
        #     raise ValueError("Mesh file does not exist, check path: ", self.meshFilePath)
        
        Light2CameraTransforms = np.array(self.Light2CameraTransforms)

        number_of_images = Light2CameraTransforms.shape[0]

        Light2CameraTransforms = Light2CameraTransforms.reshape(number_of_images*4,4)

        Solver = NearSolver(self.height,self.width,number_of_images,self.meshFilePath)

        Solver.set_camera_matrix(self.CameraMatrix)
  
        Depth_image = self.Depth

        M_copy = Solver.attenuateMatrix(Light2CameraTransforms, Depth_image, M_copy)


        if self.foreground_ind is None:
            _M = M_copy.T
        else:
            _M = M_copy[self.foreground_ind, :].T

        A, E, ite = rpsnumerics.rpca_inexact_alm(_M)    # RPCA Photometric stereo
        M_orignial[self.foreground_ind,:] = A.T

        self.M = M_orignial


       

       

        #M_vector = M_noShadow.flatten()
        if False:
            plt.imshow(M_noShadow[:,0].reshape(self.height,self.width))
            plt.title("Image with shaddows removed.")
            plt.show()

        #generateMatrices(Eigen::MatrixXd light_poses, Eigen::MatrixXd depth_image, Eigen::VectorXd measurement)

        #print("Entering solver")

        #nonzeroind = np.nonzero(M_vector)[0]

        #print("Non zero indecies of M: ", nonzeroind.shape)


        #print(M_vector)
        if False:
            plt.imshow(Depth_image)
            plt.title("Ground Truth Depth Image")
            plt.colorbar()
            plt.savefig(self.savePath + 'GTdepth_image.pdf')
            plt.show()

        
        Solver.generate_matrices(Light2CameraTransforms, Depth_image, self.M)

        #print("light2cameratransforms: ")
        #print(Light2CameraTransforms[0:3,3])
        #print(Light2CameraTransforms[0:3,3].shape)

    
        #visible_pixels = Solver.get_visible_pixels(Light2CameraTransforms[0:3,3],Depth_image)

        #imshow

        # plt.pyplot.imshow(visible_pixels)


        # plt.pyplot.show()


        unconstrained_points = Solver.get_uc()

        #save as npy

        np.save("unconstrained_pointsRPCA.npy", unconstrained_points)

       
        #numpy zero of image size200

        test_image = np.zeros((self.height,self.width))
        image1 = copy.deepcopy(self.M[:,0].reshape(self.height,self.width))

        #max value of image 1

        max_value = image1.max()

        #convert to rgb

        #print("numer of unconstrained points: ", len(unconstrained_points))
        
        for point in unconstrained_points:
            test_image[point[0],point[1]] = 1
            image1[point[0],point[1]] = point[2]
        self.unconstrained_mask = image1     

        # first_image = self.M[:,0].reshape(self.height,self.width)
        if False:
            plt.imshow(image1)
            plt.title("Light sources pr pixel")
            plt.colorbar()
            plt.savefig(self.savePath + 'unconstrained_points.pdf')
            plt.show()

        A = Solver.get_a()
        B = Solver.get_b()

        nonzeroind = np.nonzero(B)[0]

        #print("Non zero indecies of B: ", nonzeroind.shape)

        A_sparse = scipy.sparse.csc_matrix(A)
        print("Solving ....")
        start = time.time()
        #x, istop, itn, r1norm = scipy.sparse.linalg.lsqr(A_sparse, B,show=True)[:4]
        x, istop, itn, r1norm = scipy.sparse.linalg.lsmr(A_sparse, B,show=True)[:4]
        print("Time taken: ", time.time() - start)
        #print("x shape: ", x.shape)

        #x = Solver.get_x()

      
        x_reshaped = x.reshape(-1,3)

        self.NearNormals = normalize(x_reshaped, axis=1)
        #self.NearNormals = x_reshaped

       

        #psutil.disp_2_normal_maps(normal=self.N,normal2=self.NearNormals, height=self.height, width=self.width, delay=0)   
        #psutil.disp_normalmap(normal=self.NearNormals, height=self.height, width=self.width, delay=0)  



    def _solve_near_l2(self):
        """
        Lambertian Photometric stereo based on least-squares
        Woodham 1980, modified for near light source
        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        
        M_copy = copy.deepcopy(self.M)
        self.M = M_copy

        if "metal" in self.savePath:
            M_noShadow = M_copy
        else:
            M_noShadow = self.removeShadowOtsu()
        # M_noShadow = M_copy
        

        Light2CameraTransforms = np.array(self.Light2CameraTransforms)

        number_of_images = Light2CameraTransforms.shape[0]

        Light2CameraTransforms = Light2CameraTransforms.reshape(number_of_images*4,4)

        #check if meshFilePath exists
        # if not exists(self.meshFilePath):
        #     raise ValueError("Mesh file does not exist, check path: ", self.meshFilePath)

        Solver = NearSolver(self.height,self.width,number_of_images,self.meshFilePath)

        Solver.set_camera_matrix(self.CameraMatrix)

        Depth_image = self.Depth

        M_noShadow = Solver.attenuateMatrix(Light2CameraTransforms, Depth_image, M_noShadow)
  
        

        M_vector = M_noShadow.flatten()
        if False:
            plt.imshow(M_noShadow[:,0].reshape(self.height,self.width))
            plt.title("Image with shaddows removed.")
            plt.show()

        #generateMatrices(Eigen::MatrixXd light_poses, Eigen::MatrixXd depth_image, Eigen::VectorXd measurement)

        #print("Entering solver")

        #nonzeroind = np.nonzero(M_vector)[0]

        #print("Non zero indecies of M: ", nonzeroind.shape)


        #print(M_vector)
        if False:
            plt.imshow(Depth_image)
            plt.title("Ground Truth Depth Image")
            plt.colorbar()
            plt.savefig(self.savePath + 'GTdepth_image.pdf')
            plt.show()

        
        Solver.generate_matrices(Light2CameraTransforms, Depth_image, M_noShadow)

        #print("light2cameratransforms: ")
        #print(Light2CameraTransforms[0:3,3])
        #print(Light2CameraTransforms[0:3,3].shape)

    
        #visible_pixels = Solver.get_visible_pixels(Light2CameraTransforms[0:3,3],Depth_image)

        #imshow

        # plt.pyplot.imshow(visible_pixels)


        # plt.pyplot.show()


        unconstrained_points = Solver.get_uc()

        #save as npy

        np.save("unconstrained_points.npy", unconstrained_points)

       
        #numpy zero of image size200

        test_image = np.zeros((self.height,self.width))
        image1 = copy.deepcopy(self.M[:,0].reshape(self.height,self.width))

        #max value of image 1

        max_value = image1.max()

        #convert to rgb

        #print("numer of unconstrained points: ", len(unconstrained_points))
        
        for point in unconstrained_points:
            test_image[point[0],point[1]] = 1
            image1[point[0],point[1]] = point[2]
        self.unconstrained_mask = image1     

        # first_image = self.M[:,0].reshape(self.height,self.width)
        if False:
            plt.imshow(image1)
            plt.title("Light sources pr pixel")
            plt.colorbar()
            plt.savefig(self.savePath + 'unconstrained_points.pdf')
            plt.show()

        A = Solver.get_a()
        B = Solver.get_b()

        #nonzeroind = np.nonzero(B)[0]

        #print("Non zero indecies of B: ", nonzeroind.shape)

        A_sparse = scipy.sparse.csc_matrix(A)

        #x, istop, itn, r1norm = scipy.sparse.linalg.lsqr(A_sparse, B,show=True)[:4]
        x, istop, itn, r1norm = scipy.sparse.linalg.lsmr(A_sparse, B,show=True)[:4]

        #print("x shape: ", x.shape)

        #x = Solver.get_x()

      
        x_reshaped = x.reshape(-1,3)

        self.NearNormals = normalize(x_reshaped, axis=1)
        #self.NearNormals = x_reshaped

       

        #psutil.disp_2_normal_maps(normal=self.N,normal2=self.NearNormals, height=self.height, width=self.width, delay=0)   
        #psutil.disp_normalmap(normal=self.NearNormals, height=self.height, width=self.width, delay=0)   

    def _solve_l2(self):
        """
        Lambertian Photometric stereo based on least-squares
        Woodham 1980
        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)


        """
        #time the solver
        self.N = np.linalg.lstsq(self.L.T, self.M.T, rcond=None)[0].T

       
        self.N = normalize(self.N, axis=1)  # normalize to account for diffuse reflectance
        if self.background_ind is not None:
            for i in range(self.N.shape[1]):
                self.N[self.background_ind, i] = 0


   



    def _solve_l1(self):
        """
        Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        A = self.L.T
        self.N = np.zeros((self.M.shape[0], 3))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind

        for index in indices:
            b = np.array([self.M[index, :]]).T
            n = rpsnumerics.L1_residual_min(A, b)
            self.N[index, :] = n.ravel()
        self.N = normalize(self.N, axis=1)

    def _solve_l1_multicore(self):
        """
        Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        from multiprocessing import Pool
        import multiprocessing

        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        p = Pool(processes=multiprocessing.cpu_count()-1)
        normal = p.map(self._solve_l1_multicore_impl, indices)
        if self.foreground_ind is None:
            self.N = np.asarray(normal)
            self.N = normalize(self.N, axis=1)
        else:
            N = np.asarray(normal)
            N = normalize(N, axis=1)
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]
    

    def _solve_l1_multicore_impl(self, index):
        """
        Implementation of Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :param index: an index of a measurement (row of M)
        :return: a row vector of surface normal at pixel index specified by "index"
        """
        A = self.L.T
        b = np.array([self.M[index, :]]).T
        n = rpsnumerics.L1_residual_min(A, b)   # row vector of a surface normal at pixel "index"
        return n.ravel()
    

    def _solve_sbl(self):
        """
        Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        A = self.L.T
        self.N = np.zeros((self.M.shape[0], 3))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind

        for index in indices:
            b = np.array([self.M[index, :]]).T
            n = rpsnumerics.sparse_bayesian_learning(A, b)
            self.N[index, :] = n.ravel()
        self.N = normalize(self.N, axis=1)

    def _solve_sbl_multicore(self):
        """
        Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        from multiprocessing import Pool
        import multiprocessing

        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        p = Pool(processes=multiprocessing.cpu_count()-2)
        normal = p.map(self._solve_sbl_multicore_impl, indices)
        if self.foreground_ind is None:
            self.N = np.asarray(normal)
            self.N = normalize(self.N, axis=1)
        else:
            N = np.asarray(normal)
            N = normalize(N, axis=1)
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(self.N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]

    def _solve_sbl_multicore_impl(self, index):
        """
        Implementation of Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :param index: an index of a measurement (row of M)
        :return: a row vector of surface normal at pixel index specified by "index"
        """
        A = self.L.T
        b = np.array([self.M[index, :]]).T
        n = rpsnumerics.sparse_bayesian_learning(A, b)   # row vector of a surface normal at pixel "index"
        return n.ravel()

    def _solve_rpca(self):
        """
        Photometric stereo based on robust PCA.
        Lun Wu, Arvind Ganesh, Boxin Shi, Yasuyuki Matsushita, Yongtian Wang, Yi Ma:
        Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery. ACCV (3) 2010: 703-717

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """

        if self.foreground_ind is None:
            _M = self.M.T
        else:
            _M = self.M[self.foreground_ind, :].T

        A, E, ite = rpsnumerics.rpca_inexact_alm(_M)    # RPCA Photometric stereo

        if self.foreground_ind is None:
            self.N = np.linalg.lstsq(self.L.T, A, rcond=None)[0].T
            self.N = normalize(self.N, axis=1)    # normalize to account for diffuse reflectance
        else:
            N = np.linalg.lstsq(self.L.T, A, rcond=None)[0].T
            N = normalize(N, axis=1)    # normalize to account for diffuse reflectance
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(self.N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]

        #display normal map
                
        #psutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=0)
