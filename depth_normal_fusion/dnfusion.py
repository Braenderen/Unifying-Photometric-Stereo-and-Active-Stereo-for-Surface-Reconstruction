import numpy as np
import scipy.sparse as sp
import time
import math
import cv2
import matplotlib.pyplot as plt
import copy

from scipy.sparse import spdiags, csr_matrix, vstack
from scipy.sparse.linalg import cg

from tqdm.auto import tqdm

import pyvista as pv


import random


def move_left(mask): return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:] 
def move_right(mask): return np.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]
def move_top(mask): return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:] 
def move_bottom(mask): return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:]  
def move_top_left(mask): return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]  
def move_top_right(mask): return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]  
def move_bottom_left(mask): return np.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]  
def move_bottom_right(mask): return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1] 


class dnFusion:
    def __init__(self):
        """
        Depth normal fusion class

        Attributes
        ----------
        N_0 : numpy array
            Initial normal map
        height : int
            Height of normal map
        width : int
            Width of normal map
        Z_0 : numpy array
            Initial depth map
        mask : numpy array
            Binary mask of object
        lambda_p : float
            Weight for prior term
        lambda_n : float
            Weight for normal term
        lambda_s : float
            Weight for smoothness term
        f_x : float
            Focal length in the x direction
        f_y : float
            Focal length in the y direction
        c_x : float
            Optical center in the x direction
        c_y : float
            Optical center in the y direction
        mu : numpy array
            mu matrix
        A : scipy sparse matrix
            A matrix
        Amode : str
            Finite difference mode for constructing the A matrix
        __siminus1 : numpy array
            Shift invariant gaussian filter in the negative horisontal direction
        __siplus1 : numpy array
            Shift invariant gaussian filter in the positive horisontal direction
        __sjminus1 : numpy array
            Shift invariant gaussian filter in the negative vertical direction
        __sjplus1 : numpy array
            Shift invariant gaussian filter in the positive vertical direction
        sigmaGauss : float
            Standard deviation of gaussian filter
        Z_0Dense : numpy array
            Dense depth map for calculating s
        """

        self.N_0 = None # initial normal map
        self.height = None # height of normal map
        self.width = None # width of normal map
        self.Z_0 = None # initial depth map
        self.mask = None # binary mask of object
        self.lambda_p = None # weight for prior term
        self.lambda_n = None # weight for normal term
        self.lambda_s = None # weight for smoothness term
        self.f_x = None # focal length in the x direction
        self.f_y = None # focal length in the y direction
        self.c_x = None # optical center in the x direction
        self.c_y = None # optical center in the y direction
        self.mu = None # mu matrix
        self.A = None # A matrix
        self.b = None # b vector
        self.Amode = None # finite difference mode for constructing the A matrix
        self.Smode = 'ZeroPad' # mode for constructing the shift invariant gaussian filter
        self.__siminus1 = None # shift invariant gaussian filter in the negative gorisontal direction
        self.__siplus1 = None # shift invariant gaussian filter in the positive horisontal direction
        self.__sjminus1 = None # shift invariant gaussian filter in the negative vertical direction
        self.__sjplus1 = None # shift invariant gaussian filter in the positive vertical direction
        self.sigmaGauss = None # standard deviation of gaussian filter
        self.Z_0Dense = None # dense depth map for calculating s
        self.__sThreshold = 0.0001 # threshold for s divisor
        self.Z = None # reconstructed depth map
        self.maskA = True
        self.k = 2
        self.it_limit = 200
        self.tol = 1e-4
        self.Use_depth = True
        self.synthetic = False
        self.sample_factor = 0

    def loadNormalMap(self, path):
        """
        Loads the normal map from the given path

        Parameters
        ----------
        path : str
            Path to the normal map
        """
        self.N_0 = np.load(path)

    def generate_dx_dy(self,mask, nz_horizontal, nz_vertical, step_size=1):
        
        num_pixel = np.sum(mask)

        
        pixel_idx = np.zeros_like(mask, dtype=int)
       
        pixel_idx[mask] = np.arange(num_pixel)

        
        has_left_mask = np.logical_and(move_right(mask), mask)
        has_right_mask = np.logical_and(move_left(mask), mask)
        has_bottom_mask = np.logical_and(move_top(mask), mask)
        has_top_mask = np.logical_and(move_bottom(mask), mask)

       
        nz_left = nz_horizontal[has_left_mask[mask]]
        nz_right = nz_horizontal[has_right_mask[mask]]
        nz_top = nz_vertical[has_top_mask[mask]]
        nz_bottom = nz_vertical[has_bottom_mask[mask]]

     
        data = np.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
        indices = np.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
        indptr = np.concatenate([np.array([0]), np.cumsum(has_left_mask[mask].astype(int) * 2)])
        D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        data = np.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
        indices = np.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
        indptr = np.concatenate([np.array([0]), np.cumsum(has_right_mask[mask].astype(int) * 2)])
        D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        data = np.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
        indices = np.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
        indptr = np.concatenate([np.array([0]), np.cumsum(has_top_mask[mask].astype(int) * 2)])
        D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        data = np.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
        indices = np.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
        indptr = np.concatenate([np.array([0]), np.cumsum(has_bottom_mask[mask].astype(int) * 2)])
        D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

        # Return the four sparse matrices representing the partial derivatives for each direction.
        return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


    def construct_facets_from(self,mask):
        
        idx = np.zeros_like(mask, dtype=int)
        idx[mask] = np.arange(np.sum(mask))

        
        facet_move_top_mask = move_top(mask)
        facet_move_left_mask = move_left(mask)
        facet_move_top_left_mask = move_top_left(mask)

        
        facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))

       
        facet_top_right_mask = move_right(facet_top_left_mask)
        facet_bottom_left_mask = move_bottom(facet_top_left_mask)
        facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

       
        return np.stack((4 * np.ones(np.sum(facet_top_left_mask)),
                idx[facet_top_left_mask],
                idx[facet_bottom_left_mask],
                idx[facet_bottom_right_mask],
                idx[facet_top_right_mask]), axis=-1).astype(int)
    

    def map_depth_map_to_point_clouds(self,depth_map, mask, K=None, step_size=1):
  
        H, W = mask.shape
        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.flip(xx, axis=0)

        if K is None:
            vertices = np.zeros((H, W, 3))
            vertices[..., 0] = xx * step_size
            vertices[..., 1] = yy * step_size
            vertices[..., 2] = depth_map
            vertices = vertices[mask]
        else:
            u = np.zeros((H, W, 3))
            u[..., 0] = xx
            u[..., 1] = yy
            u[..., 2] = 1
            u = u[mask].T
            vertices = (np.linalg.inv(K) @ u).T * depth_map[mask, np.newaxis]

        return vertices
    
    def depthMask(self,depth_map,mask):
        depth_mask = depth_map > 0
        #set all 0 to nan

        depth_mask_zero = np.zeros_like(depth_map)

        #depth_mask_zero[depth_mask == 0] = np.nan
        depth_mask_zero[depth_mask == 1] = 1
        depth_mask_Zero = depth_mask_zero*mask
        depth_mask_zero = depth_mask_zero.astype(bool)

        depth_mask = depth_mask_zero

        return depth_mask

    def sigmoid(self,x, k=1):
        return 1 / (1 + np.exp(-k * x))

    def setNormalMap(self, N_0):
        """
        Sets the normal map

        Parameters
        ----------
        N_0 : numpy array
            Normal map
        """
        self.N_0 = N_0

    def bilateral_normal_integration(self,normal_map,
                                 normal_mask,
                                 k=100,
                                 depth_map=None,
                                 depth_mask=None,
                                 lambda1=0,
                                 K=None,
                                 step_size=1,
                                 max_iter=150,
                                 tol=1e-4,
                                 cg_max_iter=5000,
                                 cg_tol=1e-3):
    

        num_normals = np.sum(normal_mask)
 
        nx = normal_map[normal_mask, 1]
        ny = normal_map[normal_mask, 0]
        nz = - normal_map[normal_mask, 2]

       
        if K is not None: 
            img_height, img_width = normal_mask.shape[:2]

            yy, xx = np.meshgrid(range(img_width), range(img_height))
            xx = np.flip(xx, axis=0)

            cx = K[0, 2]
            cy = K[1, 2]
            fx = K[0, 0]
            fy = K[1, 1]

            uu = xx[normal_mask] - cx
            vv = yy[normal_mask] - cy

            nz_u = uu * nx + vv * ny + fx * nz
            nz_v = uu * nx + vv * ny + fy * nz
            del xx, yy, uu, vv
       

    
        A3, A4, A1, A2 = self.generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

       
        A = vstack((A1, A2, A3, A4))
        b = np.concatenate((-nx, -nx, -ny, -ny))

        W = spdiags(0.5 * np.ones(4*num_normals), 0, 4*num_normals, 4*num_normals, format="csr")
        z = np.zeros(np.sum(normal_mask))
        energy = (A @ z - b).T @ W @ (A @ z - b)

        tic = time.time()

        energy_list = []
        if depth_map is not None:
            m = depth_mask[normal_mask].astype(int)
            M = spdiags(m, 0, num_normals, num_normals, format="csr")
            z_prior = np.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]

        pbar = tqdm(range(max_iter))


        for i in pbar:
       
            A_mat = A.T @ W @ A
            b_vec = A.T @ W @ b
            if depth_map is not None:
                depth_diff = M @ (z_prior - z)
                depth_diff[depth_diff==0] = np.nan
                offset = np.nanmean(depth_diff)
                z = z + offset
                A_mat += lambda1 * M
                b_vec += lambda1 * M @ z_prior

            D = spdiags(1/np.clip(A_mat.diagonal(), 1e-5, None), 0, num_normals, num_normals, format="csr")  

            z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)

            wu = self.sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
            wv = self.sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
            W = spdiags(np.concatenate((wu, 1-wu, wv, 1-wv)), 0, 4*num_normals, 4*num_normals, format="csr")

        
            energy_old = energy
            energy = (A @ z - b).T @ W @ (A @ z - b)
            
            relative_energy = np.abs(energy - energy_old) / energy_old
          
            if relative_energy < tol:
                break
       

   
        depth_map = np.ones_like(normal_mask, float) * np.nan
        depth_map[normal_mask] = z

        if K is not None:  
            depth_map = np.exp(depth_map)
            vertices = self.map_depth_map_to_point_clouds(depth_map, normal_mask, K=K)
    
        facets = self.construct_facets_from(normal_mask)
        if normal_map[:, :, -1].mean() < 0:
            facets = facets[:, [0, 1, 4, 3, 2]]
        surface = pv.PolyData(vertices, facets)


        return depth_map, surface


    def erodedMask(self,npy_depth):
        
        #create mask for depth map

        mask = np.zeros(npy_depth.shape, dtype=np.float32)
        mask[npy_depth > 0] = 1

        # Creating kernel 
        kernel = np.ones((7, 7), np.uint8) 

        #erode mask
        mask = cv2.erode(mask, kernel)

        return mask
    
    def magnitudeToDepth(self,npy_depth):
        #convert depth map containing length of vector to point to Z depth

        for i in range(0,npy_depth.shape[0]):
            for j in range(0,npy_depth.shape[1]):
                vector_magnitude = npy_depth[i,j]

                direction_vector = np.array(((i-self.c_y)/self.f_y,(j-self.c_x)/self.f_x,1))

                #normalize direction vector
                unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)

                #multiply normalized direction vector with magnitude to get Z depth
                z_depth = vector_magnitude * unit_direction_vector

                npy_depth[i,j] = z_depth[2]
        
        return npy_depth


    def loadSparseDepthMap(self, path):
        """
        Loads the depth map from the given path

        Parameters
        ----------
        path : str
            Path to the depth map
        """
        self.Z_0 = np.load(path)

    def setSparseDepthMap(self, Z_0):
        """
        Sets the depth map

        Parameters
        ----------
        Z0 : numpy array
            Depth map
        """
        self.Z_0 = Z_0

    def loadDenseDepthMap(self, path):
        """
        Loads the dense depth map for calculating s from the given path

        Parameters
        ----------
        path : str
            Path to the depth map
        """
        self.Z_0Dense = np.load(path)

    def setDenseDepthMap(self, Z_0Dense):
        """
        Sets the dense depth map for calculating s

        Parameters
        ----------
        Z0 : numpy array
            Depth map
        """
        self.Z_0Dense = Z_0Dense

    def loadMask(self, path):
        """
        Loads the mask from the given path

        Parameters
        ----------
        path : str
            Path to the mask
        """
        mask = np.load(path)
        self.setMask(mask)

    def setMask(self, mask):
        """
        Sets the mask

        Parameters
        ----------
        mask : binary numpy array
            Mask
        """
        # get max value of mask
        max = np.max(mask)
        if max > 1:
            raise ValueError('mask must be binary')

        self.mask = mask

    def setWeights(self, lambda_p = None, lambda_n = None, lambda_s = None):
        """
        Sets the weights

        Parameters
        ----------
        lambda_p : float
            Weight for the prior term
        lambda_n : float
            Weight for the normal term
        lambda_s : float
            Weight for the smoothness term, if not set A will not include the smoothness term
        """
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n
        self.lambda_s = lambda_s

    def setCameraParams(self, f_x = None, f_y = None, c_x = None, c_y = None):
        """
        Sets the camera parameters

        Parameters
        ----------
        f_x : float
            Focal length in the x direction
        f_y : float
            Focal length in the y direction
        c_x : float
            Optical center in the x direction
        c_y : float
            Optical center in the y direction
        """
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y

    def setCameraParamsMitsuba(self, vfov, img_width, img_height):
        """
        Sets the camera parameters from mitsuba

        Parameters
        ----------
        vfov : float
            Vertical field of view of the camera in degrees
        img_width : int
            Width of the image in pixels
        img_height : int
            Height of the image in pixels

        """

        # horizontal field of view
        #hfov = vfov * img_width/img_height
        hfov = 2 * math.atan(math.tan(vfov * 0.5 * math.pi/180) * img_width/img_height) * 180/math.pi

        # focal length
        self.f_y = img_height * 0.5 / (math.tan (vfov * 0.5 * math.pi/180) )
        self.f_x = img_width * 0.5 / (math.tan (hfov * 0.5 * math.pi/180) )

        # optical center
        self.c_x = img_width/2.0 
        self.c_y = img_height/2.0
        self.width = img_width
        self.height = img_height

    def constructMu(self, width = None, height = None, f_x = None, f_y = None, c_x = None, c_y = None, timeit=False):
        """
        Constructs the mu matrix

        Parameters
        ----------
        width : int
            Number of grid points in the horizontal direction
        height : int
            Number of grid points in the vertical direction
        f_x : float
            Focal length in the x direction
        f_y : float
            Focal length in the y direction
        c_x : float
            Optical center in the x direction
        c_y : float
            Optical center in the y direction
        """
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if f_x is None:
            f_x = self.f_x
        if f_y is None:
            f_y = self.f_y
        if c_x is None:
            c_x = self.c_x
        if c_y is None:
            c_y = self.c_y

        missingCameraParams = width is None or height is None or f_x is None or f_y is None or c_x is None or c_y is None
        if missingCameraParams:
            raise ValueError('Some camera parameters are missing, check width, height, f_x, f_y, c_x, and c_y must be set')

        if timeit:
            start = time.time()
        mu = np.zeros((height, width,3))
        for i in range(height):
            for j in range(width):
                mu[i,j,0] = (i-c_x)/f_x
                mu[i,j,1] = (j-c_y)/f_y
                mu[i,j,2] = 1
        self.mu = mu
        if timeit:
            end = time.time()
            print("mu construction time: ", end-start)

    def crop(self, x, y, w, h):
        """
        Crops the depth and normal maps and the mask

        Parameters
        ----------
        x : int
            x coordinate of the top left corner of the bounding box
        y : int
            y coordinate of the top left corner of the bounding box
        w : int
            Width of the bounding box
        h : int
            Height of the bounding box
        """
        missing = self.Z_0 is None or self.Z_0Dense is None or self.N_0 is None or self.mask is None or self.mu is None
        if missing:
            raise ValueError('Z_0, Z_0Dense, N_0, mask, and mu must be set to crop')
        
        if x < 0 or y < 0 or w < 0 or h < 0:
            raise ValueError('Crop box has negative values')

        notEqualWidth = self.Z_0.shape[1] != self.Z_0Dense.shape[1] or self.Z_0.shape[1] != self.N_0.shape[1] or self.Z_0.shape[1] != self.mask.shape[1] or self.Z_0.shape[1] != self.mu.shape[1]
        notEqualHeight = self.Z_0.shape[0] != self.Z_0Dense.shape[0] or self.Z_0.shape[0] != self.N_0.shape[0] or self.Z_0.shape[0] != self.mask.shape[0] or self.Z_0.shape[0] != self.mu.shape[0]
        if notEqualWidth or notEqualHeight:
            print('Z_0 shape: ', self.Z_0.shape)
            print('Z_0Dense shape: ', self.Z_0Dense.shape)
            print('N_0 shape: ', self.N_0.shape)
            print('mask shape: ', self.mask.shape)
            print('mu shape: ', self.mu.shape)
            raise ValueError('Z_0, Z_0Dense, N_0, mask, and mu must have the same width and height to crop')

        self.Z_0 = self.Z_0[y:y+h, x:x+w]
        self.Z_0Dense = self.Z_0Dense[y:y+h, x:x+w]
        self.N_0 = self.N_0[y:y+h, x:x+w, :]
        self.mask = self.mask[y:y+h, x:x+w]
        self.mu = self.mu[y:y+h, x:x+w, :]
        self.width = self.N_0.shape[1]
        self.height = self.N_0.shape[0]

    def applyMask(self, voidDepth = 0, voidNormal = None):
        """
        Applies the mask to the depth and normal maps
        """
        missing = self.Z_0 is None or self.Z_0Dense is None or self.N_0 is None or self.mask is None or self.mu is None
        if missing:
            raise ValueError('Z_0, Z_0Dense, N_0, mask, and mu must be set to apply mask')
        
        notEqualWidth = self.Z_0.shape[1] != self.Z_0Dense.shape[1] or self.Z_0.shape[1] != self.N_0.shape[1] or self.Z_0.shape[1] != self.mask.shape[1]
        notEqualHeight = self.Z_0.shape[0] != self.Z_0Dense.shape[0] or self.Z_0.shape[0] != self.N_0.shape[0] or self.Z_0.shape[0] != self.mask.shape[0]
        if notEqualWidth or notEqualHeight:
            print('Z_0 shape: ', self.Z_0.shape)
            print('Z_0Dense shape: ', self.Z_0Dense.shape)
            print('N_0 shape: ', self.N_0.shape)
            print('mask shape: ', self.mask.shape)
            raise ValueError('Z_0, Z_0Dense, N_0, mask, and mu must have the same width and height to apply mask')

        if self.mask is None: 
            raise ValueError('mask must be set to apply mask')
        
        if np.max(self.mask) > 1:
            raise ValueError('mask must be binary')
            
        self.Z_0[self.mask == 0] = voidDepth
        self.Z_0Dense[self.mask == 0] = voidDepth

        if voidNormal is not None:
            raise NotImplementedError('voidNormal is not implemented')
        #self.N_0 = self.N_0*self.mask[:,:,np.newaxis]
    
    def constructA(self, mode = 'CentralWithFB', sigma = None ,timeit=False):
        """
        Constructs the A matrix of the least squares problem (equation 8 in the paper)
        Only includes the smoothness term if lambda_s is set
        """

        missingCameraParams = self.width is None or self.height is None or self.f_x is None or self.f_y is None or self.c_x is None or self.c_y is None
        if missingCameraParams:
            raise ValueError('Some camera parameters are missing, check width, height, f_x, f_y, c_x, and c_y must be set')

        self.Amode = mode # finite difference mode for constructing the A matrix
        if mode == 'Adaptive':
            if sigma is not None:
                self.sigmaGauss = sigma
            if self.sigmaGauss is None:
                raise ValueError('sigma must be set when using Adaptive mode')

        if timeit:
            start = time.time()  
        r1 = self._Arow1()
        end = time.time()
        if timeit:
            print("row1 time: ", end-start)

        if timeit:
            start = time.time()
        r2 = self._Arow2()
        end = time.time()
        if timeit:
            print("row2 time: ", end-start)

        if timeit:
            start = time.time()
        r3 = self._Arow3()
        end = time.time()
        if timeit:
            print("row3 time: ", end-start)

        if self.lambda_s is not None:
            start = time.time()
            r4 = self._Arow4()
            end = time.time()
            if timeit:
                print("row4 time: ", end-start)

            self.A = sp.vstack((r1,r2,r3,r4))
            self.r1 = r1
            self.r2 = r2
            self.r3 = r3
            self.r4 = r4
        else:
            self.A = sp.vstack((r1,r2,r3))
            self.r1 = r1
            self.r2 = r2
            self.r3 = r3
            self.r4 = None

    # def maskA(self):
    #     """
    #     Masks the A matrix
    #     """
    #     if self.A is None:
    #         raise ValueError('A must be set to mask')
    #     if self.b is None:
    #         raise ValueError('B must be set to mask')
    #     if self.mask is None:
    #         raise ValueError('mask must be set to mask A')

    #     if not isinstance(self.A, sp.csr_array):
    #         print(type(self.A))
    #         raise ValueError('A must be a csr array')
    #     if not isinstance(self.b, np.ndarray):
    #         print(type(self.b))
    #         raise ValueError('b must be a csr array')
    #     if not isinstance(self.mask, np.ndarray):
    #         print(type(self.mask))
    #         raise ValueError('mask must be a np array')

    #     flatMask = self.mask.flatten()

    #     print('Form: ',self.r1.has_canonical_format)

    #     # Set rows in r1 to zero where mask is zero
    #     zeroIndices = np.where(flatMask == 0)[0]
    #     r1nnz = self.r1.nonzero()  # [[rows],[cols]]
    #     print('Size of r1', self.r1.shape)
    #     print('Non zero rows in r1: ', len(r1nnz[0]))
    #     print('Zero rows in mask', len(zeroIndices))
        
    #     # Find overlap of zeroIndices and r1nnz
    #     overlap = np.intersect1d(zeroIndices, r1nnz[0])
    #     print('Overlap: ', len(overlap))
    #     print('First ten in overlap', overlap[:10])
    #     print('First ten in r1nnz', r1nnz[0][:10])

    #     print("R1 data shape")

    #     print(len(self.r1.data))
    #     print(self.r1.data[:10])

    #     print(self.r1.indices[0:10])


    #     for i in range(len(self.r1.indices)):

    #         if self.r1.indices[i] in zeroIndices:
    #             self.r1.indices
        

    #     # get entry aka [row,col] of r1nnz where row is in overlap
    #     #row, col = np.where(self.r1[overlap,:])


    #     exit()
    # def maskA(self):
    #     """
    #     Masks the A matrix
    #     """
    #     if self.A is None:
    #         raise ValueError('A must be set to mask')
    #     if self.b is None:
    #         raise ValueError('B must be set to mask')
    #     if self.mask is None:
    #         raise ValueError('mask must be set to mask A')

    #     if not isinstance(self.A, sp.csr_array):
    #         print(type(self.A))
    #         raise ValueError('A must be a csr array')
    #     if not isinstance(self.b, np.ndarray):
    #         print(type(self.b))
    #         raise ValueError('b must be a csr array')
    #     if not isinstance(self.mask, np.ndarray):
    #         print(type(self.mask))
    #         raise ValueError('mask must be a np array')

    #     flatMask = self.mask.flatten()
        
    #     # Find indices of zero elements in flatMask
    #     zeroIndices = np.where(flatMask == 0)[0]
    #     print('zeroIndices: ', zeroIndices)
    #     print('Count of zeroIndices: ', len(zeroIndices))

    #     # Create mask for row1
    #     row1Mask = np.ones(self.r1.shape[0], dtype=bool)
    #     row1Mask[zeroIndices] = False
    #     # Remove rows in r1 where mask is zero
    #     print("Size of r1 before removing zero rows: ", self.r1.shape)
    #     r1Masked = self.r1[row1Mask,:]
    #     print("Size of r1 after removing zero rows: ", r1Masked.shape)

    #     # Create mask for b
    #     bMask = np.ones(self.r1.shape[0], dtype=bool)
    #     bMask[zeroIndices] = False
    #     # Remove rows in b where mask is zero
    #     print("Size of b before removing zero rows: ", self.b.shape)
    #     tempb = self.b[0:bMask.shape[0]]
    #     bMasked = tempb[bMask]
    #     print("Size of b after removing zero rows: ", bMasked.shape)

    #     # Create mask for row2
    #     row2Mask = np.ones(self.r2.shape[0], dtype=bool)
    #     row2Mask[zeroIndices] = False
    #     # Remove rows in r2 where mask is zero
    #     print("Size of r2 before removing zero rows: ", self.r2.shape)
    #     r2Masked = self.r2[row2Mask,:]
    #     print("Size of r2 after removing zero rows: ", r2Masked.shape)
        
    #     # Create mask for row3
    #     row3Mask = np.ones(self.r3.shape[0], dtype=bool)
    #     row3Mask[zeroIndices] = False
    #     # Remove rows in r3 where mask is zero
    #     print("Size of r3 before removing zero rows: ", self.r3.shape)
    #     r3Masked = self.r3[row3Mask,:]
    #     print("Size of r3 after removing zero rows: ", r3Masked.shape)
        
    #     if self.lambda_s is not None:
    #         # Create mask for row4
    #         row4Mask = np.ones(self.r4.shape[0], dtype=bool)
    #         row4Mask[zeroIndices] = False
    #         # Remove rows in r4 where mask is zero
    #         print("Size of r4 before removing zero rows: ", self.r4.shape)
    #         r4Masked = self.r4[row4Mask,:]
    #         print("Size of r4 after removing zero rows: ", r4Masked.shape)
    #         self.A = sp.vstack([r1Masked, r2Masked, r3Masked, r4Masked])
    #     else:
    #         # Combine the masked rows
    #         self.A = sp.vstack([r1Masked, r2Masked, r3Masked])

    #     # Fix b such that there is zeros
    #     temp = np.zeros(self.A.shape[0])
    #     temp[0:bMasked.shape[0]] = bMasked
    #     self.b = temp


        #! remove this later
        # print('Non zero: ', np.count_nonzero(b1))
        # print('Zero: ', b1.shape[0] - np.count_nonzero(b1))
        # print(type(self.A))
        # print(type(self.b))

        # # Create temp mask and plot
        # tempMask = np.zeros(b1.shape[0])
        # tempMask[zeroIndices] = 1
        # tempMask = tempMask.reshape(self.Z_0.shape)
        # plt.imshow(tempMask)
        # plt.title('Mask of zero rows')
        
        # plt.figure()
        # testb = self.b[:n]
        # testb = testb.reshape(self.Z_0.shape)
        # plt.imshow(testb)
        # plt.title('b1')
        # plt.show()


    def _Arow1(self):
        """
        Constructs the first row of the A matrix for the least squares problem  
        """

        mu = self.mu
        mu = mu.reshape(mu.shape[0]*mu.shape[1], mu.shape[2])
        mu_norm = np.linalg.norm(mu, axis=1)
        row1 = self.lambda_p * mu_norm
        if self.maskA:
            row1 = row1 * self.mask.flatten()

        row1_dia = sp.diags_array(row1, offsets=0, format='csr') # returns a csr array

        return row1_dia
    
    def _Arow2(self):
        """
        Constructs the second row of the A matrix for the least squares problem  
        """
        N_0 = self.N_0
        mu = self.mu

        # Dot product of N_0 and mu for each pixel
        N_0_dot_mu = np.zeros((N_0.shape[0],N_0.shape[1]))
        for i in range(N_0.shape[0]):
            for j in range(N_0.shape[1]):
                N_0_dot_mu[i,j] = np.dot(N_0[i,j], mu[i,j])

        # Diagonal matrix of N_0_dot_mu
        if self.maskA:
            temp = N_0_dot_mu.flatten() * self.mask.flatten()
            N_0_dot_mu_dia = sp.diags_array(temp,offsets=0, format='csr') 
        else:
            N_0_dot_mu_dia = sp.diags_array(N_0_dot_mu.flatten(),offsets=0, format='csr')

        # First part of the second row of A
        #! TODO: Skal det være elementvise eller @?
        part1_dia = N_0_dot_mu_dia * self._centralDiffU_stacked()

        # x component of N_0 matrix of normal vectors
        N_0_x = N_0[:,:,0] 

        # Second part of the second row of A
        part2 = N_0_x/self.f_x
        if self.maskA:
            part2 = part2.flatten() * self.mask.flatten()
            part2_dia = sp.diags_array(part2,offsets=0, format='csr')
        else:
            part2_dia = sp.diags_array(part2.flatten(),offsets=0, format='csr')
        row2 = self.lambda_n * (part1_dia + part2_dia)
        return row2
    

    def _Arow3(self):
        """
        Constructs the third row of the A matrix of the least squares problem  
        """
        N_0 = self.N_0
        mu = self.mu

        N_0_dot_mu = np.zeros((N_0.shape[0],N_0.shape[1]))
        for i in range(N_0.shape[0]):
            for j in range(N_0.shape[1]):
                N_0_dot_mu[i,j] = np.dot(N_0[i,j], mu[i,j])

        if self.maskA:
            temp = N_0_dot_mu.flatten() * self.mask.flatten()
            N_0_dot_mu_dia = sp.diags_array(temp,offsets=0, format='csr')
        else:
            N_0_dot_mu_dia = sp.diags_array(N_0_dot_mu.flatten(),offsets=0, format='csr')

        #! TODO: Skal det være elementwise eller @?
        part1_dia = N_0_dot_mu_dia *self._centralDiffV_stacked()

        N_0_y = N_0[:,:,1] # y component of N_0 matrix of normal vectors
        part2 = N_0_y/self.f_y
        if self.maskA:
            part2 = part2.flatten() * self.mask.flatten()
            part2_dia = sp.diags_array(part2,offsets=0, format='csr')
        else:
            part2_dia = sp.diags_array(part2.flatten(),offsets=0, format='csr')
        row3 = self.lambda_n * (part1_dia + part2_dia)
        return row3
    
    def _Arow4(self):
        """
        Constructs the fourth row of the A matrix of the least squares problem  (Laplacian)
        """
        lap = self._laplacian_stacked()
        row4 = self.lambda_s * lap
        return row4

    def _shiftInvariantGaussFilter(self, p, pi):
        """
        Calculates the shift-invariant Gaussian filter for the depth
        range at the location pi given a general pixel p
        """
        raise NotImplementedError

        sigma = self.sigmaGauss

        return math.exp(-(p-pi)**2/(2*sigma**2))


    def _shiftInvariantGauss(self, direction):
        """
        Constructs s, the shift invariant gaussian filter in the given direction
        
        Parameters
        ----------
        direction : str
            Direction of the gaussian filter, either 'left', 'right', 'up', or 'down'
        sigma : float
            Standard deviation of the gaussian filter
        """
        if direction not in ['left', 'right', 'up', 'down']:
            raise ValueError('direction must be either left, right, up, or down')
        
        if self.sigmaGauss is None:
            raise ValueError('sigma must be set')
        
        if self.Z_0Dense is None:
            raise ValueError('Z_0Dense must be set')
        
        if self.Smode is None:
            raise ValueError('Smode must be set')

        mode = self.Smode
        
        sigma = self.sigmaGauss
        Z = self.Z_0Dense

        s = np.zeros(Z.shape)
        for y in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                if direction == 'left':
                    if x == 0: # if on the left edge of the grid
                        if mode == 'ZeroPad':
                            s[y,x] = 0
                        elif mode == 'replicate':
                            s[y,x] = math.exp(-(Z[y,x+1]-Z[y,x])**2/(2*sigma**2))
                        else:
                            raise ValueError('mode must be either ZeroPad or replicate')
                    else:
                        s[y,x] = math.exp(-(Z[y,x]-Z[y,x-1])**2/(2*sigma**2))
                elif direction == 'right':
                    if x == Z.shape[1]-1: # if on the right edge of the grid
                        if mode == 'ZeroPad':
                            s[y,x] = 0
                        elif mode == 'replicate':
                            s[y,x] = math.exp(-(Z[y,x-1]-Z[y,x])**2/(2*sigma**2))
                        else:
                            raise ValueError('mode must be either ZeroPad or replicate')
                    else:
                        s[y,x] = math.exp(-(Z[y,x] - Z[y,x+1])**2/(2*sigma**2))
                elif direction == 'up':
                    if y == 0: # if on the top edge of the grid
                        if mode == 'ZeroPad':
                            s[y,x] = 0
                        elif mode == 'replicate':
                            s[y,x] = math.exp(-(Z[y+1,x]-Z[y,x])**2/(2*sigma**2))
                        else:
                            raise ValueError('mode must be either ZeroPad or replicate')
                    else:
                        s[y,x] = math.exp(-(Z[y,x] - Z[y-1,x])**2/(2*sigma**2))
                elif direction == 'down':
                    if y == Z.shape[0]-1: # if on the bottom edge of the grid
                        if mode == 'ZeroPad':
                            s[y,x] = 0
                        elif mode == 'replicate':
                            s[y,x] = math.exp(-(Z[y-1,x]-Z[y,x])**2/(2*sigma**2))
                        else:
                            raise ValueError('mode must be either ZeroPad or replicate')
                    else:
                        s[y,x] = math.exp(-(Z[y,x] - Z[y+1,x])**2/(2*sigma**2))
                else:
                    raise ValueError('direction must be either left, right, up, or down')

        # binarized = s > 0.2
        # dialted = cv2.dilate(binarized.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
        # return dialted.astype(float)
        #return binarized.astype(float)
        return s


    def _centralDiffU_stacked(self):
        """
        Constructs the central difference matrix for the first derivative in the u direction when x (z) is stacked
        https://math.stackexchange.com/questions/819800/differentiation-matrix-for-central-difference-scheme

        Parameters
        ----------
        width : int
            Number of grid points in the horizontal direction
        height : int
            Number of grid points in the vertical direction
        n : int
            Number of grid points (width * height)
        mode : str
            CentralWithFB: Central differences with forward/backward differences on the edges
            Adaptive: Central differences with forward/backward differences on discontinuities
            
        """
        mode = self.Amode
        if mode == 'CentralWithFB':
            pass
        elif mode == 'Adaptive':
            if self.__siminus1 is None: # sheck if S(p-1,p) has been constructed
                self.__siminus1 = self._shiftInvariantGauss('left')
            if self.__siplus1 is None: # sheck if S(p+1,p) has been constructed
                self.__siplus1 = self._shiftInvariantGauss('right')
            SiMinus1 = self.__siminus1.flatten()
            SiPlus1 = self.__siplus1.flatten()
            threshold = self.__sThreshold
        else:
            raise NotImplementedError
        row = []
        col = []
        data = []
        n = self.width * self.height # number of grid points

        for i in range(n):
            if (i + 1) % self.width == 1:   # if i is on the left edge of the grid
                if mode == 'CentralWithFB':
                    # (i,i) = -1
                    row.append(i)
                    col.append(i)
                    data.append(-1)

                    # (i,i+1) = 1
                    row.append(i)
                    col.append(i+1)
                    data.append(1)
                elif mode == 'Adaptive':
                    divisor = SiMinus1[i]+SiPlus1[i]
                    if divisor < threshold:
                        divisor = 1

                    # (i,i)= (s(i-1,j)-s(i+1,j))/(s(i,j)+s(i+1,j))z(i)
                    row.append(i)
                    col.append(i)
                    data.append((SiMinus1[i]-SiPlus1[i])/divisor)

                    # (i,i+1) = s(i+1,j)/(s(i,j)+s(i+1,j))z(i+1)
                    row.append(i)
                    col.append(i+1)
                    data.append(SiPlus1[i]/divisor)

            elif (i +1) % self.width == 0:  # if i is on the right edge of the grid
                if mode == 'CentralWithFB':
                    # (i,i-1) = -1
                    row.append(i)
                    col.append(i-1)
                    data.append(-1)

                    # (i,i) = 1
                    row.append(i)
                    col.append(i)
                    data.append(1)

                elif mode == 'Adaptive':
                    divisor = SiMinus1[i]+SiPlus1[i]
                    if divisor < threshold:
                        divisor = 1
                    
                    # (i,i-1) = -s(i-1,j)/(s(i-1,j)+s(i,j))z(i-1)
                    row.append(i)
                    col.append(i-1)
                    data.append(-SiMinus1[i]/divisor)

                    # (i,i)= (s(i-1,j)-s(i,j))/(s(i-1,j)+s(i,j))z(i)
                    row.append(i)
                    col.append(i)
                    data.append((SiMinus1[i]-SiPlus1[i])/divisor)
                    
            else:   # if not on the left or right edge of the grid 
                if mode == 'CentralWithFB':
                    # (i,i-1) = -1
                    row.append(i)
                    col.append(i-1)
                    data.append(-1/2)

                    # (i,i+1) = 1
                    row.append(i)
                    col.append(i+1)
                    data.append(1/2)

                elif mode == 'Adaptive':
                    divisor = SiMinus1[i]+SiPlus1[i]
                    if divisor < threshold:
                        divisor = 1
                    
                    # (i,i-1) = -s(i-1,j)/(s(i-1,j)+s(i+1,j))z(i-1)
                    row.append(i)
                    col.append(i-1)
                    data.append(-SiMinus1[i]/divisor)

                    # (i,i)= (s(i-1,j)-s(i+1,j))/(s(i-1,j)+s(i+1,j))z(i)
                    row.append(i)
                    col.append(i)
                    data.append((SiMinus1[i]-SiPlus1[i])/divisor)

                    # (i,i+1) = s(i+1,j)/(s(i-1,j)+s(i+1,j))z(i+1)
                    row.append(i)
                    col.append(i+1)
                    data.append(SiPlus1[i]/divisor)

        if data == []:
            raise ValueError('data is empty')

        mat = sp.csr_array((data, (row, col)), shape=(n, n))
        #raise ValueError('fix s minus and s plus')
        
        return mat
    
    def _centralDiffV_stacked(self):
        """
        Constructs the central difference matrix for the first derivative in the V direction when x (z) is stacked
        https://math.stackexchange.com/questions/819800/differentiation-matrix-for-central-difference-scheme

        Parameters
        ----------
        width : int
            Number of grid points in the horizontal direction
        height : int
            Number of grid points in the vertical direction
        n : int
            Number of grid points (width * height)
        mode : str
            CentralWithFB: Central differences with forward/backward differences on the edges
            Adaptive: Central differences with forward/backward differences on discontinuities
            
        """
        mode = self.Amode

        if mode == 'CentralWithFB':
            pass
        elif mode == 'Adaptive':
            if self.__sjminus1 is None: # check if S(p,p-1) has been constructed
                self.__sjminus1 = self._shiftInvariantGauss('up')
            if self.__sjplus1 is None: # check if S(p,p+1) has been constructed
                self.__sjplus1 = self._shiftInvariantGauss('down')
            SjMinus1 = self.__sjminus1.flatten()
            SjPlus1 = self.__sjplus1.flatten()
            threshold = self.__sThreshold
        else:
            raise NotImplementedError

        row = []
        col = []
        data = []
        n = self.width * self.height # number of grid points
        width = self.width

        for i in range(n):
            if i  < width:  # if i is on the top edge of the grid
                if mode == 'CentralWithFB':
                    # (i , i) = -1
                    row.append(i)
                    col.append(i)
                    data.append(-1)

                    # (i , i+width) = 1
                    row.append(i)
                    col.append(i+width)
                    data.append(1)

                elif mode == 'Adaptive':
                    divisor = SjMinus1[i]+SjPlus1[i]
                    if divisor < threshold:
                        divisor = 1
                    
                    # (i,i) = 
                    row.append(i)
                    col.append(i)
                    data.append((SjMinus1[i]-SjPlus1[i])/divisor)

                    # (i,i+width) = 
                    row.append(i)
                    col.append(i+width)
                    data.append(SjPlus1[i]/divisor)

            elif i >= n-width:   # if i is on the bottm edge of the grid
                if mode == 'CentralWithFB':
                    # (i, i-width) = -1
                    row.append(i)
                    col.append(i-width)
                    data.append(-1)

                    # (i,i) = 1
                    row.append(i)
                    col.append(i)
                    data.append(1)

                elif mode == 'Adaptive':
                    divisor = SjMinus1[i]+SjPlus1[i]
                    if divisor < threshold:
                        divisor = 1
                    
                    # (i,i-width) = 
                    row.append(i)
                    col.append(i-width)
                    data.append(-SjMinus1[i]/divisor)

                    # (i,i) = 
                    row.append(i)
                    col.append(i)
                    data.append((SjMinus1[i]-SjPlus1[i])/divisor)

            else:   # if not on the top or bottom edge of the grid
                if mode == 'CentralWithFB':
                    # (i,i-width) = -1/2
                    row.append(i)
                    col.append(i-width)
                    data.append(-1/2)

                    # (i,i+width) = 1/2
                    row.append(i)
                    col.append(i+width)
                    data.append(1/2)

                elif mode == 'Adaptive':
                    divisor = SjMinus1[i]+SjPlus1[i]
                    if divisor < threshold:
                        divisor = 1

                    # (i,i-width) = 
                    row.append(i)
                    col.append(i-width)
                    data.append(-SjMinus1[i]/divisor)

                    # (i,i) = 
                    row.append(i)
                    col.append(i)
                    data.append((SjMinus1[i]-SjPlus1[i])/divisor)

                    # (i,i+width) = 
                    row.append(i)
                    col.append(i+width)
                    data.append(SjPlus1[i]/divisor)

        if data == []:
            raise ValueError('data is empty')
        
        mat = sp.csr_array((data, (row, col)), shape=(n, n))

        return mat
    
    def _laplacian_stacked(self):
        """
        Constructs the laplacian matrix when x (z) is stacked

        Parameters
        ----------
        width : int
            Number of grid points in the horizontal direction
        height : int
            Number of grid points in the vertical direction
        n : int
            Number of grid points (width * height)
        mode : str
            CentralWithFB: Central differences with forward/backward differences on the edges
            Adaptive: Central differences with forward/backward differences on discontinuities
        """

        mode = self.Amode

        # TEMP
        # self.sigmaGauss = 100
        # self.Z_0Dense = cv2.imread("/home/phmar/workspace/masterthesis/datasets/middlebury2006/Aloe/disp1.png", cv2.IMREAD_GRAYSCALE)
        # mask = copy.deepcopy(self.Z_0Dense)
        # mask[mask > 0] = 1
        # self.Z_0Dense = self.Z_0Dense + 270
        # self.Z_0Dense = 3740*160/self.Z_0Dense
        # self.Z_0Dense = self.Z_0Dense*mask
        # print('Z_0Dense shape: ', self.Z_0Dense.shape)
        # self.__siminus1 = self._shiftInvariantGauss('left')
        # self.__siplus1 = self._shiftInvariantGauss('right')
        # self.__sjminus1 = self._shiftInvariantGauss('up')
        # self.__sjplus1 = self._shiftInvariantGauss('down')

        if mode == 'CentralWithFB':
            pass
        elif mode == 'Adaptive':
            if self.__siminus1 is None: # sheck if S(p-1,p) has been constructed
                self.__siminus1 = self._shiftInvariantGauss('left')
            if self.__siplus1 is None: # sheck if S(p+1,p) has been constructed
                self.__siplus1 = self._shiftInvariantGauss('right')
            if self.__sjminus1 is None: # sheck if S(p,p-1) has been constructed
                self.__sjminus1 = self._shiftInvariantGauss('up')
            if self.__sjplus1 is None: # sheck if S(p,p+1) has been constructed
                self.__sjplus1 = self._shiftInvariantGauss('down')
            SiMinus1 = self.__siminus1.flatten()
            SiPlus1 = self.__siplus1.flatten()
            SjMinus1 = self.__sjminus1.flatten()
            SjPlus1 = self.__sjplus1.flatten()
            threshold = self.__sThreshold

            if True:
                # Plot as 2x2 grid
                fig, axs = plt.subplots(2, 3)
                axs[0, 0].imshow(self.__siminus1)
                axs[0, 0].set_title('SiMinus1')
                axs[0, 1].imshow(self.Z_0Dense)
                axs[0, 1].set_title('Z_0Dense')
                axs[0, 2].imshow(self.__siplus1)
                axs[0, 2].set_title('SiPlus1')
                axs[1, 0].imshow(self.__sjminus1)
                axs[1, 0].set_title('SjMinus1')
                axs[1, 1].imshow(self.mask)
                axs[1, 1].set_title('mask')
                axs[1, 2].imshow(self.__sjplus1)
                axs[1, 2].set_title('SjPlus1')
                plt.show()
                
                plt.figure()
                plt.imshow(self.__siminus1)
                plt.imsave('SiMinus1.png', self.__siminus1)
                plt.savefig('SiMinus1Fig.png',bbox_inches='tight')
                plt.show()

                plt.figure()
                plt.imshow(self.__sjminus1)
                plt.imsave('SjMinus1.png', self.__sjminus1)
                plt.savefig('SjMinus1Fig.png',bbox_inches='tight')
                plt.show()

                plt.figure()
                plt.imshow(self.Z_0Dense)
                plt.imsave('Z_0Dense.png', self.Z_0Dense)
                plt.savefig('Z_0DenseFig.png',bbox_inches='tight')
        else:
            raise NotImplementedError

        row = []
        col = []
        data = []
        n = self.width * self.height # number of grid points
        width = self.width

        flatMask = self.mask.flatten()
    
        for i in range(n):
            if self.maskA:
                if flatMask[i] == 0:
                    continue
            if (i + 1) % width != 0: # if i is NOT on the right edge of the grid
                if mode == 'CentralWithFB':
                    # (i, i+1) = 1
                    row.append(i)
                    col.append(i+1)
                    data.append(1)
                
                if mode == 'Adaptive':
                    divisor = SiMinus1[i]+SiPlus1[i]+SjMinus1[i]+SjPlus1[i]
                    if divisor < threshold:
                        divisor = 1

                    row.append(i)
                    col.append(i+1)
                    data.append((SiPlus1[i])/divisor)

                notImplemented = 0  #placeholder
            
            if (i + 1) % width != 1: # if i is NOT on the left edge of the grid
                if mode == 'CentralWithFB':
                    # (i, i-1) = 1
                    row.append(i)
                    col.append(i-1)
                    data.append(1)

                if mode == 'Adaptive':
                    divisor = SiMinus1[i]+SiPlus1[i]+SjMinus1[i]+SjPlus1[i]
                    if divisor < threshold:
                        divisor = 1

                    row.append(i)
                    col.append(i-1)
                    data.append((SiMinus1[i])/divisor)

            if i+1 > width: # if i is NOT on the top edge of the grid
                if mode == 'CentralWithFB':
                    # (i, i-width) = 1
                    row.append(i)
                    col.append(i-width)
                    data.append(1)

                if mode == 'Adaptive':
                    divisor = SiMinus1[i]+SiPlus1[i]+SjMinus1[i]+SjPlus1[i]
                    if divisor < threshold:
                        divisor = 1

                    row.append(i)
                    col.append(i-width)
                    data.append((SjMinus1[i])/divisor)

            if i < n-width: # if i is NOT on the bottom edge of the grid
                if mode == 'CentralWithFB':
                    # (i, i+width) = 1
                    row.append(i)
                    col.append(i+width)
                    data.append(1)

                if mode == 'Adaptive':
                    divisor = SiMinus1[i]+SiPlus1[i]+SjMinus1[i]+SjPlus1[i]
                    if divisor < threshold:
                        divisor = 1

                    row.append(i)
                    col.append(i+width)
                    data.append((SjPlus1[i])/divisor)
                    
            # Middle pixel
            if mode == 'CentralWithFB':
                #(i, i) = -4
                row.append(i)
                col.append(i)
                data.append(-4)
            elif mode == 'Adaptive':
                divisor = SiMinus1[i]+SiPlus1[i]+SjMinus1[i]+SjPlus1[i]
                if divisor < threshold:
                    divisor = 1

                row.append(i)
                col.append(i)
                data.append(-(SiMinus1[i]+SiPlus1[i]+SjMinus1[i]+SjPlus1[i])/divisor)

        mat = sp.csr_array((data, (row, col)), shape=(n, n))

        return mat

    def constructb(self,timeit=False):
        """
        Constructs the b vector of the least squares problem  
        """
        if timeit:  
            start = time.time()
        mu = self.mu
        z_0 = self.Z_0
        mu = mu.reshape(mu.shape[0]*mu.shape[1], mu.shape[2])
        mu_norm = np.linalg.norm(mu, axis=1)
        mu_dia = sp.diags_array(mu_norm, offsets=0, format='csr')
        #! TODO: Skal det være elementvise eller @?
        row1 = self.lambda_p * mu_dia @ z_0.flatten().T
        row1 = row1.reshape(row1.shape[0])

        rowZeros = np.zeros((z_0.shape[0]*z_0.shape[1]))
        if self.lambda_s is not None:
            self.b = np.concatenate((row1, rowZeros, rowZeros, rowZeros), axis=None)
        else:
            self.b = np.concatenate((row1, rowZeros, rowZeros), axis=None)
        if timeit:
            end = time.time()
            print("b construction time: ", end-start)
        
        #save b row 1
        self.br1 = row1

    def removeZeroRows(self):
        """
        Removes rows from A where b is zero in first term
        """
        if self.A is None:
            raise ValueError('A must be set')

        if self.b is None:
            raise ValueError('b must be set')

        if not isinstance(self.A, sp.csr_array):
            print(type(self.A))
            raise ValueError('A must be a csr array')
        
        if not isinstance(self.b, np.ndarray):
            print(type(self.b))
            raise ValueError('b must be a np array')

        # Find length of first term of b
        n = self.Z_0.shape[0]*self.Z_0.shape[1]
        b1 = self.b[:n]

        # Find indices of zero elements in b1
        zeroIndices = np.where(b1 == 0)[0]
        #print('zeroIndices: ', zeroIndices)
        #print('Count of zeroIndices: ', len(zeroIndices))

        self.row1Mask = np.ones(self.r1.shape[0], dtype=bool)
        self.row1Mask[zeroIndices] = False

        # Create mask for A
        mask = np.ones(self.A.shape[0], dtype=bool)
        mask[zeroIndices] = False

        #print("Size of A before removing zero rows: ", self.A.shape)
        # Remove rows in A where b is zero in first term
        self.A = self.A[mask,:]
        #print("Size of A after removing zero rows: ", self.A.shape)

        #print('Size of B before removing zero rows: ', self.b.shape[0])
        # Remove rows in b where b is zero in first term
        self.b = self.b[mask]
        #print('Size of B after removing zero rows: ', self.b.shape[0])

        #! remove this later
        # print('Non zero: ', np.count_nonzero(b1))
        # print('Zero: ', b1.shape[0] - np.count_nonzero(b1))
        # print(type(self.A))
        # print(type(self.b))

        # # Create temp mask and plot
        # tempMask = np.zeros(b1.shape[0])
        # tempMask[zeroIndices] = 1
        # tempMask = tempMask.reshape(self.Z_0.shape)
        # plt.imshow(tempMask)
        # plt.title('Mask of zero rows')
        
        # plt.figure()
        # testb = self.b[:n]
        # testb = testb.reshape(self.Z_0.shape)
        # plt.imshow(testb)
        # plt.title('b1')
        # plt.show()


    def solve(self, timeit=False):
        """
        Solves the least squares problem
        """
        if self.mask is None:
            raise ValueError('mask must be set (not used but required to ensure binary mask)')

        # Check correct format of A and b
        if not isinstance(self.A, sp.csr_array):
            print(type(self.A))
            raise ValueError('A must be a csr array')
        
        if not isinstance(self.b, np.ndarray):
            print(type(self.b))
            raise ValueError('b must be a np ndarray')

        if timeit:
            start = time.time()
        initial_guess = self.Z_0Dense.flatten()
        #print('Initial guess shape: ', initial_guess.shape)
        #print('B shape: ', self.b.shape)
        #x, istop, itn, r1norm = sp.linalg.lsqr(self.A, self.b,iter_lim = 10000, show = True, x0 = initial_guess)[:4]
        x, istop, itn, r1norm = sp.linalg.lsmr(self.A, self.b, maxiter= 10000, show=True, x0 = initial_guess)[:4]
        #x, istop, itn, r1norm = sp.linalg.lsqr(self.A, self.b, atol=1e-12, btol=1e-12)[:4]
        end = time.time()
        if timeit:
            print("lsqr time: ", end-start)

        self.Z = x.reshape(self.Z_0.shape)
        return self.Z, istop, itn, r1norm
    
    def evaluateErrorTermsAtEnd(self):
        # Row 1
        #Extract row1 from A
        pixelsInMask = np.where(self.row1Mask == 1)[0].shape[0]
        # print("pixelsinmask: ",pixelsInMask)
        r1 = self.A[:pixelsInMask,:]
        b1 = self.b[:pixelsInMask]
        row1b = r1 @ self.Z.flatten() - b1
        row1 = r1 @ self.Z.flatten()
        # print("Non zero elements in row 1: ", np.count_nonzero(row1))
        # print("Zero elements in row 1: ", len(row1) - np.count_nonzero(row1))
        # print("Shape of row1: ", row1.shape)
        # print("shape of z = ", self.Z.flatten().shape)
        # print("Shape of b1: ", b1.shape)
        # print("shape of r1: ", r1.shape)

        unmasked = np.zeros(self.row1Mask.shape)
        unmasked[self.row1Mask == 1] = row1
        # print("Shape of unmasked row1: ", unmasked.shape)
        unmaskedb = np.zeros(self.row1Mask.shape)
        unmaskedb[self.row1Mask == 1] = row1b

        # print("max unmasked row 1: ", np.max(unmasked))
        # # unMaskedFullSize = np.zeros(self.Z_0.flatten().shape)
        # # flatMask = self.mask.flatten()
        # # unMaskedFullSize[flatMask == 1] = row1
        # plt.subplot(1,3,1)
        # plt.imshow(unmasked.reshape(self.Z_0.shape))
        # plt.title("Unmasked row 1")
        # plt.colorbar()
        # plt.subplot(1,3,2)
        # plt.imshow(self.Z_0)
        # plt.title("Z_0")
        # plt.colorbar()
        # plt.subplot(1,3,3)
        # plt.imshow(unmaskedb.reshape(self.Z_0.shape))
        # plt.title("Unmasked row 1 - b")
        # plt.colorbar()
        # plt.show()

        # print("Shape of zeroremoved row1",row1.shape)
        # print("Shape of row1mask",self.row1Mask.shape)
        # print("Number of true elements in row 1 mask: ", np.count_nonzero(self.row1Mask))
        # #reconstruct row 1
        # r1mask = self.row1Mask
        # row1reconstructed = np.zeros(self.r1.shape[0])
        # row1reconstructed[r1mask] = row1
        # print("Shape of row1reconstructed",row1reconstructed.shape)
        


        #row1 = self.r1 @ self.Z.flatten()

        # Row 2
        row2 = self.r2 @ self.Z.flatten()

        # Row 3
        row3 = self.r3 @ self.Z.flatten()
        if self.r4 is not None:
            row4 = self.r4 @ self.Z.flatten()

            plt.subplot(2,2,1)
            plt.imshow(unmaskedb.reshape(self.Z_0.shape))
            plt.title('Row 1')
            plt.colorbar()
            
            plt.subplot(2,2,2)
            plt.imshow(row2.reshape(self.Z_0.shape))
            plt.title('Row 2')
            plt.colorbar()
            

            plt.subplot(2,2,3)
            plt.imshow(row3.reshape(self.Z_0.shape))
            plt.title('Row 3')
            plt.colorbar()

            plt.subplot(2,2,4)
            plt.imshow(row4.reshape(self.Z_0.shape))
            plt.title('Row 4')
            plt.colorbar()
        else:
            plt.subplot(1,3,1)
            plt.imshow(unmaskedb.reshape(self.Z_0.shape))
            plt.title('Row 1')
            plt.colorbar()
            
            plt.subplot(1,3,2)
            plt.imshow(row2.reshape(self.Z_0.shape))
            plt.title('Row 2')
            plt.colorbar()

            plt.subplot(1,3,3)
            plt.imshow(row3.reshape(self.Z_0.shape))
            plt.title('Row 3')
            plt.colorbar()

        plt.show()