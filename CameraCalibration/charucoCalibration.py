import cv2
import numpy as np
import glob
from cv2 import aruco
from scipy.spatial.transform import Rotation as Rotation
import matplotlib.pyplot as plt
from pytransform3d import transformations as tf
from pytransform3d.plot_utils import make_3d_axis

class CahrucoCalibration:
    def __init__(self):
        self.board = None
        self.arucoDict = None
        self.criteria = None
        self.stereoCriteria = None
    
    def calibrateCharuco(self, imgPath, mtx = None, dist=None, show = False):
        board = self.board
        arucoDict = self.arucoDict
        criteria = self.criteria
        
        allCorners = []
        allIds = []
        imgSize = None

        # Detecting markers
        images = sorted(glob.glob(imgPath))

        if images == []:
            print("No images found in the given path")
            print(imgPath)
            raise ValueError

        i = 0
        for fname in images:
            if show:
                print(fname, " ", i)
            i = i + 1
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict)

            # Refining detected markers
            if len(corners) > 0:
                for corner in corners:
                    cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

                ret, cCorner, cIDs  = aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if show:
                    print("corners: ", len(corners))

                if ret > 0:
                    allCorners.append(cCorner)
                    allIds.append(cIDs)
                    
                    if imgSize == None:
                        imgSize = gray.shape[::-1]
                    elif gray.shape[::-1] != imgSize:
                        Exception("All images must share the same size.")
                        exit()
                    
                    # Draw and display the corners
                    img = aruco.drawDetectedCornersCharuco(img, cCorner, cIDs)
                    if show:
                        cv2.imshow('img',img)
                        cv2.waitKey(0)
                else:
                    print("No charuco corners found")
                    print("fname: ", fname)
                    cv2.imshow('img',img)
                    cv2.waitKey(0)
            else:
                print("No markers found")
                cv2.imshow('img',img)
                cv2.waitKey(0)

        # Calibrating camera using charuco
        if mtx is None:
            flags = None
        else:
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
        ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = aruco.calibrateCameraCharucoExtended(charucoCorners=allCorners, charucoIds=allIds, board=board, imageSize=imgSize, cameraMatrix=mtx, distCoeffs=dist, flags=flags)

        # Reprojection error
        # mean_error = 0
        # for i in range(len(allCorners)):
        #     imgpoints2, _ = cv2.projectPoints(allCorners[i], rvecs[i], tvecs[i], mtx, dist)
        #     error = cv2.norm(allCorners[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        #     mean_error += error

        if show:
            print("Camera matrix : \n", mtx)
            print("dist : \n", dist)
            print("rvecs : \n", rvecs)
            print("tvecs : \n", tvecs)
            print("Standard deviation intrinsics: \n", stdDeviationsIntrinsics)
            print("Standard deviation extrinsics: \n", stdDeviationsExtrinsics)
            print("Per view errors: \n", perViewErrors)
            # print("reprojection error: ", mean_error/len(allCorners))

        #return mtx, dist, rvecs, tvecs, mean_error/len(allCorners), stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors
        return ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors
    
    def showCharucoAxis(self, imgPaths, rvecs, tvecs, mtx, dist):
        arucoDict = self.arucoDict
        board = self.board
        criteria = self.criteria
        
        # Extracting path of individual image stored in a given directory
        images = sorted(glob.glob(imgPaths))

        i = 0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            corners, ids, rejected_points = cv2.aruco.detectMarkers(gray, arucoDict)

            if len(corners) > 0:
                for corner in corners:
                    cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)
                
                ret, cCorner, cIDs = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                size_of_marker = 0.015
                rvecs, tvecs , _= aruco.estimatePoseSingleMarkers(corners, size_of_marker, mtx, dist)


                if ret > 0:
                    # Draw and display the corners
                    img = aruco.drawDetectedCornersCharuco(img, cCorner, cIDs)
                    img = aruco.drawDetectedMarkers(img, corners, ids)

                    # Draw axis
                    for i in range(len(tvecs)):
                        img = cv2.drawFrameAxes(img, mtx, dist, rvecs[i], tvecs[i], 0.01)

                    cv2.imshow('img',img)
                    cv2.waitKey(0)
                else:
                    print("No charuco corners found")
        cv2.destroyAllWindows()

    def calibrateSingleCamera(self, path, show = False, useStoredParameters = False, paramPath = None):
        if useStoredParameters:
            mtx = np.load(paramPath + 'mtx.npy')
            dist = np.load(paramPath + 'dist.npy')
        else:
            mtx = None
            dist = None

        imgPaths = path + '*.png'
        ret, mtx_est, dist_est, rvecs_est, tvecs_est, stdIntrinsics_est, stdExtrinsics_est, perViewErrors_est = self.calibrateCharuco(imgPath=imgPaths, mtx=mtx, dist=dist, show=show)
        
        if show:
            self.showCharucoAxis(imgPaths, rvecs_est, tvecs_est, mtx_est, dist_est)

        # save parameters
        np.save(path + 'mtx.npy', mtx_est)
        np.save(path + 'dist.npy', dist_est)
        np.save(path + 'rvecs.npy', rvecs_est)
        np.save(path + 'tvecs.npy', tvecs_est)
        np.save(path + 'stdIntrinsics.npy', stdIntrinsics_est)
        np.save(path + 'stdExtrinsics.npy', stdExtrinsics_est)
        np.save(path + 'perViewErrors.npy', perViewErrors_est)

        return ret, rvecs_est, tvecs_est
    
    def stereoCalibrate(self, path1, path2, gridSize):
        # Set some parameters
        criteria = self.stereoCriteria
        board = self.board
        arucoDict = self.arucoDict

        # Extracting path of individual image stored in a given directory
        images1 = sorted(glob.glob(path1 + '*.png'))
        images2 = sorted(glob.glob(path2 + '*.png'))

        imgPoints1, imgPoints2 = [], []

        for fname1 , fname2 in zip(images1, images2):
            img1 = cv2.imread(fname1)
            gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(fname2)
            gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            
            # Creating vector to store vectors of 3D and 2D points for each checkerboard image
            objpoints, imgPoints1, imgPoints2 = [], [], []
            
            size = board.getChessboardSize()

            # Defining the world coordinates for 3D points
            objp = np.zeros((1, size[0] * size[1], 3), np.float32)
            objp[0, :, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

            #Size of checkerboard square in mm
            objp = objp*gridSize

            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            corners1, ids1, rejectedImgPoints1 = aruco.detectMarkers(gray1, arucoDict)
            corners2, ids2, rejectedImgPoints2 = aruco.detectMarkers(gray2, arucoDict)
            
            # Refining detected markers
            if len(corners1) > 0 and len(corners2) > 0:
                for corner1 in corners1:
                    cv2.cornerSubPix(gray1, corner1, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)
                for corner2 in corners2:
                    cv2.cornerSubPix(gray2, corner2, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

                ret1, cCorner1, cIDs1  = aruco.interpolateCornersCharuco(corners1, ids1, gray1, board)
                ret2, cCorner2, cIDs2  = aruco.interpolateCornersCharuco(corners2, ids2, gray2, board)
                
                # Remove markers that are not in both images
                id_intersection = np.intersect1d(cIDs1, cIDs2)

                newCorners1 = []
                newIDs1 = []
                newCorners2 = []
                newIDs2 = []

                for i in range(len(cIDs1)):
                    if cIDs1[i] in id_intersection:
                        newCorners1.append(cCorner1[i])
                        newIDs1.append(cIDs1[i])
                for i in range(len(cIDs2)):
                    if cIDs2[i] in id_intersection:
                        newCorners2.append(cCorner2[i])
                        newIDs2.append(cIDs2[i])

                if len(newCorners1) < 4 or len(newCorners2) < 4:
                    print("Not enough common markers in both image " + fname1 + " and " + fname2)
                    continue

                if ret1 > 1 and ret2 > 1:
                    objPoints1, imgP1 = aruco.getBoardObjectAndImagePoints(board, np.array(newCorners1), np.array(newIDs1))
                    objPoints2, imgP2 = aruco.getBoardObjectAndImagePoints(board, np.array(newCorners2), np.array(newIDs2))
                    objpoints.append(objPoints1)
                    imgPoints1.append(imgP1)
                    imgPoints2.append(imgP2)
                elif ret1 and not ret2:
                    print("No corners found in image 2")
                elif not ret1 and ret2:
                    print("No corners found in image 1")
                else:
                    print("No corners found in image 1 or 2")
            else:
                print("No markers found in one or both images")
                cv2.imshow('img1',img1)
                cv2.imshow('img2',img2)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

        # Set and load parameters for stereo calbiration
        mtx1 = np.load(path1 + 'mtx.npy')
        dist1 = np.load(path1 + 'dist.npy')
        mtx2 = np.load(path2 + 'mtx.npy')
        dist2 = np.load(path2 + 'dist.npy')

        calibrationFlags = cv2.CALIB_FIX_INTRINSIC
        R = np.zeros((3,3))
        T = np.zeros((3,1))

        ret, _, _, _, _, R, T, E, F, rvecs, tvecs, perViewErrors = cv2.stereoCalibrateExtended(objpoints, imgPoints1, imgPoints2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1], R, T, criteria=criteria, flags=calibrationFlags)

        return ret, R, T, E, F, rvecs, tvecs, perViewErrors

    def _loadRobotPose(self, path):
        robotPoseC = np.loadtxt(path + "robotPoseC.csv", delimiter=",")
        rot = robotPoseC[:,3:6]
        rot_matrix = Rotation.from_rotvec(rot).as_matrix()
        pos = robotPoseC[:,0:3]
        #pos = pos*1000 # convert to mm
        return rot_matrix, pos

    def _inverseSignleTransformation(self, R, T):
        R_inv = np.transpose(R)
        T_inv = -R_inv @ T
        return R_inv, T_inv
    
    def _inverseBatchTransformation(self, R, T):
        R_inv = np.zeros(R.shape)
        T_inv = np.zeros(T.shape)
        for i in range(len(R)):
            R_inv[i] = np.transpose(R[i])
            T_inv[i] = -R_inv[i] @ T[i]
        return R_inv, T_inv
    
    def _rotvec2rotmatBatch(self, rotvec):
        rotmat = np.zeros((len(rotvec), 3, 3))
        for i in range(len(rotmat)):
            rotmat[i] = cv2.Rodrigues(rotvec[i])[0]
        return rotmat
    
    # draw transformations
    def drawTransformations(self, rotMat, trans, ax=None, linestyle=None):
        if ax is None:
            ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
        for i in range(len(rotMat)):
            transform = tf.transform_from(rotMat[i], trans[i])
            tf.plot_transform(ax=ax, A2B=transform, s=0.5, linestyle=linestyle)
        return ax
        
    def calibrateEyeInHand(self):
        raise NotImplementedError

    def calibrateEyeToHand(self, path, camera, method, show = False):
        # Get transformation from checkerboard to base
        R_gripper2base, t_gripper2base = self._loadRobotPose(path)
        R_base2gripper, t_base2gripper = self._inverseBatchTransformation(R_gripper2base, t_gripper2base)

        # Get transformation from camera to checkerboard
        _, rvecs_est, tvecs_est = self.calibrateSingleCamera(path + camera + '/')

        R_target2cam = np.array(self._rotvec2rotmatBatch(rvecs_est))
        t_target2cam = np.array(tvecs_est)

        R_cam2base_est, t_cam2base_est = cv2.calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, method=method)

        if show:
            raise NotImplementedError
            ax = drawTransformations(R_base2gripper, t_base2gripper, linestyle='-')
            rotm = Rotation.from_rotvec(np.squeeze(np.array(rvecs2), axis=2)).as_matrix()
            trans = np.squeeze(np.array(tvecs2), axis=2)
            ax = drawTransformations(R_cam2base_est, t_cam2base_est, ax=ax, linestyle="--")
            plt.tight_layout()
            plt.show()

        return R_cam2base_est, t_cam2base_est

if __name__ == "__main__":
    ############################################################################################################
    # Main
    ############################################################################################################
    calib = CahrucoCalibration()

    # Define the termination criteria for the corner sub-pixel algorithm
    calib.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define charuco board
    calib.arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    calib.board = aruco.CharucoBoard((10,8), 0.03, 0.022, calib.arucoDict)
    calib.board.setLegacyPattern(True)

    # Define path
    path = './calibMay11_forReport/'
    #path = './calibMay22/'

    # Define cameras
    cameras = ['rsIR1','pg'] #['rsIR1', 'rsIR2', 'rsColor', 'pg']
    cameras = ['pg']

    for camera in cameras:
        ret = calib.calibrateSingleCamera(path + camera + '/', show = False, useStoredParameters = False, paramPath = None)[0]
        print(camera + " intrinsic calibration done")
        print(camera + " intrinsic error: ", ret)


    # Stereo calibration
    if False:
        # Stereo criteria
        calib.stereoCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # gridSize
        gridSize = 0.030 # mm

        # Define paths for the two cameras, camera1 is the reference camera
        path1 = path + 'pg/'
        path2 = path + 'rsIR1/'

        # Stereo 
        retS, R, T, E, F, rvecs, tvecs, perViewErrors = calib.stereoCalibrate(path1, path2, gridSize)

        # save parameters
        np.save(path + 'R.npy', R)
        np.save(path + 'T.npy', T)
        np.save(path + 'E.npy', E)
        np.save(path + 'F.npy', F)
        np.save(path + 'rvecs.npy', rvecs)
        np.save(path + 'tvecs.npy', tvecs)
        np.save(path + 'perViewErrors.npy', perViewErrors)

        print("Stereo calibration done")
        print("Stereo error: ", retS)

    ############################################################################################################
    # Hand eye calibration NEW
    # https://github.com/opencv/opencv/blob/b5a9a6793b3622b01fe6c7b025f85074fec99491/modules/calib3d/test/test_calibration_hand_eye.cpp#L484
    ############################################################################################################
    camera = 'pg'

    R_cam2base_est, t_cam2base_est = calib.calibrateEyeToHand(path, camera, cv2.CALIB_HAND_EYE_PARK)

    T_cam2base = np.zeros((4,4))
    T_cam2base[0:3,0:3] = R_cam2base_est
    T_cam2base[0:3,3] = np.squeeze(t_cam2base_est)
    T_cam2base[3,3] = 1

    print("R: \n", R_cam2base_est)
    print("T: \n", t_cam2base_est)
    np.save(path + camera + '/T_cam2base.npy', T_cam2base)
