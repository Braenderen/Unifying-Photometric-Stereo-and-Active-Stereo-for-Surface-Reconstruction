import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy





def load_lighttxt(filename=None):
    """
    Load light file specified by filename.
    The format of lights.txt should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.txt
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.loadtxt(filename)
    return Lt.T

def load_camera_matrix(filename=None):
    """
    Load camera matrix specified by filename.
    The format of camera_matrix.npy should be
        fx 0 cx
        0 fy cy
        0 0 1

    :param filename: filename of camera_matrix.npy
    :return: camera matrix (3 \times 3)
    """
    if filename is None:
        raise ValueError("filename is None")
    K = np.load(filename)
    return K
    
def load_cam2base_matrix_npy(filename=None):
    """
    Load camera to base transformation matrix specified by filename.
    The format of cam2base_matrix.npy should be
        r11 r12 r13 t1
        r21 r22 r23 t2
        r31 r32 r33 t3
        0   0   0   1

    :param filename: filename of cam2base_matrix.npy
    :return: camera to base transformation matrix (4 \times 4)
    """
    if filename is None:
        raise ValueError("filename is None")
    T = np.load(filename)
    return T

def load_depth_npy(filename=None):
    """
    Load depth file specified by filename.
    The format of depth.npy should be
        depth1
        depth2
        ...
        depthf

    :param filename: filename of depth.npy
    :return: depth matrix (1 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    depth = np.load(filename)
    return depth

def load_light_transform_npy(filename=None):
    """
    Load light transformation matrix specified by filename.
    The format of light_transform.npy should be
        r11 r12 r13 t1
        r21 r22 r23 t2
        r31 r32 r33 t3
        0   0   0   1

    :param filename: filename of light_transform.npy
    :return: light transformation matrix (4 \times 4)
    """
    if filename is None:
        raise ValueError("filename is None")
    T = np.load(filename)
    return T

def load_cameras_matrix(filename=None):
    """
    Load camera matrix specified by filename.
    The format of camera_matrix.npy should be
        fx 0 cx
        0 fy cy
        0 0 1

    :param filename: filename of camera_matrix.npy
    :return: camera matrix (3 \times 3)
    """
    if filename is None:
        raise ValueError("filename is None")
    K = np.load(filename)
    return K

def load_baselight_npy(filename=None):
    """
    Load light file specified by filename.
    The format of lights.npy should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.npy
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.load(filename)
    return Lt.T


def load_lightnpy(filename=None):
    """
    Load light numpy array file specified by filename.
    The format of lights.npy should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.npy
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.load(filename)
    return Lt.T


def load_image(filename=None):
    """
    Load image specified by filename (read as a gray-scale)
    :param filename: filename of the image to be loaded
    :return img: loaded image
    """
    if filename is None:
        raise ValueError("filename is None")
    return cv2.imread(filename, 0)


def load_images(foldername=None, ext=None):
    """
    Load images in the folder specified by the "foldername" that have extension "ext"
    :param foldername: foldername
    :param ext: file extension
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None or ext is None:
        raise ValueError("filename/ext is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*." + ext)):
        im = cv2.imread(fname).astype(np.float64)
        if im.ndim == 3:
            # Assuming that RGBA will not be an input
            im = np.mean(im, axis=2)   # RGB -> Gray
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width


def load_npyimages(foldername=None):
    """
    Load images in the folder specified by the "foldername" in the numpy format
    :param foldername: foldername
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None:
        raise ValueError("filename is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*.npy")):
        im = np.load(fname)
        if im.ndim == 3:
            im = np.mean(im, axis=2)
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width


def disp_normalmap(normal=None, height=None, width=None, delay=0, name=None):
    """
    Visualize normal as a normal map
    :param normal: array of surface normal (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :return: None
    """
    if normal is None:
        raise ValueError("Surface normal `normal` is None")
    N_copy = copy.deepcopy(normal)
    N = np.reshape(N_copy, (height, width, 3))  # Reshape to image coordinates
    #N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
    N = (N + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map'
    #cv2.imshow(name, N)

    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(N)
    plt.show()

    # cv2.waitKey(delay)
    # cv2.destroyWindow(name)
    # cv2.waitKey(1)    # to deal with frozen window...

def disp_2_normal_maps(normal=None,normal2=None, height=None, width=None, delay=0, name=None):
 
  
    if normal is None:
        raise ValueError("Surface normal `normal` is None")
    
    Normals = copy.deepcopy(normal)
    Normals2 = copy.deepcopy(normal2)

    N = np.reshape(Normals, (height, width, 3))  # Reshape to image coordinates
    N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
    N = (N + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map'
    #cv2.imshow(name, N)
        
    if normal is None:
        raise ValueError("Surface normal `normal` is None")
    
    N2 = np.reshape(Normals2, (height, width, 3))  # Reshape to image coordinates
    N2[:, :, 0], N2[:, :, 2] = N2[:, :, 2], N2[:, :, 0].copy()  # Swap RGB <-> BGR
    N2 = (N2 + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map 2'
        
    
    # ax = plt.gca()
    # fig = plt.gcf()
        
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')


    axs[0].imshow(N)
    axs[1].imshow(N2)
    
    plt.show()


def save_normalmap_as_npy(filename=None, normal=None, height=None, width=None):
    """
    Save surface normal array as a numpy array
    :param filename: filename of the normal array
    :param normal: surface normal array (height \times width \times 3)
    :return: None
    """
    if filename is None:
        raise ValueError("filename is None")
    N = np.reshape(normal, (height, width, 3))
    np.save(filename, N)


def load_normalmap_from_npy(filename=None):
    """
    Load surface normal array (which is a numpy array)
    :param filename: filename of the normal array
    :return: surface normal (numpy array) in formatted in (height, width, 3).
    """
    if filename is None:
        raise ValueError("filename is None")
    return np.load(filename)


def evaluate_angular_error(gtnormal=None, normal=None, background=None):
    if gtnormal is None or normal is None:
        raise ValueError("surface normal is not given")
    
    #disp_normalmap(normal, 1200, 1980)
    ae = np.multiply(gtnormal, normal)
    aesum = np.sum(ae, axis=1)
    coord = np.where(aesum > 1.0)
    aesum[coord] = 1.0
    coord = np.where(aesum < -1.0)
    aesum[coord] = -1.0
    ae = np.arccos(aesum) * 180.0 / np.pi
    if background is not None:
        ae[background] = 0
    return ae

