import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import csv



def loadMask(path=None):
    mask = plt.imread(path)
    return mask

def disp_normalmap(normal=None, height=None, width=None, name=None, save_path=None, sphere_rad_px=None):
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
    # Rotate the normals by 180 degrees about y axis
    N[:, :, 0] = -N[:, :, 0]
    N[:, :, 1] = N[:, :, 1]
    N[:, :, 2] = -N[:, :, 2]

    # Rotate the normals by 180 degrees about z axis
    N[:, :, 0] = -N[:, :, 0]
    N[:, :, 1] = -N[:, :, 1]
    N[:, :, 2] = N[:, :, 2]

    # Set normals to one where N[:, :, 2] = 0 or nan
    mask = np.isnan(N[:, :, 2])
    N[mask, :] = 0
    mask = N[:, :, 2] == 0
    N[mask, :] = 1

    N = (N + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map'

    if sphere_rad_px is not None:
        sphere = generateSphereNormals(sphere_rad_px,sphere_rad_px)

        mask = np.isnan(sphere[:, :, 2])
        sphere[mask, :] = 0
        mask = sphere[:, :, 2] == 0
        sphere[mask, :] = 1
        sphere = (sphere + 1.0) / 2.0 # Rescale

        # insert into normals in upper right corner
        swidth = sphere.shape[1]
        sheight = sphere.shape[0]

        N[10:sheight+10,width-swidth-10:width-10,:] = sphere
    
    ax = plt.gca()
    fig = plt.gcf()
    ax.imshow(N)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    else:
        plt.savefig('normal_map.png',bbox_inches='tight')
    # plt.title(name)
    # plt.show()


def getCropParams(mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    borderpx = 25
    x = x - borderpx
    y = y - borderpx
    w = w + 2*borderpx
    h = h + 2*borderpx
    return x, y, w, h

def cropImage(img, x, y, w, h):
    if len(img.shape) == 3:
        return img[y:y+h, x:x+w, :]
    elif len(img.shape) == 2:
        return img[y:y+h, x:x+w]
    else:
        raise ValueError("Image has wrong dimensions")

def loadNormals(resultsPath):
    # Load normal maps
    if os.path.exists(resultsPath + 'RPCA_est_normal.npy'):
        RPCA = np.load(resultsPath + 'RPCA_est_normal.npy')
        RPCA = np.reshape(RPCA, (RPCA.shape[0], RPCA.shape[1], 3))
    else:
        RPCA = None

    if os.path.exists(resultsPath + 'LS_est_normal.npy'):
        LS = np.load(resultsPath + 'LS_est_normal.npy')
        LS = np.reshape(LS, (LS.shape[0], LS.shape[1], 3))
    else:
        LS = None

    if os.path.exists(resultsPath + 'SBL_est_normal.npy'):
        SBL = np.load(resultsPath + 'SBL_est_normal.npy')
        SBL = np.reshape(SBL, (SBL.shape[0], SBL.shape[1], 3))
    else:
        SBL = None

    if os.path.exists(resultsPath + 'Near_est_normal.npy'):
        NearLS = np.load(resultsPath + 'Near_est_normal.npy')
        NearLS = np.reshape(NearLS, (NearLS.shape[0], NearLS.shape[1], 3))
    else:
        NearLS = None
    
    if os.path.exists(resultsPath + 'NearRPCA_est_normal.npy'):
        NearRPCA = np.load(resultsPath + 'NearRPCA_est_normal.npy')
        NearRPCA = np.reshape(NearRPCA, (NearRPCA.shape[0], NearRPCA.shape[1], 3))
    else:
        NearRPCA = None

    return RPCA, LS, SBL, NearLS, NearRPCA


def generateSphereNormals(width, height):
    # Generate half sphere of normals with radius 1
    N = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            x = (i - height/2) / (height/2)
            y = (j - width/2) / (width/2)
            zsqr = 1**2 - x**2 - y**2
            if zsqr < 0:
                continue
            z = np.sqrt(zsqr)
            N[i, j, :] = [x, y, z]

    #plot in 3d scatterplot
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(N[:,:,0], N[:,:,1], N[:,:,2])
    # plt.show()


    # # Rotate the normals by 180 degrees about z axis
    # N[:, :, 0] = N[:, :, 0]
    # N[:, :, 1] = N[:, :, 1]
    # N[:, :, 2] = -N[:, :, 2]

    # # Rotate the normals by 180 degrees about y axis
    # N[:, :, 0] = N[:, :, 0]
    # N[:, :, 1] = -N[:, :, 1]
    # N[:, :, 2] = N[:, :, 2]

    # Rotate the normals by 90 degrees about z axis
    N = np.rot90(N, k=1, axes=(0, 1))
    
    # Normalize normals
    N = N / np.linalg.norm(N, axis=2)[:, :, None]

    return N

def saveNormalMapWithSphere(normal=None, height=None, width=None, name=None, save_path=None, sphere=None):
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
    if sphere is None:
        raise ValueError("Surface normal `sphere` is None")
    N_copy = copy.deepcopy(normal)
    N = np.reshape(N_copy, (height, width, 3))  # Reshape to image coordinates
    
    # Rotate the normals by 180 degrees about y axis
    N[:, :, 0] = -N[:, :, 0]
    N[:, :, 1] = N[:, :, 1]
    N[:, :, 2] = -N[:, :, 2]

    # Rotate the normals by 180 degrees about z axis
    N[:, :, 0] = -N[:, :, 0]
    N[:, :, 1] = -N[:, :, 1]
    N[:, :, 2] = N[:, :, 2]

    # Set normals to one where N[:, :, 2] = 0 or nan
    mask = np.isnan(N[:, :, 2])
    N[mask, :] = 0
    mask = N[:, :, 2] == 0
    N[mask, :] = 1

    N = (N + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map'

    # insert into normals in upper right corner
    swidth = sphere.shape[1]
    sheight = sphere.shape[0]

    N[height-sheight:height,:swidth,:] = sphere
    
    fig = plt.figure(1)
    ax = plt.gca()
    ax.imshow(N)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    else:
        plt.savefig('normal_map.png',bbox_inches='tight')
    # plt.title(name)
    # plt.show()
    # exit()


# Main function
if __name__ == "__main__":
    # Load data
    maskPath = '../data/real_objects/BuddhaDome/meta_mask.png'
    mask = loadMask(maskPath)

    x, y, w, h = getCropParams(mask)
    mask = cropImage(mask, x, y, w, h)
    mask = mask.astype(np.uint8)
    mask3D = np.stack((mask, mask, mask), axis=2)

    # Generate sphere normalsne:
    sphere = generateSphereNormals(200,200)
    smask = np.isnan(sphere[:, :, 2])
    sphere[smask, :] = 0
    smask = sphere[:, :, 2] == 0
    sphere[smask, :] = 1
    sphere = (sphere + 1.0) / 2.0 # Rescale

    paths = ["./buddhaJune2/"]
    numlights = [33]
    methods = ['LS','NearLS','SBL','RPCA','NearRPCA']
    save_path = "buddhaJune2_anaylsis/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for path, lights in zip(paths,numlights):
        RPCA, LS, SBL, NearLS, NearRPCA = loadNormals(path)

        for method in methods:
            N = locals()[method]
            if N is None:
                print("Normals for ", method, " not found")
                continue
            N = cropImage(N, x, y, w, h)

            saveNormalMapWithSphere(N, h, w, name=method, save_path=save_path+str(lights)+"_"+method+".png",sphere=sphere)

