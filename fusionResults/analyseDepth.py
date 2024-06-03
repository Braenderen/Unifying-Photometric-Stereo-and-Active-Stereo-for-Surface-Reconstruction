import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import csv
import sys

# Load ground truth
def loadGroundTruth(objectType=None):
    if objectType is None:
        raise("Object is not given")
    if objectType == "bunny":
        GT = np.load('depthBunny.npy')
    elif objectType == "armadillo":
        GT = np.load('depthArmadillo.npy')
    return GT

def loadMask(objectType=None):
    if objectType is None:
        raise("Object is not given")
    if objectType == "bunny":
        mask = plt.imread("maskBunny.png")
    elif objectType == "armadillo":
        mask = plt.imread("maskArmadillo.png")
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
    borderpx = 10
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

def loadDepths(resultsPath):
    
    # Load depth maps
    if os.path.exists(resultsPath + 'LS_depth.npy'):
        LS = np.load(resultsPath + 'LS_depth.npy')
    else:
        LS = None

    if os.path.exists(resultsPath + 'Near_depth.npy'):
        NearLS = np.load(resultsPath + 'Near_depth.npy')
    else:
        NearLS = None
    
    if os.path.exists(resultsPath + 'RPCA_depth.npy'):
        RPCA = np.load(resultsPath + 'RPCA_depth.npy')
    else:
        RPCA = None

    if os.path.exists(resultsPath + 'NearRPCA_depth.npy'):
        NearRPCA = np.load(resultsPath + 'NearRPCA_depth.npy')
    else:
        NearRPCA = None

    return LS, NearLS, RPCA, NearRPCA

def loadNormals(resultsPath):
    oldPath = "../results_oldSettings/" + resultsPath.split('.')[1]
    
    # Load normal maps
    if os.path.exists(resultsPath + 'RPCA_est_normal.npy'):
        RPCA = np.load(resultsPath + 'RPCA_est_normal.npy')
        RPCA = np.reshape(RPCA, (RPCA.shape[0], RPCA.shape[1], 3))
    else:
        RPCA = None
    
    # if os.path.exists(oldPath + 'RPCA_est_normal.npy'):
    #     RPCA = np.load(oldPath + 'RPCA_est_normal.npy')
    #     RPCA = np.reshape(RPCA, (RPCA.shape[0], RPCA.shape[1], 3))
    # else:
    #     RPCA = None 

    if os.path.exists(resultsPath + 'LS_est_normal.npy'):
        LS = np.load(resultsPath + 'LS_est_normal.npy')
        LS = np.reshape(LS, (LS.shape[0], LS.shape[1], 3))
    else:
        LS = None
    # if os.path.exists(oldPath + 'LS_est_normal.npy'):
    #     LS = np.load(oldPath + 'LS_est_normal.npy')
    #     LS = np.reshape(LS, (LS.shape[0], LS.shape[1], 3))
    # else:
    #     LS = None

    # if os.path.exists(resultsPath + 'SBL_est_normal.npy'):
    #     SBL = np.load(resultsPath + 'SBL_est_normal.npy')
    #     SBL = np.reshape(SBL, (SBL.shape[0], SBL.shape[1], 3))
    # else:
    #     SBL = None

    if os.path.exists(oldPath + 'SBL_est_normal.npy'):
        SBL = np.load(oldPath + 'SBL_est_normal.npy')
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

def saveDepthMap(depth=None, name=None, save_path=None):
    """
    Visualize depth as a depth map
    :param depth: array of depth (p \times 1)
    :param name: display name
    :return: None
    """
    if depth is None:
        raise ValueError("Depth `depth` is None")
    D = copy.deepcopy(depth)
    if name is None:
        name = 'depth map'
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(D)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    else:
        plt.savefig('depth_map.png',bbox_inches='tight')
    # plt.title(name)
    # plt.show()

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

    if sphere is not None:
        # insert into normals in upper right corner
        swidth = sphere.shape[1]
        sheight = sphere.shape[0]

        if "armadillo" in save_path:
            N[330:330+sheight,width-swidth:width,:] = sphere
        else:
            N[10:sheight+10,width-swidth-10:width-10,:] = sphere
    
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

def calculate_depth_error(gt=None, depth=None, mask=None):
    if gt is None or depth is None:
        raise ValueError("Depth is not given")
    
    gt = copy.deepcopy(gt)
    gt = gt.flatten()
    d = copy.deepcopy(depth)
    d = d.flatten()
    mask = copy.deepcopy(mask)
    mask = mask.flatten()
    if mask is not None:
        d = d*mask
        gt = gt*mask
    # d = d - np.min(d)
    # d = d / np.max(d)
    # gt = gt - np.min(gt)
    # gt = gt / np.max(gt)
    de = np.abs(gt - d)
    return de

def calculate_angular_error(gtnormal=None, normal=None, background=None):
    if gtnormal is None or normal is None:
        raise ValueError("surface normal is not given")
    
    #disp_normalmap(normal, 1200, 1980)
    gt = copy.deepcopy(gtnormal)
    
    gt = np.reshape(gt, (gt.shape[0]*gt.shape[1], 3))
    n = copy.deepcopy(normal)
    n = np.reshape(n, (n.shape[0]*n.shape[1], 3))
    mask = copy.deepcopy(background)
    mask = mask.flatten()
    ae = np.multiply(gt, n)
    aesum = np.sum(ae, axis=1)
    coord = np.where(aesum > 1.0)
    aesum[coord] = 1.0
    coord = np.where(aesum < -1.0)
    aesum[coord] = -1.0
    ae = np.arccos(aesum) * 180.0 / np.pi
    if mask is not None:
        ae = ae*mask
    return ae

def evaluate_depth_error(de):
    # plt.imshow(de.reshape(837,898))
    # plt.title('Depth Error')
    # plt.show()
    deCopy = copy.deepcopy(de)
    deNonNan = deCopy[~np.isnan(deCopy)]
    deNonZero = deNonNan[deNonNan != 0]
    mean = np.mean(deNonZero)
    median = np.median(deNonZero)
    std = np.std(deNonZero)
    mean_squa = np.mean(deNonZero**2)
    rms = np.sqrt(np.mean(deNonZero**2))
    max = np.max(deNonZero)
    min = np.min(deNonZero)
    # set 3 decimal places
    # mean = np.around(mean, 3)
    # median = np.around(median, 3)
    # std = np.around(std, 3)
    # mean_squa = np.around(mean_squa, 3)
    # rms = np.around(rms, 3)
    # max = np.around(max, 3)
    # min = np.around(min, 3)

    return mean, median, rms

def evaluate_angle_error(ae):
    # plt.imshow(ae.reshape(837,898))
    # plt.title('Angular Error')
    # plt.show()
    
    aeNon90 = ae[ae != 90]
    aeNonZero = aeNon90[aeNon90 != 0]

    mean = np.mean(aeNonZero)
    median = np.median(aeNonZero)
    std = np.std(aeNonZero)
    mean_squa = np.mean(aeNonZero**2)
    rms = np.sqrt(np.mean(aeNonZero**2))
    max = np.max(aeNonZero)
    min = np.min(aeNonZero)
    # set 3 decimal places
    # mean = np.around(mean, 3)
    # median = np.around(median, 3)
    # std = np.around(std, 3)
    # mean_squa = np.around(mean_squa, 3)
    # rms = np.around(rms, 3)
    # max = np.around(max, 3)
    # min = np.around(min, 3)

    return mean, median, std, mean_squa, rms, max, min

def saveErrorMap(ae, mask, save_path, cutoff=None):
    aeCopy = copy.deepcopy(ae)
    aeCopy = aeCopy.reshape(mask.shape)
    #invert mask
    mask = np.logical_not(mask)
    aeCopy[mask] = np.nan
    fig = plt.figure(2)
    fig.clf()
    ax = plt.axes()
    ax.cla()
    
    tempMask = np.ones_like(aeCopy)
    tempMask[aeCopy is np.nan] = 0
    # tempMask[aeCopy > np.percentile(aeCopy[~np.isnan(aeCopy)], 95)] = 0
    

    std = np.std(aeCopy[~np.isnan(aeCopy)])
    if cutoff is not None:
        tempMask[aeCopy>cutoff] = 0
    aeMasked = copy.deepcopy(aeCopy)
    aeMasked[tempMask == 0] = np.nan

    im = ax.imshow(aeMasked)
    #plt.title('Angular Error')
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    plt.savefig(save_path,bbox_inches='tight')
    # plt.title("Error map")
    # plt.show()

def saveErrorMapPercentile(ae, mask, save_path):

    aeCopy = copy.deepcopy(ae)
    aeCopy = aeCopy.reshape(mask.shape)
    #invert mask
    mask = np.logical_not(mask)
    aeCopy[mask] = np.nan
    fig = plt.figure(2)
    fig.clf()
    ax = plt.axes()
    ax.cla()
    
    tempMask = np.ones_like(aeCopy)
    tempMask[aeCopy is np.nan] = 0
    tempMask[aeCopy > np.percentile(aeCopy[~np.isnan(aeCopy)], 95)] = 0
    

    std = np.std(aeCopy[~np.isnan(aeCopy)])
    #tempMask[aeCopy>std*stdScaler] = 0
    aeMasked = copy.deepcopy(aeCopy)
    aeMasked[tempMask == 0] = np.nan

    im = ax.imshow(aeMasked)
    #plt.title('Angular Error')
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    plt.savefig(save_path,bbox_inches='tight')
    # plt.title("Error map")
    # plt.show()

# Main function
if __name__ == "__main__":
    # Define config path
    if len(sys.argv) > 1:
        testType = sys.argv[1]
    else:
        #testType = "./armadillo_difuse_k2_0.38_0.0001_depth"
        testType = "./armadillo_plastic_k2_0.38_0.0001_depth"

    if "armadillo" in testType:
        objectType = "armadillo"
    else:
        objectType = "bunny"
    
    print("Object type: ", objectType)

    SAVEDEPTH = True
    SAVEERRORMAPS = True

    # Load data
    GT = loadGroundTruth(objectType)
    mask = loadMask(objectType)

    # Crop
    x, y, w, h = getCropParams(mask)
    mask = cropImage(mask, x, y, w, h)
    mask = mask.astype(np.uint8)
    GT = cropImage(GT, x, y, w, h)
    GT = GT*mask

    methods = ['LS','NearLS','RPCA','NearRPCA']
    data_path = testType + '/'
    save_path = testType + '_analysis/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    saveDepthMap(GT, name='GT', save_path=save_path+"_GT.png")

    lsRes = []
    neaLSRes = []
    rpcaRes = []
    nearRPCARes = []

    # Create csv file
    with open(save_path+'results.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['MeanLS','MeanNLS','MeanRPCA','MeanNRPCA',
                         'MedianLS','MedianNLS','MedianRPCA','MedianNRPCA',
                         'RMSLS','RMSNLS','RMSRPCA','RMSNRPCA'])


    LS, NearLS, RPCA, NearRPCA = loadDepths(data_path)
    GT= GT*1000
    LS = LS*1000
    NearLS = NearLS*1000
    RPCA = RPCA*1000
    NearRPCA = NearRPCA*1000

    for method in methods:
        D = locals()[method]

        if D is None or np.nonzero(D)[0].shape[0] < 10000:
            print("Depth for ", method, " not found")
            if method == 'LS':
                lsRes.append(['-', '-', '-', '-'])
            elif method == 'NearLS':
                neaLSRes.append(['-', '-', '-', '-'])
            elif method == 'RPCA':
                rpcaRes.append(['-', '-', '-', '-'])
            elif method == 'NearRPCA':
                nearRPCARes.append(['-', '-', '-', '-'])
        else:
            D = cropImage(D, x, y, w, h)*mask
            if SAVEDEPTH:
                saveDepthMap(D, name=method, save_path=save_path+method+"_depth.png")
            
            de = calculate_depth_error(GT, D, mask)
            if method == 'LS':
                #aeCopy[~np.isnan(aeCopy)]
                deCopy = copy.deepcopy(de)
                deNonNan = deCopy[~np.isnan(deCopy)]
                percentile95 = np.percentile(deNonNan[deNonNan != 0], 95)
                std6 = np.std(deNonNan[deNonNan != 0])*6
                
            print("LS 95 percentile: ", percentile95)
            print("LS 6 std: ", std6)
            if SAVEERRORMAPS:
                saveErrorMap(de, mask, save_path+method+"_error_6std.png", cutoff=std6)
                saveErrorMap(de, mask, save_path+method+"_error_95percentile.png", cutoff=percentile95)
            
            mean, median, rms = evaluate_depth_error(de)
            if method == 'LS':
                lsRes.append([mean, median, rms])
            elif method == 'NearLS':
                neaLSRes.append([mean, median, rms])
            elif method == 'RPCA':
                rpcaRes.append([mean, median, rms])
            elif method == 'NearRPCA':
                nearRPCARes.append([mean, median, rms])
        print(testType, "method:", method)
    
    with open(save_path+'results.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([lsRes[0][0], neaLSRes[0][0], rpcaRes[0][0], nearRPCARes[0][0],
                        lsRes[0][1], neaLSRes[0][1], rpcaRes[0][1], nearRPCARes[0][1],
                        lsRes[0][2], neaLSRes[0][2], rpcaRes[0][2], nearRPCARes[0][2]])




