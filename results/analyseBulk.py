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
        GT_geo = np.load('../data/bunny_difuse_10lights/trueNormals.npy')
        GT_sh = np.load('bunny_trueNormals_sh.npy')
    elif objectType == "armadillo":
        GT_geo = np.load('../data/armadillo_difuse_3lights/trueNormals.npy')
        GT_sh = np.load('armadillo_trueNormals_sh.npy')
    return GT_geo, GT_sh

def loadMask(objectType=None):
    if objectType is None:
        raise("Object is not given")
    if objectType == "bunny":
        mask = plt.imread('../data/bunny_difuse_10lights/meta_mask.png')
    elif objectType == "armadillo":
        mask = plt.imread('../data/armadillo_difuse_3lights/meta_mask.png')
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

def loadNormals(resultsPath):
    oldPath = "../results_oldSettings/" + resultsPath.split('.')[1]
    
    # Load normal maps
    # if os.path.exists(resultsPath + 'RPCA_est_normal.npy'):
    #     RPCA = np.load(resultsPath + 'RPCA_est_normal.npy')
    #     RPCA = np.reshape(RPCA, (RPCA.shape[0], RPCA.shape[1], 3))
    # else:
    #     RPCA = None
    
    if os.path.exists(oldPath + 'RPCA_est_normal.npy'):
        RPCA = np.load(oldPath + 'RPCA_est_normal.npy')
        RPCA = np.reshape(RPCA, (RPCA.shape[0], RPCA.shape[1], 3))
    else:
        RPCA = None 

    # if os.path.exists(resultsPath + 'LS_est_normal.npy'):
    #     LS = np.load(resultsPath + 'LS_est_normal.npy')
    #     LS = np.reshape(LS, (LS.shape[0], LS.shape[1], 3))
    # else:
    #     LS = None
    if os.path.exists(oldPath + 'LS_est_normal.npy'):
        LS = np.load(oldPath + 'LS_est_normal.npy')
        LS = np.reshape(LS, (LS.shape[0], LS.shape[1], 3))
    else:
        LS = None

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

def saveErrorMap(ae, mask, save_path):
    aeCopy = copy.deepcopy(ae)
    aeCopy = aeCopy.reshape(mask.shape)
    #invert mask
    mask = np.logical_not(mask)
    aeCopy[mask] = np.nan
    fig = plt.figure(2)
    fig.clf()
    ax = plt.axes()
    ax.cla()
    im = ax.imshow(aeCopy)
    #plt.title('Angular Error')
    plt.axis('off')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)

    plt.savefig(save_path,bbox_inches='tight')

# Main function
if __name__ == "__main__":
    # Define config path
    if len(sys.argv) > 1:
        testType = sys.argv[1]
    else:
        testType = "./armadillo_difuse"
        #testType = "./armadillo_plastic"
        testType = "./plastic"
        #testType = "./metal"
        #testType = "./difuse"

    if "armadillo" in testType:
        objectType = "armadillo"
    else:
        objectType = "bunny"
    
    SAVENORMALMAP = True
    SAVEERRORMAPS = True

    # Load data
    GT_geo, GT_sh = loadGroundTruth(objectType)
    mask = loadMask(objectType)

    # Crop
    x, y, w, h = getCropParams(mask)
    mask = cropImage(mask, x, y, w, h)
    mask = mask.astype(np.uint8)
    mask3D = np.stack((mask, mask, mask), axis=2)
    GT_geo = cropImage(GT_geo, x, y, w, h)*mask3D
    GT_sh = cropImage(GT_sh, x, y, w, h)*mask3D
    


    # Generate sphere normalsne:
    sphere = generateSphereNormals(200,200)
    smask = np.isnan(sphere[:, :, 2])
    sphere[smask, :] = 0
    smask = sphere[:, :, 2] == 0
    sphere[smask, :] = 1
    sphere = (sphere + 1.0) / 2.0 # Rescale

    paths = [testType +'_3lights/', testType +'_4lights/', testType +'_5lights/', testType +'_6lights/', testType +'_7lights/', testType +'_8lights/', testType +'_9lights/', testType +'_10lights/', testType +'_15lights/']
    numlights = [3, 4, 5, 6, 7, 8, 9, 10, 15]
    methods = ['LS','NearLS','RPCA','NearRPCA']
    save_path = testType + '_analysis/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    saveNormalMapWithSphere(GT_geo, h, w, name='GT', save_path=save_path+"_GT_geo.png",sphere=sphere)
    saveNormalMapWithSphere(GT_sh, h, w, name='GT', save_path=save_path+"_GT_sh.png",sphere=sphere)
    aeGT = calculate_angular_error(GT_geo, GT_sh, mask)
    saveErrorMap(aeGT, mask, save_path+"_errorGT.png")
    # mean, median, std, mean_squa, rms, max, min = evaluate_angle_error(aeGT)
    # print("GT: mean: ", np.around(mean, 3), " median: ", np.around(median, 3), " std: ", np.around(std, 3), " mean_squa: ", np.around(mean_squa, 3), " rms: ", np.around(rms, 3), " min: ", np.around(min ,3), " max: ", np.around(max,3))
    # exit()

    lsGeoRes = []
    nearGeoLSRes = []
    rpcaGeoRes = []
    nearRPCAGeoRes = []

    lsSHRes = []
    nearSHLSRes = []
    rpcaSHRes = []
    nearRPCASHRes = []
    

    # Create csv file
    with open(save_path+'geoResults.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['numLights', 'MeanLS','MeanNLS','MeanRPCA','MeanNRPCA',
                         'MedianLS','MedianNLS','MedianRPCA','MedianNRPCA',
                         'StdLS','StdNLS','StdRPCA','StdNRPCA',
                         'RMSLS','RMSNLS','RMSRPCA','RMSNRPCA'])
    with open(save_path+'shResults.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['numLights', 'MeanLS','MeanNLS','MeanRPCA','MeanNRPCA',
                         'MedianLS','MedianNLS','MedianRPCA','MedianNRPCA',
                         'StdLS','StdNLS','StdRPCA','StdNRPCA',
                         'RMSLS','RMSNLS','RMSRPCA','RMSNRPCA'])
    
    for path, lights in zip(paths,numlights):
        RPCA, LS, SBL, NearLS, NearRPCA = loadNormals(path)

        for method in methods:
            N = locals()[method]

            if N is None or np.nonzero(N)[0].shape[0] < 10000:
                print("Normals for ", method, " not found")
                if method == 'LS':
                    lsGeoRes.append(['-', '-', '-', '-'])
                    lsSHRes.append(['-', '-', '-', '-'])
                elif method == 'NearLS':
                    nearGeoLSRes.append(['-', '-', '-', '-'])
                    nearSHLSRes.append(['-', '-', '-', '-'])
                elif method == 'RPCA':
                    rpcaGeoRes.append(['-', '-', '-', '-'])
                    rpcaSHRes.append(['-', '-', '-', '-'])
                elif method == 'NearRPCA':
                    nearRPCAGeoRes.append(['-', '-', '-', '-'])
                    nearRPCASHRes.append(['-', '-', '-', '-'])
            else:
                N = cropImage(N, x, y, w, h)*mask3D
                if SAVENORMALMAP:
                    saveNormalMapWithSphere(N, h, w, name=method, save_path=save_path+str(lights)+"_"+method+".png",sphere=sphere)
                
                ae = calculate_angular_error(GT_geo, N, mask)
                if SAVEERRORMAPS:
                    saveErrorMap(ae, mask, save_path+str(lights)+"_"+method+"_error.png")
                mean, median, std, mean_squa, rms, max, min = evaluate_angle_error(ae)
                aeSH = calculate_angular_error(GT_sh, N, mask)
                meanSH, medianSH, stdSH, mean_squaSH, rmsSH, maxSH, minSH = evaluate_angle_error(aeSH)
                if SAVEERRORMAPS:
                    saveErrorMap(aeSH, mask, save_path+str(lights)+"_"+method+"_errorSH.png")
                if method == 'LS':
                    lsGeoRes.append([mean, median, std, rms])
                    lsSHRes.append([meanSH, medianSH, stdSH, rmsSH])
                elif method == 'NearLS':
                    nearGeoLSRes.append([mean, median, std, rms])
                    nearSHLSRes.append([meanSH, medianSH, stdSH, rmsSH])
                elif method == 'RPCA':
                    rpcaGeoRes.append([mean, median, std, rms])
                    rpcaSHRes.append([meanSH, medianSH, stdSH, rmsSH])
                elif method == 'NearRPCA':
                    nearRPCAGeoRes.append([mean, median, std, rms])
                    nearRPCASHRes.append([meanSH, medianSH, stdSH, rmsSH])
            print(testType, "lights:", lights, "method:", method)
            #print(method, ": mean: ", np.around(mean, 3), " median: ", np.around(median, 3), " std: ", np.around(std, 3), " mean_squa: ", np.around(mean_squa, 3), " rms: ", np.around(rms, 3), " min: ", np.around(min ,3), " max: ", np.around(max,3))
    
    with open(save_path+'geoResults.csv', mode='a') as file:
        for i in range(len(numlights)):
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([numlights[i], lsGeoRes[i][0], nearGeoLSRes[i][0], rpcaGeoRes[i][0] , nearRPCAGeoRes[i][0] , 
                            lsGeoRes[i][1] , nearGeoLSRes[i][1] , rpcaGeoRes[i][1] , nearRPCAGeoRes[i][1] , 
                            lsGeoRes[i][2] , nearGeoLSRes[i][2] , rpcaGeoRes[i][2] , nearRPCAGeoRes[i][2] , 
                            lsGeoRes[i][3] , nearGeoLSRes[i][3] , rpcaGeoRes[i][3] , nearRPCAGeoRes[i][3]])
    
    with open(save_path+'shResults.csv', mode='a') as file:
        for i in range(len(numlights)):
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([numlights[i], lsSHRes[i][0], nearSHLSRes[i][0], rpcaSHRes[i][0] , nearRPCASHRes[i][0] , 
                            lsSHRes[i][1] , nearSHLSRes[i][1] , rpcaSHRes[i][1] , nearRPCASHRes[i][1] , 
                            lsSHRes[i][2] , nearSHLSRes[i][2] , rpcaSHRes[i][2] , nearRPCASHRes[i][2] , 
                            lsSHRes[i][3] , nearSHLSRes[i][3] , rpcaSHRes[i][3] , nearRPCASHRes[i][3]])



