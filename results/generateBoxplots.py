import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import csv

# Load ground truth
def loadGroundTruth():
    GT = np.load('../data/bunny_difuse_10lights/trueNormals.npy')
    return GT

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

    # insert into normals in upper right corner
    swidth = sphere.shape[1]
    sheight = sphere.shape[0]

    N[10:sheight+10,width-swidth-10:width-10,:] = sphere
    
    fig = plt.figure(1)
    ax = plt.gca()
    ax.imshow(N)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    else:
        plt.savefig('normal_map.png',bbox_inches='tight')
    plt.title(name)
    plt.show()


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
    aeCopy = aeCopy.reshape(837,898)
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
    # Load data
    GT = loadGroundTruth()
    maskPath = '../data/bunny_difuse_10lights/meta_mask.png'
    mask = loadMask(maskPath)

    x, y, w, h = getCropParams(mask)

    mask = cropImage(mask, x, y, w, h)
    mask = mask.astype(np.uint8)
    mask3D = np.stack((mask, mask, mask), axis=2)
    GT = cropImage(GT, x, y, w, h)*mask3D

    # Generate sphere normalsne:
    sphere = generateSphereNormals(200,200)
    smask = np.isnan(sphere[:, :, 2])
    sphere[smask, :] = 0
    smask = sphere[:, :, 2] == 0
    sphere[smask, :] = 1
    sphere = (sphere + 1.0) / 2.0 # Rescale

    #paths = ['difuse_3lights/', 'difuse_4lights/', 'difuse_5lights/', 'difuse_6lights/', 'difuse_7lights/', 'difuse_8lights/', 'difuse_9lights/', 'difuse_10lights/', 'difuse_15lights/']
    numlights = [3, 4, 5, 6, 7, 8, 9, 10, 15]
    #numlights = [10]

    # imageType = "plasticNarrow"
    # imageType = "plastic"
    # imageType = "metal"
    # imageType = "metalNarrow"
    imageType = 'difuseWide'
    paths = [imageType + '_3lights/', imageType + '_4lights/', imageType + '_5lights/', imageType + '_6lights/', imageType + '_7lights/', imageType + '_8lights/', imageType + '_9lights/', imageType + '_10lights/', imageType + '_15lights/']

    methods = ['LS','NearLS','RPCA','NearRPCA']
    save_path= 'boxplots/'
    
    #paths = ['difuse_10lights/']
    #numlights = [10]
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    LSAE = []
    NearLSAE = []
    RPCAAE = []
    NearRPCAAE = []
    
    for path, lights in zip(paths,numlights):
        RPCA, LS, SBL, NearLS, NearRPCA = loadNormals(path)

        for method in methods:
            N = locals()[method]
            if N is None:
                print("Normals for ", method, " not found")
                continue
            N = cropImage(N, x, y, w, h)*mask3D

            ae = calculate_angular_error(GT, N, mask)

            if method == 'LS':
                LSAE.append(ae.tolist())
            elif method == 'NearLS':
                NearLSAE.append(ae.tolist())
            elif method == 'RPCA':
                RPCAAE.append(ae.tolist())
            elif method == 'NearRPCA':
                NearRPCAAE.append(ae.tolist())

    
    ##### BOX PLOTS #####

    # there are 4 individuals, each one tested under 3 different settings

    # --- Random data, e.g. results per algorithm:
    data_group1 = LSAE
    data_group2 = NearLSAE
    data_group3 = RPCAAE
    data_group4 = NearRPCAAE

    colors = ['pink', 'lightblue', 'lightgreen', 'violet']

    # we compare the performances of the 4 individuals within the same set of 3 settings 
    data_groups = [data_group1, data_group2, data_group3, data_group4]

    # --- Labels for your data:
    labels_list = ['3','4', '5', '6', '7', '8', '9', '10', '15']
    width       = 1/len(labels_list)
    xlocations  = [ x*((1+ len(data_groups))*width) for x in range(len(data_group1)) ]

    symbol      = 'r+'
    ymin        = min ( [ val  for dg in data_groups  for data in dg for val in data ] )-5
    ymax        = max ( [ val  for dg in data_groups  for data in dg for val in data ])
    ymax = 100

    ax = plt.gca()
    ax.set_ylim(ymin,ymax)

    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    plt.xlabel('Number of lights')
    plt.ylabel('Angular error [degrees]')
    #plt.title('title')

    space = len(data_groups)/2
    offset = len(data_groups)/2


    # --- Offset the positions per group:

    group_positions = []
    for num, dg in enumerate(data_groups):    
        _off = (0 - space + (0.5+num))
        print(_off)
        group_positions.append([x+_off*(width+0.01) for x in xlocations])
    
    elements = []

    for dg, pos, c in zip(data_groups, group_positions, colors):
        elements.append( ax.boxplot(dg, 
                    sym=symbol,
                    labels=['']*len(labels_list),
        #            labels=labels_list,
                    positions=pos, 
                    widths=width, 
                    boxprops=dict(facecolor=c),
        #             capprops=dict(color=c),
        #            whiskerprops=dict(color=c),
        #            flierprops=dict(color=c, markeredgecolor=c),                       
                    medianprops=dict(color='grey'),
        #           notch=False,  
        #           vert=True, 
        #           whis=1.5,
        #           bootstrap=None, 
        #           usermedians=None, 
        #           conf_intervals=None,
                    patch_artist=True,
                    showfliers=False
                    ))
    methods = ['LS','NLS','RPCA','NRPCA']
    ax.legend([element["boxes"][0] for element in elements], methods)
    ax.set_xticks( xlocations )
    ax.set_xticklabels( labels_list, rotation=0 )
    ax.set_yticks(np.arange(0, ymax, 10))

    plt.savefig(save_path + imageType +"_boxplot.pdf", format="pdf", bbox_inches="tight")
    plt.show()


