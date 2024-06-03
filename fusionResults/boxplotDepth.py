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

# Main function
if __name__ == "__main__":
    # Define config path
    if len(sys.argv) > 1:
        testType = sys.argv[1]
    else:
        testType = "./armadillo_difuse_k2_0.38_0.0001_depth"
        # testType = "./armadillo_plastic_k2_0.38_0.0001_depth"

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
    GT = GT*1000

    methods = ['LS','NearLS','RPCA','NearRPCA']
    save_path = "./boxplots/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    LSAE = []
    NearLSAE = []
    RPCAAE = []
    NearRPCAAE = []

    paths = ["./armadillo_difuse_k2_0.38_0.0001_depth/", "./armadillo_plastic_k2_0.38_0.0001_depth/"]
    for path in paths:
        LS, NearLS, RPCA, NearRPCA = loadDepths(path)
        LS = LS*1000
        NearLS = NearLS*1000
        RPCA = RPCA*1000
        NearRPCA = NearRPCA*1000

        for method in methods:
            D = locals()[method]
            if D is None or np.nonzero(D)[0].shape[0] < 10000: 
                print("Normals for ", method, " not found")
                if method == 'LS':
                    LSAE.append([])
                elif method == 'NearLS':
                    NearLSAE.append([])
                elif method == 'RPCA':
                    RPCAAE.append([])
                elif method == 'NearRPCA':
                    NearRPCAAE.append([])
                continue
            D = cropImage(D, x, y, w, h)*mask

            ae = calculate_depth_error(GT, D, mask)
            ae = ae[~np.isnan(ae)]
            ae = ae[ae != 0]

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
    labels_list = ['Diffuse','Plastic']
    width       = 1/len(labels_list)
    xlocations  = [ x*((1+ len(data_groups))*width) for x in range(len(data_group1)) ]

    symbol      = 'r+'
    ymin        = min ( [ val  for dg in data_groups  for data in dg for val in data ] ) - 1
    ymax        = max ( [ val  for dg in data_groups  for data in dg for val in data ])
    ymax = 16
    ax = plt.gca()
    ax.set_ylim(ymin,ymax)

    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    plt.ylabel('Absolute error [mm]')
    #plt.title('title')

    space = len(data_groups)/2
    offset = len(data_groups)/2


    # --- Offset the positions per group:

    group_positions = []
    for num, dg in enumerate(data_groups):    
        _off = (0 - space + (0.5+num))
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
    #ax.set_yticks(np.arange(0, ymax))

    plt.savefig(save_path + testType +"_boxplot.pdf", format="pdf", bbox_inches="tight")
    plt.show()


