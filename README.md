# Unifying-Photometric-Stereo-and-Active-Stereo-for-Surface-Reconstruction# Unifying Photometric Stereo and Active Stereo for Surface Reconstruction
This git repository accompanies the master's thesis 

![normals](figures/normalmap.png)

![mesh](figures/mesh.png)
## Installation

This repository is depnedent on the following python packages:
- matplotlib
- mitsuba
- numpy
- Open3D
- OpenCV2
- PIL
- pybind11
- pytransform3d
- pyvista
- scipy (listed twice)
- sklearn
- tqdm



For interfacing with realsense:
- pyrealsense2

For interfacing with the Point Grey Camera:
- rotpy

For interfacing with the Universal Robots manipulator:
- ur_rtde
## Running examples
### Surface normal estimation
The surface normal estimation by photometric stereo methods can be executed by running the following

```bash 
python3 normals_test.py <optional: configPath>
```

Alternatively to run a batch of test, configure the "runNormalsTest.py" script.
```bash 
python3 runNormalsTest.py
```

### Linear depth normal fusion
The linear depth normal fusion can be executed by running the following:

```bash 
python3 linearFusion.py <optional: configPath>
```

### Photometric results evaulation
The photometric results can be executed by running the following:

```bash 
python3 analyseBulk.py <optional: testFolder>
```

For real data use the alternative script:

```bash 
python3 analyseReal.py <optional: testFolder>
```

To generate boxplots:
```bash 
python3 boxplotsBulk.py <optional: testFolder>
```

To run bulk analysis of the above scripts, configure the "runBulkAnalysis.py"
```bash 
python3 runBulkAnalysis.py 
```

### Bilateral normal integration evaulation
The bilateral normal integration results can be evaluated with the following scrips:

```bash 
python3 analyseDepth.py <optional: testFolder>
```

To generate boxplots of the depth:
```bash 
python3 boxpoltsDepth.py <optional: testFolder>
```





## Acknowledgements
Some of the works of this repository is based on the following
 - [RobustPhotometricStereo](https://github.com/yasumat/RobustPhotometricStereo)
 - [Bilateral Normal Integration](https://github.com/xucao-42/bilateral_normal_integration)
 - [Segment Anything](https://github.com/facebookresearch/segment-anything)
