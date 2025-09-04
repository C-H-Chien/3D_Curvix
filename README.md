# 3D Curvix: From Multiview 2D Edges to 3D Curve Segments (BMVC 2025)
### Research @ Brown LEMS
The code is moving towards the final stage of cleaning and organization.

## Dependencies:
(1) CMake 3.14 or higher <br />
(2) Eigen 3.3.2 (higher version is not compatible) <br />
(3) YAML-CPP, can be built from its [official repo](https://github.com/jbeder/yaml-cpp). (This is used only for parsing data from .yaml file) <br />
Note that by default the code uses C++17. Some minor code has to be changed to turn the code compatible with C++11, but we encourage to use C++17.

## How to build and compile the code
Follow the standard build and compile steps after cloning the repo
```bash
$ mkdir build && cd build
$ cmake ..
$ make -j
```
and you shall see an executive file under ``/buid/bin``. Run the executive file ``./edge_reconstruction-main`` in the ``bin`` folder as there are multiple relative paths used in the code. <br />
When running on Brown CCV Oscars server, manually add Eigen libraries cmake file to ``Eigen3_DIR``:
```bash
/gpfs/runtime/opt/eigen/3.3.2/share/eigen3/cmake/
```
and also manually add the YAML library to ``LD_LIBRARY_PATH``:
```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/XXX/bin/lib64/
```
where ``XXX`` is the installed prefix path of your YAML-CPP. 

## Outputs and Visualization
All output files are given under the ``outputs/`` folder, including some intermediate data. Note that all files will be cleared out when the code starts a new run. This can be deactivated by setting the macro ``DELETE_ALL_FILES_UNDER_OUTPUTS`` defined in the ``Edge_Reconst/definitions.h`` file as _false_.
### 3D edges 
Edges arising from a pair of hypothesis views are given in the files `3D_edges_*.txt` for 3D locations and `3D_tangents_*.txt` for 3D orientation represented by a unit vector. Both are under the world coordinate. 
### 3D curves
3D curves can be found in `final_curves.txt`. To show the curves, use `visualization/plot_curves.m` matlab file. Each curve is colored individually. <br />

## Evaluations

### Dataset
[ABC-NEF dataset](https://github.com/yunfan1202/NEF_code) provides a set of nice 3D ground-truth curves which can be sampled into 3D edges for evaluating the reconstructed 3D edges/curves. 
- We observe that the camera absolute poses of the ABC-NEF dataset are not very accurate (this can be seen by first projecting 3D ground-truth curve points onto images, and then around 1-2 pixels of shift from the projected curve points and the true object ridge are observed.) We thus provide code (can be found under the ``refine_ABC_NEF_camera_poses`` folder) for refining absolute camera poses by bundle adjustment, minimizing the nearest neighbor points between projected points and the detected third-order edges. The refined poses for all objects are collected in this Google Drive.  <br />
- From the projections of 3D sampled curve points, third-order edge correspondences across different views can be constructed. The occlusion is reasonsed by the orientation difference between the projected 3D edge orientation and the third-order edges, as well as the magnitude of projection rays. Again, we made the GT edge correspondences available in this Google Drive.

### Quantitative Evaluation
An evaluation script is customized and created under ``evaluation`` folder. For ABC-NEF dataset, you can download the ground-truth sampled curve points from [Google Drive](https://drive.google.com/drive/folders/1FH8_jykq44YA4FGJ6Par4gBMZg7Ayp1q?usp=sharing). It is encouraged to launch a conda environment before running the evaluation script. Follow the commands below to install:
```bash
conda create -n edge_sketch python=3.8
conda activate edge_sketch
pip install -r requirements.txt
```
Then, let's say ``outputs/curves_points`` is the 3D edge locations obtained from the 3D edge sketch, run the evaluation script to get the precision, recall, accuracy, completeness, and F-score:
```bash
python eval_main.py
```
Refer to ``eval_main.py`` for more information on where the ground-truth curve points directory is specified.

## Precision-Recall Experiments

## Test
There is a test file under ``test/`` which is primarily used to test part of the functionalities of the 3D edge sketch and grouping. It is compiled in conjunction with the main code, and the executable file resides uner ``/buid/bin``.

## Contributors:
Qiwu Zhang (qiwu_zhang@brown.edu) <br />
Chiang-Heng Chien (chiang-heng_chien@brown.edu) <br />



