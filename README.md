# 3D Curvix: From Multiview 2D Edges to 3D Curve Segments (BMVC 2025)
### Research @ Brown LEMS
This repository hosts the code for 3D Curvix, a 3D curve reconstruction framework based on [3D Edge Sketch](https://github.com/C-H-Chien/3D_Edge_Sketch). Specifically, 3D Curvix addresses noise and redundancies of an unorganized cloud of 3D edges (given by 3D Edge Sketch) through consolidating redundant 3D edges arising from hypothesis formation and multiple passes of hypothesis view pairs. A clean, consolidated 3D edges are then grouped for form 3D curve segments represented by a sequence of ordered 3D edges.

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

## Outputs
All output files are given under the ``outputs/`` folder, including some intermediate data. Note that all files will be cleared out when the code starts a new run. This can be deactivated by setting the macro ``DELETE_ALL_FILES_UNDER_OUTPUTS`` defined in the ``Edge_Reconst/definitions.h`` file as _false_.
### 3D edges 
Edges arising from a pair of hypothesis views are given in the files `3D_edges_*.txt` for 3D locations and `3D_tangents_*.txt` for 3D orientation represented by a unit vector. Both are under the world coordinate. 
### 3D curves
3D curves can be found in `final_curves.txt`. To show the curves, use `visualization/plot_curves.m` matlab file. Each curve is colored individually. <br />

## Evaluations
### Dataset
[ABC-NEF dataset](https://github.com/yunfan1202/NEF_code) provides a set of nice 3D ground-truth curves which can be sampled into 3D edges for evaluating the reconstructed 3D edges/curves. 
- We observe that the camera absolute poses of the ABC-NEF dataset are not very accurate (this can be seen by first projecting 3D ground-truth curve points onto images, and then around 1-2 pixels of shift from the projected curve points and the true object ridge are observed. See [this opened issue](https://github.com/yunfan1202/NEF_code/issues/12) for more information). The ground-truth rotations are also improper. We thus provide code (can be found under the ``refine_ABC_NEF_camera_poses`` folder) for refining absolute camera poses by bundle adjustment, minimizing the nearest neighbor points between projected points and the detected third-order edges. We will release a collection of refined poses for all ABC-NEF objects.  <br />

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

## Others (Can be ignored for now)
There is a test file under ``test/`` which is primarily used to test part of the functionalities of the 3D edge sketch and grouping. It is compiled in conjunction with the main code, and the executable file resides uner ``/buid/bin``.

## References
If you use the code, please cite the 3D Curvix and 3D Edge Sketch papers:
```BibTeX
@InProceedings{Zhang:Chien:Fabbri:Kimia:BMCV2025,
  title={{3D Curvix: From Multiview 2D Edges to 3D Curve Segments}},
  author={Zhang, Qiwu and Chien, Chiang-Heng and Fabbri, Ricardo and Kimia, Benjamin},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  year={2025}
}
```
```BibTeX
@inproceedings{Zheng:Chien:Fabbri:Kimia:WACV2025,
  title={{3D Edge Sketch from Multiview Images}},
  author={Zheng, Yilin and Chien, Chiang-Heng and Fabbri, Ricardo and Kimia, Benjamin},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={3196--3205},
  year={2025},
  organization={IEEE}
}
```

## TODOs
The code is a bit messay. We are working toward restructuring and reorganizing to increase the overall readability and compatibility. Also, a GPU parallel version is developing. We will release the updates soon.

## Contributors:
Qiwu Zhang (qiwu_zhang@brown.edu) <br />
Chiang-Heng Chien (chiang-heng_chien@brown.edu) <br />



