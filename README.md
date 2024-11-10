# 3D Edge Sketch and Grouping

## Dependencies:
(1) CMake 3.14 or higher <br />
(2) Eigen 3.3.2 (higher version is not compatible) <br />
(3) YAML-CPP, can be built from its [official repo](https://github.com/jbeder/yaml-cpp). (This is used only for parsing data from .yaml file) <br />

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
where ``XX`` is the installed prefix path of your YAML-CPP.

## Contributors:
Qiwu Zhang (qiwu_zhang@brown.edu) <br />
Chiang-Heng Chien (chiang-heng_chien@brown.edu) <br />

## References
Third-Order Edge Detector: [paper](https://ieeexplore.ieee.org/abstract/document/8382271) and [code](https://github.com/C-H-Chien/Third-Order-Edge-Detector). The code has been embedded to this repository.


