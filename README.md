# 3D Edge Sketch and Grouping
This is a heavily under-developed on-going research by LEMS Lab at Brown University.

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
where ``XXX`` is the installed prefix path of your YAML-CPP.

## Outputs
Some intermidiate data when running the code will be saved to files which are stored in a ``outputs/`` folder. All files will be cleared out when the code starts a new run. This can be deactivated by setting the macro ``DELETE_ALL_FILES_UNDER_OUTPUTS`` defined in the ``Edge_Reconst/definitions.h`` file as _false_.

## Test
There is a test file under ``test/`` which is primarily used to test part of the functionalities of the 3D edge sketch and grouping. It is compiled in conjunction with the main code, and the executable file resides uner ``/buid/bin``.

## Contributors:
Qiwu Zhang (qiwu_zhang@brown.edu) <br />
Chiang-Heng Chien (chiang-heng_chien@brown.edu) <br />



