#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "../Edge_Reconst/definitions.h"

Eigen::Matrix3d getSkewSymmetric(Eigen::Vector3d T) {
    Eigen::Matrix3d T_x = (Eigen::Matrix3d() << 0.,  -T(2),   T(1), T(2),  0.,  -T(0), -T(1),  T(0),   0.).finished();
    return T_x;
}

Eigen::Matrix3d getRodriguesRotationMatrix(Eigen::Vector3d v1, Eigen::Vector3d v2) {

    //> make sure that the input vectors are unit-vectors
    v1 /= v1.norm();
    v2 /= v2.norm();

    Eigen::Vector3d v1_cross_v2 = v1.cross(v2);
    double s = v1_cross_v2.norm();
    double c = v1.dot(v2);
    double coeff = 1.0 / (1.0 + c);
    Eigen::Matrix3d v_x = getSkewSymmetric(v1_cross_v2);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + v_x + coeff * v_x * v_x;
    return R;
}

int main(int argc, char **argv) {

    //> [TEST] Verify the correctness of the aligning two unit vectors in 3D by Rodrigue's formula
    Eigen::Vector3d v1(0.700390490817757, 0.114407726497277, 0.704531072763853);
    Eigen::Vector3d v2(0.714979548359578, 0.362563357150639, 0.597789308602281);
    Eigen::Matrix3d R_align_v1_to_v2 = getRodriguesRotationMatrix(v1, v2);
    Eigen::Vector3d aligned_v2 = R_align_v1_to_v2 * v1;
    std::cout << aligned_v2 << std::endl;
    Eigen::Vector3d eulerAnglesXYZ = R_align_v1_to_v2.eulerAngles(0, 1, 2);
    std::cout << "Euler angles:" << std::endl << eulerAnglesXYZ << std::endl;

    //> [TEST] Verify the correctness of projecting a 3D unit tangent vector (in world coordinate) to a 2D image
    Eigen::Vector3d Tangent_3D_w(-0.604940217207650, 0.119991911797586, 0.787178045112998);
    Eigen::Matrix3d Rot;
    Rot <<  0.283307133314224,  0.599568603826573,  0.748501541427090, \
           -0.799349013318374, -0.283603601454719,  0.529726488056670, \
            0.529885103697223, -0.748389261378974,  0.398917648559720;

    Eigen::Vector3d point_location(313.128094185699, 221.221179621611, 1.0);
    Eigen::Matrix3d K;
    K << 2584.93250981950,	0,	249.771375872214, \
         0,	2584.79186060577,	278.312679379194, \
         0,	0,	1;
    
    Eigen::Vector3d point_in_meters = K.inverse() * point_location;
    Eigen::Vector3d Tangent_3D_c = Rot * Tangent_3D_w;
    Eigen::Vector3d tangent_2D   = Tangent_3D_c - Tangent_3D_c(2) * point_in_meters;
    tangent_2D.normalize();
    std::cout << "projected tangent is (" << tangent_2D(0) << ", " << tangent_2D(1) << ", " << tangent_2D(2) << ")" << std::endl;

    return 0;
}