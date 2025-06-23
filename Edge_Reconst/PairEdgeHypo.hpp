#ifndef PAIREDGEHYPO_HPP
#define PAIREDGEHYPO_HPP
// =============================================================================
//
// ==============================================================================
#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <vector>
#include <memory>

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

#include <stdio.h>
#include <stdlib.h>

#include "util.hpp"

namespace PairEdgeHypothesis {
    
    class pair_edge_hypothesis {

    public:
        pair_edge_hypothesis( double );
        
        Eigen::MatrixXd getAp_Bp(Eigen::MatrixXd Edges_HYPO2, Eigen::Vector3d pt_edgel_HYPO1, Eigen::Matrix3d F );
        Eigen::MatrixXd getAp_Bp_Dist(Eigen::MatrixXd Edges_HYPO2, Eigen::Vector3d pt_edgel_HYPO1, Eigen::Matrix3d F );
        Eigen::MatrixXd getHYPO2_idx(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd numerOfDist);
        Eigen::MatrixXd getedgels_HYPO2(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd numerOfDist);
        Eigen::MatrixXd getHYPO2_idx_Ore(Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range);
        Eigen::MatrixXd getedgels_HYPO2_Ore(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range);
        Eigen::MatrixXd getHYPO2_idx_Ore_sted(Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range);
        Eigen::MatrixXd getHYPO2_idx_Ore_fixed(Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range);
        Eigen::MatrixXd getedgels_HYPO2_Ore_fixed(Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd OreListdegree, std::pair<double,double> epip_angle_range);
        Eigen::MatrixXd edgelsHYPO2correct(Eigen::MatrixXd edgels_HYPO2,  Eigen::MatrixXd edgel_HYPO1, Eigen::Matrix3d F21, Eigen::Matrix3d F12, Eigen::MatrixXd HYPO2_idx_raw);
        // Eigen::MatrixXd edgelsHYPO2correct_post_validation(Eigen::MatrixXd edgels_HYPO2,  Eigen::MatrixXd edgel_HYPO1, Eigen::Matrix3d F21, Eigen::Matrix3d F12, Eigen::MatrixXd HYPO2_idx_raw);

        Eigen::MatrixXd edgelsHYPO2_epipolar_correction(Eigen::MatrixXd edgels_HYPO2,  Eigen::MatrixXd edgel_HYPO1, Eigen::Matrix3d F21, Eigen::Matrix3d F12, Eigen::MatrixXd HYPO2_idx_raw);

    private:
        std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
        double reproj_dist_thresh;
        int circle_R;
        

    };

}


#endif
