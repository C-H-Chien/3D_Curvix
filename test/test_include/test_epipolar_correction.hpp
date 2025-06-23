#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>

#include "../Edge_Reconst/definitions.h"
#include "../Edge_Reconst/util.hpp"
#include "../Edge_Reconst/file_reader.hpp"
#include "../Edge_Reconst/getOrientationList.hpp"
#include "../Edge_Reconst/PairEdgeHypo.hpp"

std::vector< std::pair<Eigen::Vector3d, Eigen::Vector3d> > edges_epipolar_correction( Eigen::MatrixXd edgels_HYPO2, Eigen::MatrixXd edgel_HYPO1, Eigen::Matrix3d F21, std::vector<int> &status, MultiviewGeometryUtil::multiview_geometry_util util )
{
    std::vector< std::pair<Eigen::Vector3d, Eigen::Vector3d> > original_and_correct_edge_pair;
    Eigen::Vector3d epip_coeffs = util.getEpipolarLineCoeffs( edgel_HYPO1, 0, F21 );
        
    for(int idx_hypo2 = 0; idx_hypo2 < edgels_HYPO2.rows(); idx_hypo2++){

        Eigen::Vector3d Original_Edge( edgels_HYPO2(idx_hypo2, 0), edgels_HYPO2(idx_hypo2, 1), edgels_HYPO2(idx_hypo2, 2) );
        Eigen::Vector3d xy1_H2( edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), 1.0 );
        
        double corrected_x, corrected_y, corrected_theta;
        double epiline_x, epiline_y;
        double normal_distance_epiline = util.getNormalDistance2EpipolarLine( epip_coeffs, xy1_H2, epiline_x, epiline_y );

        if (normal_distance_epiline < LOCATION_PERTURBATION){
            //> If normal distance is small, move directly to epipolar line
            corrected_x = epiline_x;
            corrected_y = epiline_y;
            corrected_theta = edgels_HYPO2(idx_hypo2, 2);
            status.push_back(0);
        }
        else {
            double x_intersection, y_intersection;
            Eigen::Vector3d isolated_H2( edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), edgels_HYPO2(idx_hypo2,2) );
            double dist_diff_edg2 = util.getTangentialDistance2EpipolarLine( epip_coeffs, isolated_H2, x_intersection, y_intersection );
            double theta = edgels_HYPO2(idx_hypo2,2);

            //> Inner two cases: 
            if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH) {
                //> (i) if the displacement after epipolar shift is less than EPIP_TANGENCY_DISPL_THRESH, then feel free to shift it along its direction vector
                corrected_x = x_intersection;
                corrected_y = y_intersection;
                corrected_theta = edgels_HYPO2(idx_hypo2, 2);
                status.push_back(1);
            }
            else {
                //> (ii) if not, then perturb the edge orientation first before shifting the edge along its direction vector
                // double p_theta = a1_line * cos(theta) + b1_line * sin(theta);
                // double derivative_p_theta = -a1_line * sin(theta) + b1_line * cos(theta);
                double p_theta = epip_coeffs(0) * cos(theta) + epip_coeffs(1) * sin(theta);
                double derivative_p_theta = -epip_coeffs(0) * sin(theta) + epip_coeffs(1) * cos(theta);

                //> decide how theta should be perturbed by observing the signs of p_theta and derivative_p_theta
                if (p_theta > 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
                else if (p_theta < 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
                else if (p_theta > 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;
                else if (p_theta < 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;

                //> Calculate the intersection between the tangent and epipolar line
                Eigen::Vector3d isolated_H2_( edgels_HYPO2(idx_hypo2,0), edgels_HYPO2(idx_hypo2,1), theta );
                dist_diff_edg2 = util.getTangentialDistance2EpipolarLine( epip_coeffs, isolated_H2_, x_intersection, y_intersection );

                if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH) {
                    corrected_x = x_intersection;
                    corrected_y = y_intersection;
                    corrected_theta = theta;
                    status.push_back(2);
                } 
                else {
                    continue;
                }
            }
        }

        Eigen::Vector3d Epip_Corrected_Edge( corrected_x, corrected_y, corrected_theta );

        original_and_correct_edge_pair.push_back( std::make_pair(Original_Edge, Epip_Corrected_Edge) );
    }

    return original_and_correct_edge_pair;
}

//MARK: TEST EPIPOLAR CORRECTION MAIN
void test_epipolar_correction_main( MultiviewGeometryUtil::multiview_geometry_util util ) {

    std::string Dataset_Path = "/gpfs/data/bkimia/Datasets/";
    std::string Dataset_Name = "ABC-NEF";
    std::string Scene_Name = "00000006";
    const int Total_Num_Of_Images = 50;
    const int hyp01_view_indx = 48;
    const int hyp02_view_indx = 28;

    //> random number engine
    std::mt19937_64 rnd_eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    //> classes
    file_reader Load_Data(Dataset_Path, Dataset_Name, Scene_Name, Total_Num_Of_Images);
    GetOrientationList::get_OrientationList getOre( 0.3, 800, 800 );
    PairEdgeHypothesis::pair_edge_hypothesis PairHypo( 2.0 );

    std::vector<Eigen::Matrix3d> All_R;
    std::vector<Eigen::Vector3d> All_T;
    std::vector<Eigen::MatrixXd> All_Edgels; 
    Eigen::Matrix3d K;
    Eigen::MatrixXd Edges_HYPO1;
    Eigen::MatrixXd Edges_HYPO2;
    Eigen::Matrix3d Rot_HYPO1;
    Eigen::Matrix3d Rot_HYPO2;
    Eigen::Vector3d Transl_HYPO1;
    Eigen::Vector3d Transl_HYPO2;
    Eigen::Matrix3d R21, R12;
    Eigen::Vector3d T21, T12;

    //> Read absolute camera rotation matrices (all under world coordinate)
    Load_Data.readRmatrix( All_R );

    //> Read absolute camera translation vectors (all under world coordinate)
    Load_Data.readTmatrix( All_T );

    Load_Data.read_All_Edgels( All_Edgels, 1 );

    K << 1111.11136542426, 0,	399.500000000000, 0, 1111.11136542426, 399.500000000000, 0, 0, 1;

    //> set up the hypothesis view pair
    Rot_HYPO1       = All_R[hyp01_view_indx];
    Rot_HYPO2       = All_R[hyp02_view_indx];
    Transl_HYPO1    = All_T[hyp01_view_indx];
    Transl_HYPO2    = All_T[hyp02_view_indx];
    Edges_HYPO1     = All_Edgels[hyp01_view_indx];
    Edges_HYPO2     = All_Edgels[hyp02_view_indx];

    util.getRelativePoses( Rot_HYPO1, Transl_HYPO1, Rot_HYPO2, Transl_HYPO2, R21, T21, R12, T12 );
    Eigen::Matrix3d F21 = util.getFundamentalMatrix(K.inverse(), K.inverse(), R21, T21); 

    //> Compute epipolar wedge angles between HYPO1 and HYPO2 and valid angle range in HYPO1 for fast indexing from edges of HYPO2
    Eigen::MatrixXd OreListdegree = getOre.getOreList(hyp01_view_indx, hyp02_view_indx, Edges_HYPO2, All_R, All_T, K, K);
    auto result = getOre.getOreListBar(Edges_HYPO1, All_R, All_T, K, K, hyp02_view_indx, hyp01_view_indx);
    Eigen::MatrixXd OreListBardegree = result.first;
    Eigen::Vector3d epipole = result.second;

    //> Define the distribution for the desired range for the random index of H1 edges
    const int max_val = Edges_HYPO1.rows();
    std::uniform_int_distribution<int> dist(0, max_val);

    //> write data to the files
    std::ofstream file_Epip_Geom_H1_Edges("../../outputs/test_epipolar_line_from_H1_edges.txt");
    std::ofstream file_H2_Edges_Status("../../outputs/test_H2_edges_status.txt");

    unsigned H1_edge_count = 0;
    const unsigned num_of_test_rounds = 10;

    while ( H1_edge_count < num_of_test_rounds ) {

        //> randomly pick one H1 edge
        int H1_edge_idx = dist(rnd_eng);

        std::pair<double, double> epip_angle_range_from_H1_edge = std::make_pair(OreListBardegree(H1_edge_idx, 0), OreListBardegree(H1_edge_idx, 1));

        //> Find the corresponding H2 edge using the epipolar angle range
        //> (i) H2 edge index
        Eigen::MatrixXd HYPO2_idx_raw = PairHypo.getHYPO2_idx_Ore(OreListdegree, epip_angle_range_from_H1_edge);
        if (HYPO2_idx_raw.rows() == 0) continue;
        //> (ii) H2 edge location and orientation
        Eigen::MatrixXd edgels_HYPO2 = PairHypo.getedgels_HYPO2_Ore(Edges_HYPO2, OreListdegree, epip_angle_range_from_H1_edge);

        std::vector<int> status;
        std::vector< std::pair< Eigen::Vector3d, Eigen::Vector3d > > original_and_shifted_edges = \
            edges_epipolar_correction( edgels_HYPO2, Edges_HYPO1.row(H1_edge_idx), F21, status, util );
        if (original_and_shifted_edges.size() == 0) continue;

        //> check if there is any H2 edge whose orientation is perturbed for epipolar correction
        bool b_has_orient_perturbed = false;
        for (int si = 0; si < status.size(); si++) {
            if (status[si] == 2) { 
                b_has_orient_perturbed = true;
                break;
            }
        }
        if (!b_has_orient_perturbed) continue;

        //> write to the files
        // Eigen::MatrixXd picked_H1_edge = Edges_HYPO1.row(H1_edge_idx);
        Eigen::Vector3d epip_coeffs = util.getEpipolarLineCoeffs( Edges_HYPO1, H1_edge_idx, F21 );
        // Eigen::Vector3d epip_coeffs = util.getEpipolarLineCoeffs( picked_H1_edge, F21 );
        file_Epip_Geom_H1_Edges << Edges_HYPO1(H1_edge_idx, 0) << "\t" << Edges_HYPO1(H1_edge_idx, 1) << "\t" << Edges_HYPO1(H1_edge_idx, 2) << "\t";
        file_Epip_Geom_H1_Edges << epip_coeffs(0) << "\t" << epip_coeffs(1) << "\t" << epip_coeffs(2) << "\n";

        for(int i = 0; i < original_and_shifted_edges.size(); i++) {
            std::pair< Eigen::Vector3d, Eigen::Vector3d > H2_edge_pair = original_and_shifted_edges[i];
            Eigen::Vector3d original_H2_edge = H2_edge_pair.first;
            Eigen::Vector3d corrected_H2_edge = H2_edge_pair.second;

            file_H2_Edges_Status << H1_edge_count << "\t" << original_H2_edge(0) << "\t" << original_H2_edge(1) << "\t" << original_H2_edge(2) << "\t";
            file_H2_Edges_Status << corrected_H2_edge(0) << "\t" << corrected_H2_edge(1) << "\t" << corrected_H2_edge(2) << "\t" << status[i] << "\n";
        }

        H1_edge_count++;
    }

    file_Epip_Geom_H1_Edges.close();
    file_H2_Edges_Status.close();
}