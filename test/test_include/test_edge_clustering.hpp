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
#include "../Edge_Reconst/EdgeClusterer.hpp"

std::vector<int> find_Unique_Sorted_Numbers( std::vector<int> vec ) 
{
    std::vector<int> unique_sorted_vec = vec;
    std::sort(unique_sorted_vec.begin(), unique_sorted_vec.end());

    //> Move unique elements to the front
    auto last = std::unique(unique_sorted_vec.begin(), unique_sorted_vec.end());

    //> Erase the duplicate elements at the end
    unique_sorted_vec.erase(last, unique_sorted_vec.end());

    return unique_sorted_vec;
}

//MARK: TEST EDGE CLUSTERING MAIN
void test_edge_clustering_main( MultiviewGeometryUtil::multiview_geometry_util util ) {

    const std::string Dataset_Path = "/gpfs/data/bkimia/Datasets/";
    const std::string Dataset_Name = "ABC-NEF";
    const std::string Scene_Name = "00000006";
    const int Total_Num_Of_Images = 50;
    const int hyp01_view_indx = 25;
    const int hyp02_view_indx = 49;

    //> write data to the files
    std::ofstream file_Epip_Geom_H1_Edges("../../outputs/test_epipolar_line_from_H1_edges.txt");
    std::ofstream file_H2_Edges_Status("../../outputs/test_H2_edges_status_w_clustering.txt");

    //> random number engine
    std::mt19937_64 rnd_eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    //> class constructors
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

    unsigned H1_edge_count = 0;
    const unsigned num_of_test_rounds = 1;

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

        //> Do epipolar correction on H2 edges
        std::vector<int> status;
        std::vector< std::pair< Eigen::Vector3d, Eigen::Vector3d > > original_and_shifted_edges = \
            edges_epipolar_correction( edgels_HYPO2, Edges_HYPO1.row(H1_edge_idx), F21, status, util );
        if (original_and_shifted_edges.size() == 0) continue;
        int Num_Of_Epipolar_Corrected_H2_Edges = original_and_shifted_edges.size();

        //> Organize the data for the preparation of edge clustering
        std::vector< Eigen::Vector3d > corrected_edges_HYPO2;
        for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; i++) {
            corrected_edges_HYPO2.push_back(original_and_shifted_edges[i].second);
        }
        Eigen::MatrixXd edgels_HYPO2_corrected;

        // //> (optional) check if there is any H2 edge whose orientation is perturbed for epipolar correction
        // bool b_has_orient_perturbed = false;
        // for (int si = 0; si < status.size(); si++) {
        //     if (status[si] == 2) { 
        //         b_has_orient_perturbed = true;
        //         break;
        //     }
        // }
        // if (!b_has_orient_perturbed) continue;

        //> Do clustering on epipolar corrected H2 edges
        EdgeClusterer edge_cluster_engine(Num_Of_Epipolar_Corrected_H2_Edges, corrected_edges_HYPO2, H1_edge_idx);
        Eigen::MatrixXd HYPO2_idx = edge_cluster_engine.performClustering( HYPO2_idx_raw, Edges_HYPO2, edgels_HYPO2_corrected, false );

        Eigen::MatrixXd clustered_edges_HYPO2 = edge_cluster_engine.Epip_Correct_H2_Edges;
        int Num_Of_Clusters_per_H1_Edge = edge_cluster_engine.Num_Of_Clusters;
        std::vector<int> cluster_labels = edge_cluster_engine.cluster_labels;

        //> Renumbering the cluster labels into 0, 1, 2, etc
        std::vector<int> renumbered_cluster_labels = cluster_labels;
        std::vector<int> unique_cluster_labels = find_Unique_Sorted_Numbers( cluster_labels );
        for (int i = 0; i < unique_cluster_labels.size(); i++) {
            for (int j = 0; j < cluster_labels.size(); j++) {
                if (cluster_labels[j] == unique_cluster_labels[i]) {
                    renumbered_cluster_labels[j] = i;
                }
            }
        }

        //> Assertion check: the size of the original cluster labels and the renumbered version should match
        assert( renumbered_cluster_labels.size() == cluster_labels.size() );
        // for (int i = 0; i < cluster_labels.size(); i++)
        //     std::cout << cluster_labels[i] << " -> " << renumbered_cluster_labels[i] << std::endl;

        std::map<int, std::vector<int> > label_to_cluster;
        for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
            label_to_cluster[cluster_labels[i]].push_back(i); 
        }

        unsigned cluster_label_count = 0;
        for (const auto& cluster_pair : label_to_cluster) {
            std::cout << "Cluster " << cluster_label_count << " -> Edge Indices: [";

            //> Iterate through the vector associated with the current key
            for (size_t i = 0; i < cluster_pair.second.size(); ++i) {
                std::cout << cluster_pair.second[i];
                if (i < cluster_pair.second.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
            cluster_label_count++;
        }

        //> Write to the files
        // Eigen::MatrixXd picked_H1_edge = Edges_HYPO1.row(H1_edge_idx);
        Eigen::Vector3d epip_coeffs = util.getEpipolarLineCoeffs( Edges_HYPO1, H1_edge_idx, F21 );
        file_Epip_Geom_H1_Edges << Edges_HYPO1(H1_edge_idx, 0) << "\t" << Edges_HYPO1(H1_edge_idx, 1) << "\t" << Edges_HYPO1(H1_edge_idx, 2) << "\t";
        file_Epip_Geom_H1_Edges << epip_coeffs(0) << "\t" << epip_coeffs(1) << "\t" << epip_coeffs(2) << "\n";

        for(int i = 0; i < original_and_shifted_edges.size(); i++) {
            std::pair< Eigen::Vector3d, Eigen::Vector3d > H2_edge_pair = original_and_shifted_edges[i];
            Eigen::Vector3d original_H2_edge = H2_edge_pair.first;
            Eigen::Vector3d corrected_H2_edge = H2_edge_pair.second;

            file_H2_Edges_Status << H1_edge_count << "\t" << original_H2_edge(0) << "\t" << original_H2_edge(1) << "\t" << original_H2_edge(2) << "\t";
            file_H2_Edges_Status << corrected_H2_edge(0) << "\t" << corrected_H2_edge(1) << "\t" << corrected_H2_edge(2) << "\t";
            file_H2_Edges_Status << clustered_edges_HYPO2(i,0) << "\t" << clustered_edges_HYPO2(i,1) << "\t" << clustered_edges_HYPO2(i,2) << "\t";
            file_H2_Edges_Status << renumbered_cluster_labels[i] << "\t" << Num_Of_Clusters_per_H1_Edge << "\n";
        }

        H1_edge_count++;
    }

    file_Epip_Geom_H1_Edges.close();
    file_H2_Edges_Status.close();
}