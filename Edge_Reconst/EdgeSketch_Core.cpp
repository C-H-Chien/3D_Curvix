#ifndef EDGESKETCH_CORE_CPP
#define EDGESKETCH_CORE_CPP
// =============================================================================================================================
//
// ChangLogs
//    
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =============================================================================================================================
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <vector>

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

//> YAML file data reader
#include <yaml-cpp/yaml.h>

#include "EdgeSketch_Core.hpp"
#include "getReprojectedEdgel.hpp"
#include "util.hpp"
#include "definitions.h"
#include "../Edge_Reconst/mvt.hpp"


//> Constructor
EdgeSketch_Core::EdgeSketch_Core(YAML::Node Edge_Sketch_Setting_File)
    : Edge_Sketch_Setting_YAML_File(Edge_Sketch_Setting_File)
{
    //> Parse data from the YAML file
    //> (1) 3D Edge Sketch Settings
    Num_Of_OMP_Threads                              = Edge_Sketch_Setting_YAML_File["Num_Of_OMP_Threads"].as<int>();
    hyp01_view_indx                                 = Edge_Sketch_Setting_YAML_File["Init_Hypo1_View_Index"].as<int>();
    hyp02_view_indx                                 = Edge_Sketch_Setting_YAML_File["Init_Hypo2_View_Index"].as<int>();
    Edge_Loc_Pertubation                            = Edge_Sketch_Setting_YAML_File["delta"].as<double>();
    Orien_Thresh                                    = Edge_Sketch_Setting_YAML_File["delta_theta"].as<double>();
    Max_Num_Of_Support_Views                        = Edge_Sketch_Setting_YAML_File["Max_Num_Of_Support_Views"].as<int>();
    Edge_Detection_Init_Thresh                      = Edge_Sketch_Setting_YAML_File["Multi_Thresh_Init_Thresh"].as<int>();
    Edge_Detection_Final_Thresh                     = Edge_Sketch_Setting_YAML_File["Multi_Thresh_Final_Thresh"].as<int>();
    Parallel_Epipolar_Line_Angle_Deg                = Edge_Sketch_Setting_YAML_File["Parallel_Epipolar_Line_Angle"].as<double>();
    Reproj_Dist_Thresh                              = Edge_Sketch_Setting_YAML_File["Reproj_Dist_Thresh"].as<double>();
    Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges   = Edge_Sketch_Setting_YAML_File["Ratio_Of_Claimed_Edges_to_Stop"].as<double>();
    Max_3D_Edge_Sketch_Passes                       = Edge_Sketch_Setting_YAML_File["Max_Num_Of_3D_Edge_Sketch_Passes"].as<int>();
    circleR                                         = Edge_Sketch_Setting_YAML_File["circleR"].as<double>(); //> Unknown setting
    //> (2) Dataset Settings
    Dataset_Path                        = Edge_Sketch_Setting_YAML_File["Dataset_Path"].as<std::string>();
    Dataset_Name                        = Edge_Sketch_Setting_YAML_File["Dataset_Name"].as<std::string>();
    Scene_Name                          = Edge_Sketch_Setting_YAML_File["Scene_Name"].as<std::string>();
    Num_Of_Total_Imgs                   = Edge_Sketch_Setting_YAML_File["Total_Num_Of_Images"].as<int>();
    Img_Rows                            = Edge_Sketch_Setting_YAML_File["Image_Rows"].as<int>();
    Img_Cols                            = Edge_Sketch_Setting_YAML_File["Image_Cols"].as<int>();
    Use_Multiple_K                      = Edge_Sketch_Setting_YAML_File["Use_Multiple_K"].as<bool>();
    fx                                  = Edge_Sketch_Setting_YAML_File["fx"].as<double>();
    fy                                  = Edge_Sketch_Setting_YAML_File["fy"].as<double>();
    cx                                  = Edge_Sketch_Setting_YAML_File["cx"].as<double>();
    cy                                  = Edge_Sketch_Setting_YAML_File["cy"].as<double>();
    //> (3) Other Settings
    Delta_FileName_Str                  = Edge_Sketch_Setting_YAML_File["deltastr"].as<std::string>();

    //> Initialization
    edge_sketch_time = 0.0;
    enable_aborting_3D_edge_sketch = false;
    num_of_nonveridical_edge_pairs = 0;
    pair_edges_time = 0.0;
    finalize_edge_pair_time = 0.0;
    find_next_hypothesis_view_time = 0.0;

    //> Class objects
    Load_Data       = std::shared_ptr<file_reader>(new file_reader(Dataset_Path, Dataset_Name, Scene_Name, Num_Of_Total_Imgs));
    util            = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());
    PairHypo        = std::shared_ptr<PairEdgeHypothesis::pair_edge_hypothesis>(new PairEdgeHypothesis::pair_edge_hypothesis( Reproj_Dist_Thresh, circleR ));
    getReprojEdgel  = std::shared_ptr<GetReprojectedEdgel::get_Reprojected_Edgel>(new GetReprojectedEdgel::get_Reprojected_Edgel());
    getSupport      = std::shared_ptr<GetSupportedEdgels::get_SupportedEdgels>(new GetSupportedEdgels::get_SupportedEdgels( Orien_Thresh ));
    getOre          = std::shared_ptr<GetOrientationList::get_OrientationList>(new GetOrientationList::get_OrientationList( Edge_Loc_Pertubation, Img_Rows, Img_Cols ));
    edgeMapping     = std::shared_ptr<EdgeMapping>(new EdgeMapping());

    //> Set up OpenMP threads
    omp_set_num_threads(Num_Of_OMP_Threads);
    // int ID = omp_get_thread_num();
#if SHOW_OMP_NUM_OF_THREADS
    std::cout << "Using " << Num_Of_OMP_Threads << " threads for OpenMP parallelization." << std::endl;
#endif
}

void EdgeSketch_Core::Read_Camera_Data() {
    
    //> Read absolute camera rotation matrices (all under world coordinate)
    Load_Data->readRmatrix( All_R );

    //> Read absolute camera translation vectors (all under world coordinate)
    Load_Data->readTmatrix( All_T );

    //> Read camera intrinsic matrix
    if (Use_Multiple_K)
        Load_Data->readK( All_K );
    else
        K << fx, 0,	cx, 0, fy, cy, 0, 0, 1;   
}

void EdgeSketch_Core::Set_Hypothesis_Views_Camera() {
    
    Rot_HYPO1       = All_R[hyp01_view_indx];
    Rot_HYPO2       = All_R[hyp02_view_indx];
    Transl_HYPO1    = All_T[hyp01_view_indx];
    Transl_HYPO2    = All_T[hyp02_view_indx];

    if (Use_Multiple_K) {
        K_HYPO1 = All_K[hyp01_view_indx];
        K_HYPO2 = All_K[hyp02_view_indx];
    }
    else {
        K_HYPO1 = K;
        K_HYPO2 = K;
    }
    util->getRelativePoses( Rot_HYPO1, Transl_HYPO1, Rot_HYPO2, Transl_HYPO2, R21, T21, R12, T12 );
    F21 = util->getFundamentalMatrix(K_HYPO1.inverse(), K_HYPO2.inverse(), R21, T21); 
    F12 = util->getFundamentalMatrix(K_HYPO2.inverse(), K_HYPO1.inverse(), R12, T12);

    //> generate a list of validation view indices which is valid_view_index (data type: std::vector<int>)
    get_validation_view_index_list();

    //> A list of history hypothesis views
    history_hypothesis_views_index.push_back(hyp01_view_indx);
    history_hypothesis_views_index.push_back(hyp02_view_indx);
}



///////////////////////////// cluster related /////////////////////////////
void EdgeSketch_Core::Read_Edgels_Data() {
    //> Read edgels detected at a specific threshold 
    Load_Data->read_All_Edgels( All_Edgels, thresh_EDG );
}

void EdgeSketch_Core::reset_hypo2_clusters() {
    hypo2_clusters.clear();
}


std::vector<int> EdgeSketch_Core::get_edges_in_same_cluster(int hypo1_edge, int hypo2_edge) {
    // Check if the edge exists in our mapping
    auto it = hypo2_clusters.find(std::make_pair(hypo1_edge, hypo2_edge));
    if (it != hypo2_clusters.end()) {
        return it->second;
    }
    
    // If not found in any cluster, return a vector containing only this edge
    return {hypo2_edge};
}
///////////////////////////// cluster related /////////////////////////////



void EdgeSketch_Core::Set_Hypothesis_Views_Edgels() {

    Edges_HYPO1.resize(0, 0); 
    Edges_HYPO2.resize(0, 0); 
    paired_edge.resize(0, 0); 
    OreListdegree.resize(0, 0); 
    OreListBardegree.resize(0, 0); 
    epipole.setZero(); 

    Edges_HYPO1     = All_Edgels[hyp01_view_indx];
    Edges_HYPO2     = All_Edgels[hyp02_view_indx];

    //> Initialize a list of paired edges between HYPO1 and HYPO2
    paired_edge         = Eigen::MatrixXd::Constant(Edges_HYPO1.rows()*Num_Of_Total_Imgs, Num_Of_Total_Imgs, -2);
    
    //> Compute epipolar wedge angles between HYPO1 and HYPO2 and valid angle range in HYPO1 for fast indexing from edges of HYPO2
    OreListdegree       = getOre->getOreList(hyp01_view_indx, hyp02_view_indx, Edges_HYPO2, All_R, All_T, K_HYPO1, K_HYPO2);

    auto result = getOre->getOreListBar(Edges_HYPO1, All_R, All_T, K_HYPO1, K_HYPO2, hyp02_view_indx, hyp01_view_indx);
    OreListBardegree = result.first;
    epipole = result.second;
}

void EdgeSketch_Core::Run_3D_Edge_Sketch() {

    itime = omp_get_wtime();
    reset_hypo2_clusters();

    #pragma omp parallel
    {

        //> Local array stacking all supported indices
        std::vector<Eigen::MatrixXd> local_thread_supported_indices;
        // Thread-local clusters 
        std::unordered_map<int, std::vector<int>> thread_local_clusters;

        int H1_edge_idx;
       
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< First loop: loop over all edgels from hypothesis view 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
        //<<<<<<<<<<< Identify pairs of edge, correct the positions of the edges from Hypo2, and store the paired edges >>>>>>>>>>>>>>>>//
        #pragma omp for schedule(static, Num_Of_OMP_Threads)
        for (H1_edge_idx = 0; H1_edge_idx < Edges_HYPO1.rows() ; H1_edge_idx++) {

            if ( Skip_this_Edge( H1_edge_idx ) ){
                // if(abs(Edges_HYPO1(H1_edge_idx,0) - 462.853)<0.001  && abs(Edges_HYPO1(H1_edge_idx,1) - 434.987)<0.001){
                //     std::cout<<"hypo 1 skipped"<<std::endl;
                //     std::cout<<"edge index is: "<<H1_edge_idx<<std::endl;
                //     exit(0);
                // }
                continue;
            } 
            

            //> Get the current edge from HYPO1
            Eigen::Vector3d pt_edgel_HYPO1;
            pt_edgel_HYPO1 << Edges_HYPO1(H1_edge_idx,0), Edges_HYPO1(H1_edge_idx,1), 1;

            //Get angle Thresholds from OreListBar (in Degree) 
            double thresh_ore21_1 = OreListBardegree(H1_edge_idx, 0);
            double thresh_ore21_2 = OreListBardegree(H1_edge_idx, 1);
            Eigen::Vector3d corresponding_epipole = epipole;

            // if(abs(Edges_HYPO1(H1_edge_idx,0) - 462.853)<0.001  && abs(Edges_HYPO1(H1_edge_idx,1) - 434.987)<0.001){
            //     std::cout<<"thresh_ore21_1: "<<thresh_ore21_1<<std::endl;
            //     std::cout<<"thresh_ore21_2: "<<thresh_ore21_2<<std::endl;
            //     std::cout<<"corresponding_epipole: "<<corresponding_epipole<<std::endl;
            //     exit(0);
            // }

            //> Find the corresponding edgel in HYPO2 based on the epipolar angle
            Eigen::MatrixXd HYPO2_idx_raw = PairHypo->getHYPO2_idx_Ore(OreListdegree, thresh_ore21_1, thresh_ore21_2);
            if (HYPO2_idx_raw.rows() == 0){
                // if(abs(pt_edgel_HYPO1(0) - 462.853)<0.001  && abs(pt_edgel_HYPO1(1) - 434.987)<0.001){
                //     std::cout<<"HYPO2_idx_raw row is 0"<<std::endl;
                //     exit(0);
                // }
                continue;
            } 
            //> Retrieve Hypo2 Edgels
            Eigen::MatrixXd edgels_HYPO2 = PairHypo->getedgels_HYPO2_Ore(Edges_HYPO2, OreListdegree, thresh_ore21_1, thresh_ore21_2);
            //> Correct Edgels in Hypo2 Based on Epipolar Constraints
            Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgels_HYPO2, Edges_HYPO1.row(H1_edge_idx), F21, F12, HYPO2_idx_raw);
            //Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct(edgels_HYPO2, Edges_HYPO1.row(H1_edge_idx), F21, F12, HYPO2_idx_raw);
            
            //> Organize the final edge data (hypothesis edge pairs)
            Eigen::MatrixXd Edges_HYPO1_final(edgels_HYPO2_corrected.rows(), 4);
            Edges_HYPO1_final << edgels_HYPO2_corrected.col(0), edgels_HYPO2_corrected.col(1), edgels_HYPO2_corrected.col(2), edgels_HYPO2_corrected.col(3);
            Eigen::MatrixXd Edges_HYPO2_final(edgels_HYPO2_corrected.rows(), 4);
            Edges_HYPO2_final << edgels_HYPO2_corrected.col(4), edgels_HYPO2_corrected.col(5), edgels_HYPO2_corrected.col(6), edgels_HYPO2_corrected.col(7);

            if (Edges_HYPO2_final.rows() == 0) continue;

            Eigen::Vector2d pt_H1 = Edges_HYPO1_final.row(0);
            Eigen::Vector2d pt_H2 = Edges_HYPO2_final.row(0);

            bool exit_flag = false;
            // if (abs(pt_H1(0) - 462.853)<0.001  && abs(pt_H1(1) - 434.987)<0.001){
            //     std::cout<<"thresh_ore21_1 is: "<<thresh_ore21_1<<std::endl;
            //     std::cout<<"thresh_ore21_2 is: "<<thresh_ore21_2<<std::endl;
            //     std::cout<<"corresponding_epipole is: "<<corresponding_epipole<<std::endl;
            //     std::cout<<"before correction: "<<edgels_HYPO2<<std::endl;
            //     std::cout << "Before clustering (Edges_HYPO2_final):\n" << Edges_HYPO2_final << std::endl;
            //     exit_flag = true;
            // }
            // std::cout<<"hypothesis 1 edge is: "<<Edges_HYPO1.row(H1_edge_idx)<<std::endl;
           ////////////////////////////////////// cluster hypothesis 2's edges //////////////////////////////////////
            Eigen::MatrixXd HYPO2_idx;
            int N = Edges_HYPO2_final.rows();

            double cluster_threshold = 1;
            double orientation_threshold = 20.0;
            int max_cluster_size = 10; 

            std::vector<int> cluster_labels(N);
            std::iota(cluster_labels.begin(), cluster_labels.end(), 0); // Each point starts in its own cluster

            // Track average orientations for each cluster
            std::unordered_map<int, double> cluster_avg_orientations;
            for (int i = 0; i < N; ++i) {
                cluster_avg_orientations[i] = Edges_HYPO2_final(i, 2);
            }
            
            //////////////////////////// Helper functions ////////////////////////////
            double gaussian_sigma = 2.0; 
            double max_reliable_shift = 5.0; 

            // Store original positions before correction for distance calculation
            Eigen::MatrixXd original_positions(edgels_HYPO2.rows(), 2);
            for (int i = 0; i < edgels_HYPO2.rows(); ++i) {
                original_positions(i, 0) = edgels_HYPO2(i, 0);
                original_positions(i, 1) = edgels_HYPO2(i, 1);
            }

            //get cluster size
            std::function<int(int, int)> getClusterSize = [&cluster_labels](int label, int N) -> int {
                int size = 0;
                for (int i = 0; i < N; ++i) {
                    if (cluster_labels[i] == label) size++;
                }
                return size;
            };
            // check if two orientations are within thredhold
            std::function<bool(double, double)> areSimilarOrientations = [orientation_threshold](double orient1, double orient2) -> bool {
                double orient1_deg = orient1 * (180.0 / M_PI);
                double orient2_deg = orient2 * (180.0 / M_PI);
                double diff = std::abs(orient1_deg - orient2_deg);
                return diff < orientation_threshold;
            };

            // Calculate Gaussian weight based on shift distance
            std::function<double(int)> getShiftWeight = [&original_positions, &Edges_HYPO2_final, gaussian_sigma](int idx) -> double {
                double shift_distance = (original_positions.row(idx) - Edges_HYPO2_final.row(idx).head<2>()).norm();
                return std::exp(-0.5 * std::pow(shift_distance / gaussian_sigma, 2));
            };

            // Update cluster average orientation
            std::function<double(int, int)> updateAvgOrientation = [&cluster_labels, &cluster_avg_orientations, &Edges_HYPO2_final, &getShiftWeight, N](int label1, int label2) -> double {
                double sum_orin = 0;
                int count = 0;
                
                for (int i = 0; i < N; ++i) {
                    if (cluster_labels[i] == label1 || cluster_labels[i] == label2) {
                        double orientation = Edges_HYPO2_final(i, 2);
                        double weight = getShiftWeight(i);

                        sum_orin += weight * orientation;
                        count++;
                    }
                }
                
                return sum_orin/count;
            };

            //////////////////////////// Helper functions ////////////////////////////

            // Merge clusters starting from closest pairs
            bool merged = true;
            while (merged) {
                merged = false;

                // For each point, find its nearest neighbor and merge if within threshold
                for (int i = 0; i < N; ++i) {
                    double min_dist = std::numeric_limits<double>::max();
                    int nearest = -1;
                    
                    // QIWU TODO: CHECK WHY EDGES WITH HUGE ORIENTATION DIFFERENCE ARE MERGED TOGETHER
                    // Find the nearest edge to the current edge
                    for (int j = 0; j < N; ++j) {
                        if (cluster_labels[i] != cluster_labels[j]) {
                            double dist = (Edges_HYPO2_final.row(i).head<2>() - Edges_HYPO2_final.row(j).head<2>()).norm();
                            double orient_i = cluster_avg_orientations[cluster_labels[i]];
                            double orient_j = cluster_avg_orientations[cluster_labels[j]];
                            if (dist < min_dist && dist < cluster_threshold && areSimilarOrientations(orient_i, orient_j)) {
                                min_dist = dist;
                                nearest = j;
                            }
                        }
                    }
                    // If found a nearest edge within threshold, merge clusters
                    if (nearest != -1) {
                        int old_label = cluster_labels[nearest];
                        int new_label = cluster_labels[i];
                        int size_old = getClusterSize(old_label, N);
                        int size_new = getClusterSize(new_label, N);
                        if (size_old + size_new <= max_cluster_size) {
                            // Calculate new average orientation for the merged cluster
                            double merged_avg_orientation = updateAvgOrientation(old_label, new_label);
                            // Update all points in the smaller cluster
                            for (int k = 0; k < N; ++k) {
                                if (cluster_labels[k] == old_label) {
                                    cluster_labels[k] = new_label;
                                }
                            }
                            // Update the average orientation of the merged cluster
                            cluster_avg_orientations[new_label] = merged_avg_orientation;

                            merged = true;
                            break;
                        }
                    }
                }
            }

            // Group indices by their cluster label
            std::map<int, std::vector<int> > label_to_cluster;
            for (int i = 0; i < N; ++i) {
                label_to_cluster[cluster_labels[i]].push_back(i);
            }

            //////////// push to clusters////////////
            thread_local_clusters.clear(); // Clear for this H1 edge

            std::map<int, std::vector<int> >::iterator kv_it;
            for (kv_it = label_to_cluster.begin(); kv_it != label_to_cluster.end(); ++kv_it) {
                std::vector<int> original_indices;
                
                for (size_t i = 0; i < kv_it->second.size(); ++i) {
                    int local_idx = kv_it->second[i];
                    if (local_idx >= 0 && local_idx < HYPO2_idx_raw.rows()) {
                        int original_idx = static_cast<int>(HYPO2_idx_raw(local_idx));
                        if (original_idx >= 0 && original_idx < Edges_HYPO2.rows()) {
                            original_indices.push_back(original_idx);
                        }
                    }
                }
                
                for (size_t i = 0; i < original_indices.size(); ++i) {
                    int original_idx = original_indices[i];
                    thread_local_clusters[original_idx] = original_indices;
                }
            }

            #pragma omp critical
            {
                std::unordered_map<int, std::vector<int> >::iterator kv_it;
                for (kv_it = thread_local_clusters.begin(); kv_it != thread_local_clusters.end(); ++kv_it) {
                    hypo2_clusters[std::make_pair(H1_edge_idx, kv_it->first)] = kv_it->second;
                }
            }
            //////////// push to clusters////////////

            std::vector<std::vector<int> > clusters;
            std::map<int, std::vector<int> >::iterator it;
            for (it = label_to_cluster.begin(); it != label_to_cluster.end(); ++it) {
                clusters.push_back(it->second);
            }

            // Initialize the HYPO2_idx matrix with the correct size
            HYPO2_idx.resize(N, 1);

            // For each cluster, compute the average edge and update all edges in the cluster
            for (size_t c = 0; c < clusters.size(); ++c) {
                const std::vector<int>& cluster = clusters[c];
                if (cluster.empty()) continue;
                
                // First, check if there are any edges with orientations close to ±90 degrees
                std::vector<double> orientations;
                int posNinetyCount = 0;
                int negNinetyCount = 0;

                //std::cout<<cluster.size()<<std::endl;
                
                for (size_t i = 0; i < cluster.size(); ++i) {
                    int idx = cluster[i];
                    double orientation_rad = Edges_HYPO2_final(idx, 2);
                    double orientation_deg = orientation_rad * (180.0 / M_PI);
                    orientations.push_back(orientation_rad); //all edges orientations are in -pi/2 to pi/2
                    
                    if (orientation_deg >= 0) {
                        posNinetyCount++;
                    } else{
                        negNinetyCount++;
                    }
                }
                
                bool majorityIsPositive = (posNinetyCount > negNinetyCount);
                // std::cout<<"majorityIsPositive is: "<<majorityIsPositive<<std::endl;
                
                bool flip_flag = false;

                // Process orientations before averaging
                for (size_t i = 0; i < cluster.size(); ++i) {
                    int idx = cluster[i];
                    double orientation_rad = Edges_HYPO2_final(idx, 2);
                    double orientation_deg = orientation_rad * (180.0 / M_PI);
                    double flipped_orientation;
                    
                    
                   // Check if this edge needs orientation flipping
                    if ((orientation_deg >= 70 && orientation_deg <= 90 && !majorityIsPositive) ||
                        (orientation_deg <= -70 && orientation_deg >= -90 && majorityIsPositive)) {
                    
                        flip_flag = true;

                        if (orientation_deg >= 70) {
                            flipped_orientation = orientation_rad - M_PI; 
                        }else {
                            flipped_orientation = orientation_rad + M_PI; // This will be close to +π/2
                        }
                        
                        // Update the orientation in the edge data
                        Edges_HYPO2_final(idx, 2) = flipped_orientation;
                        
                        std::cout << "Flipped edge orientation from " << orientation_deg << " to " << (flipped_orientation * 180.0 / M_PI)<< " degrees in cluster " << c << std::endl;
                        
                    }
                }
                if(flip_flag){
                    for (size_t j = 0; j < cluster.size(); ++j) {
                            std::cout<<"after flipping orientation: "<<Edges_HYPO2_final(cluster[j], 2) * (180.0 / M_PI)<<std::endl;
                        }
                    exit(0);
                }
                
                // Compute average position and orientation
                Eigen::RowVector4d cluster_avg = Eigen::RowVector4d::Zero();
                for (size_t i = 0; i < cluster.size(); ++i) {
                    int idx = cluster[i];
                    cluster_avg += Edges_HYPO2_final.row(idx);
                }
                cluster_avg /= static_cast<double>(cluster.size());
                
                // DEBUGGING: Check if average orientation differs significantly from original orientations
                // for (size_t i = 0; i < cluster.size(); ++i) {
                //     int idx = cluster[i];
                //     double original_orientation = Edges_HYPO2_final(idx, 2);
                //     double average_orientation = cluster_avg(2);
                //     double angle_diff_rad = std::abs(original_orientation - average_orientation);
                //     double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
                    
                //     // Normalize to range [0, 180]
                //     if (angle_diff_deg > 180) {
                //         angle_diff_deg = 360 - angle_diff_deg;
                //     }
                    
                //     // Alert if difference is greater than 20 degrees
                //     if (angle_diff_deg > 20.0) {
                //         std::cout << "WARNING: Large orientation difference in cluster!" << std::endl;
                //         for (size_t j = 0; j < cluster.size(); ++j) {
                //             int idx_inner = cluster[j];
                //             std::cout << "  Edge in the cluster before epipolar correction:( " << edgels_HYPO2(idx_inner, 0) << ", " << edgels_HYPO2(idx_inner, 1) << "), orientation: " << edgels_HYPO2(idx_inner, 2)<<", "<< edgels_HYPO2(idx_inner, 2) * (180.0 / M_PI) << " degrees" << std::endl;
                //             std::cout << "  Edge in the cluster after epipolar correction:( " << Edges_HYPO2_final(idx_inner, 0) << ", " << Edges_HYPO2_final(idx_inner, 1) << "), orientation: " << Edges_HYPO2_final(idx_inner, 2)<<", "<< Edges_HYPO2_final(idx_inner, 2) * (180.0 / M_PI) << " degrees" << std::endl;
                //         }
                //         std::cout << "  Average orientation: " << average_orientation * (180.0 / M_PI) << " degrees" << std::endl;
                //         std::cout << "  Difference: " << angle_diff_deg << " degrees" << std::endl;
                //         std::cout << "  Cluster centroid location: ("<<cluster_avg(0) << ", " << cluster_avg(1) << "), orientation: " << cluster_avg(2) << std::endl;
                //     }
                // }
                
                // Find the edge closest to the average to use as the representative
                double min_dist = std::numeric_limits<double>::max();
                int closest_idx = -1;
                for (size_t i = 0; i < cluster.size(); ++i) {
                    int idx = cluster[i];
                    double dist = (Edges_HYPO2_final.row(idx).head<2>() - cluster_avg.head<2>()).norm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_idx = idx;
                    }
                }
                
                // Update all edges in the cluster with the average edge
                for (size_t i = 0; i < cluster.size(); ++i) {
                    int idx = cluster[i];
                    Edges_HYPO2_final.row(idx) = cluster_avg;
                    // Preserve the original index for reference
                    if (edgels_HYPO2_corrected.cols() > 8) {
                        HYPO2_idx(idx, 0) = edgels_HYPO2_corrected(closest_idx, 8);
                    } else {
                        HYPO2_idx(idx, 0) = -2;
                    }
                }
            }
            ////////////////////////////////////// cluster hypothesis 2's edges //////////////////////////////////////
            if(exit_flag){
                std::cout << "After clustering (Edges_HYPO2_final):\n" << Edges_HYPO2_final << std::endl;
                exit(0);
            }



            int valid_view_counter = 0;
            int stack_idx = 0;
            Eigen::MatrixXd supported_indices;
            supported_indices.conservativeResize(edgels_HYPO2.rows(), Num_Of_Total_Imgs-2);
            Eigen::MatrixXd supported_indice_current;
            supported_indice_current.conservativeResize(edgels_HYPO2.rows(),1);
            Eigen::MatrixXi supported_indices_stack;

            bool isempty_link = true;

            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Second loop:loop over all validation views >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
            for (int VALID_INDX = 0; VALID_INDX < Num_Of_Total_Imgs; VALID_INDX++) {
                //> Skip the two hypothesis views
                if (VALID_INDX == hyp01_view_indx || VALID_INDX == hyp02_view_indx) continue;


                //> Get camera pose and other info for current validation view
                Eigen::MatrixXd TO_Edges_VALID = All_Edgels[VALID_INDX];
                Eigen::Matrix3d R3             = All_R[VALID_INDX];
                Eigen::Vector3d T3             = All_T[VALID_INDX];
                Eigen::MatrixXd VALI_Orient    = TO_Edges_VALID.col(2);
                Eigen::MatrixXd Tangents_VALID;
                Tangents_VALID.conservativeResize(TO_Edges_VALID.rows(),2);
                Tangents_VALID.col(0)          = (VALI_Orient.array()).cos();
                Tangents_VALID.col(1)          = (VALI_Orient.array()).sin();
                Eigen::Matrix3d K3 = (Use_Multiple_K) ? All_K[VALID_INDX] : K;

                //> Relative pose between hypothesis view 1 and validation view
                Eigen::Matrix3d R31 = util->getRelativePose_R21(Rot_HYPO1, R3);
                Eigen::Vector3d T31 = util->getRelativePose_T21(Rot_HYPO1, R3, Transl_HYPO1, T3);

                
                //> Find the reprojected edgels
                Eigen::MatrixXd edge_loc_tgt_gamma3 = getReprojEdgel->getGamma3LocationAndTangent(hyp01_view_indx, hyp02_view_indx, Edges_HYPO1_final, Edges_HYPO2_final, All_R, All_T, VALID_INDX, K_HYPO1, K_HYPO2);
                Eigen::MatrixXd edge_tgt_gamma3     = edge_loc_tgt_gamma3.block(0, edge_loc_tgt_gamma3.cols() - 2, edge_loc_tgt_gamma3.rows(), 2);
                
                //> Calculate the epipolar angle range (Hypo1 --> Vali)
                auto result1 = getOre->getOreListBar(Edges_HYPO1_final, All_R, All_T, K_HYPO1, K3, VALID_INDX, hyp01_view_indx);
                Eigen::MatrixXd OreListBardegree31 = result1.first;
                //Eigen::MatrixXd OreListBardegree31 = getOre->getOreListBar(Edges_HYPO1_final, All_R, All_T, K_HYPO1, K3, VALID_INDX, hyp01_view_indx);
                Eigen::MatrixXd OreListdegree31    = getOre->getOreListVali(TO_Edges_VALID, All_R, All_T, K_HYPO1, K3, VALID_INDX, hyp01_view_indx);

                //> Calculate the epipolar angle range (Hypo2 --> Vali)
                auto result2 = getOre->getOreListBar(Edges_HYPO2_final, All_R, All_T, K_HYPO2, K3, VALID_INDX, hyp02_view_indx);
                Eigen::MatrixXd OreListBardegree32 = result2.first;
                Eigen::MatrixXd OreListdegree32    = getOre->getOreListVali(TO_Edges_VALID, All_R, All_T, K_HYPO2, K3, VALID_INDX, hyp02_view_indx);
                
                //> Find the epipolar angle range of the epipolar wedge on the validation view arising from Edges_HYPO1_final, and parse the corresponding edgels on the validation view
                double thresh_ore31_1 = OreListBardegree31(0,0);
                double thresh_ore31_2 = OreListBardegree31(0,1);
                Eigen::Vector3d epipole1 = result1.second;

                Eigen::MatrixXd vali_idx31 = PairHypo->getHYPO2_idx_Ore(OreListdegree31, thresh_ore31_1, thresh_ore31_2);
                Eigen::MatrixXd edgels_31  = PairHypo->getedgels_HYPO2_Ore(TO_Edges_VALID, OreListdegree31, thresh_ore31_1, thresh_ore31_2);

                //> An array indicating if the two epipolar edges are "almost" parallel (if so, discard the edge pair as too much uncertainty is given)
                Eigen::VectorXd isparallel = Eigen::VectorXd::Ones(Edges_HYPO2_final.rows());

                //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Third loop: loop over each edge from Hypo2 <<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>//
                for (int idx_pair = 0; idx_pair < Edges_HYPO2_final.rows(); idx_pair++) {

                    //> Epipolar angle range of the epipolar wedge on the validation view arising from Edges_HYPO2_final
                    double thresh_ore32_1 = OreListBardegree32(idx_pair,0);
                    double thresh_ore32_2 = OreListBardegree32(idx_pair,1);
                    Eigen::Vector3d epipole2 = result2.second;
                    
                    //> Find the epipolar angle range of the epipolar wedge on the validation view arising from Edges_HYPO2_final, and parse the corresponding edgels on the validation view
                    Eigen::MatrixXd vali_idx32 = PairHypo->getHYPO2_idx_Ore(OreListdegree32, thresh_ore32_1, thresh_ore32_2);
                    Eigen::MatrixXd edgels_32  = PairHypo->getedgels_HYPO2_Ore(TO_Edges_VALID, OreListdegree32, thresh_ore32_1, thresh_ore32_2);
                    
                    //> Check if the two epipolar wedges are almost parallel
                    if ( is_Epipolar_Wedges_in_Parallel( thresh_ore31_1, thresh_ore31_2, thresh_ore32_1, thresh_ore32_2, idx_pair, isparallel, supported_indice_current ) )
                        continue;

                    //> Find all the edges fall inside the two epipolar wedges intersection on validation view, (Hypo1 --> Vali) && (Hypo2 --> Vali)
                    std::vector<double> v_intersection;
                    std::vector<double> v1(vali_idx31.data(), vali_idx31.data() + vali_idx31.rows());
                    std::vector<double> v2(vali_idx32.data(), vali_idx32.data() + vali_idx32.rows());
                    set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v_intersection));
                    Eigen::VectorXd idxVector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v_intersection.data(), v_intersection.size());
                    Eigen::MatrixXd inliner(idxVector);

                    //> Calculate orientation of gamma 3 (the reprojected edge)
                    Eigen::Vector2d edgels_tgt_reproj = {edge_tgt_gamma3(idx_pair,0), edge_tgt_gamma3(idx_pair,1)};
                    //> Get support from validation view for this gamma 3: 
                    //>   >> if the validation edge supports the hypothesis edge pair, return the supporting edge index of the validation view
                    //>   >> if not, return -2
                    int supported_link_indx = getSupport->getSupportIdx(edgels_tgt_reproj, Tangents_VALID, inliner);


                    /////////////////////////////////////////// check if validation edge can be corrected properly //////////////////////////////
                    // if (supported_link_indx != -2) {
                    //     Eigen::Matrix3d Rot_HYPO1_val       = All_R[hyp01_view_indx];
                    //     Eigen::Matrix3d Rot_HYPO3       = All_R[VALID_INDX];
                    //     Eigen::Vector3d Transl_HYPO1_val    = All_T[hyp01_view_indx];
                    //     Eigen::Vector3d Transl_HYPO3    = All_T[VALID_INDX];
                    //     Eigen::Matrix3d R13;
                    //     Eigen::Vector3d T13;
  
                    //     //> Relative pose between hypothesis view 1 and validation view
                    //     Eigen::Matrix3d R31 = util->getRelativePose_R21(Rot_HYPO1, Rot_HYPO3);
                    //     Eigen::Vector3d T31 = util->getRelativePose_T21(Rot_HYPO1, Rot_HYPO3, Transl_HYPO1, Transl_HYPO3);


                    //     util->getRelativePoses(Rot_HYPO1_val, Transl_HYPO1_val, Rot_HYPO3, Transl_HYPO3, R31, T31, R13, T13);
                    //     Eigen::Matrix3d F31 = util->getFundamentalMatrix(K_HYPO1.inverse(), K_HYPO2.inverse(), R31, T31); 
                    //     Eigen::Matrix3d F13 = util->getFundamentalMatrix(K_HYPO2.inverse(), K_HYPO1.inverse(), R13, T13);

                    //     Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct_post_validation(All_Edgels[VALID_INDX].row(supported_link_indx), Edges_HYPO1_final, F31, F13, HYPO2_idx_raw);
                    //     //Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct(All_Edgels[VALID_INDX].row(supported_link_indx), Edges_HYPO1_final, F31, F13, HYPO2_idx_raw);
                    //     if (corrected_validation_edge.rows() == 0) {
                    //         supported_link_indx = -2;
                    //     }
                    // }

                    // if (abs(Edges_HYPO1(H1_edge_idx,0) - 529) < 0.001 && abs(Edges_HYPO1(H1_edge_idx,1) - 398.495) < 0.001 &&
                    //     abs(Edges_HYPO2_final(idx_pair,0) - 422.715) < 0.001 && //clustered value
                    //     abs(Edges_HYPO2_final(idx_pair,1) - 376.059) < 0.001) {
                        
                    //     if (inliner.rows() > 0) {
                    //         std::cout<<"------------------------------------"<<std::endl;
                    //         std::cout << "Validation View Index: " << VALID_INDX << "\n";
                    //         std::cout << "Epipole 1: " << epipole1.transpose() << "\n";
                    //         std::cout << "Angle Range Hypothesis 1: [" << thresh_ore31_1 << ", " << thresh_ore31_2 << "]\n";
                    //         std::cout << "Epipole 2: " << epipole2.transpose() << "\n";
                    //         std::cout << "Angle Range Hypothesis 2: [" << thresh_ore32_1 << ", " << thresh_ore32_2 << "]\n";
                            
                    //         std::cout << "reprojection: "<<VALID_INDX << " " << edge_loc_tgt_gamma3(idx_pair, 0) << " " << edge_loc_tgt_gamma3(idx_pair, 1) << " " << edge_loc_tgt_gamma3(idx_pair, 2) << ";" << std::endl;
                    //         std::cout << "Validation edge location and orientation:" << std::endl;
                    //         for (int idx_inline = 0; idx_inline < inliner.rows(); idx_inline++) {
                    //             // Check if edge coordinates are non-zero
                    //             if (abs(TO_Edges_VALID(inliner(idx_inline), 0)) > 1e-6 || 
                    //                 abs(TO_Edges_VALID(inliner(idx_inline), 1)) > 1e-6) {
                                    
                    //                 std::cout << VALID_INDX << " " << TO_Edges_VALID(inliner(idx_inline), 0) << " " << TO_Edges_VALID(inliner(idx_inline), 1) << " " << TO_Edges_VALID(inliner(idx_inline), 2)<< ";"  << std::endl;
                    //             }
                    //         }
                    //     }
                    // }
                   
                    /////////////////////////////////////////// check if validation edge can be corrected properly //////////////////////////////
                   
                    //> Get the supporting edge index from this validation view
                    supported_indice_current.row(idx_pair) << supported_link_indx;

                    //> If the current hypothesis edge pair is supported by the validation view
                    if (supported_link_indx != -2) {
                        //> CHANGE IT TO INTEGER, SHALL WE?
                        supported_indices_stack.conservativeResize(stack_idx+1,2);
                        supported_indices_stack.row(stack_idx) << idx_pair, supported_link_indx;
                        // supported_indices_stack.row(stack_idx) << double(idx_pair), double(supported_link_indx);
                        isempty_link = false;
                        stack_idx++;
                    }
                }
                //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  End of third loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
                supported_indices.col(valid_view_counter) << supported_indice_current.col(0);
                valid_view_counter++;
            } 
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  End of second loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

            //> Now, for each local thread, stack the supported indices
            local_thread_supported_indices.push_back(supported_indices);

            //> Check for Empty Supported Indices
            if (isempty_link) continue;

            //> Find a list of H2 edge indices (which are paired up with the current H1 edge) where each H2 edge index is unique => indices_stack_unique
            std::vector<double> indices_stack(supported_indices_stack.data(), supported_indices_stack.data() + supported_indices_stack.rows());
            std::vector<double> indices_stack_unique = indices_stack;
            std::sort(indices_stack_unique.begin(), indices_stack_unique.end());
            std::vector<double>::iterator it1;
            it1 = std::unique(indices_stack_unique.begin(), indices_stack_unique.end());
            indices_stack_unique.resize( std::distance(indices_stack_unique.begin(),it1) );

            //> Count the frequency of each hypothesis edge pair 
            Eigen::VectorXd rep_count;
            rep_count.conservativeResize(indices_stack_unique.size(),1);

            for (int unique_idx = 0; unique_idx < indices_stack_unique.size(); unique_idx++) {
                rep_count.row(unique_idx) << count(indices_stack.begin(), indices_stack.end(), indices_stack_unique[unique_idx]);
            }

            //> Find all edge pairs that have more than 4 supporting validation views
            std::vector<int> valid_pairs;
            for (int i = 0; i < rep_count.size(); i++) {
                if (rep_count(i) >= Max_Num_Of_Support_Views) { 
                    valid_pairs.push_back(i);
                }
            }
            if (valid_pairs.empty()) {continue;}
            for (int valid_idx : valid_pairs) {
                int finalpair = int(indices_stack_unique[valid_idx]);
            
                // Find the next available row in paired_edge for this H1 edge
                int pair_row_idx = -1;
                for (int row = H1_edge_idx * Num_Of_Total_Imgs; row < (H1_edge_idx + 1) * Num_Of_Total_Imgs; row++) {
                    if (paired_edge(row, 0) == -2) {  // Find first empty row for this H1 edge
                        pair_row_idx = row;
                        break;
                    }
                }
            
                if (pair_row_idx != -1) {
                    // Store the edge pair information
                    paired_edge.row(pair_row_idx) << H1_edge_idx, HYPO2_idx(finalpair), supported_indices.row(finalpair);
                } 
            }

            // if (abs(pt_H1(0) - 462.853) <0.01 && abs(pt_H1(1) - 434.987) <0.01) {
            // //if (abs(pt_H1(0) - 529) <0.001 && abs(pt_H1(1) - 398.495) <0.001 ) { //abs(pt_H2(0) - 424.879) <0.001 && abs(pt_H2(1) - 374.357) <0.001
            //     std::cout<<"rep count is: "<<rep_count<<std::endl;
            //     for (int iu = 0; iu < indices_stack_unique.size();iu++){
            //         int edge_idx = indices_stack_unique[iu];
            //         std::cout<<Edges_HYPO2_final(edge_idx,0)<<", "<<Edges_HYPO2_final(edge_idx,1)<<std::endl;
            //     }
            // }

            // //> Find the maximal number of validation view supports and check if this number is over the threshold
            // //>   >> If not over the threshold, go to the next H1 edge
            // Eigen::VectorXd::Index maxIndex;
            // int max_num_of_valid_supports = int(rep_count.maxCoeff(&maxIndex));
            // if( max_num_of_valid_supports < Max_Num_Of_Support_Views ){
            //     continue;
            // }
   
            // int finalpair = -2;

            // //> If there are more than 1 hypothesis edge pair that has maximal validation view supports
            // int num_of_H2_edges_with_max_valid_supports = std::count(rep_count.data(), rep_count.data()+rep_count.size(), max_num_of_valid_supports);
            // if (num_of_H2_edges_with_max_valid_supports > 1) {
            //     std::vector<double> rep_count_vec(rep_count.data(), rep_count.data() + rep_count.rows());
            //     std::vector<int> max_index;
            //     auto start_it = begin(rep_count_vec);
            //     while (start_it != end(rep_count_vec)) {
            //         start_it = std::find(start_it, end(rep_count_vec), max_num_of_valid_supports);
            //         if (start_it != end(rep_count_vec)) {
            //             auto const pos = std::distance(begin(rep_count_vec), start_it);
            //             max_index.push_back(int(pos));
            //             ++start_it;
            //         }
            //     }

            //     //Select the Final Paired Edge
            //     Eigen::Vector3d coeffs;
            //     coeffs = F21 * pt_edgel_HYPO1;
            //     Eigen::MatrixXd Edge_Pts;
            //     Edge_Pts.conservativeResize(max_index.size(),2);
            //     for(int maxidx = 0; maxidx<max_index.size(); maxidx++){
            //         //std::cout<<indices_stack_unique[max_index[maxidx]]<<std::endl;
            //         Edge_Pts.row(maxidx) << Edges_HYPO2_final(indices_stack_unique[max_index[maxidx]], 0), \
            //                                 Edges_HYPO2_final(indices_stack_unique[max_index[maxidx]], 1);
            //     }
            //     Eigen::VectorXd Ap = coeffs(0)*Edge_Pts.col(0);
            //     Eigen::VectorXd Bp = coeffs(1)*Edge_Pts.col(1);
            //     Eigen::VectorXd numDist = Ap + Bp + Eigen::VectorXd::Ones(Ap.rows())*coeffs(2);
            //     double denomDist = coeffs(0)*coeffs(0) + coeffs(1)*coeffs(1);
            //     denomDist = sqrt(denomDist);
            //     Eigen::VectorXd dist = numDist.cwiseAbs()/denomDist;
            //     Eigen::VectorXd::Index   minIndex;
            //     double min_dist = dist.minCoeff(&minIndex);
            //     //> ignore if the distance from the reprojected edge to the epipolar line is greater than some threshold
            //     if (min_dist > Reproj_Dist_Thresh) continue;

            //     finalpair = int(indices_stack_unique[max_index[minIndex]]);
            // }
            // else {
            //     finalpair = int(indices_stack_unique[int(maxIndex)]);
            // }

            //paired_edge.row(H1_edge_idx) << H1_edge_idx, HYPO2_idx(finalpair), supported_indices.row(finalpair);

            //> paired_edge is a row vector continaing (hypo1 edge index), (paired hypo2 edge index), (paired validation edge indices)
            //paired_edge.row(H1_edge_idx*Num_Of_Total_Imgs +Num_Of_Total_Imgs) << H1_edge_idx, HYPO2_idx(finalpair), supported_indices.row(finalpair);

        }
        //> A critical session to stack all local supported indices
        #pragma omp critical
        all_supported_indices.insert(all_supported_indices.end(), local_thread_supported_indices.begin(), local_thread_supported_indices.end());
    }
    pair_edges_time += omp_get_wtime() - itime;
}

// void EdgeSketch_Core::saveBestMatchesToFile(const std::unordered_map<int, int>& hypothesis1ToBestMatch,
//                            const std::unordered_map<int, int>& hypothesis2ToBestMatch,
//                            const std::string& filename) {
//     std::ofstream outFile(filename);

//     if (!outFile.is_open()) {
//         std::cerr << "Error: Could not open file " << filename << " for writing.\n";
//         return;
//     }
//     // Find and write mutual matches
//     outFile << "Mutual Matches\n";
//     outFile << "--------------------------\n";
//     for (const auto& h1_pair : hypothesis1ToBestMatch) {
//         int h1 = h1_pair.first;
//         int h2 = h1_pair.second;

//         // Check if the reverse match exists
//         auto it = hypothesis2ToBestMatch.find(h2);
//         if (it != hypothesis2ToBestMatch.end() && it->second == h1) {
//             outFile << "H1:H2 " << h1 <<" "<< h2 << "\n";
//         }
//     }

//     outFile.close();
//     std::cout << "Best matches and mutual matches have been saved to " << filename << "\n";
// }


std::unordered_map<int, int> EdgeSketch_Core::saveBestMatchesToFile(const std::unordered_map<int, int>& hypothesis1ToBestMatch,
                                                                    const std::unordered_map<int, int>& hypothesis2ToBestMatch,
                                                                    const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return {}; // Return empty map if file cannot be opened
    }

    std::unordered_map<int, int> mutualMatches;

    // Find and write mutual matches
    outFile << "Mutual Matches\n";
    outFile << "--------------------------\n";
    for (const auto& h1_pair : hypothesis1ToBestMatch) {
        int h1 = h1_pair.first;
        int h2 = h1_pair.second;

        // Check if the reverse match exists
        auto it = hypothesis2ToBestMatch.find(h2);
        if (it != hypothesis2ToBestMatch.end() && it->second == h1) {
            outFile << "H1:H2 " << h1 << " " << h2 << "\n";
            mutualMatches[h2] = h1; // Store mutual match with H2 as the key
        }
    }

    outFile.close();
    std::cout << "Best matches and mutual matches have been saved to " << filename << "\n";

    return mutualMatches; // Return the map of mutual matches
}




void EdgeSketch_Core::Finalize_Edge_Pairs_and_Reconstruct_3D_Edges(std::shared_ptr<EdgeMapping> edgeMapping) {

    //std::unordered_map<int, int> mutualMatches = saveBestMatchesToFile(hypothesis1_best_match, hypothesis2_best_match, "../../outputs/best_matches.txt");
    itime = omp_get_wtime();

    std::vector<int> valid_pair_index;
    std::vector<int> valid_pair_indices;
    
    for (int pair_idx = 0; pair_idx < paired_edge.rows(); pair_idx++) {
        int H1_index = paired_edge(pair_idx, 0);
        if (H1_index != -2 && H1_index != -3) {
            valid_pair_indices.push_back(pair_idx);
        }
    }

    int pair_num = valid_pair_indices.size();
    paired_edge_final = Eigen::MatrixXd::Constant(pair_num, paired_edge.cols(), -2);

    for (int i = 0; i < pair_num; i++) {
        int row_idx = valid_pair_indices[i];

        paired_edge_final(i, 0) = paired_edge(row_idx, 0); // Hypothesis 1 edge index
        paired_edge_final(i, 1) = paired_edge(row_idx, 1); // Hypothesis 2 edge index

        for (int col = 2; col < paired_edge.cols(); col++) {
            paired_edge_final(i, col) = paired_edge(row_idx, col); // Validation view indices
        }

        // Debugging: Print validation indices for a specific hypothesis edge
        int H1_edge_idx = paired_edge_final(i, 0);
        int H2_edge_idx = paired_edge_final(i, 1);
    }

    std::string info_str = "Number of valid edge pairs: " + std::to_string(pair_num);
    LOG_INFOR_MESG(info_str);

#if DEBUG_PAIRED_EDGES
    std::ofstream debug_file_paired_edges;
    std::string Output_File_Path_Paired_Edges = "../../outputs/paired_edge_final.txt";
    debug_file_paired_edges.open(Output_File_Path_Paired_Edges);
    debug_file_paired_edges << paired_edge_final;
    debug_file_paired_edges.close();

    std::ofstream paired_edges_locations_file;
    std::string Output_File_Path_Edge_Locations = "../../outputs/paired_edges_final_" + std::to_string(hyp01_view_indx) + "_" + std::to_string(hyp02_view_indx) + ".txt";
    paired_edges_locations_file.open(Output_File_Path_Edge_Locations);


    for (int pair_idx = 0; pair_idx < paired_edge_final.rows(); ++pair_idx) {
        // Write Hypothesis 1 and Hypothesis 2 edge locations
        int hypo1_idx = paired_edge_final(pair_idx, 0);
        int hypo2_idx = paired_edge_final(pair_idx, 1);
            
        Eigen::Vector2d edge_hypo1 = Edges_HYPO1.row(hypo1_idx).head<2>();
        Eigen::Vector2d edge_hypo2 = Edges_HYPO2.row(hypo2_idx).head<2>();

        Eigen::MatrixXd edgel_HYPO1   = Edges_HYPO1.row(int(paired_edge_final(pair_idx,0)));  //> edge index in hypo 1
        Eigen::MatrixXd edgel_HYPO2   = Edges_HYPO2.row(int(paired_edge_final(pair_idx,1)));  //> edge index in hypo 2
        Eigen::MatrixXd HYPO2_idx_raw = Edges_HYPO2.row(int(paired_edge_final(pair_idx,1)));

        Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);
        //Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);
        Eigen::MatrixXd Edges_HYPO1_final(edgels_HYPO2_corrected.rows(), 4);
        Edges_HYPO1_final << edgels_HYPO2_corrected.col(0), edgels_HYPO2_corrected.col(1), edgels_HYPO2_corrected.col(2), edgels_HYPO2_corrected.col(3);
        Eigen::MatrixXd Edges_HYPO2_final(edgels_HYPO2_corrected.rows(), 4);
        Edges_HYPO2_final << edgels_HYPO2_corrected.col(4), edgels_HYPO2_corrected.col(5), edgels_HYPO2_corrected.col(6), edgels_HYPO2_corrected.col(7);

        Eigen::Vector2d pt_H1 = Edges_HYPO1_final.row(0);
        Eigen::Vector2d pt_H2 = Edges_HYPO2_final.row(0);
        
        if (HYPO2_idx_raw.rows() == 0 || edgels_HYPO2_corrected.rows() == 0) {
            std::cout << "No valid matches found for edge " << pair_idx << " at threshold " << thresh_EDG << std::endl;
            continue;
        }

        paired_edges_locations_file << "Pair " << pair_idx + 1 << ":\n";
        Eigen::RowVectorXd R_vector1 = Eigen::Map<Eigen::RowVectorXd>(All_R[hyp01_view_indx].data(), All_R[hyp01_view_indx].size());
        Eigen::RowVectorXd R_vector2 = Eigen::Map<Eigen::RowVectorXd>(All_R[hyp02_view_indx].data(), All_R[hyp02_view_indx].size());
    
        paired_edges_locations_file << pt_H1(0) << " " << pt_H1(1) << " " << R_vector1 << " " << All_T[hyp01_view_indx].transpose() << "\n";
        paired_edges_locations_file << pt_H2(0) << " " << pt_H2(1) << " " << R_vector2 << " " << All_T[hyp02_view_indx].transpose() << "\n";

        int val_count = 0;
        int val_count_pre = 0;
        // Loop through validation views and write actual edge locations and R, T matrices
        for (int col = 2; col < paired_edge_final.cols(); col++) {
            
            int val_idx = valid_view_index[col - 2]; 
            int support_idx = paired_edge_final(pair_idx, col);
            if (support_idx != -2) {
                val_count_pre ++;
                /////////////////////////////////// epipolar correcting validation view edges ///////////////////////////////////
                Eigen::RowVectorXd R_vector = Eigen::Map<Eigen::RowVectorXd>(All_R[val_idx].data(), All_R[val_idx].size());
                Eigen::MatrixXd edgel_VALID = All_Edgels[val_idx].row(support_idx);
                Eigen::Matrix3d Rot_HYPO1_val       = All_R[hyp01_view_indx];
                Eigen::Matrix3d Rot_HYPO3       = All_R[val_idx];
                Eigen::Vector3d Transl_HYPO1_val    = All_T[hyp01_view_indx];
                Eigen::Vector3d Transl_HYPO3    = All_T[val_idx];
                Eigen::Matrix3d R31;
                Eigen::Vector3d T31;
                Eigen::Matrix3d R13;
                Eigen::Vector3d T13;

                if (Use_Multiple_K) {
                    K_HYPO1 = All_K[hyp01_view_indx];
                    K_HYPO2 = All_K[hyp02_view_indx];
                }
                else {
                    K_HYPO1 = K;
                    K_HYPO2 = K;
                }
                util->getRelativePoses(Rot_HYPO1_val, Transl_HYPO1_val, Rot_HYPO3, Transl_HYPO3, R31, T31, R13, T13);
                Eigen::Matrix3d F31 = util->getFundamentalMatrix(K_HYPO1.inverse(), K_HYPO2.inverse(), R31, T31); 
                Eigen::Matrix3d F13 = util->getFundamentalMatrix(K_HYPO2.inverse(), K_HYPO1.inverse(), R13, T13);

                Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct(edgel_VALID, edgel_HYPO1, F31, F13, HYPO2_idx_raw);
                //Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct_post_validation(edgel_VALID, edgel_HYPO1, F31, F13, HYPO2_idx_raw);
                Eigen::MatrixXd Edges_VAL_final(corrected_validation_edge.rows(), 4);
                Edges_VAL_final << corrected_validation_edge.col(4), corrected_validation_edge.col(5), corrected_validation_edge.col(6), corrected_validation_edge.col(7);
                if (corrected_validation_edge.rows() == 0) {
                    continue;
                }

                if (corrected_validation_edge.rows() > 0) {
                    Eigen::Vector2d pt_VAL = Edges_VAL_final.row(0);
                    paired_edges_locations_file << pt_VAL(0) << " " << pt_VAL(1) << " " << R_vector << " " << All_T[val_idx].transpose() << "\n";
                    // if (abs(pt_H1(0) - 440.429) <0.001 && abs(pt_H1(1) - 414.716) <0.001  && abs(pt_H2(0) - 369.995) <0.001 && abs(pt_H2(1) - 457.536) <0.001) {
                    // //if (abs(pt_H1(0) - 349.01) <0.001 && abs(pt_H1(1) - 313.545) <0.001) {
                    //     int hypo2_index = int(paired_edge_final(pair_idx,1));
                    //     //std::cout<<pt_H2<<std::endl;
                    //     std::cout << "validation view: " << val_idx << " point: "<<Edges_VAL_final.row(0)<< std::endl;
                    // }
                }
                val_count++;
                /////////////////////////////////// epipolar correcting validation view edges ///////////////////////////////////

                Eigen::Vector2d supporting_edge = All_Edgels[val_idx].row(support_idx).head<2>();
                
            }

        }
        if (val_count<4 || val_count_pre < 4) {
            std::cout << "!!!ERROR!!!"<< std::endl;
        }
        paired_edges_locations_file << "\n"; // Newline between pairs
    }
    paired_edges_locations_file.close();

#endif

    std::vector<Eigen::Matrix3d> Rs;
    Rs.push_back(R21);
    std::vector<Eigen::Vector3d> Ts;
    Ts.push_back(T21);
    
    std::vector<Eigen::Matrix3d> abs_Rs;
    std::vector<Eigen::Vector3d> abs_Ts;
    abs_Rs.push_back(All_R[hyp01_view_indx]);
    abs_Rs.push_back(All_R[hyp02_view_indx]);
    abs_Ts.push_back(All_T[hyp01_view_indx]);
    abs_Ts.push_back(All_T[hyp02_view_indx]);

    int hypo1_identifier = 0;
    int previous_hypo1 = -1; 
    Gamma1s.conservativeResize(paired_edge_final.rows(),3);
    tangent3Ds.conservativeResize(paired_edge_final.rows(), 3);

    int count = 0;
    int valid_pair_idx = 0;


    for (int pair_idx = 0; pair_idx < paired_edge_final.rows(); pair_idx++) {
        
        // int hypo1_idx = paired_edge_final(pair_idx, 0);
        // int hypo2_idx = paired_edge_final(pair_idx, 1);
        // auto it = mutualMatches.find(hypo1_idx);
        // if (it == mutualMatches.end() || it->second != hypo2_idx) {
        //     count +=1;
        //     std::cout<<count<<std::endl;
        //     continue; // Skip this pair if it's not mutual
        // }
        // Get the hypothesis 1 edge index
        int current_hypo1 = int(paired_edge_final(pair_idx, 0));

        // Check if this is a new hypothesis 1 edge
        if (current_hypo1 != previous_hypo1) {
            hypo1_identifier++;
        }
        previous_hypo1 = current_hypo1;

        Eigen::MatrixXd edgel_HYPO1   = Edges_HYPO1.row(int(paired_edge_final(pair_idx,0)));  //> edge index in hypo 1
        Eigen::MatrixXd edgel_HYPO2   = Edges_HYPO2.row(int(paired_edge_final(pair_idx,1)));  //> edge index in hypo 2
        Eigen::MatrixXd HYPO2_idx_raw = Edges_HYPO2.row(int(paired_edge_final(pair_idx,1)));
        int hypo1_idx = int(paired_edge_final(pair_idx,0));
        int hypo2_idx = int(paired_edge_final(pair_idx,1));

        /////////// get cluster member ///////////
        std::vector<int> cluster_members = get_edges_in_same_cluster(hypo1_idx, hypo2_idx);
        // Eigen::Vector2d hypo2_location = Edges_HYPO2.row(hypo2_idx).head<2>();
        // std::cout << "Hypo2 edge " << hypo2_idx << " location: (" 
        //           << hypo2_location(0) << ", " << hypo2_location(1) << ")" << std::endl;
        
        // std::cout << "Cluster members locations:" << std::endl;
        // for (int member_idx : cluster_members) {
        //     if (member_idx < 0 || member_idx >= Edges_HYPO2.rows()) {
        //         std::cout << "  Member " << member_idx << ": Invalid index" << std::endl;
        //         continue;
        //     }
            
        //     Eigen::Vector2d member_location = Edges_HYPO2.row(member_idx).head<2>();
        //     std::cout << "  Member " << member_idx << " location: (" 
        //               << member_location(0) << ", " << member_location(1) << ")" << std::endl;
        // }
        // exit(0);
        /////////// get cluster member ///////////

        Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);
        //Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);

        if (HYPO2_idx_raw.rows() == 0 || edgels_HYPO2_corrected.rows() == 0) {
            std::cout << "No valid matches found for edge " << pair_idx << " at threshold " << thresh_EDG << std::endl;
            continue;
        }

        Eigen::MatrixXd Edges_HYPO1_final(edgels_HYPO2_corrected.rows(),4);
        Edges_HYPO1_final << edgels_HYPO2_corrected.col(0), edgels_HYPO2_corrected.col(1), edgels_HYPO2_corrected.col(2), edgels_HYPO2_corrected.col(3);
        Eigen::MatrixXd Edges_HYPO2_final(edgels_HYPO2_corrected.rows(),4);
        Edges_HYPO2_final << edgels_HYPO2_corrected.col(4), edgels_HYPO2_corrected.col(5), edgels_HYPO2_corrected.col(6), edgels_HYPO2_corrected.col(7);

        Eigen::Vector2d pt_H1 = Edges_HYPO1_final.row(0);
        Eigen::Vector2d pt_H2 = Edges_HYPO2_final.row(0);
        std::vector<Eigen::Vector2d> pts;
        pts.push_back(pt_H1);
        pts.push_back(pt_H2);

        //> The resultant edge_pt_3D is 3D edges "under the first hypothesis view coordinate"
        Eigen::Vector3d edge_pt_3D = util->linearTriangulation(2, pts, Rs, Ts, K_HYPO1);


        if (edge_pt_3D.hasNaN()) {
            LOG_ERROR("NaN values detected in edge_pt_3D for pair_idx: ");
            Gamma1s.row(pair_idx)<< 0, 0, 0;  //> TBD
            std::cerr << pair_idx << std::endl;
            continue;
        }

        Eigen::Vector3d edge_pt_3D_world = util->transformToWorldCoordinates(edge_pt_3D, All_R[hyp01_view_indx], All_T[hyp01_view_indx]);

        // if (abs(pt_H2(0) - 461.653) <0.01 && abs(pt_H2(1) - 429.427) <0.01) {
        //     std::cout<<"matched pt_H2 is: "<<pt_H2.transpose()<<std::endl;
        //     std::cout<<"world 3D edge is: "<<edge_pt_3D_world.transpose()<<std::endl;
        //     //std::cout<<"triangulated 3d edge is: "<< edge_pt_3D.transpose()<<std::endl;
        //     //exit(0);
        // }

        Gamma1s.row(valid_pair_idx) << edge_pt_3D(0), edge_pt_3D(1), edge_pt_3D(2);

        // const double EPSILON = 1e-4;  // Increase tolerance from 1e-6 to 1e-4
        // if (std::abs(edge_pt_3D_world(0) - 0.330075) < EPSILON &&        // if (std::abs(edge_pt_3D(0) - 0.187817) < EPSILON &&
        //     std::abs(edge_pt_3D_world(1) - 0.346706) < EPSILON &&        //     std::abs(edge_pt_3D(1) + 0.373794) < EPSILON &&
        //     std::abs(edge_pt_3D_world(2) - 0.368642) < EPSILON) {        //     std::abs(edge_pt_3D(2) + 3.68693) < EPSILON) {
        //     std::cout << "Matched tangents_3D: " << Gamma1s.row(valid_pair_idx) << std::endl;
        //     std::cout<<"hyp1 is: " <<Edges_HYPO1_final.transpose() <<std::endl;
        //     std::cout<<"hyp2 is: "<<pt_H2.transpose()<<", original hyp2 is: "<<edgel_HYPO2<<std::endl;
        //     for (int member_idx : cluster_members) {
        //         Eigen::Vector3d member_location = Edges_HYPO2.row(member_idx).head<3>();
        //         std::cout<<"hypo2 cluster member location is: "<<member_location.transpose()<<std::endl;
        //     }

        //     exit(0);
        // }
        ///////////////////// debug /////////////////////
       

        //> Triangulate edge orientations and make them in the world coordinate
        Eigen::Vector3d Edgel_View1(edgel_HYPO1(0),  edgel_HYPO1(1), edgel_HYPO1(2));
        Eigen::Vector3d Edgel_View2(edgel_HYPO2(0),  edgel_HYPO2(1), edgel_HYPO2(2));
        Eigen::Vector3d tangents_3D_world = util->get3DTangentFromTwo2Dtangents(Edgel_View1, Edgel_View2, K_HYPO1, K_HYPO2, All_R[hyp01_view_indx], All_T[hyp01_view_indx], All_R[hyp02_view_indx], All_T[hyp02_view_indx]);
        tangent3Ds.row(valid_pair_idx) = tangents_3D_world;

        valid_pair_idx++;
        
        Eigen::Vector3d edge_uncorrected_hyp1 = edgel_HYPO1.block<1,3>(0,0).transpose();
        Eigen::Vector3d edge_uncorrected_hyp2 = edgel_HYPO2.block<1,3>(0,0).transpose();
        edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D_world, tangents_3D_world, pt_H1, edge_uncorrected_hyp1, hyp01_view_indx, hypo1_idx, All_R[hyp01_view_indx], All_T[hyp01_view_indx]);
        edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D_world, tangents_3D_world, pt_H2, edge_uncorrected_hyp2, hyp02_view_indx, hypo2_idx, All_R[hyp02_view_indx], All_T[hyp02_view_indx]);

        //std::cout<<"hypo2 edge is: "<<edge_uncorrected_hyp2.transpose()<<std::endl;
        ////////// include all cluster member in the edge mapping ///////////
        for (int member_idx : cluster_members) {
            if (member_idx < 0 || member_idx >= Edges_HYPO2.rows() || member_idx == hypo2_idx) {
                //std::cout << "  Member " << member_idx << ": Invalid index" << std::endl;
                continue;
            }

            Eigen::Vector3d member_location = Edges_HYPO2.row(member_idx).head<3>();
            //std::cout<<"cluster member location is: "<<member_location.transpose()<<std::endl;

            // if (abs(pt_H1(0) - 274.278) <0.001 && abs(pt_H1(1) - 410.995) <0.001  && abs(pt_H2(0) - 284.165) <0.001 && abs(pt_H2(1) - 422.119) <0.001) {
            //     std::cout<<"cluster member location is: "<<member_location.transpose()<<std::endl;
            // }

            edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D_world, tangents_3D_world, pt_H2, member_location, hyp02_view_indx, member_idx, All_R[hyp02_view_indx], All_T[hyp02_view_indx]);
        }
        //exit(0);
        ////////// include all cluster member in the edge mapping ///////////
        
        ///////////////////////////////// Add support from validation views /////////////////////////////////
        std::vector<std::pair<int, Eigen::Vector2d>> validation_support_edges;

        // Loop through validation views to find the supporting edges
        int val_indx_in_paired_edge_array = 2; // +2 accounts for the first two columns for HYPO1 and HYPO2
        for (int val_idx = 0; val_idx < Num_Of_Total_Imgs; ++val_idx) {
            if (val_idx == hyp01_view_indx || val_idx == hyp02_view_indx) {
                continue;  // Skip hypothesis views
            }

            // Retrieve support index from paired_edge for the current validation view
            int support_idx = paired_edge_final(pair_idx, val_indx_in_paired_edge_array);  
            if (support_idx != -2) {
                // Retrieve the supporting edge from the validation view
                Eigen::MatrixXd edges_for_val_frame = All_Edgels[val_idx];

                if (edges_for_val_frame.rows() <= support_idx) {
                    LOG_ERROR("Something buggy here!\n");
                    std::cout << "(pair_idx, val_idx, edges_for_val_frame.rows(), support_idx) = (" << pair_idx << ", " << val_idx << ", " << edges_for_val_frame.rows() << ", " << support_idx << ")" << std::endl;
                    Eigen::Vector2d supporting_edge = edges_for_val_frame.row(support_idx).head<2>();
                }

                Eigen::Vector2d supporting_edge = edges_for_val_frame.row(support_idx).head<2>();
                Eigen::Vector3d supporting_edge_mapping = edges_for_val_frame.row(support_idx).head<3>();

                /////////////////////////////// epipolar correct validation edges ////////////////////////////////
                Eigen::Matrix3d Rot_HYPO1_val       = All_R[hyp01_view_indx];
                Eigen::Matrix3d Rot_HYPO3       = All_R[val_idx];
                Eigen::Vector3d Transl_HYPO1_val    = All_T[hyp01_view_indx];
                Eigen::Vector3d Transl_HYPO3    = All_T[val_idx];
                Eigen::Matrix3d R13;
                Eigen::Vector3d T13;
                Eigen::Matrix3d R31;
                Eigen::Vector3d T31;
                if (Use_Multiple_K) {
                    K_HYPO1 = All_K[hyp01_view_indx];
                    K_HYPO2 = All_K[hyp02_view_indx];
                }
                else {
                    K_HYPO1 = K;
                    K_HYPO2 = K;
                }

                util->getRelativePoses(Rot_HYPO1_val, Transl_HYPO1_val, Rot_HYPO3, Transl_HYPO3, R31, T31, R13, T13);
                Eigen::Matrix3d F31 = util->getFundamentalMatrix(K_HYPO1.inverse(), K_HYPO2.inverse(), R31, T31); 
                Eigen::Matrix3d F13 = util->getFundamentalMatrix(K_HYPO2.inverse(), K_HYPO1.inverse(), R13, T13);
                //Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct(edges_for_val_frame.row(support_idx), Edges_HYPO1_final, F31, F13, HYPO2_idx_raw);
                Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct_post_validation(edges_for_val_frame.row(support_idx), Edges_HYPO1_final, F31, F13, HYPO2_idx_raw);
                Eigen::MatrixXd Edges_VAL_final(corrected_validation_edge.rows(), 4);
                Edges_VAL_final << corrected_validation_edge.col(4), corrected_validation_edge.col(5), corrected_validation_edge.col(6), corrected_validation_edge.col(7);
                if (corrected_validation_edge.rows() > 0) {
                    Eigen::Vector2d pt_VAL = Edges_VAL_final.row(0);
                    //> Qiwu: Add the supporting edge to the edgeMapping for the 3D edge
                    Eigen::Vector2d corrected_val = pt_VAL.segment(0,2);

                    //> Qiwu: Store validation view and the supporting edge
                    validation_support_edges.emplace_back(val_idx, supporting_edge);
                    //> Qiwu: Add the supporting edge to the edgeMapping for the 3D edge
                    //edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D_world, supporting_edge, val_idx, All_R[val_idx], All_T[val_idx]);
                    edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D_world, tangents_3D_world, corrected_val, supporting_edge_mapping, val_idx, support_idx, All_R[val_idx], All_T[val_idx]);
                }
            }
            val_indx_in_paired_edge_array++;
        }
    }
    finalize_edge_pair_time += omp_get_wtime() - itime;
    //std::cout << "EdgeMapping in EdgeSketch_Core: " << edgeMapping.get() << std::endl;

    std::cout << "edge sketch size of edge_3D_to_supporting_edges: " << edgeMapping->edge_3D_to_supporting_edges.size() << std::endl;
}

void EdgeSketch_Core::Stack_3D_Edges() {
    
    Eigen::Matrix3d R_ref = All_R[hyp01_view_indx];
    Eigen::Vector3d T_ref = All_T[hyp01_view_indx];

    //> Transform the 3D edges from the first hypothesis view coordinate (Gamma1s) to the world coordinate (Gamma1s_world)
    Eigen::MatrixXd Gamma1s_world(Gamma1s.rows(), 3);
    for (int i = 0; i < Gamma1s.rows(); ++i) {
        Eigen::Vector3d point_camera = Gamma1s.row(i).transpose();
        Eigen::Vector3d point_world = util->transformToWorldCoordinates(point_camera, R_ref, T_ref);
        Gamma1s_world.row(i) = point_world.transpose();
        
    }

#if WRITE_3D_EDGES
    std::ofstream myfile2;
    std::string Output_File_Path2 = "../../outputs/3D_edges_" + Dataset_Name + "_" + Scene_Name + "_hypo1_" + std::to_string(hyp01_view_indx) \
                                    + "_hypo2_" + std::to_string(hyp02_view_indx) + "_t" + std::to_string(Edge_Detection_Init_Thresh) + "to" \
                                    + std::to_string(Edge_Detection_Final_Thresh) + "_delta" + Delta_FileName_Str + "_theta" + std::to_string(Orien_Thresh) \
                                    + "_N" + std::to_string(Max_Num_Of_Support_Views) + ".txt";
    std::cout << "Writing 3D edge locations to: " << Output_File_Path2 << std::endl;
    myfile2.open (Output_File_Path2);
    myfile2 << Gamma1s_world;
    myfile2.close();

    std::ofstream tangents_file;
    std::string tangents_output_path = "../../outputs/3D_tangents_" + Dataset_Name + "_" + Scene_Name + "_hypo1_" + std::to_string(hyp01_view_indx) \
                                       + "_hypo2_" + std::to_string(hyp02_view_indx) + "_t" + std::to_string(Edge_Detection_Init_Thresh) + "to" \
                                       + std::to_string(Edge_Detection_Final_Thresh) + "_delta" + Delta_FileName_Str + "_theta" + std::to_string(Orien_Thresh) \
                                       + "_N" + std::to_string(Max_Num_Of_Support_Views) + ".txt";
    std::cout << "Writing 3D edge tangents to: " << tangents_output_path << std::endl;
    tangents_file.open(tangents_output_path);
    tangents_file << tangent3Ds;
    tangents_file.close();
#endif

    Eigen::MatrixXd mvt_3d_edges = NViewsTrian::mvt(hyp01_view_indx, hyp02_view_indx, Scene_Name, Edge_Detection_Init_Thresh, Edge_Detection_Final_Thresh);

    //> Concatenate reconstructed 3D edges
    if (all_3D_Edges.rows() == 0) {
        //all_3D_Edges = Gamma1s_world;
        all_3D_Edges = mvt_3d_edges;
    } 
    else {
        all_3D_Edges.conservativeResize(all_3D_Edges.rows() + mvt_3d_edges.rows(), 3);
        all_3D_Edges.bottomRows(mvt_3d_edges.rows()) = mvt_3d_edges;
        // all_3D_Edges.conservativeResize(all_3D_Edges.rows() + Gamma1s_world.rows(), 3);
        // all_3D_Edges.bottomRows(Gamma1s_world.rows()) = Gamma1s_world;
    }

    //std::cout<<"mvt_3d_edges numer of rows:"<<(double)(mvt_3d_edges.rows())<<", total number of 3d edges: "<<(double)all_3D_Edges.rows()<<std::endl;

}


void EdgeSketch_Core::Calculate_Edge_Support_Ratios_And_Select_Next_Views(std::shared_ptr<EdgeMapping> edgeMapping) {

    itime = omp_get_wtime();
    Load_Data->read_All_Edgels(All_Edgels, Edge_Detection_Final_Thresh);
    
    // For each view, track which edges are supporting any 3D edge
    std::vector<std::set<int>> supportedEdgesIndices(Num_Of_Total_Imgs);
    
    // Iterate through all 3D edges and their supporting 2D edges
    for (auto it = edgeMapping->edge_3D_to_supporting_edges.begin(); 
        it != edgeMapping->edge_3D_to_supporting_edges.end(); ++it) {
        const auto& supportList = it->second;
        
        // Add all supporting edges to their respective view sets
        for (const auto& supportData : supportList) {
            int imageIndex = supportData.image_number;
            int edgeIndex = supportData.edge_idx;
            supportedEdgesIndices[imageIndex].insert(edgeIndex);
        }
    }
    

    std::vector<double> ratios;
    std::vector<std::pair<int, double>> frameSupportCounts;
    
    for (int i = 0; i < Num_Of_Total_Imgs; i++) {
        int totalEdges = All_Edgels[i].rows();
        int supportedEdges = supportedEdgesIndices[i].size();
        
        double ratio = static_cast<double>(supportedEdges) / totalEdges;
        ratios.push_back(ratio);
        frameSupportCounts.push_back({i, ratio});
        
        // std::cout << "View " << i << ": " << supportedEdges << "/" << totalEdges << " edges supported (" << (ratio * 100) << "%)" << std::endl;
    }
    
    avg_ratio = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
    std::cout << "Average ratio of claimed edges: " << avg_ratio << std::endl;
    
    // Select next best hypothesis views
    // Sort views by their claim ratio (ascending)
    std::sort(frameSupportCounts.begin(), frameSupportCounts.end(), 
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second < b.second;
              });
    
    // Select the two views with the lowest claim ratio
    int bestView1 = frameSupportCounts[0].first;
    int bestView2 = frameSupportCounts[1].first;
    
    // Avoid reusing views that have been used before
    int keep_finding_counter = 0;
    while (true) {
        int find_existence_HYPO1 = std::count(history_hypothesis_views_index.begin(), 
                                              history_hypothesis_views_index.end(), bestView1);
        int find_existence_HYPO2 = std::count(history_hypothesis_views_index.begin(), 
                                              history_hypothesis_views_index.end(), bestView2);
        
        if (find_existence_HYPO1 == 0 && find_existence_HYPO2 == 0)
            break;
        
        if (find_existence_HYPO1 > 0 && find_existence_HYPO2 > 0) {
            bestView1 = frameSupportCounts[2 + keep_finding_counter].first;
            bestView2 = frameSupportCounts[3 + keep_finding_counter].first;
        } else if (find_existence_HYPO1 > 0) {
            bestView1 = frameSupportCounts[1 + keep_finding_counter].first;
            bestView2 = frameSupportCounts[2 + keep_finding_counter].first;
        } else if (find_existence_HYPO2 > 0) {
            bestView2 = frameSupportCounts[2 + keep_finding_counter].first;
        }
        
        keep_finding_counter++;
        
        // Prevent infinite loop if we can't find unused views
        if (keep_finding_counter > frameSupportCounts.size() - 2) {
            std::cout << "Warning: Unable to find unused views. Reusing some previous views." << std::endl;
            break;
        }
    }
    
    // Assign the selected views
    hyp01_view_indx = bestView1;
    hyp02_view_indx = bestView2;
    
    std::cout << "Selected next hypothesis views: " << bestView1 << " and " << bestView2 << std::endl;
    
    // Check if we've reached the stopping ratio
    if (avg_ratio > Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges) {
        enable_aborting_3D_edge_sketch = true;
        std::cout << "Reached Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges value" << std::endl;
    } else {
        enable_aborting_3D_edge_sketch = false;
    }
    
    find_next_hypothesis_view_time += omp_get_wtime() - itime;
}


void EdgeSketch_Core::Clear_Data() {
    all_supported_indices.clear();
    All_Edgels.clear();
    claimedEdgesList.clear();
    num_of_nonveridical_edge_pairs = 0;
}

EdgeSketch_Core::~EdgeSketch_Core() { }



#endif 