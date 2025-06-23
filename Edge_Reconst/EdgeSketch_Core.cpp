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
    //> (2) Dataset Settings
    Dataset_Path                                    = Edge_Sketch_Setting_YAML_File["Dataset_Path"].as<std::string>();
    Dataset_Name                                    = Edge_Sketch_Setting_YAML_File["Dataset_Name"].as<std::string>();
    Scene_Name                                      = Edge_Sketch_Setting_YAML_File["Scene_Name"].as<std::string>();
    Num_Of_Total_Imgs                               = Edge_Sketch_Setting_YAML_File["Total_Num_Of_Images"].as<int>();
    Img_Rows                                        = Edge_Sketch_Setting_YAML_File["Image_Rows"].as<int>();
    Img_Cols                                        = Edge_Sketch_Setting_YAML_File["Image_Cols"].as<int>();
    Use_Multiple_K                                  = Edge_Sketch_Setting_YAML_File["Use_Multiple_K"].as<bool>();
    fx                                              = Edge_Sketch_Setting_YAML_File["fx"].as<double>();
    fy                                              = Edge_Sketch_Setting_YAML_File["fy"].as<double>();
    cx                                              = Edge_Sketch_Setting_YAML_File["cx"].as<double>();
    cy                                              = Edge_Sketch_Setting_YAML_File["cy"].as<double>();

    //> Define post file name for files to which 3D edge locations and tangents are written
    Post_File_Name_Str = Dataset_Name + "_" + Scene_Name + "_hypo1_" + std::to_string(hyp01_view_indx) + "_hypo2_" + std::to_string(hyp02_view_indx) \
                        + "_t" + std::to_string(Edge_Detection_Init_Thresh) + "to" + std::to_string(Edge_Detection_Final_Thresh) + "_theta" \
                        + std::to_string(Orien_Thresh) + "_N" + std::to_string(Max_Num_Of_Support_Views) + ".txt";

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
    PairHypo        = std::shared_ptr<PairEdgeHypothesis::pair_edge_hypothesis>(new PairEdgeHypothesis::pair_edge_hypothesis( Reproj_Dist_Thresh ));
    getReprojEdgel  = std::shared_ptr<GetReprojectedEdgel::get_Reprojected_Edgel>(new GetReprojectedEdgel::get_Reprojected_Edgel());
    getSupport      = std::shared_ptr<GetSupportedEdgels::get_SupportedEdgels>(new GetSupportedEdgels::get_SupportedEdgels( Orien_Thresh ));
    getOre          = std::shared_ptr<GetOrientationList::get_OrientationList>(new GetOrientationList::get_OrientationList( Edge_Loc_Pertubation, Img_Rows, Img_Cols ));
    edgeMapping     = std::shared_ptr<EdgeMapping>(new EdgeMapping());

    //> Set up OpenMP threads. Reset it if the upper bound of threads is lower than what is set.
    Num_Of_OMP_Threads = (Num_Of_OMP_Threads > omp_get_max_threads()) ? omp_get_max_threads() : Num_Of_OMP_Threads;
    omp_set_num_threads(Num_Of_OMP_Threads);

    local_hypo2_clusters.resize(Num_Of_OMP_Threads);


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
        // TODO:DOCUMENT THIS: Thread-local clusters 
        std::unordered_map<int, std::vector<int>> thread_local_clusters;
        int H1_edge_idx;
       
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< First loop: loop over all edgels from hypothesis view 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
        #pragma omp for schedule(static, Num_Of_OMP_Threads)
        for (H1_edge_idx = 0; H1_edge_idx < Edges_HYPO1.rows() ; H1_edge_idx++) {

            int thread_id = omp_get_thread_num();

            //> Check if the H1 edge is not close to the image boundary or if it has been visited before
            if ( Skip_this_Edge( H1_edge_idx ) ) continue;

            //> Get the epipolar angle range in H2 arising from the H1 edge
            // double thresh_ore21_1 = OreListBardegree(H1_edge_idx, 0);
            // double thresh_ore21_2 = OreListBardegree(H1_edge_idx, 1);
            std::pair<double, double> epip_angle_range_from_H1_edge = std::make_pair(OreListBardegree(H1_edge_idx, 0), OreListBardegree(H1_edge_idx, 1));

            //> Find the corresponding H2 edge using the epipolar angle range
            //> (i) H2 edge index
            Eigen::MatrixXd HYPO2_idx_raw = PairHypo->getHYPO2_idx_Ore(OreListdegree, epip_angle_range_from_H1_edge);
            if (HYPO2_idx_raw.rows() == 0) continue;
            //> (ii) H2 edge location and orientation
            Eigen::MatrixXd edgels_HYPO2 = PairHypo->getedgels_HYPO2_Ore(Edges_HYPO2, OreListdegree, epip_angle_range_from_H1_edge);

            //> Correct the corresponding H2 edges by shifting to the epipolar line
            // Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgels_HYPO2, Edges_HYPO1.row(H1_edge_idx), F21, F12, HYPO2_idx_raw);
            Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2_epipolar_correction(edgels_HYPO2, Edges_HYPO1.row(H1_edge_idx), F21, F12, HYPO2_idx_raw);
            
            //> Organize the final edge data (hypothesis edge pairs)
            Eigen::MatrixXd Edges_HYPO1_final(edgels_HYPO2_corrected.rows(), 4);
            Edges_HYPO1_final << edgels_HYPO2_corrected.col(0), edgels_HYPO2_corrected.col(1), edgels_HYPO2_corrected.col(2), edgels_HYPO2_corrected.col(3);
            Eigen::MatrixXd Edges_HYPO2_final(edgels_HYPO2_corrected.rows(), 4);
            Edges_HYPO2_final << edgels_HYPO2_corrected.col(4), edgels_HYPO2_corrected.col(5), edgels_HYPO2_corrected.col(6), edgels_HYPO2_corrected.col(7);
            if ( !any_H2_Edge_Surviving_Epipolar_Corrected(Edges_HYPO2_final) ) continue;

            //> Store the Hypo2 Indices
            //Eigen::MatrixXd HYPO2_idx(edgels_HYPO2_corrected.rows(), 1); 
            //HYPO2_idx << edgels_HYPO2_corrected.col(8);
            // if (HYPO2_idx.rows() == 0) continue;
            // std::cout << "Before clustering (Edges_HYPO2_final):\n" << Edges_HYPO2_final << std::endl;
            //std::cout << "Before clustering (HYPO2_idx):\n" << HYPO2_idx_raw.transpose() << std::endl;
            // Eigen::Vector2d pt_H1 = Edges_HYPO1_final.row(0);
            // Eigen::Vector2d pt_H2 = Edges_HYPO2_final.row(0);

            // if (abs(pt_H1(0) - 529) <0.001 && abs(pt_H1(1) - 398.495) <0.001){
            //     std::cout << "Before clustering (Edges_HYPO2_final):\n" << Edges_HYPO2_final << std::endl;
            //     // std::cout<<"edgels_HYPO2 is: "<<edgels_HYPO2<<std::endl;
            //     // exit(0);
            // }

            ////////////////////////////////////// cluster hypothesis 2's edges //////////////////////////////////////

            int Num_Of_Epipolar_Corrected_H2_Edges = Edges_HYPO2_final.rows();

            //> =========== CLUSTERING H2 EDGES ===========
            EdgeClusterer edge_cluster_engine(Num_Of_Epipolar_Corrected_H2_Edges, Edges_HYPO2_final, H1_edge_idx);
            Eigen::MatrixXd HYPO2_idx = edge_cluster_engine.performClustering( HYPO2_idx_raw, Edges_HYPO2, edgels_HYPO2_corrected );

            Edges_HYPO2_final = edge_cluster_engine.Epip_Correct_H2_Edges;
            local_hypo2_clusters[thread_id].push_back(edge_cluster_engine.H2_Clusters);
            //> =========== CLUSTERING H2 EDGES ===========

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
                // double thresh_ore31_1 = OreListBardegree31(0,0);
                // double thresh_ore31_2 = OreListBardegree31(0,1);
                std::pair<double, double> epip_angle_range_from_H1_edge_to_V1 = std::make_pair(OreListBardegree31(0,0), OreListBardegree31(0,1));
                Eigen::Vector3d epipole1 = result1.second;

                Eigen::MatrixXd vali_idx31 = PairHypo->getHYPO2_idx_Ore(OreListdegree31, epip_angle_range_from_H1_edge_to_V1);
                Eigen::MatrixXd edgels_31  = PairHypo->getedgels_HYPO2_Ore(TO_Edges_VALID, OreListdegree31, epip_angle_range_from_H1_edge_to_V1);

                //> An array indicating if the two epipolar edges are "almost" parallel (if so, discard the edge pair as too much uncertainty is given)
                Eigen::VectorXd isparallel = Eigen::VectorXd::Ones(Edges_HYPO2_final.rows());

                //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Third loop: loop over each edge from Hypo2 <<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>//
                for (int idx_pair = 0; idx_pair < Edges_HYPO2_final.rows(); idx_pair++) {

                    //> Epipolar angle range of the epipolar wedge on the validation view arising from Edges_HYPO2_final
                    // double thresh_ore32_1 = OreListBardegree32(idx_pair,0);
                    // double thresh_ore32_2 = OreListBardegree32(idx_pair,1);
                    std::pair<double, double> epip_angle_range_from_H2_edge_to_V1 = std::make_pair(OreListBardegree32(idx_pair,0), OreListBardegree32(idx_pair,1));
                    Eigen::Vector3d epipole2 = result2.second;
                    
                    //> Find the epipolar angle range of the epipolar wedge on the validation view arising from Edges_HYPO2_final, and parse the corresponding edgels on the validation view
                    Eigen::MatrixXd vali_idx32 = PairHypo->getHYPO2_idx_Ore(OreListdegree32, epip_angle_range_from_H2_edge_to_V1);
                    Eigen::MatrixXd edgels_32  = PairHypo->getedgels_HYPO2_Ore(TO_Edges_VALID, OreListdegree32, epip_angle_range_from_H2_edge_to_V1);
                    
                    //> Check if the two epipolar wedges are almost parallel
                    if ( is_Epipolar_Wedges_in_Parallel( epip_angle_range_from_H1_edge_to_V1, epip_angle_range_from_H2_edge_to_V1, idx_pair, isparallel, supported_indice_current ) )
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

                    //> Get the supporting edge index from this validation view
                    supported_indice_current.row(idx_pair) << supported_link_indx;

                    //> If the current hypothesis edge pair is supported by the validation view
                    if (supported_link_indx != -2) {
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
            if (valid_pairs.empty()) continue;
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
        }
        //> A critical session to stack all local supported indices
        #pragma omp critical
        all_supported_indices.insert(all_supported_indices.end(), local_thread_supported_indices.begin(), local_thread_supported_indices.end());
    } //> end of #pragma omp parallel

    //> Merge thread-local maps into a single final map (sequentially)
    for (const auto& map_per_thread : local_hypo2_clusters) {
        for (const auto& thread_map : map_per_thread) {
            for (const auto& pair : thread_map) {
                //> Merge elements
                // hypo2_clusters_CH[pair.first] = pair.second;
                hypo2_clusters[pair.first] = pair.second;
            }
        }
    }

    // if (hypo2_clusters_CH == hypo2_clusters) { LOG_INFOR_MESG("YAH! The hypo2 clusteres are equal!"); }
    // else { LOG_INFOR_MESG("SADLY! The hypo2 clusteres are NOT equal!"); }

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

        //> CH: WHY IS HERE ANOTHER EPIPOLAR CORRECTION?????
        // Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);
        Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2_epipolar_correction(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);
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
                // /////////////////////////////////// epipolar correcting validation view edges ///////////////////////////////////
                Eigen::RowVectorXd R_vector = Eigen::Map<Eigen::RowVectorXd>(All_R[val_idx].data(), All_R[val_idx].size());
                Eigen::MatrixXd edgel_VALID = All_Edgels[val_idx].row(support_idx);
                // Eigen::Matrix3d Rot_HYPO1_val       = All_R[hyp01_view_indx];
                // Eigen::Matrix3d Rot_HYPO3       = All_R[val_idx];
                // Eigen::Vector3d Transl_HYPO1_val    = All_T[hyp01_view_indx];
                // Eigen::Vector3d Transl_HYPO3    = All_T[val_idx];
                // Eigen::Matrix3d R31;
                // Eigen::Vector3d T31;
                // Eigen::Matrix3d R13;
                // Eigen::Vector3d T13;

                // if (Use_Multiple_K) {
                //     K_HYPO1 = All_K[hyp01_view_indx];
                //     K_HYPO2 = All_K[hyp02_view_indx];
                // }
                // else {
                //     K_HYPO1 = K;
                //     K_HYPO2 = K;
                // }
                // util->getRelativePoses(Rot_HYPO1_val, Transl_HYPO1_val, Rot_HYPO3, Transl_HYPO3, R31, T31, R13, T13);
                // Eigen::Matrix3d F31 = util->getFundamentalMatrix(K_HYPO1.inverse(), K_HYPO2.inverse(), R31, T31); 
                // Eigen::Matrix3d F13 = util->getFundamentalMatrix(K_HYPO2.inverse(), K_HYPO1.inverse(), R13, T13);

                // Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct(edgel_VALID, edgel_HYPO1, F31, F13, HYPO2_idx_raw);
                // Eigen::MatrixXd Edges_VAL_final(corrected_validation_edge.rows(), 4);
                // Edges_VAL_final << corrected_validation_edge.col(4), corrected_validation_edge.col(5), corrected_validation_edge.col(6), corrected_validation_edge.col(7);
                // if (corrected_validation_edge.rows() == 0) {
                //     continue;
                // }

                // if (corrected_validation_edge.rows() > 0) {
                    // Eigen::Vector2d pt_VAL = Edges_VAL_final.row(0);
                    // paired_edges_locations_file << pt_VAL(0) << " " << pt_VAL(1) << " " << R_vector << " " << All_T[val_idx].transpose() << "\n";
                // }
                // /////////////////////////////////// epipolar correcting validation view edges ///////////////////////////////////
                // Eigen::Vector2d supporting_edge = All_Edgels[val_idx].row(support_idx).head<2>();

                paired_edges_locations_file << edgel_VALID(0) << " " << edgel_VALID(1) << " " << R_vector << " " << All_T[val_idx].transpose() << "\n";
                val_count++;
            }

        }
        if (val_count < 4 || val_count_pre < 4) LOG_ERROR("Something's buggy here!");
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

        // Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);
        Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2_epipolar_correction(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);

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

        if (abs(pt_H1(0) - 504.003) <0.01 && abs(pt_H1(1) - 399.142) <0.01) {
            std::cout<<"matched pt_H2 is: "<<pt_H2.transpose()<<std::endl;
            std::cout<<"world 3D edge is: "<<edge_pt_3D_world.transpose()<<std::endl;
            std::cout<<"triangulated 3d edge is: "<< edge_pt_3D.transpose()<<std::endl;
        }

        Gamma1s.row(valid_pair_idx) << edge_pt_3D(0), edge_pt_3D(1), edge_pt_3D(2);

        // const double EPSILON = 1e-4;  // Increase tolerance from 1e-6 to 1e-4
        // if (std::abs(edge_pt_3D(0) + 0.520866) < EPSILON &&        // if (std::abs(edge_pt_3D(0) - 0.187817) < EPSILON &&
        //     std::abs(edge_pt_3D(1) - 0.00430003) < EPSILON &&        //     std::abs(edge_pt_3D(1) + 0.373794) < EPSILON &&
        //     std::abs(edge_pt_3D(2) + 4.38449) < EPSILON) {        //     std::abs(edge_pt_3D(2) + 3.68693) < EPSILON) {
        //     std::cout << "Matched tangents_3D: " << Gamma1s.row(valid_pair_idx) << std::endl;
        //     std::cout<<"hyp1 is: " <<pt_H1.transpose() <<std::endl;
        //     std::cout<<"hyp2 is: "<<pt_H2.transpose()<<", original hyp2 is: "<<edgel_HYPO2<<std::endl;
        // }
        ///////////////////// debug /////////////////////
        // const double EPSILON = 1e-4; 
        // if (std::abs(edge_pt_3D_world(0) - 0.254774) < EPSILON &&        // if (std::abs(point_world(0) - 0.411971) < EPSILON &&
        //     std::abs(edge_pt_3D_world(1) - 0.0508765) < EPSILON &&        //     std::abs(point_world(1) - 0.0771234) < EPSILON &&
        //     std::abs(edge_pt_3D_world(2) - 0.516086) < EPSILON) {        //     std::abs(point_world(2) - 0.323085) < EPSILON) {
        //         std::cout<<"hyp1 is: " <<pt_H1.transpose() <<std::endl;
        //         std::cout<<"hyp2 is: "<<pt_H2.transpose()<<", original hyp2 is: "<<edgel_HYPO2<<std::endl;
        // }

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

        ////////// include all cluster member in the edge mapping ///////////
        for (int member_idx : cluster_members) {
            if (member_idx < 0 || member_idx >= Edges_HYPO2.rows() || member_idx == hypo2_idx) {
                //std::cout << "  Member " << member_idx << ": Invalid index" << std::endl;
                continue;
            }

            Eigen::Vector3d member_location = Edges_HYPO2.row(member_idx).head<3>();

            edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D_world, tangents_3D_world, pt_H2, member_location, hyp02_view_indx, member_idx, All_R[hyp02_view_indx], All_T[hyp02_view_indx]);
        }
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
                // Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct_post_validation(edges_for_val_frame.row(support_idx), Edges_HYPO1_final, F31, F13, HYPO2_idx_raw);
                Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2_epipolar_correction(edges_for_val_frame.row(support_idx), Edges_HYPO1_final, F31, F13, HYPO2_idx_raw);
                

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

    // std::cout << "edge sketch size of edge_3D_to_supporting_edges: " << edgeMapping->edge_3D_to_supporting_edges.size() << std::endl;
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
    std::string Output_File_Path2 = "../../outputs/3D_edges_" + Post_File_Name_Str;
    std::cout << "Writing 3D edge locations to: " << Output_File_Path2 << std::endl;
    myfile2.open (Output_File_Path2);
    myfile2 << Gamma1s_world;
    myfile2.close();

    std::ofstream tangents_file;
    std::string tangents_output_path = "../../outputs/3D_tangents_" + Post_File_Name_Str;
    std::cout << "Writing 3D edge tangents to: " << tangents_output_path << std::endl;
    tangents_file.open(tangents_output_path);
    tangents_file << tangent3Ds;
    tangents_file.close();
#endif

    Eigen::MatrixXd mvt_3d_edges = NViewsTrian::mvt( hyp01_view_indx, hyp02_view_indx, Post_File_Name_Str );

    //> Concatenate reconstructed 3D edges
    if (all_3D_Edges.rows() == 0) {
        all_3D_Edges = Gamma1s_world;
        // all_3D_Edges = mvt_3d_edges;
    } 
    else {
        // all_3D_Edges.conservativeResize(all_3D_Edges.rows() + mvt_3d_edges.rows(), 3);
        // all_3D_Edges.bottomRows(mvt_3d_edges.rows()) = mvt_3d_edges;
        all_3D_Edges.conservativeResize(all_3D_Edges.rows() + Gamma1s_world.rows(), 3);
        all_3D_Edges.bottomRows(Gamma1s_world.rows()) = Gamma1s_world;
    }
}

// void EdgeSketch_Core::Calculate_Edge_Support_Ratios_And_Select_Next_Views(std::shared_ptr<EdgeMapping> edgeMapping) {

//     itime = omp_get_wtime();
//     Load_Data->read_All_Edgels(All_Edgels, Edge_Detection_Final_Thresh);
    
//     // For each view, track which edges are supporting any 3D edge
//     std::vector<std::set<int>> supportedEdgesIndices(Num_Of_Total_Imgs);
    
//     // Iterate through all 3D edges and their supporting 2D edges
//     for (auto it = edgeMapping->edge_3D_to_supporting_edges.begin(); 
//         it != edgeMapping->edge_3D_to_supporting_edges.end(); ++it) {
//         const auto& supportList = it->second;
        
//         // Add all supporting edges to their respective view sets
//         for (const auto& supportData : supportList) {
//             int imageIndex = supportData.image_number;
//             int edgeIndex = supportData.edge_idx;
//             supportedEdgesIndices[imageIndex].insert(edgeIndex);
//         }
//     }
    

//     std::vector<double> ratios;
//     std::vector<std::pair<int, double>> frameSupportCounts;
    
//     for (int i = 0; i < Num_Of_Total_Imgs; i++) {
//         int totalEdges = All_Edgels[i].rows();
//         int supportedEdges = supportedEdgesIndices[i].size();
        
//         double ratio = static_cast<double>(supportedEdges) / totalEdges;
//         ratios.push_back(ratio);
//         frameSupportCounts.push_back({i, ratio});
        
//         // std::cout << "View " << i << ": " << supportedEdges << "/" << totalEdges << " edges supported (" << (ratio * 100) << "%)" << std::endl;
//     }
    
//     avg_ratio = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
//     std::cout << "Average ratio of claimed edges: " << avg_ratio << std::endl;
    
//     // Select next best hypothesis views
//     // Sort views by their claim ratio (ascending)
//     std::sort(frameSupportCounts.begin(), frameSupportCounts.end(), 
//               [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
//                   return a.second < b.second;
//               });
    
//     // Select the two views with the lowest claim ratio
//     int bestView1 = frameSupportCounts[0].first;
//     int bestView2 = frameSupportCounts[1].first;
    
//     // Avoid reusing views that have been used before
//     int keep_finding_counter = 0;
//     while (true) {
//         int find_existence_HYPO1 = std::count(history_hypothesis_views_index.begin(), 
//                                               history_hypothesis_views_index.end(), bestView1);
//         int find_existence_HYPO2 = std::count(history_hypothesis_views_index.begin(), 
//                                               history_hypothesis_views_index.end(), bestView2);
        
//         if (find_existence_HYPO1 == 0 && find_existence_HYPO2 == 0)
//             break;
        
//         if (find_existence_HYPO1 > 0 && find_existence_HYPO2 > 0) {
//             bestView1 = frameSupportCounts[2 + keep_finding_counter].first;
//             bestView2 = frameSupportCounts[3 + keep_finding_counter].first;
//         } else if (find_existence_HYPO1 > 0) {
//             bestView1 = frameSupportCounts[1 + keep_finding_counter].first;
//             bestView2 = frameSupportCounts[2 + keep_finding_counter].first;
//         } else if (find_existence_HYPO2 > 0) {
//             bestView2 = frameSupportCounts[2 + keep_finding_counter].first;
//         }
        
//         keep_finding_counter++;
        
//         // Prevent infinite loop if we can't find unused views
//         if (keep_finding_counter > frameSupportCounts.size() - 2) {
//             std::cout << "Warning: Unable to find unused views. Reusing some previous views." << std::endl;
//             break;
//         }
//     }
    
//     // Assign the selected views
//     hyp01_view_indx = bestView1;
//     hyp02_view_indx = bestView2;
    
//     std::cout << "Selected next hypothesis views: " << bestView1 << " and " << bestView2 << std::endl;
    
//     // Check if we've reached the stopping ratio
//     if (avg_ratio > Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges) {
//         enable_aborting_3D_edge_sketch = true;
//         std::cout << "Reached Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges value" << std::endl;
//     } else {
//         enable_aborting_3D_edge_sketch = false;
//     }
    
//     find_next_hypothesis_view_time += omp_get_wtime() - itime;
// }

void EdgeSketch_Core::Project_3D_Edges_and_Find_Next_Hypothesis_Views() {

    //> First read all edges with the final-run threshold (TODO: is this step necessary?)
    itime = omp_get_wtime();
    Load_Data->read_All_Edgels( All_Edgels, Edge_Detection_Final_Thresh );

    //> Loop over all views
    for (int i = 0; i < Num_Of_Total_Imgs; i++) {

        //> Project the 3D edges to each view indexed by i
        Eigen::MatrixXd projectedEdges = project3DEdgesToView(all_3D_Edges, All_R[i], All_T[i], K, All_R[hyp01_view_indx], All_T[hyp01_view_indx]);

        //> Claim the projected edges by the observed edges
        int num_of_claimed_edges = claim_Projected_Edges(projectedEdges, All_Edgels[i], Reproj_Dist_Thresh);
        claimedEdgesList.push_back(num_of_claimed_edges);
    }

    //> Use the selectBestViews function to determine the two frames with the least claimed edges
    // std::pair<int, int> bestViews = selectBestViews(claimedEdgesList);
    std::pair<int, int> next_hypothesis_views;
    select_Next_Best_Hypothesis_Views( claimedEdgesList, All_Edgels, next_hypothesis_views, history_hypothesis_views_index );
    
    //> Assign the best views to the hypothesis indices
    hyp01_view_indx = next_hypothesis_views.first;
    hyp02_view_indx = next_hypothesis_views.second;

    //> Check if the claimed edges is over a ratio of total observed edges
    enable_aborting_3D_edge_sketch = (avg_ratio > Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges) ? (true) : (false);
    std::cout<<"average ratio of claimed edges: "<<avg_ratio<<std::endl;
    if (avg_ratio > Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges){
        enable_aborting_3D_edge_sketch = true;
        std::cout<<"reached Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges value"<<std::endl;
    }else{
        enable_aborting_3D_edge_sketch = false;
    }
    find_next_hypothesis_view_time += omp_get_wtime() - itime;
}



int EdgeSketch_Core::claim_Projected_Edges(const Eigen::MatrixXd& projectedEdges, const Eigen::MatrixXd& observedEdges, double threshold) {
    
    int num_of_claimed_observed_edges = 0;

    //> Loop over all observed edges
    for (int i = 0; i < observedEdges.rows(); ++i) {

        //> Loop over all projected edges
        for (int j = 0; j < projectedEdges.rows(); ++j) {

            //> Calculate the Euclidean distance
            double dist = (projectedEdges.row(j) - observedEdges.row(i).head<2>()).norm();

            //> If the projected edge and the observed edge has Euclidean distance less than the "threshold"
            if (dist < threshold) {
                num_of_claimed_observed_edges++;
                break;
            }
        }
    }

    return num_of_claimed_observed_edges;
}

Eigen::MatrixXd EdgeSketch_Core::project3DEdgesToView(const Eigen::MatrixXd& edges3D, const Eigen::Matrix3d& R, const Eigen::Vector3d& T, const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_hyp01, const Eigen::Vector3d& T_hpy01) {

    Eigen::MatrixXd edges2D(edges3D.rows(), 2);

    for (int i = 0; i < edges3D.rows(); ++i) {
        Eigen::Vector3d point3D = edges3D.row(i).transpose();
        Eigen::Vector3d point_camera = R * point3D + T;

        // Check if the Z value is zero to avoid division by zero
        if (point_camera(2) == 0) {
            std::cout << "Error: Point " << i << " is located at infinity (Z=0) after camera transformation.\n"<<std::endl;
            continue;  
        }
        
        Eigen::Vector3d point_image = K * point_camera;
        edges2D(i, 0) = point_image(0) / point_image(2);
        edges2D(i, 1) = point_image(1) / point_image(2);
    }
    
    return edges2D;
}

void EdgeSketch_Core::select_Next_Best_Hypothesis_Views( 
    const std::vector<int>& claimedEdges, 
    std::vector<Eigen::MatrixXd> All_Edgels, 
    std::pair<int, int>& next_hypothesis_views, 
    std::vector<int> history_hypothesis_views_index ) 
{
    std::vector<std::pair<int, double>> frameSupportCounts;
    std::vector<double> all_ratio_of_claimed_over_unclaimed;

    double ratio_claimed_over_unclaimed;
    for (int i = 0; i < claimedEdges.size(); i++) {
        ratio_claimed_over_unclaimed = (double)(claimedEdges[i]) / (double)(All_Edgels[i].rows());
        frameSupportCounts.push_back({i, ratio_claimed_over_unclaimed});
        all_ratio_of_claimed_over_unclaimed.push_back(ratio_claimed_over_unclaimed);
        //std::cout<<"image #"<<i<<" claimed edge ratio is: "<<ratio_claimed_over_unclaimed<<", total number of claimed edges: "<<(double)(claimedEdges[i])<<", total number of edgels: "<<(double)(All_Edgels[i].rows())<<std::endl;
    }

    std::sort(frameSupportCounts.begin(), frameSupportCounts.end(), 
            [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                return a.second < b.second;
            });

    int bestView1 = frameSupportCounts[0].first;
    int bestView2 = frameSupportCounts[1].first;

    //> Calculate the average ratio of claimed 2D edges over unclaimed 2D edges
    avg_ratio = std::accumulate(all_ratio_of_claimed_over_unclaimed.begin(), all_ratio_of_claimed_over_unclaimed.end(), 0.0) / (double)(all_ratio_of_claimed_over_unclaimed.size());

    //> If the selected hypothesis view for the next round is repeated based on the history data, pick another one instead
    int keep_finding_counter = 0;
    while (true) {
        int find_existence_HYPO1 = std::count(history_hypothesis_views_index.begin(), history_hypothesis_views_index.end(), bestView1);
        int find_existence_HYPO2 = std::count(history_hypothesis_views_index.begin(), history_hypothesis_views_index.end(), bestView2);

        if (find_existence_HYPO1 == 0 && find_existence_HYPO2 == 0)
            break;

        if (find_existence_HYPO1 > 0 && find_existence_HYPO2 > 0) {
            bestView1 = frameSupportCounts[2 + keep_finding_counter].first;
            bestView2 = frameSupportCounts[3 + keep_finding_counter].first;
        }
        if (find_existence_HYPO1 > 0) {
            bestView1 = frameSupportCounts[1 + keep_finding_counter].first;
            bestView2 = frameSupportCounts[2 + keep_finding_counter].first;
        }
        if (find_existence_HYPO2 > 0) {
            bestView2 = frameSupportCounts[2 + keep_finding_counter].first;
        }
        keep_finding_counter++;
    }
    
    next_hypothesis_views = std::make_pair(bestView1, bestView2);
}


void EdgeSketch_Core::Clear_Data() {
    all_supported_indices.clear();
    All_Edgels.clear();
    claimedEdgesList.clear();
    num_of_nonveridical_edge_pairs = 0;
}

EdgeSketch_Core::~EdgeSketch_Core() { }



#endif 