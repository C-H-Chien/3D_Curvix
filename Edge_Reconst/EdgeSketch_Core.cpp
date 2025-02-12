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
}

void EdgeSketch_Core::Read_Edgels_Data() {
    //> Read edgels detected at a specific threshold 
    Load_Data->read_All_Edgels( All_Edgels, thresh_EDG );
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
    //paired_edge         = Eigen::MatrixXd::Constant(Edges_HYPO1.rows(), Num_Of_Total_Imgs, -2);
    paired_edge         = Eigen::MatrixXd::Constant(Edges_HYPO1.rows()*(Num_Of_Total_Imgs), Num_Of_Total_Imgs, -2);

    //> Compute epipolar wedge angles between HYPO1 and HYPO2 and valid angle range in HYPO1 for fast indexing from edges of HYPO2
    OreListdegree       = getOre->getOreList(hyp01_view_indx, hyp02_view_indx, Edges_HYPO2, All_R, All_T, K_HYPO1, K_HYPO2);

    auto result = getOre->getOreListBar(Edges_HYPO1, All_R, All_T, K_HYPO1, K_HYPO2, hyp02_view_indx, hyp01_view_indx);
    OreListBardegree = result.first;
    epipole = result.second;
}

void EdgeSketch_Core::Run_3D_Edge_Sketch(int hypothesis) {

    itime = omp_get_wtime();
    #pragma omp parallel
    {
        //> Local array stacking all supported indices
        std::vector<Eigen::MatrixXd> local_thread_supported_indices;
        int edge_idx;

        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< First loop: loop over all edgels from hypothesis view 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
        //<<<<<<<<<<< Identify pairs of edge, correct the positions of the edges from Hypo2, and store the paired edges >>>>>>>>>>>>>>>>//
        #pragma omp for schedule(static, Num_Of_OMP_Threads)
        for (edge_idx = 0; edge_idx < Edges_HYPO1.rows() ; edge_idx++) {

            if ( Skip_this_Edge( edge_idx ) ) continue;

            ///////////////////////////////////////////////// incremental method ////////////////////////////////////////////////////
            // bool skip_edge = false;
            // for (const auto& paired_edge_matrix : paired_edge_final_all) {
            //     for (int row = 0; row < paired_edge_matrix.rows(); ++row) {
            //         if (paired_edge_matrix(row, hyp01_view_indx) == edge_idx) { 
            //             //std::cout<<"hyp1 skip"<<std::endl;
            //             skip_edge = true;
            //             continue;
            //         }
            //     }
            //     if (skip_edge) break; 
            // }
            // if (skip_edge) continue;
            ///////////////////////////////////////////////// incremental method ////////////////////////////////////////////////////
            
            //> TODO: Summarize the following piece of code into get HYPO1 edgel and the corresponding HYPO2 edgels
            
            //> Get the current edge from HYPO1
            Eigen::Vector3d pt_edgel_HYPO1;
            pt_edgel_HYPO1 << Edges_HYPO1(edge_idx,0), Edges_HYPO1(edge_idx,1), 1;

            //Get angle Thresholds from OreListBar (in Degree) 
            double thresh_ore21_1 = OreListBardegree(edge_idx, 0);
            double thresh_ore21_2 = OreListBardegree(edge_idx, 1);
            Eigen::Vector3d corresponding_epipole = epipole;

            //> Find the corresponding edgel in HYPO2 based on the epipolar angle
            Eigen::MatrixXd HYPO2_idx_raw = PairHypo->getHYPO2_idx_Ore(OreListdegree, thresh_ore21_1, thresh_ore21_2);
            if (HYPO2_idx_raw.rows() == 0) continue;
            //> Retrieve Hypo2 Edgels
            Eigen::MatrixXd edgels_HYPO2 = PairHypo->getedgels_HYPO2_Ore(Edges_HYPO2, OreListdegree, thresh_ore21_1, thresh_ore21_2);
            //> Correct Edgels in Hypo2 Based on Epipolar Constraints
            Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgels_HYPO2, Edges_HYPO1.row(edge_idx), F21, F12, HYPO2_idx_raw);

            // if (abs(Edges_HYPO1(edge_idx,0)-520.6008)<0.001  && abs(Edges_HYPO1(edge_idx,1)-428.9978)<0.001){
            //     // std::cout<<HYPO2_idx_raw<<std::endl;
            //     // std::cout<<edgels_HYPO2<<std::endl;
            //     // std::cout<<edgels_HYPO2_corrected<<std::endl;
            //     std::cout<<thresh_ore21_1<<std::endl;
            //     std::cout<<thresh_ore21_2<<std::endl;
            //     std::cout<<corresponding_epipole<<std::endl;
            // }
            
            //> Organize the final edge data (hypothesis edge pairs)
            Eigen::MatrixXd Edges_HYPO1_final(edgels_HYPO2_corrected.rows(), 4);
            Edges_HYPO1_final << edgels_HYPO2_corrected.col(0), edgels_HYPO2_corrected.col(1), edgels_HYPO2_corrected.col(2), edgels_HYPO2_corrected.col(3);
            Eigen::MatrixXd Edges_HYPO2_final(edgels_HYPO2_corrected.rows(), 4);
            Edges_HYPO2_final << edgels_HYPO2_corrected.col(4), edgels_HYPO2_corrected.col(5), edgels_HYPO2_corrected.col(6), edgels_HYPO2_corrected.col(7);

            //> Store the Hypo2 Indices
            Eigen::MatrixXd HYPO2_idx(edgels_HYPO2_corrected.rows(), 1); 
            HYPO2_idx << edgels_HYPO2_corrected.col(8);
            if (HYPO2_idx.rows() == 0) continue;

            // ///////////////////////////////////////////////// incremental method ////////////////////////////////////////////////////
            // for (int i = 0; i < HYPO2_idx.rows(); i ++){
            //     int idx_hypo2 = HYPO2_idx(i);
            //     for (const auto& paired_edge_matrix : paired_edge_final_all) {
            //         for (int row = 0; row < paired_edge_matrix.rows(); ++row) {
            //             if (paired_edge_matrix(row, hyp02_view_indx) == idx_hypo2) { 
            //                 //std::cout<<"hyp2 skip"<<std::endl;
            //                 skip_edge = true;
            //                 continue;
            //             }
            //         }
            //         if (skip_edge) break; 
            //     }
            //     if (skip_edge) break; 
            // }
            // if (skip_edge) continue;
            ///////////////////////////////////////////////// incremental method ////////////////////////////////////////////////////
            
            int supported_edge_idx = 0;
            int stack_idx = 0;
            Eigen::MatrixXd supported_indices;
            supported_indices.conservativeResize(edgels_HYPO2.rows(), Num_Of_Total_Imgs-2);
            Eigen::MatrixXd supported_indice_current;
            supported_indice_current.conservativeResize(edgels_HYPO2.rows(),1);
            Eigen::MatrixXd supported_indices_stack;

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
                Eigen::MatrixXd edge_tgt_gamma3    = getReprojEdgel->getGamma3Tgt(hyp01_view_indx, hyp02_view_indx, Edges_HYPO1_final, Edges_HYPO2_final, All_R, All_T, VALID_INDX, K_HYPO1, K_HYPO2);
                
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

                    ////////////////////////////////////////////// Record Wedge ////////////////////////////////////////////
                    if (abs(Edges_HYPO1_final(idx_pair,0)-274.394)<0.001  && abs(Edges_HYPO1_final(idx_pair,1)-405.998)<0.001 &&	
                    abs(Edges_HYPO2_final(idx_pair,0)-284.514)<0.001  && abs(Edges_HYPO2_final(idx_pair,1)-418.369)<0.001 &&
                    !v_intersection.empty()){
                        //std::cout<<"hyp 1 edge_idx: "<<edge_idx<<std::endl;
                        std::cout<<"found support in valid# "<<VALID_INDX<<std::endl;
                        // std::ofstream epipole_file;
                        // std::string Epipole_File_Path = "../../outputs/epipole_and_angle_range_val_" + std::to_string(VALID_INDX) + ".txt";
                        // epipole_file.open(Epipole_File_Path);

                        // // Write to the file
                        // epipole_file << "Validation View Index: " << VALID_INDX << "\n";
                        // epipole_file << "Epipole 1: " << epipole1.transpose() << "\n";
                        // epipole_file << "Angle Range Hypothesis 1: [" << thresh_ore31_1 << ", " << thresh_ore31_2 << "]\n";
                        // epipole_file << "Epipole 2: " << epipole2.transpose() << "\n";
                        // epipole_file << "Angle Range Hypothesis 2: [" << thresh_ore32_1 << ", " << thresh_ore32_2 << "]\n";
                        // epipole_file << "-------------------------\n";
                        // epipole_file.close();
                    }
                    ////////////////////////////////////////////// Record Wedge ////////////////////////////////////////////

                    
                    //> Calculate orientation of gamma 3 (the reprojected edge)
                    Eigen::Vector2d edgels_tgt_reproj = {edge_tgt_gamma3(idx_pair,0), edge_tgt_gamma3(idx_pair,1)};
                    //> Get support from validation view for this gamma 3
                    double supported_link_indx = getSupport->getSupportIdx(edgels_tgt_reproj, Tangents_VALID, inliner);


                    //> Get the supporting edge idx from this validation view
                    supported_indice_current.row(idx_pair) << supported_link_indx;
                    if (supported_link_indx != -2) {
                        supported_indices_stack.conservativeResize(stack_idx+1,2);
                        supported_indices_stack.row(stack_idx) << double(idx_pair), double(supported_link_indx);
                        isempty_link = false;
                        stack_idx++;
                    }
                }
                supported_indices.col(supported_edge_idx) << supported_indice_current.col(0);
                supported_edge_idx++;
            } 
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  End of second loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//

            //> Now, for each local thread, stack the supported indices
            local_thread_supported_indices.push_back(supported_indices);

            //> Check for Empty Supported Indices
            if (isempty_link) continue;

            //> Create a Stack of Supported Indices
            std::vector<double> indices_stack(supported_indices_stack.data(), supported_indices_stack.data() + supported_indices_stack.rows());
            std::vector<double> indices_stack_unique = indices_stack;
            std::sort(indices_stack_unique.begin(), indices_stack_unique.end());
            std::vector<double>::iterator it1;
            it1 = std::unique(indices_stack_unique.begin(), indices_stack_unique.end());
            indices_stack_unique.resize( std::distance(indices_stack_unique.begin(),it1) );

            //> Count the Occurrence of Each Unique Index
            Eigen::VectorXd rep_count;
            rep_count.conservativeResize(indices_stack_unique.size(),1);

            for (int unique_idx = 0; unique_idx<indices_stack_unique.size(); unique_idx++) {
                rep_count.row(unique_idx) << int(count(indices_stack.begin(), indices_stack.end(), indices_stack_unique[unique_idx]));
            }

            //> Find the maximal number of supports from validation views and check if this number is over the threshold
            Eigen::VectorXd::Index maxIndex;
            double max_support = rep_count.maxCoeff(&maxIndex);
            if( max_support < Max_Num_Of_Support_Views ){
                continue;
            }

            int finalpair = -2;
            int numofmax = std::count(rep_count.data(), rep_count.data()+rep_count.size(), max_support);
            // Find all edges within the distance threshold
            std::vector<int> valid_indices;
            Eigen::Vector3d coeffs;
            coeffs = F21 * pt_edgel_HYPO1;
            Eigen::MatrixXd Edge_Pts;

            for(int a = 0; a < rep_count.rows(); a++){
                if(rep_count(a) < Max_Num_Of_Support_Views){
                    continue;
                }
                
                Eigen::Vector2d Edge_Pt;
                Edge_Pt << Edges_HYPO2_final.row(indices_stack_unique[a])(0), Edges_HYPO2_final.row(indices_stack_unique[a])(1);
                
                double Ap = coeffs(0) * Edge_Pt(0);
                double Bp = coeffs(1) * Edge_Pt(1);
                double numDist = Ap + Bp + coeffs(2);
                double denomDist = coeffs(0)*coeffs(0) + coeffs(1)*coeffs(1);
                denomDist = sqrt(denomDist);
                double dist = std::abs(numDist) / denomDist;

                if (dist > Reproj_Dist_Thresh) continue;
                valid_indices.push_back(int(indices_stack_unique[a]));
            }
            // If no edges are within the threshold, skip
            if (valid_indices.empty()) continue;
   

            for (int valid_idx : valid_indices) {
                paired_edge.row(edge_idx*Num_Of_Total_Imgs+valid_idx) << edge_idx, HYPO2_idx(valid_idx), supported_indices.row(valid_idx);

                if (std::abs(Edges_HYPO1(edge_idx,0)- 274.394) < 0.001 &&
                    std::abs(Edges_HYPO1(edge_idx,1) - 405.998) < 0.001 &&
                    std::abs(Edges_HYPO2(HYPO2_idx(valid_idx),0)- 284.506) < 0.001 &&
                    std::abs(Edges_HYPO2(HYPO2_idx(valid_idx),1) - 418.5) < 0.001 ) {
                    std::cout << "Supporting Validation View Indices for Hypothesis Edge (274.394, 405.998): ";
                    
                    // Iterate over the columns of supported_indices for this valid_idx
                    for (int col = 0; col < supported_indices.cols(); col++) {
                        int support_index = supported_indices(valid_idx, col);
                        if (support_index != -2) {  // Ignore invalid values
                            std::cout << col<< " ";
                        }
                    }
                    std::cout << std::endl;
                }
            }
     
            if (numofmax > 1) {
                std::vector<double> rep_count_vec(rep_count.data(), rep_count.data() + rep_count.rows());
                //std::cout<< "here"<<std::endl;
                std::vector<int> max_index;
                auto start_it = begin(rep_count_vec);
                while (start_it != end(rep_count_vec)) {
                    start_it = std::find(start_it, end(rep_count_vec), max_support);
                    if (start_it != end(rep_count_vec)) {
                        auto const pos = std::distance(begin(rep_count_vec), start_it);
                        max_index.push_back(int(pos));
                        ++start_it;
                    }
                }

                //Select the Final Paired Edge
                Eigen::Vector3d coeffs;
                coeffs = F21 * pt_edgel_HYPO1;
                Eigen::MatrixXd Edge_Pts;
                Edge_Pts.conservativeResize(max_index.size(),2);
                for(int maxidx = 0; maxidx<max_index.size(); maxidx++){
                    //std::cout<<indices_stack_unique[max_index[maxidx]]<<std::endl;
                    Edge_Pts.row(maxidx) << Edges_HYPO2_final(indices_stack_unique[max_index[maxidx]], 0), \
                                            Edges_HYPO2_final(indices_stack_unique[max_index[maxidx]], 1);
                }
                Eigen::VectorXd Ap = coeffs(0)*Edge_Pts.col(0);
                Eigen::VectorXd Bp = coeffs(1)*Edge_Pts.col(1);
                Eigen::VectorXd numDist = Ap + Bp + Eigen::VectorXd::Ones(Ap.rows())*coeffs(2);
                double denomDist = coeffs(0)*coeffs(0) + coeffs(1)*coeffs(1);
                denomDist = sqrt(denomDist);
                Eigen::VectorXd dist = numDist.cwiseAbs()/denomDist;
                Eigen::VectorXd::Index   minIndex;
                double min_dist = dist.minCoeff(&minIndex);
                //> ignore if the distance from the reprojected edge to the epipolar line is greater than some threshold
                if (min_dist > Reproj_Dist_Thresh) continue;

                finalpair = int(indices_stack_unique[max_index[minIndex]]);
            }
            else {
                finalpair = int(indices_stack_unique[int(maxIndex)]);
            }

            if(hypothesis == 1){
                hypothesis1_best_match[edge_idx] = HYPO2_idx(finalpair);
            }else{
                hypothesis2_best_match[edge_idx] = HYPO2_idx(finalpair);
            }

            //> paired_edge is a row vector continaing (hypo1 edge index), (paired hypo2 edge index), (paired validation edge indices)
            //paired_edge.row(edge_idx*Num_Of_Total_Imgs +Num_Of_Total_Imgs) << edge_idx, HYPO2_idx(finalpair), supported_indices.row(finalpair);

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




void EdgeSketch_Core::Finalize_Edge_Pairs_and_Reconstruct_3D_Edges() {

    //std::unordered_map<int, int> mutualMatches = saveBestMatchesToFile(hypothesis1_best_match, hypothesis2_best_match, "../../outputs/best_matches.txt");

    itime = omp_get_wtime();
    int pair_num = 0;
    std::vector<int> valid_pair_index;
    for (int pair_idx = 0; pair_idx < paired_edge.rows(); pair_idx++) {
      if(paired_edge(pair_idx,0) != -2 && paired_edge(pair_idx,0) != -3) {
        valid_pair_index.push_back(pair_idx);
        pair_num++;
      }
    }
    paired_edge_final = Eigen::MatrixXd::Constant(pair_num, Num_Of_Total_Imgs, -2);
    for (int i = 0; i < pair_num; i++){
      paired_edge_final.row(i) << paired_edge.row(valid_pair_index[i]);
      int edge_idx = paired_edge_final(i,0);

    //   if (std::abs(Edges_HYPO1(edge_idx,0) - 274.394) < 0.001 &&
    //         std::abs(Edges_HYPO1(edge_idx, 1) - 405.998) < 0.001) {
    //         std::cout << "Validation Indices for Hypothesis 1 Edge (274.394, 405.998): ";
    //         for (int col = 2; col < paired_edge_final.cols(); col++) {
    //             int val_index = paired_edge_final(i, col);
    //             if (val_index != -2) { // Ignore invalid values
    //                 std::cout << col << " ";
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    }

    std::string info_str = "Number of pairs is: " + std::to_string(pair_num);
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

        // Loop through validation views and write actual edge locations and R, T matrices
        int val_indx_in_paired_edge_array = 2;
        for (int val_idx = 0; val_idx < Num_Of_Total_Imgs; ++val_idx) {
            if (val_idx == hyp01_view_indx || val_idx == hyp02_view_indx) {
                continue;  // Skip hypothesis views
            }

            int support_idx = paired_edge_final(pair_idx, val_indx_in_paired_edge_array);
            if (support_idx != -2) {
                if (abs(pt_H1(0) - 274.394) <0.001 && abs(pt_H1(1) - 405.998) <0.001 && abs(pt_H2(0) - 284.514) <0.001 && abs(pt_H2(1) - 418.369) <0.001) {
                    std::cout<<"Finalize_Edge_Pairs_and_Reconstruct_3D_Edges validation index: "<<val_idx<<std::endl;
                }
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

                Eigen::MatrixXd corrected_validation_edge = PairHypo->edgelsHYPO2correct_post_validation(edgel_VALID, edgel_HYPO1, F31, F13, HYPO2_idx_raw);
                Eigen::MatrixXd Edges_VAL_final(corrected_validation_edge.rows(), 4);
                Edges_VAL_final << corrected_validation_edge.col(4), corrected_validation_edge.col(5), corrected_validation_edge.col(6), corrected_validation_edge.col(7);
                if (corrected_validation_edge.rows() == 0) {
                    continue;
                }
                if (corrected_validation_edge.rows() > 0) {
                    Eigen::Vector2d pt_VAL = Edges_VAL_final.row(0);
                    paired_edges_locations_file << pt_VAL(0) << " " << pt_VAL(1) << " " << R_vector << " " << All_T[val_idx].transpose() << "\n";
                    // if (abs(pt_H1(0) - 274.394) <0.001 && abs(pt_H1(1) - 405.998) <0.001 &&
                    //     abs(pt_H2(0) - 284.514) <0.001 && abs(pt_H2(1) - 418.369) <0.001) {
                    //     std::cout << "validation view: " << val_idx << " point: "<<pt_VAL(0)<<","<<pt_VAL(1)<<" original point: "<<edgel_VALID(0)<<","<<edgel_VALID(1) << std::endl;
                    // }
                }
                /////////////////////////////////// epipolar correcting validation view edges ///////////////////////////////////

                //Eigen::Vector2d supporting_edge = All_Edgels[val_idx].row(support_idx).head<2>();
                
                // if (abs(edge_hypo1(0)-519.863)<0.001  && abs(edge_hypo1(1)-399.004)<0.001 && abs(pt_H2(0)-517.955)<0.001  && abs(pt_H2(1)- 384.822)<0.001){
                //     std::cout<<"valid # "<<val_idx<<": "<<supporting_edge(0) << " " << supporting_edge(1) <<" " << supporting_edge(2) <<std::endl;
                //     std::cout<<"hyp1 orientation: "<< edgel_HYPO1(0,2) << std::endl;
                //     std::cout<<"hyp2 orientation: "<< edgel_HYPO2(0,2) <<std::endl;
                // }
            }
            val_indx_in_paired_edge_array++;
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

    // int num_mutual_matches = 0;
    // for (int row = 0; row < paired_edge_final.rows(); ++row) {
    //     auto pair = paired_edge_final.row(row);

    //     int hypo1_idx = pair(0);
    //     int hypo2_idx = pair(1);

    //     // Check if it's a mutual match
    //     auto it = mutualMatches.find(hypo1_idx);
    //     if (it != mutualMatches.end() && it->second == hypo2_idx) {
    //         num_mutual_matches++;
    //     }
    // }

    // // Resize Gamma1s and tangent3Ds based on the number of mutual matches
    // Gamma1s.resize(num_mutual_matches, 3);
    // tangent3Ds.resize(num_mutual_matches, 3);


    int count = 0;
    int valid_pair_idx = 0;

    // std::ofstream hypo2_coords_before("/gpfs/data/bkimia/zqiwu/3D/3D_Edge_Sketch_and_Grouping/outputs/hypo2_coords_before_correction.txt");
    // std::ofstream hypo2_coords_after("/gpfs/data/bkimia/zqiwu/3D/3D_Edge_Sketch_and_Grouping/outputs/hypo2_coords_after_correction.txt");

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

        //Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation_post_validation(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);
        Eigen::MatrixXd edgels_HYPO2_corrected = PairHypo->edgelsHYPO2correct_post_validation(edgel_HYPO2, edgel_HYPO1, F21, F12, HYPO2_idx_raw);

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

        Gamma1s.row(valid_pair_idx) << edge_pt_3D(0), edge_pt_3D(1), edge_pt_3D(2);
        //Gamma1s.row(valid_pair_idx) << edge_pt_3D(0), edge_pt_3D(1), edge_pt_3D(2), hypo1_identifier;
        // const double EPSILON = 1e-3;  // Increase tolerance from 1e-6 to 1e-4
        // if (std::abs(edge_pt_3D(0) - 0.48281) < EPSILON &&
        //     std::abs(edge_pt_3D(1) + 0.0250759) < EPSILON &&
        //     std::abs(edge_pt_3D(2) + 4.288) < EPSILON) {
        //     std::cout << "Matched tangents_3D: " << Gamma1s.row(valid_pair_idx) << std::endl;
        //     std::cout<<"hyp1 is: " <<pt_H1 <<" hyp2 is: "<<pt_H2<<std::endl;
        // }

        Eigen::MatrixXd tangents_3D = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Edgel_View1(edgel_HYPO1(0),  edgel_HYPO1(1), edgel_HYPO1(2));
        Eigen::Vector3d Edgel_View2(edgel_HYPO2(0),  edgel_HYPO2(1), edgel_HYPO2(2));
        Eigen::Matrix3d R1_test = All_R[hyp01_view_indx];
        Eigen::Matrix3d R21_test;
        Eigen::Matrix3d R12_test;
        Eigen::Vector3d T12_test;
        Eigen::Vector3d T21_test;   

        util->getRelativePoses(All_R[hyp01_view_indx], All_T[hyp01_view_indx], All_R[hyp02_view_indx], All_T[hyp02_view_indx], R21_test, T21_test, R12_test, T12_test);

        Compute_3D_Tangents(Edgel_View1, Edgel_View2, K_HYPO1, K_HYPO2, R21_test, T21_test, tangents_3D);
        Eigen::Vector3d tangents_3D_world = (All_R[hyp01_view_indx]).transpose() * tangents_3D;
        tangent3Ds.row(valid_pair_idx) = tangents_3D_world;
        valid_pair_idx++;
        
        edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D, pt_H1, hyp01_view_indx);
        edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D, pt_H2, hyp02_view_indx);

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

                //> Qiwu: Store validation view and the supporting edge
                validation_support_edges.emplace_back(val_idx, supporting_edge);
                //> Qiwu: Add the supporting edge to the edgeMapping for the 3D edge
                edgeMapping->add3DToSupportingEdgesMapping(edge_pt_3D, supporting_edge, val_idx);

                
            }
            val_indx_in_paired_edge_array++;
        }
    }
    finalize_edge_pair_time += omp_get_wtime() - itime;
}


void EdgeSketch_Core::test_3D_tangent() {
    // Given Rotation and Translation Matrices
    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d T1, T2;

    R1 << 0.229714, 0.973258, -1.29728e-07,
         -0.220415, 0.0520239, 0.974018,
         -0.947971, 0.223746, -0.226472;

    T1 << -0.601486, -0.402814, -3.93395;

    R2 << 0.159322, -0.987227, -3.64782e-08,
         -0.0530916, -0.00856814, 0.998553,
          0.985798, 0.159092, 0.0537785;

    T2 << 0.413953, -0.468447, -3.95085;

    // Camera Intrinsic Matrix (example, adjust if needed)
    Eigen::Matrix3d K;
    K << 2584.93250981950, 0, 249.771375872214,
         0, 2584.79186060577, 278.312679379194,
         0, 0, 1;

    // Point coordinates from two images
    std::vector<Eigen::Vector2d> points_img1 = {
        {520.590800, 428.498217}, {520.581695, 427.998471}, {520.573102, 427.498616},
        {520.564490, 426.998731}, {520.555647, 426.498837}, {520.645733, 430.996517},
        {520.634568, 430.496660}, {520.623192, 429.996899}, {520.611786, 429.497305},
        {520.600795, 428.997801}};
    std::vector<Eigen::Vector2d> points_img2 = {
        {240.855540, 429.000920}, {240.849805, 428.001160}, {240.839928, 427.001388},
        {240.839928, 427.001388}, {240.834057, 426.501440}, {240.885916, 432.001500},
        {240.872839, 431.001501}, {240.866802, 430.501425}, {240.861729, 430.001200},
        {240.855540, 429.000920}};

    // Relative Rotation and Translation
    Eigen::Matrix3d R21 = R2 * R1.transpose();
    Eigen::Vector3d T21 = T2 - R21 * T1;

    // Store 3D points and tangents
    std::vector<Eigen::Vector3d> points_3D;
    std::vector<Eigen::Vector3d> tangents_3D;

    for (size_t i = 0; i < points_img1.size(); ++i) {
        // Prepare the corresponding points
        Eigen::Vector2d pt1 = points_img1[i];
        Eigen::Vector2d pt2 = points_img2[i];
        Eigen::Vector3d pt1_h(pt1(0), pt1(1), 1.0);
        Eigen::Vector3d pt2_h(pt2(0), pt2(1), 1.0);

        // Triangulate the 3D point
        std::vector<Eigen::Vector2d> pts_test = {pt1, pt2};
        std::vector<Eigen::Matrix3d> Rs_test = {R21};
        std::vector<Eigen::Vector3d> Ts_test = {T21};

        Eigen::Vector3d edge_pt_3D = util->linearTriangulation(2, pts_test, Rs_test, Ts_test, K);
        Eigen::Vector3d edge_pt_3D_world = util->transformToWorldCoordinates(edge_pt_3D, R1, T1);

        points_3D.push_back(edge_pt_3D_world);

        // Compute the 3D tangent
        Eigen::MatrixXd tangent_3D;
        Compute_3D_Tangents(pt1_h, pt2_h, K, K, R21, T21, tangent_3D);
        Eigen::Vector3d tangent_3D_world = R1.transpose() * tangent_3D;

        tangents_3D.push_back(tangent_3D_world);

        // Print results
        std::cout << "Point " << i + 1 << ": (" << edge_pt_3D_world(0) << ", " << edge_pt_3D_world(1) << ", " << edge_pt_3D_world(2) << ")" << std::endl;
        std::cout << "Tangent " << i + 1 << ": (" << tangent_3D_world(0) << ", " << tangent_3D_world(1) << ", " << tangent_3D_world(2) << ")" << std::endl;
    }

    std::cout << "Triangulation and tangent computation completed for all points." << std::endl;
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
        // const double EPSILON = 1e-4;  // Increase tolerance from 1e-6 to 1e-4
        // if (std::abs(point_world(0) - 0.501451) < EPSILON &&
        //     std::abs(point_world(1) - 0.995733) < EPSILON &&
        //     std::abs(point_world(2) - 0.448107) < EPSILON) {
        //     std::cout << "Matched tangents_3D_world: " << point_world.transpose() << std::endl;
        //     std::cout << "Corresponding tangents_3D: " << point_camera.transpose() << std::endl;
        // }
    }
    // Eigen::Matrix3d R_ref = All_R[hyp01_view_indx];
    // Eigen::Vector3d T_ref = All_T[hyp01_view_indx];

    // Eigen::MatrixXd Gamma1s_world(Gamma1s.rows(), 4);
    // for (int i = 0; i < Gamma1s.rows(); ++i) {
    //     Eigen::Vector3d point_camera = Gamma1s.row(i).head<3>().transpose();
    //     Eigen::Vector3d point_world = util->transformToWorldCoordinates(point_camera, R_ref, T_ref);
    //     Gamma1s_world.row(i).head<3>() = point_world.transpose();
    //     Gamma1s_world(i, 3) = Gamma1s(i, 3);
    // }


#if WRITE_3D_EDGES
    std::ofstream myfile2;
    std::string Output_File_Path2 = "../../outputs/3D_edges_" + Dataset_Name + "_" + Scene_Name + "_hypo1_" + std::to_string(hyp01_view_indx) \
                                    + "_hypo2_" + std::to_string(hyp02_view_indx) + "_t" + std::to_string(Edge_Detection_Init_Thresh) + "to" \
                                    + std::to_string(Edge_Detection_Final_Thresh) + "_delta" + Delta_FileName_Str + "_theta" + std::to_string(Orien_Thresh) \
                                    + "_N" + std::to_string(Max_Num_Of_Support_Views) + ".txt";
    std::cout << Output_File_Path2 << std::endl;
    myfile2.open (Output_File_Path2);
    myfile2 << Gamma1s_world;
    myfile2.close();

    std::ofstream tangents_file;
    std::string tangents_output_path = "../../outputs/3D_tangents_" + Dataset_Name + "_" + Scene_Name + "_hypo1_" + std::to_string(hyp01_view_indx) \
                                       + "_hypo2_" + std::to_string(hyp02_view_indx) + "_t" + std::to_string(Edge_Detection_Init_Thresh) + "to" \
                                       + std::to_string(Edge_Detection_Final_Thresh) + "_delta" + Delta_FileName_Str + "_theta" + std::to_string(Orien_Thresh) \
                                       + "_N" + std::to_string(Max_Num_Of_Support_Views) + ".txt";
    std::cout << "Writing 3D tangents to: " << tangents_output_path << std::endl;
    tangents_file.open(tangents_output_path);
    tangents_file << tangent3Ds;
    tangents_file.close();
#endif

    //> Concatenate reconstructed 3D edges
    if (all_3D_Edges.rows() == 0) {
        all_3D_Edges = Gamma1s_world;
    } 
    else {
        all_3D_Edges.conservativeResize(all_3D_Edges.rows() + Gamma1s_world.rows(), 3);
        all_3D_Edges.bottomRows(Gamma1s_world.rows()) = Gamma1s_world;
    }
}



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
    select_Next_Best_Hypothesis_Views( claimedEdgesList, All_Edgels, next_hypothesis_views, least_ratio );
    
    //> Assign the best views to the hypothesis indices
    hyp01_view_indx = next_hypothesis_views.first;
    hyp02_view_indx = next_hypothesis_views.second;

    //> Check if the claimed edges is over a ratio of total observed edges
    enable_aborting_3D_edge_sketch = (least_ratio > Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges) ? (true) : (false);
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
        //Eigen::Vector3d world_point3D = R_hyp01.transpose() * (point3D - T_hpy01);
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
    double& least_ratio) 
{
    std::vector<std::pair<int, double>> frameSupportCounts;

    double ratio_claimed_over_unclaimed;
    for (int i = 0; i < claimedEdges.size(); i++) {
        ratio_claimed_over_unclaimed = (double)(claimedEdges[i]) / (double)(All_Edgels[i].rows());
        frameSupportCounts.push_back({i, ratio_claimed_over_unclaimed});
    }

    std::sort(frameSupportCounts.begin(), frameSupportCounts.end(), 
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second < b.second;
              });

    int bestView1 = frameSupportCounts[0].first;
    int bestView2 = frameSupportCounts[1].first;

    // If the new best views are the same as hyp1 and hyp2, randomly select two new views
    if ((bestView1 == hyp01_view_indx && bestView2 == hyp02_view_indx) || (bestView1 == hyp02_view_indx && bestView2 == hyp01_view_indx)) {
        std::vector<int> availableViews;
        for (int i = 0; i < claimedEdges.size(); i++) {
            if (i != hyp01_view_indx && i != hyp02_view_indx) {
                availableViews.push_back(i);
            }
        }

        // Randomly select two unique views from the available views
        if (availableViews.size() >= 2) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, availableViews.size() - 1);

            int randomIndex1 = dis(gen);
            int randomIndex2;
            do {
                randomIndex2 = dis(gen);
            } while (randomIndex2 == randomIndex1);

            bestView1 = availableViews[randomIndex1];
            bestView2 = availableViews[randomIndex2];
        }
    }

    next_hypothesis_views = std::make_pair(bestView1, bestView2);
    least_ratio = frameSupportCounts[0].second;
}


void EdgeSketch_Core::Clear_Data() {
    all_supported_indices.clear();
    All_Edgels.clear();
    claimedEdgesList.clear();
    num_of_nonveridical_edge_pairs = 0;
}

EdgeSketch_Core::~EdgeSketch_Core() { }


void EdgeSketch_Core::Compute_3D_Tangents(
    const Eigen::MatrixXd& pt_edge_view1,
    const Eigen::MatrixXd& pt_edge_view2,
    const Eigen::Matrix3d& K1,
    const Eigen::Matrix3d& K2,
    const Eigen::Matrix3d& R21,
    const Eigen::Vector3d& T21,
    Eigen::MatrixXd& tangents_3D)
{
    tangents_3D.resize(1, 3);

    Eigen::Vector3d e1  = {1,0,0};
    Eigen::Vector3d e3  = {0,0,1};

    // Normalize edge points
    Eigen::Vector3d Gamma1 = K1.inverse() * Eigen::Vector3d(pt_edge_view1(0), pt_edge_view1(1), 1.0);
    Eigen::Vector3d Gamma2 = K2.inverse() * Eigen::Vector3d(pt_edge_view2(0), pt_edge_view2(1), 1.0);

    Eigen::Vector3d tgt1(cos(pt_edge_view1(2)), sin(pt_edge_view1(2)), 0.0);
    Eigen::Vector3d tgt2(cos(pt_edge_view2(2)), sin(pt_edge_view2(2)), 0.0);
    Eigen::Vector3d tgt1_meters = K1.inverse() * tgt1;
    Eigen::Vector3d tgt2_meters = K2.inverse() * tgt2;

    double rho1 = (double(e1.transpose() * T21) - double(e3.transpose() * T21) * double(e1.transpose() *Gamma2))/(double(e3.transpose() * R21 * Gamma1)* double(e1.transpose() * Gamma2) - double(e1.transpose() * R21 * Gamma1));

    Eigen::Vector3d n1 = tgt1_meters.cross(Gamma1);
    Eigen::Vector3d n2 = R21.transpose() * tgt2_meters.cross(Gamma2);

    Eigen::Vector3d T3D = n1.cross(n2) / (n1.cross(n2) ).norm();
    tangents_3D = T3D;
}



#endif
