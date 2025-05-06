#ifndef EDGESKETCH_CORE_HPP
#define EDGESKETCH_CORE_HPP
// =============================================================================
//
// ==============================================================================
#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

//> Eigen library
#include <Eigen/Core>
#include <Eigen/Dense>

//> YAML file data reader
#include <yaml-cpp/yaml.h>

//> shared class pointers
#include "file_reader.hpp"
#include "util.hpp"
#include "PairEdgeHypo.hpp"
#include "getReprojectedEdgel.hpp"
#include "getSupportedEdgels.hpp"
#include "getOrientationList.hpp"
#include "edge_mapping.hpp"
#include "../Edge_Reconst/mvt.hpp"

//> mutual best support
#include <unordered_map>
#include <utility>

    
class EdgeSketch_Core {

public:
    std::shared_ptr<EdgeMapping> edgeMapping = nullptr;
    std::vector<Eigen::MatrixXd> paired_edge_final_all;

    // For each edge index, store all other edge indices in the same cluster
    std::unordered_map<int, std::vector<int>> hypo2_clusters;

    //> Constructor
    EdgeSketch_Core( YAML::Node );
    void Read_Camera_Data();
    void Read_Edgels_Data();
    void Set_Hypothesis_Views_Camera();
    void Set_Hypothesis_Views_Edgels();
    void Run_3D_Edge_Sketch();
    void Finalize_Edge_Pairs_and_Reconstruct_3D_Edges(std::shared_ptr<EdgeMapping> edgeMapping);
    void Clear_Data();
    void Stack_3D_Edges();
    void Project_3D_Edges_and_Find_Next_Hypothesis_Views();

    std::unordered_map<int, int> saveBestMatchesToFile(const std::unordered_map<int, int>& hypothesis1ToBestMatch,
                           const std::unordered_map<int, int>& hypothesis2ToBestMatch,
                           const std::string& filename);

    // Function to get all edges in the same cluster as a given edge
    std::vector<int> get_edges_in_same_cluster(int edge_index);
    // Function to reset clusters for a new hypo1-hypo2 iteration
    void reset_hypo2_clusters();

    //> Destructor
    ~EdgeSketch_Core();
    
    bool Skip_this_Edge( const int edge_idx ) {
      //> Edge Boundary Check: not too close to boundary
      if ( Edges_HYPO1(edge_idx,0) < 10 || Edges_HYPO1(edge_idx,0) > Img_Cols-10 || Edges_HYPO1(edge_idx,1) < 10 || Edges_HYPO1(edge_idx,1) > Img_Rows-10)
        return true;
      
      //> Paired Edge Check: not yet been paired
      if ( paired_edge(edge_idx,0) != -2 )
        return true;
      return false;
    }

    bool is_Epipolar_Wedges_in_Parallel(double thresh_ore31_1, double thresh_ore31_2, double thresh_ore32_1, double thresh_ore32_2, int idx_pair, Eigen::VectorXd &isparallel, Eigen::MatrixXd &supported_indice_current) {
      Eigen::MatrixXd anglediff(4,1);
      anglediff << fabs(thresh_ore31_1 - thresh_ore32_1), fabs(thresh_ore31_1 - thresh_ore32_2), \
                   fabs(thresh_ore31_2 - thresh_ore32_1), fabs(thresh_ore31_2 - thresh_ore32_2);
      if (anglediff.maxCoeff() <= Parallel_Epipolar_Line_Angle_Deg) {
          isparallel.row(idx_pair) << 0;
          supported_indice_current.row(idx_pair) << -2;
          return true;
      }
      else {
        return false;
      }
    }

    Eigen::MatrixXd paired_edge_final;
    double edge_sketch_time;
    int thresh_EDG;
    int hyp01_view_indx;
    int hyp02_view_indx;
    int Edge_Detection_Init_Thresh;
    int Edge_Detection_Final_Thresh;
    int Max_3D_Edge_Sketch_Passes;

    //> Edges and camera intrinsix/extrinsic matrices of all images
    std::vector<Eigen::Matrix3d> All_R;
    std::vector<Eigen::Vector3d> All_T;
    std::vector<Eigen::Matrix3d> All_K;
    std::vector<Eigen::MatrixXd> All_Edgels; 
    Eigen::Matrix3d K;

    //> Input Dataset Settings
    std::string Dataset_Path;
    std::string Dataset_Name;
    std::string Scene_Name;
    int Num_Of_Total_Imgs;
    int Img_Rows;
    int Img_Cols;
    bool Use_Multiple_K;
    double fx;
    double fy;
    double cx;
    double cy;
    std::string Delta_FileName_Str;

    std::vector< Eigen::MatrixXd > all_supported_indices;
    Eigen::MatrixXd Gamma1s;
    Eigen::MatrixXd tangent3Ds;
    
    Eigen::MatrixXd all_3D_Edges;
    std::vector< int > claimedEdgesList;
    double avg_ratio;
    bool enable_aborting_3D_edge_sketch;
    int num_of_nonveridical_edge_pairs;
    std::vector<int> history_hypothesis_views_index;

    //> timer
    double itime, pair_edges_time;
    double finalize_edge_pair_time;
    double find_next_hypothesis_view_time;
    
private:
    //> sharing the classes
    std::shared_ptr<file_reader> Load_Data = nullptr;
    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    std::shared_ptr<PairEdgeHypothesis::pair_edge_hypothesis> PairHypo = nullptr;
    std::shared_ptr<GetReprojectedEdgel::get_Reprojected_Edgel> getReprojEdgel = nullptr;
    std::shared_ptr<GetSupportedEdgels::get_SupportedEdgels> getSupport = nullptr;
    std::shared_ptr<GetOrientationList::get_OrientationList> getOre = nullptr;
    //std::shared_ptr<EdgeMapping> edgeMapping = nullptr;

    Eigen::MatrixXd project3DEdgesToView(const Eigen::MatrixXd& edges3D, const Eigen::Matrix3d& R, const Eigen::Vector3d& T, const Eigen::Matrix3d& K, const Eigen::Matrix3d& R_hyp01, const Eigen::Vector3d& T_hpy01);

    
    int claim_Projected_Edges(const Eigen::MatrixXd& projectedEdges, const Eigen::MatrixXd& observedEdges, double threshold);
    void select_Next_Best_Hypothesis_Views( 
      const std::vector< int >& claimedEdges, std::vector<Eigen::MatrixXd> All_Edgels,
      std::pair<int, int> &next_hypothesis_views, std::vector<int> history_hypothesis_views_index
    );

    bool get_H2_edge_indices_passing_dist2EL_thresh(
      std::vector<int> &H2_edge_indices_passing_dist2EL_thresh, std::vector<double> indices_stack_unique,
      Eigen::Vector3d pt_edgel_HYPO1, Eigen::VectorXd rep_count, Eigen::MatrixXd Edges_HYPO2_final ) 
    {
      Eigen::Vector3d coeffs = F21 * pt_edgel_HYPO1;

      //> Loop over all hypothesis edge pairs, pick the pairs where the H2 edge is close to the epipolar line arising from the H1 edge
      for(int a = 0; a < rep_count.rows(); a++) {
          //> Ignore if the current hypothesis edge pair has validation view supports less than Max_Num_Of_Support_Views
          if (rep_count(a) < Max_Num_Of_Support_Views) continue;
          
          Eigen::Vector2d Edge_Pt;
          Edge_Pt << Edges_HYPO2_final.row(indices_stack_unique[a])(0), Edges_HYPO2_final.row(indices_stack_unique[a])(1);
          
          double Ap = coeffs(0) * Edge_Pt(0);
          double Bp = coeffs(1) * Edge_Pt(1);
          double numDist = Ap + Bp + coeffs(2);
          double denomDist = coeffs(0)*coeffs(0) + coeffs(1)*coeffs(1);
          denomDist = sqrt(denomDist);
          double dist = std::abs(numDist) / denomDist;

          if (dist > Reproj_Dist_Thresh) continue;
          H2_edge_indices_passing_dist2EL_thresh.push_back(int(indices_stack_unique[a]));
      }
      //> If no H2 edges are within the point-to-epipolar line distance threshold, go to the next H1 edge
      return (H2_edge_indices_passing_dist2EL_thresh.empty()) ? (true) : (false);
    }

    //> generate a list of validation view indices
    void get_validation_view_index_list() {
      //> reset the vector
      valid_view_index.clear();
      for (int VALID_INDX = 0; VALID_INDX < Num_Of_Total_Imgs; VALID_INDX++) {
        if (VALID_INDX == hyp01_view_indx || VALID_INDX == hyp02_view_indx) continue;
        valid_view_index.push_back(VALID_INDX);
      }
    }
    

    //> YAML file data parser
    YAML::Node Edge_Sketch_Setting_YAML_File;

    //> Input 3D Edge Sketch Settings
    int Num_Of_OMP_Threads;
    double Edge_Loc_Pertubation;
    double Orien_Thresh;
    int Max_Num_Of_Support_Views;
    double Parallel_Epipolar_Line_Angle_Deg;
    double Reproj_Dist_Thresh;
    double Stop_3D_Edge_Sketch_by_Ratio_Of_Claimed_Edges;
    int circleR; //> Unknown setting


    //> Edges and camera intrinsix/extrinsic matrices of the two hypothesis views
    Eigen::Matrix3d K_HYPO1;
    Eigen::Matrix3d K_HYPO2;
    Eigen::MatrixXd Edges_HYPO1;
    Eigen::MatrixXd Edges_HYPO2;
    Eigen::Matrix3d Rot_HYPO1;
    Eigen::Matrix3d Rot_HYPO2;
    Eigen::Vector3d Transl_HYPO1;
    Eigen::Vector3d Transl_HYPO2;
    //> Relative poses and fundmental matrices
    Eigen::Matrix3d R21;
    Eigen::Vector3d T21;
    Eigen::Matrix3d F21;
    Eigen::Matrix3d R12;
    Eigen::Vector3d T12;  
    Eigen::Matrix3d F12;

    Eigen::MatrixXd paired_edge;
    Eigen::MatrixXd OreListdegree;
    Eigen::MatrixXd OreListBardegree;
    Eigen::Vector3d epipole;

    // data structure for tracking best matches
    std::unordered_map<int, int> hypothesis1_best_match;
    std::unordered_map<int, int> hypothesis2_best_match;

    //> a list of validation view indices
    std::vector<int> valid_view_index;
};




#endif