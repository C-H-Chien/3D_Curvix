// =============================================================================================================================
//> This N-view triangulation code is built on top of the publicly released certifiable solver for multiple views triangulation
//  https://github.com/C-H-Chien/FastNViewTriangulation arises from the paper:
//  Garcia-Salguero, Mercedes, and Javier Gonzalez-Jimenez. "Certifiable solver for real-time N-view triangulation." 
//  IEEE Robotics and Automation Letters 8, no. 4 (2023): 1999-2005.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =============================================================================================================================
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <malloc.h>
#include <random>

////////////// 3D RECONSTRUCTION /////////////////
#include <iterator> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <numeric>
#include <set>
////////////// 3D RECONSTRUCTION /////////////////

#include "NViewsTypes.h"
#include "NViewsUtils.h"
#include "NViewsClass.h"
#include "definitions.h"
#include "generatePointCloud.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>  
#include "edge_mapping.hpp"

//> Macros
#define VERBOSE (false)

namespace NViewsTrian{

//> structured data for feature tracks
struct Feature_Track {

    unsigned Length;                                //> feature track length = number of features
    std::vector<Eigen::Vector3d> Locations;         //> feature point locations in pixels
    std::vector<Eigen::Matrix3d> Abs_Rots;          //> relative rotations
    std::vector<Eigen::Vector3d> Abs_Transls;       //> relative translations
    std::vector<Eigen::Vector3d> Optimal_Locations; //> corrected feature point locations in pixels
    bool is_sol_global_optimal;
    std::vector<double> Reprojection_Errors;
    Eigen::Vector3d Gamma;

}; //> End of struct Feature_Track



void Multiview_Triangulation( Feature_Track& feature_track_, Eigen::Matrix3d K ) {

    Eigen::Matrix3d inverse_K = K.inverse();
    
    //> Construct pairwise epipolar constraints
    int M_cameras = feature_track_.Length;
    Eigen::MatrixXd idx_matrix;
    int n_comb = generateM2Comb(M_cameras, idx_matrix);

    //> Construct pairwise relative poses and feature correspondences
    std::vector< PairObj > Feature_Track_Connections;
    for (int jj = 0; jj < n_comb; jj++) {
        PairObj feature_corr_i; 
        int id1 = idx_matrix(0, jj); 
        int id2 = idx_matrix(1, jj); 

        //> Compute relative poses
        Matrix3 R1 = feature_track_.Abs_Rots[id1];
        Matrix3 R2 = feature_track_.Abs_Rots[id2]; 
        Vector3 t1 = feature_track_.Abs_Transls[id1]; 
        Vector3 t2 = feature_track_.Abs_Transls[id2]; 
        Matrix3 Rrel = R2 * R1.transpose();
        Vector3 trel = t2 - R2*(R1.transpose())*t1;
        trel.normalize();
                
        //> Compute essential matrices
        Matrix3 Ess = Matrix3::Identity(); 
        Matrix3 Tx = Matrix3::Zero(); 
        Tx << 0, -trel(2), trel(1), trel(2), 0, -trel(0), -trel(1), trel(0), 0; 
        Ess = Tx * Rrel;

        //> Organize feature connection information
        feature_corr_i.id1 = id1; 
        feature_corr_i.id2 = id2; 
        feature_corr_i.F = Ess; 
        feature_corr_i.p1 = inverse_K * feature_track_.Locations[id1]; //str_out.obs[0].col(id1); 
        feature_corr_i.p2 = inverse_K * feature_track_.Locations[id2];
        Feature_Track_Connections.push_back(feature_corr_i);
    }
    //> Run N-view triangulation!
    NViewsClass corr_N_view; 
    //std::cout<<"mvt line 99"<<std::endl; 
    //> (1) Create constraint matrices
    corr_N_view.createProblemMatrices(Feature_Track_Connections, M_cameras);
    //std::cout<<"mvt line 101"<<std::endl; 
       
    //> (2) Run correction and check whether the corrected features are globally optimal solutions from N-view triangulation
    NViewsOptions options_corr; 
    options_corr.save_val_constr = false;
    options_corr.debug           = false; 
    options_corr.max_iters       = 50;
    NViewsResult Feature_Corrections = corr_N_view.correctObservations(options_corr);
    bool certified_global_optimum = corr_N_view.certified_global_optimum;
    feature_track_.is_sol_global_optimal = certified_global_optimum;
#if VERBOSE
    std::cout << "Is solution a global optimum? " << (certified_global_optimum ? std::string("Yes") : std::string("No")) << std::endl;
#endif


    //> (3) Make corrections to the observed feature track point locations
    std::vector<Matrix4> proj_s; 
    proj_s.clear();
    //std::cout << "Reserving " << M_cameras << " elements for proj_s" << std::endl;
    proj_s.reserve(M_cameras);
    //std::cout << "Reservation successful" << std::endl;

    
    std::vector<Vector3> Corrected_Features_in_Metric;
    std::vector<Vector3> Observed_Features_in_Metric;
        
    for (int jc=0; jc < M_cameras;jc++) {
        //> Projection matrix
        Matrix3 R = feature_track_.Abs_Rots[jc];  
        Vector3 t = feature_track_.Abs_Transls[jc]; 
        Matrix4 P1 = Matrix4::Identity(); 

        P1.block<3,3>(0,0) = R; 
        P1.block<3,1>(0,3) = t;

        proj_s.push_back(P1);
        // Eigen::Matrix4d identity_matrix = Eigen::Matrix4d::Identity();
        // proj_s.push_back(identity_matrix);
                
        //> update observation by the correction from N-view triangulation 
        Vector3 pt = inverse_K * feature_track_.Locations[jc];  
        Vector3 delta_ref; 
        delta_ref << Feature_Corrections.sol_final( jc*2), Feature_Corrections.sol_final(jc*2 + 1), 0;

        Corrected_Features_in_Metric.push_back( pt + delta_ref );
        Observed_Features_in_Metric.push_back( pt );
        feature_track_.Optimal_Locations.push_back( K * Corrected_Features_in_Metric[jc] ); 
    }

    //> (4) Triangulate corrected points to a common 3D point via linear triangulation and reproject to the images
    Vector3 Gamma_Corrected;            //> 3D corrected point under the world coordinate
    Eigen::VectorXd Depths_Corrected;   //> (?)
    double Linearity_Err = triangulateNPoint(proj_s, Corrected_Features_in_Metric, Gamma_Corrected, Depths_Corrected);
    std::vector<double> Reprojection_Errors = reproject_to_images( proj_s, feature_track_.Locations, K, Gamma_Corrected, false );
    feature_track_.Reprojection_Errors = Reprojection_Errors;
    feature_track_.Gamma = Gamma_Corrected;
}


// Function to load R and T matrices from files
void loadRTMatrices(const std::string& R_file_path, const std::string& T_file_path, 
                    std::vector<Eigen::Matrix3d>& R_matrices, std::vector<Eigen::Vector3d>& T_matrices) {
    std::ifstream R_file(R_file_path);
    std::ifstream T_file(T_file_path);

    if (!R_file.is_open() || !T_file.is_open()) {
        throw std::runtime_error("Failed to open R or T matrix file.");
    }

    // Load R matrices
    double value;
    while (R_file >> value) {
        Eigen::Matrix3d R;
        R(0, 0) = value;
        for (int i = 1; i < 9; ++i) {
            R_file >> value;
            R(i / 3, i % 3) = value;
        }
        R_matrices.push_back(R);
    }

    // Load T matrices
    while (T_file >> value) {
        Eigen::Vector3d T;
        T(0) = value;
        for (int i = 1; i < 3; ++i) {
            T_file >> value;
            T(i) = value;
        }
        T_matrices.push_back(T);
    }
}


/////////////////////////////////// 3D RECONSTRUCTION ///////////////////////////////////////////////////////
// Struct to hold edge points, rotation matrix, and translation vector
struct EdgeData {
    Eigen::Vector3d edgePoint;
    Eigen::Matrix3d rotationMatrix;
    Eigen::Vector3d translationVector;
};

std::vector<std::vector<EdgeData>> parseFile(const std::string& filePath) {
    std::vector<std::vector<EdgeData>> data;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return data;
    }

    std::string line;
    std::vector<EdgeData> currentPairData;

    while (std::getline(file, line)) {
        // Check for the start of a new pair
        if (line.find("Pair") != std::string::npos) {
            if (!currentPairData.empty()) {
                data.push_back(currentPairData);
                currentPairData.clear();
            }
            continue;
        }

        // Skip empty or invalid lines
        if (line.empty() || std::isspace(line[0])) {
            continue;
        }

        std::istringstream iss(line);
        std::vector<double> values;
        double value;

        // Read all values in the line
        while (iss >> value) {
            values.push_back(value);
        }

        // Check if the line has exactly 14 values
        if (values.size() != 14) {
            std::cerr << "Invalid line format: " << line << std::endl;
            continue;
        }

        // Extract data and create an EdgeData object
        EdgeData edgeData;
        edgeData.edgePoint = Eigen::Vector3d(values[0], values[1], 1);

        // Extract rotation matrix
        edgeData.rotationMatrix << values[2], values[5], values[8],
                                   values[3], values[6], values[9],
                                   values[4], values[7], values[10];

        // Extract translation vector
        edgeData.translationVector = Eigen::Vector3d(values[11], values[12], values[13]);

        currentPairData.push_back(edgeData);
    }

    // Push the last pair if any data is left
    if (!currentPairData.empty()) {
        data.push_back(currentPairData);
    }

    file.close();
    return data;
}


// Read tangent file
std::vector<Eigen::Vector3d> readTangentFile(const std::string& tangentFilePath) {
    std::vector<Eigen::Vector3d> tangents;
    std::ifstream tangentFile(tangentFilePath);
    if (!tangentFile.is_open()) {
        std::cerr << "Failed to open tangent file: " << tangentFilePath << std::endl;
        exit(-1);
    }

    double x, y, z;
    while (tangentFile >> x >> y >> z) {
        tangents.emplace_back(x, y, z);
    }
    tangentFile.close();
    return tangents;
}

//Write to tangent file
void writeTangentFile(const std::string& outputTangentFilePath, const std::vector<Eigen::Vector3d>& tangents) {
    std::ofstream tangentFile(outputTangentFilePath);
    if (!tangentFile.is_open()) {
        std::cerr << "Failed to open tangent output file: " << outputTangentFilePath << std::endl;
        exit(-1);
    }

    for (const auto& tangent : tangents) {
        tangentFile << tangent(0) << "\t" << tangent(1) << "\t" << tangent(2) << "\n";
    }
    tangentFile.flush();
    tangentFile.close();
}

/////////////////////////////////// 3D RECONSTRUCTION ///////////////////////////////////////////////////////


Eigen::MatrixXd mvt(int hyp1, int hyp2, std::string Scene_Name, int init_toed_thresh, int final_toed_thresh) {

    std::string basePath = "../../outputs/";
    std::string filePath = basePath + "paired_edges_final_" + std::to_string(hyp1) + "_" + std::to_string(hyp2) + ".txt";
    std::string outputFilePath = basePath + "triangulated_3D_edges_ABC-NEF_" + Scene_Name + "_hypo1_" + 
                                std::to_string(hyp1) + "_hypo2_" + std::to_string(hyp2) + 
                                "_t" + std::to_string(init_toed_thresh) + "to" + std::to_string(final_toed_thresh) 
                                + "_delta03_theta15.000000_N4.txt";
    std::string points3DFile = basePath + "3D_edges_ABC-NEF_" + Scene_Name + "_hypo1_" + 
                            std::to_string(hyp1) + "_hypo2_" + std::to_string(hyp2) + 
                            "_t" + std::to_string(init_toed_thresh) + "to" + std::to_string(final_toed_thresh) 
                            + "_delta03_theta15.000000_N4.txt";
    std::string tangentFilePath = basePath + "3D_tangents_ABC-NEF_" + Scene_Name + "_hypo1_" + 
                            std::to_string(hyp1) + "_hypo2_" + std::to_string(hyp2) + 
                            "_t" + std::to_string(init_toed_thresh) + "to" + std::to_string(final_toed_thresh) 
                            + "_delta03_theta15.000000_N4.txt";
    std::string outputTangentFilePath = "../../outputs/updated_tangents.txt"; 
    std::vector<Eigen::Vector3d> points3D;

    //////////////////// Read 3D points from the file //////////////////////
    std::ifstream points3DStream(points3DFile);
    if (!points3DStream.is_open()) {
        std::cerr << "[Error]: Failed to open file: " << points3DFile << std::endl;
        exit(0);
    }


    double x, y, z;
    while (points3DStream >> x >> y >> z) {
        points3D.emplace_back(x, y, z);
    }
    points3DStream.close();

    // Read tangent file
    std::vector<Eigen::Vector3d> tangents = readTangentFile(tangentFilePath);
    if (tangents.size() != points3D.size()) {
        std::cout << "[Error]: Mismatch between 3D points and tangent data size." << std::endl;
        exit(0);
    }

    std::ofstream outFile;
    outFile.open (outputFilePath);
    
    // Placeholder for updated tangent data
    std::vector<Eigen::Vector3d> updatedTangents;

    ////////////////////// Read 3D points from the file //////////////////////
    std::vector<std::vector<EdgeData>> allPairsData = parseFile(filePath);

    //> Intrinsic matrix
    double fx = 1111.11136542426;
    double fy = 1111.11136542426;
    double cx = 399.500000000000;
    double cy = 399.500000000000;
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    Eigen::MatrixXd GammaMatrix(3, allPairsData.size());  // Matrix to store all Gamma values
    int gammaIndex = 0;

    //std::cout<<"line 369"<<std::endl;

    // Output the parsed data for verification
    for (size_t i = 0; i < allPairsData.size(); ++i) {
        //> define input data to multiview triangulation
        Feature_Track feature_track_;
        feature_track_.Length = allPairsData[i].size();

        //std::cout<<"original length is: "<<feature_track_.Length<<std::endl;

        for (size_t j = 0; j < allPairsData[i].size(); ++j) {
            Eigen::Vector3d edge_point = allPairsData[i][j].edgePoint;
            Eigen::Matrix3d R = allPairsData[i][j].rotationMatrix;
            Eigen::Vector3d t = allPairsData[i][j].translationVector;

            feature_track_.Locations.push_back(edge_point);
            feature_track_.Abs_Rots.push_back(R);
            feature_track_.Abs_Transls.push_back(t);

            Eigen::Vector3d point3D = points3D[i];
            Eigen::Vector3d point_image = K * (R * point3D + t);// Project the 3D point to 2D
            Eigen::Vector2d edges2D(point_image(0) / point_image(2), point_image(1) / point_image(2));
            Eigen::Vector2d edges_2D_point(edge_point(0), edge_point(1));
            feature_track_.Reprojection_Errors.push_back(std::sqrt((edges2D - edges_2D_point).squaredNorm()));
        }

        // Step 1: Filter top 10% and errors > 3 pixels
        std::vector<size_t> indices(feature_track_.Reprojection_Errors.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return feature_track_.Reprojection_Errors[a] > feature_track_.Reprojection_Errors[b];
        });

        size_t removeCount = std::max(indices.size() / 10, size_t(1));
        std::set<size_t> toRemove(indices.begin(), indices.begin() + removeCount);

        for (size_t idx = 0; idx < feature_track_.Reprojection_Errors.size(); ++idx) {
            if (feature_track_.Reprojection_Errors[idx] > 3.0) {
                toRemove.insert(idx);
            }
        }

        Feature_Track filtered_track;
        filtered_track.Length = feature_track_.Length - toRemove.size();
        for (size_t idx = 0; idx < feature_track_.Length; ++idx) {
            if (toRemove.find(idx) == toRemove.end()) {
                filtered_track.Locations.push_back(feature_track_.Locations[idx]);
                filtered_track.Abs_Rots.push_back(feature_track_.Abs_Rots[idx]);
                filtered_track.Abs_Transls.push_back(feature_track_.Abs_Transls[idx]);
            }
            
        }


        Multiview_Triangulation(filtered_track, K);

        // Step 3: Recheck reprojection errors (Include all points)
        std::vector<double> finalReprojectionErrors(feature_track_.Length, 0.0);

        for (size_t idx = 0; idx < feature_track_.Length; ++idx) {
            Eigen::Vector3d point3D = filtered_track.Gamma; // Use Gamma from filtered_track
            Eigen::Vector3d projected = K * (feature_track_.Abs_Rots[idx] * point3D + feature_track_.Abs_Transls[idx]);
            Eigen::Vector2d reprojected(projected(0) / projected(2), projected(1) / projected(2));
            finalReprojectionErrors[idx] = std::sqrt((reprojected - feature_track_.Locations[idx].head<2>()).squaredNorm());
        }

        Feature_Track rechecked_track;
        for (size_t idx = 0; idx < feature_track_.Length; ++idx) {
            if (finalReprojectionErrors[idx] <= 1) {
                rechecked_track.Locations.push_back(feature_track_.Locations[idx]);
                rechecked_track.Abs_Rots.push_back(feature_track_.Abs_Rots[idx]);
                rechecked_track.Abs_Transls.push_back(feature_track_.Abs_Transls[idx]);
            }
        }
       // std::cout<<"reprojected length is: "<<rechecked_track.Length<<std::endl;

        // Update the track length
        rechecked_track.Length = rechecked_track.Locations.size();
        //std::cout<<rechecked_track.Length<<std::endl;
        if (rechecked_track.Length < 6){
            //std::cout<<"length<6"<<std::endl;
            continue;
        }

        // Perform MVT again on the rechecked data
        Multiview_Triangulation(rechecked_track, K);

        // Save the final result
        outFile << rechecked_track.Gamma(0) << "\t" << rechecked_track.Gamma(1) << "\t" << rechecked_track.Gamma(2) << "\n";
        //std::cout<<rechecked_track.Gamma(0) << "\t" << rechecked_track.Gamma(1) << "\t" << rechecked_track.Gamma(2) << std::endl;
        updatedTangents.push_back(tangents[i]);

        // Store Gamma values in matrix
        GammaMatrix(0, gammaIndex) = rechecked_track.Gamma(0);
        GammaMatrix(1, gammaIndex) = rechecked_track.Gamma(1);
        GammaMatrix(2, gammaIndex) = rechecked_track.Gamma(2);
        gammaIndex++;
    }

    outFile.close();
    writeTangentFile(outputTangentFilePath, updatedTangents);

    GammaMatrix.conservativeResize(3, gammaIndex);  // Resize to keep only valid columns

    return GammaMatrix.transpose();  // Return only the filled columns
}

void grouped_mvt(const std::vector<std::vector<EdgeMapping::SupportingEdgeData>>& all_groups, 
                 const std::string& outputFilePath, const std::string& tangentOutputFilePath) {

    //> Intrinsic matrix
    double fx = 1111.11136542426;
    double fy = 1111.11136542426;
    double cx = 399.500000000000;
    double cy = 399.500000000000;
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    std::ofstream outFile(outputFilePath);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFilePath << std::endl;
        exit(0);
    }
    std::cout << "[INFO] Saving grouped triangulated 3D edges to: " << outputFilePath << std::endl;

    std::ofstream tangentOutFile(tangentOutputFilePath);
    if (!tangentOutFile.is_open()) {
        std::cerr << "Failed to open tangent output file: " << tangentOutputFilePath << std::endl;
        exit(0);
    }
    std::cout << "[INFO] Saving 3D tangents to: " << tangentOutputFilePath << std::endl;

    int triangulated_count = 0;

    std::vector<std::vector<EdgeMapping::SupportingEdgeData>> local_groups = all_groups;

    for (size_t start_idx = 0; start_idx < all_groups.size(); start_idx += 1) {

        const auto& group = all_groups[start_idx];

        // if (group.size() < 6) {
        //     std::cout << "[WARNING]: Skipping group " << start_idx << " (not enough edges for triangulation)." << std::endl;
        //     continue;
        // }

        Feature_Track feature_track_;
        //feature_track_.Length = std::min(group.size(), static_cast<size_t>(50));

        // for (const auto& edge_data : group) {
        //     feature_track_.Locations.emplace_back(edge_data.edge_uncorrected(0), edge_data.edge_uncorrected(1), 1); // Convert 2D edge to homogeneous
        //     feature_track_.Abs_Rots.emplace_back(edge_data.rotation);
        //     feature_track_.Abs_Transls.emplace_back(edge_data.translation);
        // }
        Eigen::Matrix3d prev_R = Eigen::Matrix3d::Zero();
        Eigen::Vector3d prev_t = Eigen::Vector3d::Zero();
        bool first_edge = true;

        for (const auto& edge_data : group) {
            const Eigen::Matrix3d& R_new = edge_data.rotation;
            const Eigen::Vector3d& t_new = edge_data.translation;

            if (!first_edge &&
                (R_new - prev_R).norm() < 1e-8 &&
                (t_new - prev_t).norm() < 1e-8) {
                continue;  // Skip duplicate view
            }

            feature_track_.Locations.emplace_back(edge_data.edge_uncorrected(0), edge_data.edge_uncorrected(1), 1);
            feature_track_.Abs_Rots.emplace_back(R_new);
            feature_track_.Abs_Transls.emplace_back(t_new);

            prev_R = R_new;
            prev_t = t_new;
            first_edge = false;
        }

        feature_track_.Length = feature_track_.Locations.size();
        if (feature_track_.Length < 6) {
            std::cout << "[WARNING]: Skipping group " << start_idx << " (fewer than 6 unique views)." << std::endl;
            continue;
        }


        Multiview_Triangulation(feature_track_, K);

        /////////// Randomly select 2 2D edges for calculating 3D tangent ///////////
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, feature_track_.Locations.size() - 1);
        int idx1 = distrib(gen);
        int idx2 = distrib(gen);

        while (idx2 == idx1) {
            idx2 = distrib(gen);
        }

        Eigen::Vector3d p1 = feature_track_.Locations[idx1];
        Eigen::Vector3d p2 = feature_track_.Locations[idx2];
        Eigen::Matrix3d R1 = feature_track_.Abs_Rots[idx1];
        Eigen::Matrix3d R2 = feature_track_.Abs_Rots[idx2];
        Eigen::Vector3d T1 = feature_track_.Abs_Transls[idx1];
        Eigen::Vector3d T2 = feature_track_.Abs_Transls[idx2];
        Eigen::Matrix3d R21 = R2 * R1.transpose();
        Eigen::Vector3d T21 = T2 - R21 * T1;
        Eigen::MatrixXd tangent_3D;
        double orientation1 = group[idx1].edge_uncorrected(2);
        double orientation2 = group[idx2].edge_uncorrected(2);

        tangent_3D.resize(1, 3);
        Eigen::Vector3d e1  = {1,0,0};
        Eigen::Vector3d e3  = {0,0,1};
        // Normalize edge points
        Eigen::Vector3d Gamma1 = K.inverse() * Eigen::Vector3d(p1(0), p1(1), 1.0);
        Eigen::Vector3d Gamma2 = K.inverse() * Eigen::Vector3d(p2(0), p2(1), 1.0);
        Eigen::Vector3d tgt1(cos(orientation1), sin(orientation1), 0.0);
        Eigen::Vector3d tgt2(cos(orientation2), sin(orientation2), 0.0);
        Eigen::Vector3d tgt1_meters = K.inverse() * tgt1;
        Eigen::Vector3d tgt2_meters = K.inverse() * tgt2;
        double rho1 = (double(e1.transpose() * T21) - double(e3.transpose() * T21) * double(e1.transpose() *Gamma2))/(double(e3.transpose() * R21 * Gamma1)* double(e1.transpose() * Gamma2) - double(e1.transpose() * R21 * Gamma1));
        Eigen::Vector3d n1 = tgt1_meters.cross(Gamma1);
        Eigen::Vector3d n2 = R21.transpose() * tgt2_meters.cross(Gamma2);
        Eigen::Vector3d T3D = n1.cross(n2) / (n1.cross(n2) ).norm();
        tangent_3D = T3D;
        Eigen::Vector3d tangent_3D_world = R1.transpose() * tangent_3D;

        /////////// Write 3D tangent to file ///////////
        tangentOutFile << std::fixed << tangent_3D_world(0) << "\t" << tangent_3D_world(1) << "\t" << tangent_3D_world(2) << "\n";
        tangentOutFile.flush();
        /////////// End tangent computation and writing ///////////

        outFile << std::fixed << feature_track_.Gamma(0) << "\t" << feature_track_.Gamma(1) << "\t" << feature_track_.Gamma(2) << "\n";
        
        ////////// Check if the Gamma matches the target values //////////
        // Eigen::Vector3d targetGamma(0.70455, 0.122517, 0.357381);
        // if ((feature_track_.Gamma - targetGamma).norm() < 1e-4) {
        //     std::cout << "[MATCH FOUND]\n";
        //     std::cout << "Gamma: " << feature_track_.Gamma.transpose() << std::endl;

        //     for (size_t i = 0; i < feature_track_.Locations.size(); ++i) {
        //         std::cout << "Edge " << i << ":\n";
        //         std::cout << "  Location:       " << feature_track_.Locations[i].transpose() << "\n";
        //         //std::cout << "  Rotation:\n" << feature_track_.Abs_Rots[i] << "\n";
        //         std::cout << "  Translation:    " << feature_track_.Abs_Transls[i].transpose() << "\n";
        //     }
        // }
        ////////// Check if the Gamma matches the target values //////////


        triangulated_count++;
        outFile.flush();
    }

    outFile.close();
    tangentOutFile.close();
    std::cout << "[INFO] Saved " << triangulated_count << " triangulated 3D edges to " << outputFilePath << std::endl;
    std::cout << "[INFO] Saved 3D tangents to " << tangentOutputFilePath << std::endl;
}




}