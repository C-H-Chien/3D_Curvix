#ifndef MVT_HPP
#define MVT_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <set>

// Namespace for N-View Triangulation
namespace NViewsTrian {

// Struct for feature tracks
struct Feature_Track {
    unsigned Length;
    std::vector<Eigen::Vector3d> Locations;
    std::vector<Eigen::Matrix3d> Abs_Rots;
    std::vector<Eigen::Vector3d> Abs_Transls;
    std::vector<Eigen::Vector3d> Optimal_Locations;
    bool is_sol_global_optimal;
    std::vector<double> Reprojection_Errors;
    Eigen::Vector3d Gamma;
};

// Struct to hold edge points, rotation matrix, and translation vector
struct EdgeData {
    Eigen::Vector3d edgePoint;
    Eigen::Matrix3d rotationMatrix;
    Eigen::Vector3d translationVector;
};

// Function prototypes
void Multiview_Triangulation(Feature_Track& feature_track_, Eigen::Matrix3d K);
void loadRTMatrices(const std::string& R_file_path, const std::string& T_file_path, 
                     std::vector<Eigen::Matrix3d>& R_matrices, std::vector<Eigen::Vector3d>& T_matrices);
std::vector<std::vector<EdgeData>> parseFile(const std::string& filePath);
std::vector<Eigen::Vector3d> readTangentFile(const std::string& tangentFilePath);
void writeTangentFile(const std::string& outputTangentFilePath, const std::vector<Eigen::Vector3d>& tangents);
Eigen::MatrixXd mvt(int hyp1, int hyp2, std::string Scene_Name);
void grouped_mvt(const std::vector<std::vector<EdgeMapping::SupportingEdgeData>>& all_groups, const std::string& outputFilePath, const std::string& tangentOutputFilePath);

} // namespace NViewsTrian

#endif // MVT_HPP
