#ifndef EDGE_MAPPING_HPP
#define EDGE_MAPPING_HPP

#include <Eigen/Core>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>

// Hash function for Eigen::Vector3d to use in unordered_map
struct HashEigenVector3d {
    std::size_t operator()(const Eigen::Vector3d& vec) const {
        std::size_t seed = 0;
        for (int i = 0; i < vec.size(); ++i) {
            seed ^= std::hash<double>()(vec[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Hash function for Eigen::Vector2d to use in unordered_map
struct HashEigenVector2d {
    std::size_t operator()(const Eigen::Vector2d& vec) const {
        std::size_t h1 = std::hash<double>()(vec(0));
        std::size_t h2 = std::hash<double>()(vec(1));
        return h1 ^ (h2 << 1); // Combine hashes
    }
};

class EdgeMapping {
public:
    struct SupportingEdgeData {  
        Eigen::Vector2d edge;
        int image_number;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;

        bool operator<(const SupportingEdgeData& other) const {
            if (image_number != other.image_number) return image_number < other.image_number;
            if (edge.x() != other.edge.x()) return edge.x() < other.edge.x();
            return edge.y() < other.edge.y();
        }
    };

    void add3DToSupportingEdgesMapping(const Eigen::Vector3d &edge_3D, 
                                       const Eigen::Vector2d &supporting_edge, 
                                       int image_number,
                                       const Eigen::Matrix3d &rotation,
                                       const Eigen::Vector3d &translation);

    void add3DToFrameMapping(const Eigen::Vector3d& edge_3D, 
                             const Eigen::Vector2d& supporting_edge, 
                             int frame);

    void printFirst10Edges();
    // void write_edge_linking_to_file();

    std::vector<std::vector<SupportingEdgeData>> findMergable2DEdgeGroups();

    std::unordered_map<Eigen::Vector3d, std::vector<SupportingEdgeData>, HashEigenVector3d> edge_3D_to_supporting_edges;
    std::unordered_map<int, std::unordered_map<Eigen::Vector2d, std::vector<Eigen::Vector3d>, HashEigenVector2d>> frame_to_edge_to_3D_map;
};

#endif  // EDGE_MAPPING_HPP
