#ifndef EDGE_MAPPING_HPP
#define EDGE_MAPPING_HPP

#include <Eigen/Core>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

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


struct FuzzyVector3dEqual {
    bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
        return (a - b).norm() < 1e-6;
    }
};

struct FuzzyVector3dPairEqual {
    bool operator()(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& a,
                    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& b) const {
        FuzzyVector3dEqual eq;
        return (eq(a.first, b.first) && eq(a.second, b.second)) ||
               (eq(a.first, b.second) && eq(a.second, b.first));
    }
};



/////////// For creating weighted map of 3D edges
struct HashEigenVector3dPair {
    std::size_t operator()(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& p) const {
        HashEigenVector3d hash_fn;
        return hash_fn(p.first) ^ (hash_fn(p.second) << 1);
    }
};



class EdgeMapping {
public:
    struct SupportingEdgeData {  
        Eigen::Vector2d edge;
        Eigen::Vector3d edge_uncorrected;
        int image_number;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        Eigen::Vector3d tangents_3D_world;

        bool operator<(const SupportingEdgeData& other) const {
            if (image_number != other.image_number) return image_number < other.image_number;
            if (edge.x() != other.edge.x()) return edge.x() < other.edge.x();
            return edge.y() < other.edge.y();
        }
    };

    //////////////////////////// mapping 2D edge to 3D ////////////////////////////
    struct Uncorrected2DEdgeMappingData {
        Eigen::Vector3d edge_3D;
        Eigen::Vector3d tangents_3D_world;
        Eigen::Vector2d supporting_edge;
    };


    struct Uncorrected2DEdgeKey {
        Eigen::Vector3d edge_uncorrected;
        int image_number;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;

        bool operator==(const Uncorrected2DEdgeKey& other) const {
            return edge_uncorrected == other.edge_uncorrected &&
                image_number == other.image_number &&
                rotation.isApprox(other.rotation, 1e-8) &&
                translation.isApprox(other.translation, 1e-8);
        }
    };
    //////////////////////////// mapping 2D edge to 3D ////////////////////////////

    
    // Custom hash function
    struct HashUncorrected2DEdgeKey {
        std::size_t operator()(const Uncorrected2DEdgeKey& key) const {
            std::size_t seed = 0;

            // Hash Eigen::Vector3d edge_uncorrected
            for (int i = 0; i < 3; ++i)
                seed ^= std::hash<double>()(key.edge_uncorrected[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            seed ^= std::hash<int>()(key.image_number) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash Eigen::Matrix3d rotation
            for (int i = 0; i < 9; ++i)
                seed ^= std::hash<double>()(key.rotation(i)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash translation
            for (int i = 0; i < 3; ++i)
                seed ^= std::hash<double>()(key.translation[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };



    void add3DToSupportingEdgesMapping(const Eigen::Vector3d &edge_3D, 
                                       const Eigen::Vector3d &tangents_3D_world,
                                       const Eigen::Vector2d &supporting_edge, 
                                       const Eigen::Vector3d &supporting_edge_uncorrected, 
                                       int image_number,
                                       const Eigen::Matrix3d &rotation,
                                       const Eigen::Vector3d &translation);

    void add3DToFrameMapping(const Eigen::Vector3d& edge_3D, 
                             const Eigen::Vector2d& supporting_edge, 
                             int frame);

    void printFirst10Edges();
    // void write_edge_linking_to_file();

    std::vector<std::vector<SupportingEdgeData>> findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d>& all_R,const std::vector<Eigen::Vector3d>& all_T,int Num_Of_Total_Imgs);
    std::unordered_map<Eigen::Vector3d, std::vector<SupportingEdgeData>, HashEigenVector3d> edge_3D_to_supporting_edges;
    std::unordered_map<int, std::unordered_map<Eigen::Vector2d, std::vector<Eigen::Vector3d>, HashEigenVector2d>> frame_to_edge_to_3D_map;
    std::unordered_map<Uncorrected2DEdgeKey, std::vector<Uncorrected2DEdgeMappingData>, HashUncorrected2DEdgeKey> map_Uncorrected2DEdge_To_SupportingData();
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>
    build3DEdgeWeightedGraph(const std::unordered_map<Uncorrected2DEdgeKey, std::vector<Uncorrected2DEdgeMappingData>, HashUncorrected2DEdgeKey>& uncorrected_map);
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                   HashEigenVector3dPair, FuzzyVector3dPairEqual>
    computeGraphEdgeDistanceAndAngleStats(
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                       HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
    double lambda1, double lambda2);

};

<<<<<<< HEAD
#endif  // EDGE_MAPPING_HPP
=======
#endif  // EDGE_MAPPING_HPP
>>>>>>> e6c6edf843f3f019e8d3420d15a62864487d5a8f
