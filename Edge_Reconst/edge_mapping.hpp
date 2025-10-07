#ifndef EDGE_MAPPING_HPP
#define EDGE_MAPPING_HPP

#include <Eigen/Core>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <memory>

//> YAML file data reader
#include <yaml-cpp/yaml.h>

#include "util.hpp"
#include "file_reader.hpp"
#include "definitions.h"

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
        int edge_idx;
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
        int edge_idx;
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

    struct EdgeNode {
        int index;
        Eigen::Vector3d location;  
        Eigen::Vector3d orientation; 
        std::vector<std::pair<int, EdgeNode*>> neighbors;
        bool has_orientation_fixed_in_connectivity_graph;
    };

    struct ConnectivityGraphNode {
        Eigen::Vector3d location;  
        Eigen::Vector3d orientation;      
        std::pair<int, Eigen::Vector3d> left_neighbor;  
        std::pair<int, Eigen::Vector3d> right_neighbor; 
        int curve_index;
        bool used = false;
    };

    ConnectivityGraphNode connectivity_graph_node_member;

    //> A `Curve` as a sequence of edge node indices and information for curve consolidation
    struct Curve {
        std::vector<int> edge_indices;
        int index;
        bool b_loops_back_on_left = false;
        bool b_loops_back_on_right = false;
        int to_be_merged_left_edge_index = -1;
        int to_be_merged_right_edge_index = -1;
        int to_be_merged_left_curve_index = -1;
        int to_be_merged_right_curve_index = -1;
        int consolidation_set_from_left = -1;
        int consolidation_set_from_right = -1;
    };

    using PointerNeighborMap = std::unordered_map<const Eigen::Vector3d*, std::vector<std::pair<const Eigen::Vector3d*, std::pair<Eigen::Vector3d, Eigen::Vector3d>>>>;
    
    //> This is used to map an edge index to the corresponding connectivity graph node struct
    using ConnectivityGraph = std::unordered_map<int, ConnectivityGraphNode>;

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
                                       int edge_idx,
                                       const Eigen::Matrix3d &rotation,
                                       const Eigen::Vector3d &translation);

    void add3DToFrameMapping(const Eigen::Vector3d& edge_3D, 
                             const Eigen::Vector2d& supporting_edge, 
                             int frame);

    void Setup_Data_Parameters( YAML::Node Edge_Sketch_Setting_File );
    using EdgeNodeList = std::vector<std::unique_ptr<EdgeNode>>;

    //> main code for consolidating 3D edges and form 3D curves
    void consolidate_3D_edges(const std::vector<Eigen::Matrix3d> all_R,const std::vector<Eigen::Vector3d> all_T, const Eigen::Matrix3d K, const int Num_Of_Total_Imgs);
    
    std::unordered_map<Eigen::Vector3d, std::vector<SupportingEdgeData>, HashEigenVector3d> edge_3D_to_supporting_edges;
    std::unordered_map<Uncorrected2DEdgeKey, std::vector<Uncorrected2DEdgeMappingData>, HashUncorrected2DEdgeKey> map_Uncorrected2DEdge_To_SupportingData();
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>
    build3DEdgeWeightedGraph(const std::unordered_map<Uncorrected2DEdgeKey, std::vector<Uncorrected2DEdgeMappingData>, HashUncorrected2DEdgeKey>& uncorrected_map, 
                            const std::vector<Eigen::MatrixXd> All_Edgels, std::vector<EdgeCurvelet> all_curvelets,
                            const std::vector<Eigen::Matrix3d> All_R, const std::vector<Eigen::Vector3d> All_T);
    
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                       HashEigenVector3dPair, FuzzyVector3dPairEqual>
    pruneEdgeGraph_by_3DProximityAndOrientation(std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                           HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph );

    EdgeNodeList buildEdgeNodeGraph(const std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int,
                                HashEigenVector3dPair, FuzzyVector3dPairEqual>& pruned_graph);

    void align3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes);

    void createConnectivityGraph(EdgeNodeList& edge_nodes);

    void writeConnectivityGraphToFile(const ConnectivityGraph& graph, const std::string& file_name);
    std::vector<Curve> buildCurvesFromConnectivityGraph( std::vector<Curve>& curves );
    void writeCurvesToFile(const std::vector<Curve>& curves, const std::string& file_name, bool b_write_curve_info);
    
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual> pruneEdgeGraphbyProjections(
                                                                                                                                                    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                                                                                                                                                    HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
                                                                                                                                                    const std::vector<Eigen::Matrix3d> All_R,
                                                                                                                                                    const std::vector<Eigen::Vector3d> All_T,
                                                                                                                                                    const Eigen::Matrix3d K,
                                                                                                                                                    const int Num_Of_Total_Imgs
                                                                                                                                                    );
    std::vector<EdgeCurvelet> read_curvelets();
    std::vector<Eigen::MatrixXd> read_edgels();

private:
    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    std::shared_ptr<file_reader> file_reader_ptr = nullptr;

    void write_edge_graph( std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph, std::string file_name );
    void writeCurveIndiciesForMerging( const std::vector<std::vector<int>> curve_indices_for_merging, const std::string& file_name );

    void reset_curve(Curve &curve) {
        curve.edge_indices.clear();
        curve.index = -1;
        curve.b_loops_back_on_left = false;
        curve.b_loops_back_on_right = false;
        curve.to_be_merged_left_edge_index = -1;
        curve.to_be_merged_right_edge_index = -1;
        curve.to_be_merged_left_curve_index = -1;
        curve.to_be_merged_right_curve_index = -1;
        curve.consolidation_set_from_left = -1;
        curve.consolidation_set_from_right = -1;
    }

    std::pair<int, int> make_canonical_pair(int a, int b) {
        return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    }

    ConnectivityGraph connectivity_graph;

    //> Curve extentions and merging
    std::vector<int> merge_multiple_curves(const std::vector<int> curve_indices, const std::vector<Curve> all_curves);
    std::vector<int> make_curve_orientation_consistent(const std::vector<int>& curve1, const std::vector<int>& curve2);
    std::vector<int> merge_curve_pair(const std::vector<int>& curve1, const std::vector<int>& curve2);

    std::set<std::pair<int, int>> check_duplicate_curve_ids(const std::vector<std::vector<int>>& curve_id_set);

    bool b_is_in_first_or_last_two(const std::vector<int>& vec, int num);

    //> Dataset configurations
    std::string Dataset_Path;
    std::string Dataset_Name;
    std::string Scene_Name;
    int Num_Of_Images;
    int thresh_EDG;
};

#endif  // EDGE_MAPPING_HPP