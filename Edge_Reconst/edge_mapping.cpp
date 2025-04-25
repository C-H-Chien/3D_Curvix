#include "edge_mapping.hpp"
#include "EdgeSketch_Core.hpp"
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <set>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>

#include "../Edge_Reconst/definitions.h"
#include "../Edge_Reconst/util.hpp"


void EdgeMapping::add3DToSupportingEdgesMapping(const Eigen::Vector3d &edge_3D, 
                                                const Eigen::Vector3d &tangents_3D_world,
                                                const Eigen::Vector2d &supporting_edge, 
                                                const Eigen::Vector3d &supporting_edge_uncorrected, 
                                                int image_number, 
                                                const Eigen::Matrix3d &rotation,
                                                const Eigen::Vector3d &translation
                                                ) {
    if (edge_3D_to_supporting_edges.find(edge_3D) == edge_3D_to_supporting_edges.end()) {
        edge_3D_to_supporting_edges[edge_3D] = {};
    }
    edge_3D_to_supporting_edges[edge_3D].push_back({supporting_edge, supporting_edge_uncorrected, image_number, rotation, translation, tangents_3D_world});
}

///////////////////////// convert 3d->2d relationship to 2d uncorrect -> 3d /////////////////////////
std::unordered_map<EdgeMapping::Uncorrected2DEdgeKey, std::vector<EdgeMapping::Uncorrected2DEdgeMappingData>, EdgeMapping::HashUncorrected2DEdgeKey>
EdgeMapping::map_Uncorrected2DEdge_To_SupportingData() {
    std::unordered_map<Uncorrected2DEdgeKey, std::vector<Uncorrected2DEdgeMappingData>, HashUncorrected2DEdgeKey> map;

    for (const auto& [edge_3D, support_list] : edge_3D_to_supporting_edges) {
        for (const auto& data : support_list) {
            Uncorrected2DEdgeKey key{
                data.edge_uncorrected,
                data.image_number,
                data.rotation,
                data.translation
            };

            Uncorrected2DEdgeMappingData record{
                edge_3D,
                data.tangents_3D_world,
                data.edge
            };

            map[key].push_back(record);
        }
    }
    return map;
}
///////////////////////// convert 3d->2d relationship to 2d uncorrect -> 3d /////////////////////////



///////////////////////// Create the 3D edge weight graph /////////////////////////
std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::build3DEdgeWeightedGraph(const std::unordered_map<EdgeMapping::Uncorrected2DEdgeKey, std::vector<EdgeMapping::Uncorrected2DEdgeMappingData>, EdgeMapping::HashUncorrected2DEdgeKey>& uncorrected_map) {

    std::unordered_map<
    std::pair<Eigen::Vector3d, Eigen::Vector3d>,
    int,
    HashEigenVector3dPair,
    FuzzyVector3dPairEqual
    > graph;


    for (const auto& [key, records] : uncorrected_map) {
        int size = records.size();
        if (size < 2) continue;

        for (int i = 0; i < size-1; ++i) {
            for (int j = i+1; j < size; ++j){
                Eigen::Vector3d a = records[i].edge_3D;
                Eigen::Vector3d b = records[j].edge_3D;

                // Enforce consistent ordering for pair hashing
                if ((a.x() > b.x()) || (a.x() == b.x() && a.y() > b.y()) || 
                    (a.x() == b.x() && a.y() == b.y() && a.z() > b.z())) {
                    std::swap(a, b);
                }

                std::pair<Eigen::Vector3d, Eigen::Vector3d> edge_pair = {a, b};
                graph[edge_pair]++;
            }
        }
    }
    return graph;
}
///////////////////////// Create the 3D edge weight graph /////////////////////////

void EdgeMapping::write_edge_graph(std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
                                    std::string file_name ) {
    std::string file_path = "../../outputs/" + file_name + ".txt";
    // LOG_INFOR_MESG("Writing 3D edge graph to a file.");
    std::ofstream edge_graph_file_out(file_path);
    if (!edge_graph_file_out.is_open()) {
        LOG_ERROR("Could not open 3D_edge_graph.txt file!");
        return;
    }
    for (const auto& [pair, weight] : graph) {
        const Eigen::Vector3d& p1 = pair.first;
        const Eigen::Vector3d& p2 = pair.second;

        auto it1 = edge_3D_to_supporting_edges.find(p1);
        auto it2 = edge_3D_to_supporting_edges.find(p2);
        if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end()) continue;

        Eigen::Vector3d t1 = it1->second.front().tangents_3D_world.normalized();
        Eigen::Vector3d t2 = it2->second.front().tangents_3D_world.normalized();

        edge_graph_file_out << p1(0) << "\t" << p1(1) << "\t" << p1(2) << "\t";
        edge_graph_file_out << p2(0) << "\t" << p2(1) << "\t" << p2(2) << "\t";
        edge_graph_file_out << t1(0) << "\t" << t1(1) << "\t" << t1(2) << "\t";
        edge_graph_file_out << t2(0) << "\t" << t2(1) << "\t" << t2(2) << "\n";
    }
    edge_graph_file_out.close();
}

///////////////////////// compute weighted graph stats /////////////////////////
template <typename T>
T clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(v, hi));
}

std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                   HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::pruneEdgeGraph_by_3DProximityAndOrientation(std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph ){
    //> write the 3D edge graph before pruning
    write_edge_graph( graph, "3D_edge_graph" );

    std::vector<double> distances;
    std::vector<double> angles;
    std::vector<int> edge_link_indx;
    std::vector<int> invalid_link_indx;
    std::unordered_map< std::pair<Eigen::Vector3d, Eigen::Vector3d>,
                        int, 
                        HashEigenVector3dPair,
                        FuzzyVector3dPairEqual> pruned_graph;

    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    int edge_link_counter = 0;
    for (const auto& [pair, weight] : graph) {
        const Eigen::Vector3d& p1 = pair.first;
        const Eigen::Vector3d& p2 = pair.second;

        auto it1 = edge_3D_to_supporting_edges.find(p1);
        auto it2 = edge_3D_to_supporting_edges.find(p2);
        if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end()) {
            invalid_link_indx.push_back(edge_link_counter);
            edge_link_counter++;
            continue;
        }

        Eigen::Vector3d t1 = it1->second.front().tangents_3D_world.normalized();
        Eigen::Vector3d t2 = it2->second.front().tangents_3D_world.normalized();

        //> Compute the orientation between the two (in degrees)
        angles.push_back( std::acos(fabs(t1.dot(t2)))*180/PI );

        //> compute the distance between the two
        //> (here I find the tangential distance rather than the Euclidean distance)
        Eigen::Vector3d closest_point = util->findClosestVectorFromPointToLine(p2, t2, p1);
        distances.push_back( closest_point.norm() );

        edge_link_indx.push_back(edge_link_counter);
        edge_link_counter++;
    }

    auto mean_std = [](const std::vector<double>& values) -> std::pair<double, double> {
        if (values.empty()) return {0.0, 0.0};
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sq_sum = std::accumulate(values.begin(), values.end(), 0.0,
                            [mean](double acc, double val) { return acc + (val - mean) * (val - mean); });
        double stddev = std::sqrt(sq_sum / values.size());
        return {mean, stddev};
    };

    auto [mu1, sigma1] = mean_std(distances);
    auto [mu2, sigma2] = mean_std(angles);

    std::cout << "[GRAPH STATS BEFORE PRUNING]\n";
    std::cout << "μ1 (meters): " << mu1 << ", σ1: " << sigma1 << "\n";
    std::cout << "μ2 (degrees): " << mu2 << ", σ2: " << sigma2 << "\n";
    std::cout << "End of CH's work" << std::endl;

    edge_link_counter = 0;
    int access_index = 0;
    int removed_count = 0;
    for (const auto& [pair, weight] : graph) {

        if (std::find(invalid_link_indx.begin(), invalid_link_indx.end(), edge_link_counter) != invalid_link_indx.end()) {
            edge_link_counter++;
            continue;
        }
        
        int index = edge_link_indx[access_index];
        double angle_diff = angles[index];
        double proximity_diff = distances[index];

        access_index++;
        edge_link_counter++;

        if ((proximity_diff < mu1 + sigma1*PRUNE_3D_EDGE_GRAPH_LAMBDA1 && angle_diff < mu2 + sigma2*PRUNE_3D_EDGE_GRAPH_LAMBDA2) || weight > 1) {
            pruned_graph[pair]++;
        }
    }

    //> write the 3D edge graph after pruning
    write_edge_graph( pruned_graph, "3D_edge_pruned_graph" );

    return pruned_graph;
}
///////////////////////// compute weighted graph stats /////////////////////////

///////////////////////// Start of pruning weighted graph by projecting to all views /////////////////////
std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::pruneEdgeGraphbyProjections(std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
                                        const std::vector<Eigen::Matrix3d> All_R,
                                        const std::vector<Eigen::Vector3d> All_T,
                                        const Eigen::Matrix3d K,
                                        const int Num_Of_Total_Imgs )
                                    {
    std::unordered_map<
    std::pair<Eigen::Vector3d, Eigen::Vector3d>,
    int,
    HashEigenVector3dPair,
    FuzzyVector3dPairEqual
    > pruned_graph_by_proj;

    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    // std::cout << "Begin of edge graph pruning by projections ..." << std::endl;

    for (const auto& [pair, weight] : graph) {

        //> Find the locations and tangents of the 3D edge pair
        const Eigen::Vector3d& Edge3D_1_Location = pair.first;
        const Eigen::Vector3d& Edge3D_2_Location = pair.second;

        // if (fabs(Edge3D_2_Location(0) - 0.517675) < 0.0001 && fabs(Edge3D_2_Location(1) - 0.267291) < 0.0001 && fabs(Edge3D_2_Location(2) - 0.351295) < 0.0001 && \
        //     fabs(Edge3D_1_Location(0) - 0.489324) < 0.0001 && fabs(Edge3D_1_Location(1) - 0.24528) < 0.0001 && fabs(Edge3D_1_Location(2) - 0.295694) < 0.0001) {

        auto it1 = edge_3D_to_supporting_edges.find(Edge3D_1_Location);
        auto it2 = edge_3D_to_supporting_edges.find(Edge3D_2_Location);

        //> Skip if the 3D edge does not exist in the structure edge_3D_to_supporting_edges (is it possible?)
        if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end()) continue;

        Eigen::Vector3d unit_Tangent3D_1 = it1->second.front().tangents_3D_world.normalized();
        Eigen::Vector3d unit_Tangent3D_2 = it2->second.front().tangents_3D_world.normalized();

        bool prune_flag = false;
        double ore_diff_threshold = cos(double(PRUNE_BY_PROJ_ORIE_THRESH)/180*PI);
        
        //> Project to all views 
        for (unsigned vi = 0; vi < Num_Of_Total_Imgs; vi++) {

            //> Project 3D edge point to the 2D image
            Eigen::Vector3d proj_edge_1_location = K * (All_R[vi] * Edge3D_1_Location + All_T[vi]);
            Eigen::Vector3d proj_edge_2_location = K * (All_R[vi] * Edge3D_2_Location + All_T[vi]);
            proj_edge_1_location = util->getNormalizedProjectedPoint( proj_edge_1_location );
            proj_edge_2_location = util->getNormalizedProjectedPoint( proj_edge_2_location );

            // std::cout << "[Image #" << vi << "] ";
            // std::cout << "Projected locations: (" << proj_edge_1_location(0) << ", " << proj_edge_1_location(1) << ", " << proj_edge_1_location(2) << "), (" \
            //           << proj_edge_2_location(0) << ", " << proj_edge_2_location(1) << ", " << proj_edge_2_location(2) << ")" << ", norm = " << (proj_edge_1_location - proj_edge_2_location).norm() << " ";

            //> Project 3D tangent to the 2D image
            Eigen::Vector3d proj_tangent_1 = util->project_3DTangent_to_Image(All_R[vi], K, unit_Tangent3D_1, proj_edge_1_location);
            Eigen::Vector3d proj_tangent_2 = util->project_3DTangent_to_Image(All_R[vi], K, unit_Tangent3D_2, proj_edge_2_location);

            //> Rule out if the distance between the projected 3D edge points are over a threshold
            //> TODO: make the threhold as a Macro
            prune_flag = ((proj_edge_1_location - proj_edge_2_location).norm() > PRUNE_BY_PROJ_PROX_THRESH) ? (true) : (false);
            if (prune_flag) break;

            // std::cout << "Projected tangents: (" << proj_tangent_1(0) << ", " << proj_tangent_1(1) << ", " << proj_tangent_1(2) << "), (" \
            //           << proj_tangent_2(0) << ", " << proj_tangent_2(1) << ", " << proj_tangent_2(2) << ")" << std::endl;

            //> Rule out if the orientation difference is over a threshold
            //> TODO: make the threhold as a Macro
            double abs_dot_prod = fabs(proj_tangent_1(0)*proj_tangent_2(0) + proj_tangent_1(1)*proj_tangent_2(1));
            prune_flag = (abs_dot_prod < ore_diff_threshold) ? (true) : (false);

            if (prune_flag) break;
        }

        // std::cout << "prune_flag = " << prune_flag << std::endl;

        if (!prune_flag) 
            pruned_graph_by_proj[pair]++;
        // }
    }

    //> write the 3D edge graph after pruning by projection
    write_edge_graph( pruned_graph_by_proj, "3D_edge_pruned_graph_by_proj" );
    // return graph;
    return pruned_graph_by_proj;
}


///////////////////////// End of pruning weighted graph by projecting to all views /////////////////////

///////////////////////// build a map of 3d edges with its neighbors /////////////////////////
EdgeMapping::EdgeNodeList EdgeMapping::buildEdgeNodeGraph(const std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int,
                                                            HashEigenVector3dPair, FuzzyVector3dPairEqual>& pruned_graph) {
                                                            std::unordered_map<Eigen::Vector3d, EdgeNode*, HashEigenVector3d, FuzzyVector3dEqual> node_map;
    
    // Create node list 
    EdgeNodeList nodes;

    // Create node objects 
    for (const auto& [edge, support_list] : edge_3D_to_supporting_edges) {
        std::unique_ptr<EdgeNode> node(new EdgeNode());
        node->location = edge;

        Eigen::Vector3d avg_tangent = Eigen::Vector3d::Zero();
        for (const auto& s : support_list) {
            avg_tangent += s.tangents_3D_world;
        }
        if (!support_list.empty()) {
            avg_tangent.normalize();
        }
        node->orientation = avg_tangent;

        node_map[edge] = node.get();
        nodes.push_back(std::move(node));
    }

    // Fill neighbor links
    for (const auto& [pair, _] : pruned_graph) {
        auto a_it = node_map.find(pair.first);
        auto b_it = node_map.find(pair.second);
        if (a_it == node_map.end() || b_it == node_map.end()) continue;

        EdgeNode* a = a_it->second;
        EdgeNode* b = b_it->second;

        a->neighbors.push_back(b);
        b->neighbors.push_back(a);
    }

    return nodes;
}
///////////////////////// build a map of 3d edges with its neighbors /////////////////////////



///////////////////////// Smoothing 3d edges with its neighbors /////////////////////////
void EdgeMapping::align3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes, int iterations, double step_size) {

    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::ofstream before_out("../../outputs/3D_edges_before_smoothing.txt");
    std::unordered_set<int> target_indices = {559, 561, 910, 1425, 1900, 5552, 6140, 6690, 8079,11132, 11432, 11481, 11482, 11496}; //8079 is the target edge
    //std::unordered_set<int> target_indices = { 5552, 6140, 6690, 8079,11132, 11432};

    ///////////////////// find neighbor connection //////////////////////////////////
    std::vector<int> target_indices_vec(target_indices.begin(), target_indices.end());
    std::ofstream neighbor_file("../../outputs/target_indices_neighbors.txt");
    for (size_t idx = 0; idx < target_indices_vec.size(); idx++) {
        int i = target_indices_vec[idx];
        if (i >= edge_nodes.size()) continue; // Safety check
        
        const auto& node = edge_nodes[i];
        std::cout << idx << " "; // Print index in target_indices array instead of actual node index
        
        // Check which target indices are neighbors of this node
        for (const auto* neighbor : node->neighbors) {
            // Find the neighbor's index
            for (size_t j = 0; j < edge_nodes.size(); ++j) {
                if (edge_nodes[j].get() == neighbor) { // Compare pointers
                    // Check if this neighbor is in target_indices
                    if (target_indices.count(j)) {
                        // Find the index of j in target_indices_vec
                        for (size_t neighbor_idx = 0; neighbor_idx < target_indices_vec.size(); neighbor_idx++) {
                            if (target_indices_vec[neighbor_idx] == j) {
                                std::cout << neighbor_idx << " "; // Print index in target_indices array
                                neighbor_file << node->location.transpose() << " " << neighbor->location.transpose() << std::endl;
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
        std::cout << "\n";
        neighbor_file << std::endl;
    }
    neighbor_file.close();
    ///////////////////// find neighbor connection //////////////////////////////////


    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        const auto& node = edge_nodes[i];
        //std::cout << node->location.transpose() << "; ";
        before_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
        if (target_indices.count(i)){
            //before_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
            //std::cout << node->location.transpose()<<" "<<node->orientation.transpose()<< "; "<<std::endl;
            std::cout << node->orientation.transpose()<<std::endl;
        }
    }
    //std::cout << "\n";
    before_out.close();
    std::ofstream after_out("../../outputs/3D_edges_after_smoothing.txt");
    std::cout << "Start aligning edges..." << std::endl;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Eigen::Vector3d> new_locations(edge_nodes.size());
        std::vector<Eigen::Vector3d> new_orientations(edge_nodes.size());

        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            const auto& node = edge_nodes[i];
            if (node->neighbors.empty()) {
                new_locations[i] = node->location;
                new_orientations[i] = node->orientation;
                continue;
            }

            Eigen::Vector3d sum_force = Eigen::Vector3d::Zero();
            for (const auto& neighbor : node->neighbors) {
                const Eigen::Vector3d& p = neighbor->location;
                const Eigen::Vector3d& t_neighbor = neighbor->orientation;
                const Eigen::Vector3d& B = node->location;

                Eigen::Vector3d tangential_dist = util->findClosestVectorFromPointToLine(p, t_neighbor, B);
                sum_force += tangential_dist;
            }
            sum_force /= static_cast<double>(node->neighbors.size());
            
            new_locations[i] = node->location + step_size * sum_force;

            //> Orientation aligning
            Eigen::Vector3d sum_tangent = Eigen::Vector3d::Zero();
            Eigen::Vector3d sum_euler_angles = Eigen::Vector3d::Zero();
            for (const auto& neighbor : node->neighbors) {
                Eigen::Vector3d euler_angles = util->getShortestAlignEulerAnglesDegrees(node->orientation, neighbor->orientation);
                sum_euler_angles += euler_angles;
            }

            sum_euler_angles /= static_cast<double>(node->neighbors.size());
            sum_euler_angles *= step_size;
            //sum_euler_angles *= 0.5;

            //> convert from degrees to radians
            sum_euler_angles = sum_euler_angles * M_PI / 180.0;
            Eigen::Matrix3d R_align = util->euler_to_rotation_matrix(sum_euler_angles(0), sum_euler_angles(1), sum_euler_angles(2));
            new_orientations[i] = R_align * node->orientation;
        }

        //> Update all edge locations and orientations
        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            edge_nodes[i]->location = new_locations[i];
            edge_nodes[i]->orientation = new_orientations[i];
            const auto& node = edge_nodes[i];

            // std::cout << node->location.transpose()<< "; ";
            //std::cout << node->orientation.transpose() << ";";

            if(iter == iterations-1){
                after_out << node->location.transpose() << " " << node->orientation.transpose() << "\n";
                if (target_indices.count(i)){
                    //after_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
                    //std::cout << node->location.transpose()<<" "<<node->orientation.transpose()<< "; "<<std::endl;
                }
            }
        }
        //std::cout << "\n";
    }

    ///////////////////// find neighbor connection //////////////////////////////////
    std::ofstream neighbor_file_after("../../outputs/target_indices_neighbors_after.txt");
    for (auto i : target_indices) {
        if (i >= edge_nodes.size()) continue; // Safety check 
        const auto& node = edge_nodes[i];
        // Check which target indices are neighbors of this node
        for (const auto* neighbor : node->neighbors) {
            // Find the neighbor's index
            for (size_t j = 0; j < edge_nodes.size(); ++j) {
                if (edge_nodes[j].get() == neighbor) { // Compare pointers
                    // Check if this neighbor is in target_indices
                    if (target_indices.count(j)) {
                        neighbor_file_after << node->location.transpose() << " "<<neighbor->location.transpose() << std::endl;
                        //std::cout << "Target " << i << " is a neighbor of Target " << j << std::endl;
                    }
                    break;
                }
            }
        }
        neighbor_file_after << std::endl;
    }
    neighbor_file_after.close();
    ///////////////////// find neighbor connection //////////////////////////////////

    after_out.close();
    std::string msg = "[ALIGNMENT COMPLETE] Aligned edges written to file after " + std::to_string(iterations) + " iterations with step size " + std::to_string(step_size); 
    LOG_GEN_MESG(msg);
}
///////////////////////// Smoothing 3d edges with its neighbors /////////////////////////


////////////////// test //////////////////
struct CompareVector3d {
    bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
        if (a.x() != b.x()) return a.x() < b.x();
        if (a.y() != b.y()) return a.y() < b.y();
        return a.z() < b.z();
    }
};

struct NeighborWithOrientation {
    Eigen::Vector3d location;
    Eigen::Vector3d orientation;

    bool operator<(const NeighborWithOrientation& other) const {
        if (!location.isApprox(other.location, 1e-8)) {
            if (location.x() != other.location.x()) return location.x() < other.location.x();
            if (location.y() != other.location.y()) return location.y() < other.location.y();
            return location.z() < other.location.z();
        }
        return orientation.norm() < other.orientation.norm();  // arbitrary tiebreaker
    }
};



EdgeMapping::EdgeNodeList EdgeMapping::createEdgeNodesFromEdges(const std::vector<Eigen::Vector3d>& locations,
                                                                const std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>& pruned_graph) 
                                                            {
    EdgeNodeList node_list;
    std::vector<EdgeNode*> raw_ptrs;
    FuzzyVector3dPairEqual fuzzy_equal;

    // Step 1: Create EdgeNodes from input
    for (const auto& location : locations) {
        for (const auto& [stored_loc, data] : edge_3D_to_supporting_edges) {
            if (location.isApprox(stored_loc, 1e-5)) { 
                //std::cout << "found tangent for " << location.transpose() << std::endl;
                std::unique_ptr<EdgeNode> node(new EdgeNode());
                node->location = location;
                node->orientation = data.front().tangents_3D_world.normalized();
                raw_ptrs.push_back(node.get());
                node_list.push_back(std::move(node));
                break;
            }
        }
    }

    // Step 2: Check and establish neighbor relationships from pruned_graph
    for (const auto& [pair, weight] : pruned_graph) {
        Eigen::Vector3d a = pair.first;
        Eigen::Vector3d b = pair.second;

        EdgeNode* node_a = nullptr;
        EdgeNode* node_b = nullptr;

        for (auto* node : raw_ptrs) {
            if (!node_a && node->location.isApprox(a, 1e-6)) node_a = node;
            if (!node_b && node->location.isApprox(b, 1e-6)) node_b = node;
            if (node_a && node_b) break;
        }

        if (node_a && node_b) {
            node_a->neighbors.push_back(node_b);
            node_b->neighbors.push_back(node_a);
        }
    }

    // Step 3: Print final node list
    std::cout << "\n[INFO] Node Info:\n";
    for (size_t i = 0; i < raw_ptrs.size(); ++i) {
        const auto* node = raw_ptrs[i];
        Eigen::Vector3d target_node = Eigen::Vector3d(0.562798, 0.0414557,  0.354671);
        // if(node->location.isApprox(target_node, 1e-5)){
        //     std::cout << "  Location:    " << node->location.transpose() << std::endl;
        //     std::cout << "  Orientation: " << node->orientation.transpose() << std::endl;
        //     std::cout<<"neighbors: "<<std::endl;
        //     for (const auto* neighbor : node->neighbors) {
        //         std::cout << "    - " << neighbor->location.transpose() << "   "<<neighbor->orientation.transpose()<<std::endl;
        //     }
        // }
        // std::cout << "Node " << i + 1 << ":\n";
        // std::cout << "  Location:    " << node->location.transpose() << "\n";
        // std::cout << "  Orientation: " << node->orientation.transpose() << "\n";
        // std::cout << "  Neighbors:\n";
        // for (const auto* neighbor : node->neighbors) {
        //     std::cout << "    - " << neighbor->location.transpose() << "\n";
        // }
        // std::cout << std::endl;
    }
    return node_list;
}


EdgeMapping::EdgeNodeList EdgeMapping::createEdgeNodesFromFiles(const std::string& points_file, 
                                                                const std::string& tangents_file, 
                                                                const std::string& connections_file) {
    // Initialize the EdgeNodeList and a map to store nodes by index for easy referencing
    EdgeNodeList node_list;
    std::vector<EdgeNode*> node_ptrs;
    
    // Read line points file
    std::ifstream points_infile(points_file);
    if (!points_infile.is_open()) {
        std::cerr << "Failed to open file: " << points_file << std::endl;
        return node_list;
    }
    
    // Read line tangents file
    std::ifstream tangents_infile(tangents_file);
    if (!tangents_infile.is_open()) {
        std::cerr << "Failed to open file: " << tangents_file << std::endl;
        return node_list;
    }
    
    // Create nodes with their locations and orientations
    double x, y, z;
    size_t index = 0;
    
    while (points_infile >> x >> y >> z) {
        Eigen::Vector3d location(x, y, z);
        
        // Read tangent
        double tx, ty, tz;
        if (!(tangents_infile >> tx >> ty >> tz)) {
            std::cerr << "Error: Mismatch between points and tangents files at index " << index << std::endl;
            return node_list;
        }
        Eigen::Vector3d tangent(tx, ty, tz);
        
        std::unique_ptr<EdgeNode> node(new EdgeNode());
        node->location = location;
        node->orientation = tangent.normalized();
        
        node_ptrs.push_back(node.get());
        
        node_list.push_back(std::move(node));
        index++;
    }
    
    points_infile.close();
    tangents_infile.close();
    
    // Read connections file and establish neighbor relationships
    std::ifstream connections_infile(connections_file);
    if (!connections_infile.is_open()) {
        std::cerr << "Failed to open file: " << connections_file << std::endl;
        return node_list;
    }
    
    std::string line;
    index = 0;
    
    while (std::getline(connections_infile, line)) {
        if (index >= node_ptrs.size()) {
            std::cerr << "Error: More connection lines than nodes at index " << index << std::endl;
            break;
        }
        
        EdgeNode* current_node = node_ptrs[index];
        
        // Parse the line to get neighbor indices
        std::istringstream iss(line);
        int neighbor_index;
        
        while (iss >> neighbor_index) {
            // Skip the current node's own index
            if (neighbor_index != index && neighbor_index >= 0 && neighbor_index < node_ptrs.size()) {
                // Add neighbor to current node
                current_node->neighbors.push_back(node_ptrs[neighbor_index]);
            }
        }
        index++;
    }
    
    connections_infile.close();
    
    // Print node list information
    std::cout << "\n[NODE LIST SUMMARY]" << std::endl;
    std::cout << "Total nodes: " << node_list.size() << std::endl;
    std::cout << "[INFO] Created " << node_list.size() << " edge nodes with connections from files." << std::endl;
    return node_list;
}
////////////////// test //////////////////


std::vector<std::vector<EdgeMapping::SupportingEdgeData>> EdgeMapping::findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d> all_R,
                                                                                                const std::vector<Eigen::Vector3d> all_T,
                                                                                                const Eigen::Matrix3d K,
                                                                                                const int Num_Of_Total_Imgs) { 


    int num_iteration = 1000;
    double step_size = 0.1;

    // convert 3D-> 2D relationship to 2D->3D relationship
    auto uncorrected_map = map_Uncorrected2DEdge_To_SupportingData();
    // {3D edge 1, 3D edge 2} -> weight (how many 2d edges they have in common)
    auto graph = build3DEdgeWeightedGraph(uncorrected_map);
    auto pruned_graph = pruneEdgeGraph_by_3DProximityAndOrientation(graph);
    auto pruned_graph_by_projection = pruneEdgeGraphbyProjections(pruned_graph, all_R, all_T, K, Num_Of_Total_Imgs);

    // crate 3d -> its neighbor data structure
    auto neighbor_map = buildEdgeNodeGraph(pruned_graph_by_projection);
    align3DEdgesUsingEdgeNodes(neighbor_map, num_iteration, step_size); //call it align

    /////////// TEST: manually create and iteratively smooth 2 test EdgeNodes  /////////// 

    /////////// Create EdgeNodes from input files ///////////
    std::string points_file = "../../files/line_noisy_points.txt";
    std::string tangents_file = "../../files/line_noisy_tangents.txt";
    std::string connections_file = "../../files/line_connections.txt";
    
    EdgeNodeList test_node = createEdgeNodesFromFiles(points_file, tangents_file, connections_file);

    //align3DEdgesUsingEdgeNodes(test_node, num_iteration); //call it align

    std::vector<std::vector<SupportingEdgeData>> merged_groups;

    // Projection view's extrinsics
    Eigen::Matrix3d R_view = all_R[7];
    Eigen::Vector3d T_view = all_T[7];
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::ofstream proj_outfile("../../outputs/3D_edge_groups_projected.txt");
    if (!proj_outfile.is_open()) {
        std::cerr << "Could not open 3D edge group projection output file!" << std::endl;
        return merged_groups;
    }

    std::ofstream edge_3d_outfile("../../outputs/3D_edge_groups.txt");
    if (!edge_3d_outfile.is_open()) {
        std::cerr << "Could not open 3D edge projection output file!" << std::endl;
        return merged_groups;
    }

    Eigen::Vector3d target1(0.576663, 0.929135, 0.64199);
    // Eigen::Vector3d target2(0.710306, 0.877993, 0.353316);
    // Eigen::Vector3d target3(0.704316, 0.884627,  0.35257);

    // Set to collect neighbors of target1 and target2
    std::set<NeighborWithOrientation> neighbors_of_target1;
    // std::set<NeighborWithOrientation> neighbors_of_target2;
    // std::set<NeighborWithOrientation> neighbors_of_target3;

    int group_id = 0;
    for (const auto& [pair, weight] : pruned_graph_by_projection) {
        const Eigen::Vector3d& p1 = pair.first;
        const Eigen::Vector3d& p2 = pair.second;

        // Retrieve tangents
        Eigen::Vector3d tangent1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d tangent2 = Eigen::Vector3d::Zero();

        auto it1 = edge_3D_to_supporting_edges.find(p1);
        auto it2 = edge_3D_to_supporting_edges.find(p2);

        bool valid_tangents = false;
        if (it1 != edge_3D_to_supporting_edges.end() && !it1->second.empty() &&
            it2 != edge_3D_to_supporting_edges.end() && !it2->second.empty()) {
            tangent1 = it1->second.front().tangents_3D_world.normalized();
            tangent2 = it2->second.front().tangents_3D_world.normalized();
            valid_tangents = true;
        }

        ///////////// find neighbors of the target point /////////////
        if (valid_tangents){
            const Eigen::Vector3d& a = pair.first;
            const Eigen::Vector3d& b = pair.second;

            // if (b.isApprox(Eigen::Vector3d(0.499562, -0.00378834, 0.541387), 1e-6)) {
            //     std::cout << "{Eigen::Vector3d(0.499562, -0.00378834, 0.541387), Eigen::Vector3d(" << tangent2.transpose() << ")}," << std::endl;
            // }

            if (a.isApprox(target1, 1e-6)) 
                neighbors_of_target1.insert({b, tangent2});
            else if (b.isApprox(target1, 1e-6)) 
                neighbors_of_target1.insert({a, tangent1});

            // if (a.isApprox(target2, 1e-6)) 
            //     neighbors_of_target2.insert({b, tangent2});
            // else if (b.isApprox(target2, 1e-6)) 
            //     neighbors_of_target2.insert({a, tangent1});

            // if (a.isApprox(target3, 1e-6)) 
            //     neighbors_of_target3.insert({b, tangent2});
            // else if (b.isApprox(target3, 1e-6)) 
            //     neighbors_of_target3.insert({a, tangent1});
        }
        ///////////// find neighbors of the target point /////////////

    }

    // smoothed_proj_out.close();

    // std::cout<<"neighbors of target 1 are: "<<std::endl;
    // for (auto& n1 : neighbors_of_target1) {
    //     //for (const auto& n2 : neighbors_of_target2) {
    //     //    if (n1.isApprox(n2, 1e-6)) {
    //     std::cout << n1.location.transpose() << "\n";
    //     //    }
    //    // }
    // }
    
    return merged_groups;
}
