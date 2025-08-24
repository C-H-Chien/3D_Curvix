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


void EdgeMapping::add3DToSupportingEdgesMapping(const Eigen::Vector3d &edge_3D, 
                                                const Eigen::Vector3d &tangents_3D_world,
                                                const Eigen::Vector2d &supporting_edge, 
                                                const Eigen::Vector3d &supporting_edge_uncorrected, 
                                                int image_number, 
                                                int edge_idx,
                                                const Eigen::Matrix3d &rotation,
                                                const Eigen::Vector3d &translation
                                                ) {
    if (edge_3D_to_supporting_edges.find(edge_3D) == edge_3D_to_supporting_edges.end()) {
        edge_3D_to_supporting_edges[edge_3D] = {};
    }
    edge_3D_to_supporting_edges[edge_3D].push_back({supporting_edge, supporting_edge_uncorrected, image_number, edge_idx, rotation, translation, tangents_3D_world});
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
                data.edge_idx,
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

void EdgeMapping::Setup_Data_Parameters( YAML::Node Edge_Sketch_Setting_File ) {
    thresh_EDG    = Edge_Sketch_Setting_File["Multi_Thresh_Final_Thresh"].as<int>();
    Dataset_Path  = Edge_Sketch_Setting_File["Dataset_Path"].as<std::string>();
    Dataset_Name  = Edge_Sketch_Setting_File["Dataset_Name"].as<std::string>();
    Scene_Name    = Edge_Sketch_Setting_File["Scene_Name"].as<std::string>();
    Num_Of_Images = Edge_Sketch_Setting_File["Total_Num_Of_Images"].as<int>();
}

////////////////////////////////////////////////// read curvelets and edgels //////////////////////////////////////////////////
std::vector<EdgeCurvelet> EdgeMapping::read_curvelets() {
    std::vector<EdgeCurvelet> all_curvelets;
    
    // Initialize file_reader if needed
    if (!file_reader_ptr) {      
        file_reader_ptr = std::make_shared<file_reader>(
            Dataset_Path, Dataset_Name, Scene_Name, Num_Of_Images);
    }
    
    // Read all curvelets
    file_reader_ptr->read_All_Curvelets(all_curvelets, thresh_EDG);
    
    // std::cout << "Read " << all_curvelets.size() << " curvelets" << std::endl;
    return all_curvelets;
}

std::vector<Eigen::MatrixXd> EdgeMapping::read_edgels() {
    std::vector<Eigen::MatrixXd> all_edgels;
    
    // Initialize file_reader if needed
    if (!file_reader_ptr) {
        file_reader_ptr = std::make_shared<file_reader>(
            Dataset_Path, Dataset_Name, Scene_Name, Num_Of_Images);
    }
    
    // Read all edgels
    file_reader_ptr->read_All_Edgels(all_edgels, thresh_EDG);
    
    // std::cout << "Read " << all_edgels.size() << " edgel files" << std::endl;
    return all_edgels;
}
////////////////////////////////////////////////// read curvelets and edgels //////////////////////////////////////////////////

///////////////////////// Create the 3D edge weight graph /////////////////////////
std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::build3DEdgeWeightedGraph(const std::unordered_map<EdgeMapping::Uncorrected2DEdgeKey, std::vector<EdgeMapping::Uncorrected2DEdgeMappingData>, EdgeMapping::HashUncorrected2DEdgeKey>& uncorrected_map,
                                      const std::vector<Eigen::MatrixXd> All_Edgels, std::vector<EdgeCurvelet> all_curvelets,
                                      const std::vector<Eigen::Matrix3d> All_R, const std::vector<Eigen::Vector3d> All_T) {

    std::unordered_map<
    std::pair<Eigen::Vector3d, Eigen::Vector3d>,
    int,
    HashEigenVector3dPair,
    FuzzyVector3dPairEqual
    > graph;


    //std::cout << "Number of curvelets: " << all_curvelets.size() << std::endl;
    int skipped_curvelets = 0;

    for (const auto& curvelet : all_curvelets) {
        int image_number = curvelet.image_number;
        int edge_idx = curvelet.self_edge_index;
        int neighbor_idx = curvelet.neighbor_edge_index;

        if (image_number >= All_Edgels.size()) {
            //std::cout << "Skipping curvelet: image_number " << image_number << " is out of bounds (All_Edgels size: " << All_Edgels.size() << ")" << std::endl;
            //skipped_curvelets++;
            continue;
        }
    
        Eigen::Vector3d edge_uncorrected = All_Edgels[image_number].row(edge_idx).head<3>();
        Eigen::Vector3d neighbor_uncorrected = All_Edgels[image_number].row(neighbor_idx).head<3>();

        // Create keys for looking up in uncorrected_map
        Uncorrected2DEdgeKey key_edge{
            edge_uncorrected,
            image_number,
            edge_idx,
            All_R[image_number],
            All_T[image_number]
        };
        
        Uncorrected2DEdgeKey key_neighbor{
            neighbor_uncorrected,
            image_number,
            neighbor_idx,
            All_R[image_number],
            All_T[image_number]
        };

        auto it_edge = uncorrected_map.find(key_edge);
        auto it_neighbor = uncorrected_map.find(key_neighbor);
        std::vector<Uncorrected2DEdgeMappingData> records_all;

        if (it_edge != uncorrected_map.end()) {
            records_all.insert(records_all.end(), it_edge->second.begin(), it_edge->second.end());
        }
        
        if (it_neighbor != uncorrected_map.end()) {
            records_all.insert(records_all.end(), it_neighbor->second.begin(), it_neighbor->second.end());
        }
    
        if (records_all.size() < 2) {
            //std::cout << "Skipping curvelet: insufficient records (records_all size: " << records_all.size() << ")" << std::endl;
            continue;
        }

        // for all 3d edges that have connected 2d edges
        for (int i = 0; i < records_all.size()-1; ++i) {
            for (int j = i+1; j < records_all.size(); ++j){
                Eigen::Vector3d a = records_all[i].edge_3D;
                Eigen::Vector3d b = records_all[j].edge_3D;

                // Enforce consistent ordering for pair hashing
                if ((a.x() > b.x()) || (a.x() == b.x() && a.y() > b.y()) || 
                    (a.x() == b.x() && a.y() == b.y() && a.z() > b.z())) {
                    std::swap(a, b);
                }

                std::pair<Eigen::Vector3d, Eigen::Vector3d> edge_pair = {a, b};
                if (a.x()!=b.x() && a.y()!=b.y() && a.z()!=b.z()){
                    graph[edge_pair]++;
                }
                
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

std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::pruneEdgeGraph_by_3DProximityAndOrientation(std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph ){
#if WRITE_3D_EDGE_GRAPH
    //> write the 3D edge graph before pruning
    write_edge_graph( graph, "3D_edge_graph" );
#endif

    std::vector<double> normal_distances;
    std::vector<double> tangential_distances;
    std::vector<double> angles;
    std::vector<int> edge_link_indx;
    std::vector<int> invalid_link_indx;
    std::unordered_map< std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual> pruned_graph;

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

        //> Compute the orientation between the two (in degrees);
        double dot_product = fabs(t1.dot(t2));
        // Clamp to [-1.0, 1.0] to avoid nan values
        dot_product = std::min(1.0, std::max(-1.0, dot_product));
        angles.push_back(std::acos(dot_product) * 180/PI);

        //> Normal distance
        Eigen::Vector3d closest_point = util->findClosestVectorFromPointToLine(p2, t2, p1);
        normal_distances.push_back( closest_point.norm() );
        
        //> Tangential distance
        Eigen::Vector3d projected_p2 = p2 + closest_point;
        tangential_distances.push_back((p1 - p2).norm());

        edge_link_indx.push_back(edge_link_counter);
        edge_link_counter++;
    }

    auto mean_std = [](const std::vector<double>& values) -> std::pair<double, double> {
        if (values.empty()) return {0.0, 0.0};
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sq_sum = std::accumulate(values.begin(), values.end(), 0.0, [mean](double acc, double val) { return acc + (val - mean) * (val - mean); });
        double stddev = std::sqrt(sq_sum / values.size());
        return {mean, stddev};
    };

    auto [mu_t, sigma_t] = mean_std(tangential_distances);
    auto [mu_n, sigma_n] = mean_std(normal_distances);
    auto [mu_theta, sigma_theta] = mean_std(angles);
    
    std::cout << "[GRAPH STATS BEFORE PRUNING]\n";
    std::cout << "μ_t     (meters) : " << mu_t      << ", σ_t:     " << sigma_t << "\n";
    std::cout << "μ_n     (meters) : " << mu_n      << ", σ_n:     " << sigma_n << "\n";
    std::cout << "μ_theta (degrees): " << mu_theta  << ", σ_theta: " << sigma_theta << "\n";

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
        double proximity_diff = normal_distances[index];
        double tangential_dist_fiff = tangential_distances[index];

        access_index++;
        edge_link_counter++;

        // if (angle_diff < mu_theta + sigma_theta*PRUNE_3D_EDGE_GRAPH_LAMBDA2 ) { //
        //     pruned_graph[pair]++;
        // }
        // if ((proximity_diff < mu_n + sigma_n*PRUNE_3D_EDGE_GRAPH_LAMBDA1 
        //     || tangential_dist_fiff < mu_t + sigma_t*PRUNE_3D_EDGE_GRAPH_LAMBDA3) ) { //
        //     pruned_graph[pair]++;
        // }
        if ( proximity_diff < mu_n + sigma_n*PRUNE_3D_EDGE_GRAPH_LAMBDA1 && tangential_dist_fiff < mu_t + sigma_t*PRUNE_3D_EDGE_GRAPH_LAMBDA3 && \
             angle_diff < mu_theta + sigma_theta*PRUNE_3D_EDGE_GRAPH_LAMBDA2 ) { //
            pruned_graph[pair]++;
        }
    }

#if WRITE_3D_EDGE_GRAPH
    //> write the 3D edge graph after pruning
    write_edge_graph( pruned_graph, "3D_edge_pruned_graph" );
#endif

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

    std::cout << "Begin of edge graph pruning by projections ..." << std::endl;

    for (const auto& [pair, weight] : graph) {

        //> Find the locations and tangents of the 3D edge pair
        const Eigen::Vector3d& Edge3D_1_Location = pair.first;
        const Eigen::Vector3d& Edge3D_2_Location = pair.second;

        auto it1 = edge_3D_to_supporting_edges.find(Edge3D_1_Location);
        auto it2 = edge_3D_to_supporting_edges.find(Edge3D_2_Location);

        //> Skip if the 3D edge does not exist in the structure edge_3D_to_supporting_edges (is it possible?)
        if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end()) continue;

        // Eigen::Vector3d unit_Tangent3D_1 = it1->second.front().tangents_3D_world.normalized();
        // Eigen::Vector3d unit_Tangent3D_2 = it2->second.front().tangents_3D_world.normalized();

        bool prune_flag = false;
        // double ore_diff_threshold = cos(double(PRUNE_BY_PROJ_ORIE_THRESH)/180*PI);
        
        //> Project to all views 
        for (unsigned vi = 0; vi < Num_Of_Total_Imgs; vi++) {

            //> Project 3D edge point to the 2D image
            Eigen::Vector3d proj_edge_1_location = K * (All_R[vi] * Edge3D_1_Location + All_T[vi]);
            Eigen::Vector3d proj_edge_2_location = K * (All_R[vi] * Edge3D_2_Location + All_T[vi]);
            proj_edge_1_location = util->getNormalizedProjectedPoint( proj_edge_1_location );
            proj_edge_2_location = util->getNormalizedProjectedPoint( proj_edge_2_location );

            //> Project 3D tangent to the 2D image
            // Eigen::Vector3d proj_tangent_1 = util->project_3DTangent_to_Image(All_R[vi], K, unit_Tangent3D_1, proj_edge_1_location);
            // Eigen::Vector3d proj_tangent_2 = util->project_3DTangent_to_Image(All_R[vi], K, unit_Tangent3D_2, proj_edge_2_location);

            //> Rule out if the distance between the projected 3D edge points are over a threshold
            prune_flag = ((proj_edge_1_location - proj_edge_2_location).norm() > PRUNE_BY_PROJ_PROX_THRESH) ? (true) : (false);

            if (prune_flag) break;

            //> Rule out if the orientation difference is over a threshold
            // double abs_dot_prod = fabs(proj_tangent_1(0)*proj_tangent_2(0) + proj_tangent_1(1)*proj_tangent_2(1));
            // prune_flag = (abs_dot_prod < ore_diff_threshold) ? (true) : (false);
        }

        if (!prune_flag) 
            pruned_graph_by_proj[pair]++;
    }

#if WRITE_3D_EDGE_GRAPH
    //> write the 3D edge graph after pruning by projection
    write_edge_graph( pruned_graph_by_proj, "3D_edge_pruned_graph_by_proj" );
#endif
    return pruned_graph_by_proj;
}
///////////////////////// End of pruning weighted graph by projecting to all views /////////////////////



///////////////////////// build a map of 3d edges with its neighbors /////////////////////////
EdgeMapping::EdgeNodeList EdgeMapping::buildEdgeNodeGraph(const std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int,
                                                            HashEigenVector3dPair, FuzzyVector3dPairEqual>& pruned_graph) {
    
    // Create node list 
    EdgeNodeList nodes;
    std::unordered_map<Eigen::Vector3d, std::pair<EdgeNode*, int>, HashEigenVector3d, FuzzyVector3dEqual> node_map;
    int node_index = 0;

    // Create node objects 
    for (const auto& [edge, support_list] : edge_3D_to_supporting_edges) {
        std::unique_ptr<EdgeNode> node(new EdgeNode());
        node->location = edge;
        node->has_orientation_fixed_in_connectivity_graph = false;

        Eigen::Vector3d avg_tangent = Eigen::Vector3d::Zero();
        for (const auto& s : support_list) {
            avg_tangent += s.tangents_3D_world;
        }
        if (!support_list.empty()) {
            avg_tangent.normalize();
        }
        node->orientation = avg_tangent;
        node->index = node_index;

        node_map[edge] = std::make_pair(node.get(), node_index);
        nodes.push_back(std::move(node));
        node_index++;
    }

    // Fill neighbor links
    for (const auto& [pair, _] : pruned_graph) {
        auto a_it = node_map.find(pair.first);
        auto b_it = node_map.find(pair.second);
        if (a_it == node_map.end() || b_it == node_map.end()) continue;

        EdgeNode* a = a_it->second.first;
        EdgeNode* b = b_it->second.first;
        int a_index = a_it->second.second;
        int b_index = b_it->second.second;

        a->neighbors.push_back(std::make_pair(b_index, b));
        b->neighbors.push_back(std::make_pair(a_index, a));
    }

    return nodes;
}
///////////////////////// build a map of 3d edges with its neighbors /////////////////////////

///////////////////////// Smoothing 3d edges with its neighbors /////////////////////////
void EdgeMapping::align3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes) {

    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::ofstream before_out("../../outputs/3D_edges_before_smoothing.txt");

    double step_size_force = INIT_FROCE_STEP_SIZE;
    double step_size_torque = INIT_TORQUE_STEP_SIZE;

    auto mean_std = [](const std::vector<double>& values) -> std::pair<double, double> {
        if (values.empty()) return {0.0, 0.0};
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sq_sum = std::accumulate(values.begin(), values.end(), 0.0, [mean](double acc, double val) { return acc + (val - mean) * (val - mean); });
        double stddev = std::sqrt(sq_sum / values.size());
        return {mean, stddev};
    };

    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        const auto& node = edge_nodes[i];
        before_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
    }
    std::cout << "\n";
    before_out.close();
    std::ofstream after_out("../../outputs/3D_edges_after_smoothing.txt");
    std::cout << "Start aligning edges..." << std::endl;

    for (int iter = 0; iter < NUM_OF_ITERATIONS; ++iter) {
        std::vector<Eigen::Vector3d> new_locations(edge_nodes.size());
        std::vector<Eigen::Vector3d> new_orientations(edge_nodes.size());

        //increase force and torque every 50 iterations
#if ENABLE_EXPO_FORCE_AND_TORQUE
        if(iter % 200 == 0){
            FROCE_STEP_SIZE = step_size_force * EXPO_INCREASE_FACTOR;
            step_size_torque = step_size_torque * EXPO_INCREASE_FACTOR;
        }
#endif

        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            const auto& node = edge_nodes[i];
            if (node->neighbors.empty()) {
                new_locations[i] = node->location;
                new_orientations[i] = node->orientation;
                continue;
            }

            //> Calculate distances to all neighbors for this node
            std::vector<double> distances;
            for (const auto& neighbor_pair : node->neighbors) {
                const EdgeNode* neighbor = neighbor_pair.second;
                const Eigen::Vector3d& p = neighbor->location;
                const Eigen::Vector3d& B = node->location;
                
                // Calculate Euclidean distance between node and neighbor
                double euclidean_distance = (p - B).norm();
                distances.push_back(euclidean_distance);
            }
            auto [mu_d, sigma_d] = mean_std(distances);

            // If standard deviation is too small, set a minimum value to avoid division issues
            double sigma = std::max(sigma_d, 0.001);
            
            //> Location aligning
            Eigen::Vector3d sum_force = Eigen::Vector3d::Zero();
            //> Orientation aligning
            Eigen::Vector3d sum_euler_angles = Eigen::Vector3d::Zero();

            //> Total weight for normalization
            double total_weight_force = 0.0;
            double total_weight_torque = 0.0;

            // for (const auto& neighbor_pair : node->neighbors) {
            for (size_t j = 0; j < node->neighbors.size(); ++j) {
                const auto& neighbor_pair = node->neighbors[j];
                const EdgeNode* neighbor = neighbor_pair.second;
                const Eigen::Vector3d& p = neighbor->location;
                const Eigen::Vector3d& t_neighbor = neighbor->orientation;
                const Eigen::Vector3d& B = node->location;

                //> Calculate force component (tangential distance)
                Eigen::Vector3d tangential_dist = util->findClosestVectorFromPointToLine(p, t_neighbor, B);

                //> Calculate euler angles for orientation alignment
                Eigen::Vector3d euler_angles = util->getShortestAlignEulerAnglesDegrees(node->orientation, neighbor->orientation);

                //> Calculate weight using Gaussian function based on distance
                double euclidean_distance = distances[j];
                double gaussian_weight = std::exp(-0.5 * std::pow((euclidean_distance - mu_d) / sigma, 2));
                
                //> Apply weighted force and torque
                sum_force += gaussian_weight * tangential_dist;
                sum_euler_angles += gaussian_weight * euler_angles;
                
                //> Accumulate weights for normalization
                total_weight_force += gaussian_weight;
                total_weight_torque += gaussian_weight;
            }

            // Normalize by total weight if it's not zero
            if (total_weight_force > 0) {
                sum_force /= static_cast<double>(node->neighbors.size());
            }
            if (total_weight_torque > 0) {
                sum_euler_angles /= static_cast<double>(node->neighbors.size());
            }

            //> Update edge location
            new_locations[i] = node->location + step_size_force * sum_force;

            //> Update edge orientation
            sum_euler_angles *= step_size_torque;
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

            //> Write the final updated edge locations and orientations to a file
            if(iter == NUM_OF_ITERATIONS-1){
                after_out << node->location.transpose() << " " << node->orientation.transpose() << "\n";
            }
        }
    }

    // Write edge-neighbor pairs to pruned_graph_aligned.txt
    std::ofstream pruned_graph_aligned("../../outputs/pruned_graph_aligned.txt");
    if (!pruned_graph_aligned.is_open()) {
        LOG_ERROR("Could not open pruned_graph_aligned.txt file!");
        return;
    }
    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        const auto& node = edge_nodes[i];
       
        // Skip nodes with no neighbors
        if (node->neighbors.empty()) {
            continue;
        }
       
        // Record all neighbors for this node
        for (const auto& neighbor_pair : node->neighbors) {
            const EdgeNode* neighbor = neighbor_pair.second;
            pruned_graph_aligned << node->location(0) << "\t" << node->location(1) << "\t" << node->location(2) << "\t";
            pruned_graph_aligned << neighbor->location(0) << "\t" << neighbor->location(1) << "\t" << neighbor->location(2) << "\t";
            pruned_graph_aligned << node->orientation(0) << "\t" << node->orientation(1) << "\t" << node->orientation(2) << "\t";
            pruned_graph_aligned << neighbor->orientation(0) << "\t" << neighbor->orientation(1) << "\t" << neighbor->orientation(2) << "\t\n";
        }
    }
    pruned_graph_aligned.close();
    std::cout << "Write all edge-neighbor pairs to pruned_graph_aligned.txt" << std::endl;

    //after_out.close();
    std::string msg = "[ALIGNMENT COMPLETE] Aligned edges written to file after " + std::to_string(NUM_OF_ITERATIONS) + " iterations with force step size " + std::to_string(step_size_force) + " and torque step size " + std::to_string(step_size_torque);
    LOG_GEN_MESG(msg);
}
///////////////////////// Smoothing 3d edges with its neighbors /////////////////////////


////////////////// test //////////////////
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

void EdgeMapping::createConnectivityGraph(EdgeNodeList& edge_nodes) {

    if (!util) {util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());}

    //> Make sure that the connectivity graph is clean
    connectivity_graph.clear();
    
    std::cout<<"Creating connectivity graph for " + std::to_string(edge_nodes.size()) + " edge nodes..."<<std::endl;

    //> Loop over each node in the neighborhood graph
    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        EdgeNode* node = edge_nodes[i].get();

        ConnectivityGraphNode graph_node;
        graph_node.location = node->location;
        graph_node.orientation = node->orientation;
        graph_node.left_neighbor = std::make_pair(-1, Eigen::Vector3d::Zero());  //> Default: no left neighbor
        graph_node.right_neighbor = std::make_pair(-1, Eigen::Vector3d::Zero()); //> Default: no right neighbor
        graph_node.curve_index = -1;                                             //> Default: does not belong to any curve
        
        if (!node->neighbors.empty()) {
            std::vector<std::pair<int, Eigen::Vector3d>> updated_neighbors;
            std::vector<double> proj_neighbor;

            //> First traverse all neighbors and see if there is anyone whose orientation has fixed in its own connectivity graph
            for (const auto& neighbor_pair : node->neighbors) {
                const EdgeNode* neighbor = neighbor_pair.second;
                if ( neighbor->has_orientation_fixed_in_connectivity_graph ) {
                    double factor = (util->checkOrientationConsistency(neighbor->orientation, node->orientation)) ? 1.0 : -1.0;
                    if (factor < 0) { edge_nodes[i]->orientation = -node->orientation; }
                    break;
                }
            }
            node->has_orientation_fixed_in_connectivity_graph = true;
            
            // Process each neighbor
            for (const auto& neighbor_pair : node->neighbors) {
                const int neighbor_index = neighbor_pair.first;
                EdgeNode* neighbor = neighbor_pair.second;
                
                //> Check if the neighbor node needs to flip the orientation
                double factor = (util->checkOrientationConsistency(node->orientation, neighbor->orientation)) ? 1.0 : -1.0;
                if (factor < 0) { edge_nodes[neighbor_index]->orientation = -neighbor->orientation; }
                
                Eigen::Vector3d adjusted_orientation = edge_nodes[neighbor_index]->orientation;
                updated_neighbors.push_back(std::make_pair(neighbor_index, adjusted_orientation));
                
                double line_latent_variable = util->getLineVariable(node->location, node->orientation, neighbor->location);
                proj_neighbor.push_back(line_latent_variable);

                edge_nodes[neighbor_index]->has_orientation_fixed_in_connectivity_graph = true;
            }
            
            // Find left and right neighbors
            int left_idx = -1;
            double min_positive = std::numeric_limits<double>::max();
            
            int right_idx = -1;
            double max_negative = -std::numeric_limits<double>::max();
            
            for (size_t j = 0; j < proj_neighbor.size(); ++j) {
                if (proj_neighbor[j] >= 0 && proj_neighbor[j] < min_positive) {
                    min_positive = proj_neighbor[j];
                    left_idx = j;
                }
                if (proj_neighbor[j] < 0 && proj_neighbor[j] > max_negative) {
                    max_negative = proj_neighbor[j];
                    right_idx = j;
                }
            }
            
            // Update the connectivity graph node with left/right information
            if (left_idx >= 0) {
                int left_neighbor_index = updated_neighbors[left_idx].first;
                Eigen::Vector3d left_neighbor_orientation = updated_neighbors[left_idx].second;
                graph_node.left_neighbor = std::make_pair(left_neighbor_index, left_neighbor_orientation);
                
                // Update the orientation of the left neighbor in the edge_nodes
                if (left_neighbor_index >= 0 && left_neighbor_index < edge_nodes.size()) {
                    edge_nodes[left_neighbor_index]->orientation = left_neighbor_orientation;
                }
            }
            
            if (right_idx >= 0) {
                int right_neighbor_index = updated_neighbors[right_idx].first;
                Eigen::Vector3d right_neighbor_orientation = updated_neighbors[right_idx].second;
                graph_node.right_neighbor = std::make_pair(right_neighbor_index, right_neighbor_orientation);
                
                // Update the orientation of the right neighbor in the edge_nodes
                if (right_neighbor_index >= 0 && right_neighbor_index < edge_nodes.size()) {
                    edge_nodes[right_neighbor_index]->orientation = right_neighbor_orientation;
                }
            }
            
            // Update all neighbor orientations to maintain consistency
            for (const auto& updated_pair : updated_neighbors) {
                int nbr_idx = updated_pair.first;
                if (nbr_idx >= 0 && nbr_idx < edge_nodes.size()) {
                    edge_nodes[nbr_idx]->orientation = updated_pair.second;
                }
            }
        }
        
        // Add the node to the connectivity graph
        connectivity_graph[i] = graph_node;
    }

    writeConnectivityGraphToFile(connectivity_graph, "connectivity_graph");
}

std::vector<EdgeMapping::Curve> EdgeMapping::buildCurvesFromConnectivityGraph( ) 
{
    std::vector<Curve> curves;
    std::unordered_set<int> assigned_edges;
    int curve_count = 0;

    //> Loop until all edges are assigned to a curve
    for (const auto& [node_index, node] : connectivity_graph) {
        //> Skip if this edge is already assigned to a curve
        if (assigned_edges.find(node_index) != assigned_edges.end()) {
            continue;
        }
        
        //> Create a new curve starting with this edge
        Curve curve;
        curve.edge_indices.push_back(node_index);
        assigned_edges.insert(node_index);
        
        //> Forward direction: follow right neighbors until we can't go further
        int current_index = node_index;
        while (true) {
            ConnectivityGraphNode current_node = connectivity_graph.at(current_index);
            int right_neighbor_index = current_node.right_neighbor.first;
            (connectivity_graph.at(current_index)).curve_index = curve_count;

            //> Stop if no right neighbor exists
            if (right_neighbor_index < 0 || connectivity_graph.find(right_neighbor_index) == connectivity_graph.end() ) {
                break;
            }

            //> If the right neighbor is already assigned, mark that the current curve needs to be merged in the later stage
            if (assigned_edges.find(right_neighbor_index) != assigned_edges.end()) {

                //> If the trace loops back to itself
                if (std::find(curve.edge_indices.begin(), curve.edge_indices.end(), right_neighbor_index) != curve.edge_indices.end()) {
                    curve.b_loops_back_on_right = true;
                }
                else {
                    //> If it does not loop back to itself, record the right neighbor index in case there is a need to extend the curve
                    curve.to_be_merged_right_edge_index = right_neighbor_index;
                    curve.to_be_merged_right_curve_index = (connectivity_graph.at(right_neighbor_index)).curve_index;
                }
                break;
            }
            
            //> Mark the right neighbor as assigned
            curve.edge_indices.push_back(right_neighbor_index);
            assigned_edges.insert(right_neighbor_index);

            current_index = right_neighbor_index;
        }
        
        //> Backward direction: follow left neighbors until we can't go further
        current_index = node_index;
        while (true) {
            ConnectivityGraphNode current_node = connectivity_graph.at(current_index);
            int left_neighbor_index = current_node.left_neighbor.first;
            (connectivity_graph.at(current_index)).curve_index = curve_count;
            
            //> Stop if no left neighbor exists
            if (left_neighbor_index < 0 || connectivity_graph.find(left_neighbor_index) == connectivity_graph.end() ) {
                break;
            }

            //> If the left neighbor is already assigned, mark that the current curve needs to be merged in the later stage
            if (assigned_edges.find(left_neighbor_index) != assigned_edges.end()) {

                //> If the trace loops back to itself
                if (std::find(curve.edge_indices.begin(), curve.edge_indices.end(), left_neighbor_index) != curve.edge_indices.end()) {
                    curve.b_loops_back_on_left = true;
                }
                else {
                    //> If it does not loop back to itself, record the left neighbor index in case there is a need to extend the curve
                    curve.to_be_merged_left_edge_index = left_neighbor_index;
                    curve.to_be_merged_left_curve_index = (connectivity_graph.at(left_neighbor_index)).curve_index;
                }
                break;
            }
            
            //> Mark the left neighbor as assigned
            curve.edge_indices.insert(curve.edge_indices.begin(), left_neighbor_index);
            assigned_edges.insert(left_neighbor_index);

            current_index = left_neighbor_index;
        }
        
        curve_count++;
        curves.push_back(curve);
    }

    //> CURVE EXTENSION 
    std::cout << "Starting curves extension ..." << std::endl;
    
    int iteration = 0;
    bool b_extensions_occurred = true;

    bool b_flag = false;
    
    //> Keep iterating until no more extensions can be made
    while (b_extensions_occurred) {
        b_extensions_occurred = false;
        iteration++;
        std::cout << "Extension iteration " << iteration << std::endl;
        
        //> Process each curve and extend it by adding target curves
        for (size_t i = 0; i < curves.size(); ++i) {

            Curve& current_curve = curves[i];

            if (i <= 2) {
                bool cond1 = (!current_curve.b_loops_back_on_left && current_curve.to_be_merged_left_curve_index >= 0 && current_curve.to_be_merged_left_curve_index < curves.size());
                bool cond2 = (!current_curve.b_loops_back_on_right && current_curve.to_be_merged_right_curve_index >= 0 && current_curve.to_be_merged_right_curve_index < curves.size());
                std::cout << "cond1 = " << cond1 << ", cond2 = " << cond2 << std::endl;
                std::cout << "number of edges = " << current_curve.edge_indices.size() << std::endl;
            }
            
            //> Skip if target curve is already absorbed
            if (current_curve.edge_indices.empty()) {
                continue;
            }
            
            //> Extend with "left" target curve
            if (!current_curve.b_loops_back_on_left && current_curve.to_be_merged_left_curve_index >= 0 && current_curve.to_be_merged_left_curve_index < curves.size()) {
                
                int target_idx = current_curve.to_be_merged_left_curve_index;
                Curve& target_curve = curves[target_idx];
                
                //> Skip if target curve is empty (already absorbed) and clear the merge reference since the target is gone
                if (target_curve.edge_indices.empty()) {
                    current_curve.to_be_merged_left_curve_index = -1;
                    current_curve.to_be_merged_left_edge_index = -1;
                }
                else {
                    //> Check if the edge in the target_curve is either the first two or the last two edges
                    if (b_is_in_first_or_last_two(target_curve.edge_indices, current_curve.to_be_merged_left_edge_index)) {
                           
                        //> Get edges from target curve 
                        std::vector<int> edges_to_add;
                        for (size_t j = 0; j < target_curve.edge_indices.size(); ++j) {
                            // if (j != connect_pos) {
                                edges_to_add.push_back(target_curve.edge_indices[j]);
                            // }
                        }
                        
                        //> Check to see if the target curve should be added to the "beginning" or the "end" of the curve
                        Eigen::Vector3d first_current_edge = connectivity_graph.at( current_curve.edge_indices.front() ).location;
                        Eigen::Vector3d last_current_edge  = connectivity_graph.at( current_curve.edge_indices.back() ).location;
                        Eigen::Vector3d first_target_edge  = connectivity_graph.at( target_curve.edge_indices.front() ).location;
                        Eigen::Vector3d last_target_edge   = connectivity_graph.at( target_curve.edge_indices.back() ).location;
                        std::vector<double> link_dists = { (first_current_edge - last_target_edge).norm(), \
                                                            (first_current_edge - first_target_edge).norm(), \
                                                            (last_current_edge - first_target_edge).norm(), \
                                                            (last_current_edge - last_target_edge).norm() };
                        auto min_it = std::min_element(link_dists.begin(), link_dists.end());
                        //> Get the index corresponding to the minimum value
                        int min_index = std::distance(link_dists.begin(), min_it);
                        switch (min_index) {
                            case 0:
                                current_curve.edge_indices.insert(current_curve.edge_indices.begin(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            case 1:
                                //> Reverse the order of the target curve edges and add them to the beginning of current curve
                                std::reverse(edges_to_add.begin(), edges_to_add.end());
                                current_curve.edge_indices.insert(current_curve.edge_indices.begin(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            case 2:
                                //> Add target curve edges to the end of current curve
                                current_curve.edge_indices.insert(current_curve.edge_indices.end(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            case 3:
                                //> Reverse the order of the target curve edges and add them to the end of current curve
                                std::reverse(edges_to_add.begin(), edges_to_add.end());
                                current_curve.edge_indices.insert(current_curve.edge_indices.end(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            default:
                                LOG_ERROR("Something's wrong here...");
                        }

                        // Add target curve edges to the beginning of current curve
                        // current_curve.edge_indices.insert(current_curve.edge_indices.begin(), edges_to_add.begin(), edges_to_add.end());
                        
                        // Inherit target curve's properties for the current curve
                        current_curve.b_loops_back_on_left = target_curve.b_loops_back_on_left;
                        current_curve.to_be_merged_left_edge_index = target_curve.to_be_merged_left_edge_index;
                        current_curve.to_be_merged_left_curve_index = target_curve.to_be_merged_left_curve_index;
                        
                        // Reset the target curve to default values
                        target_curve.edge_indices.clear();
                        target_curve.index = -1;
                        target_curve.b_loops_back_on_left = false;
                        target_curve.b_loops_back_on_right = false;
                        target_curve.to_be_merged_left_edge_index = -1;
                        target_curve.to_be_merged_right_edge_index = -1;
                        target_curve.to_be_merged_left_curve_index = -1;
                        target_curve.to_be_merged_right_curve_index = -1;
                        
                        b_extensions_occurred = true;
                        // std::cout << "  Extended curve " << i << " with curve " << target_idx << " (left extension, added " << edges_to_add.size() << " edges)" << std::endl;
                    }
                }
            }
            
            //> Extend with "right" target curve
            if (!current_curve.b_loops_back_on_right && current_curve.to_be_merged_right_curve_index >= 0 && current_curve.to_be_merged_right_curve_index < curves.size()) {
                
                int target_idx = current_curve.to_be_merged_right_curve_index;
                Curve& target_curve = curves[target_idx];
                
                //> Skip if target curve is already absorbed and clear the merge reference since the target is gone
                if (target_curve.edge_indices.empty()) {
                    current_curve.to_be_merged_right_curve_index = -1;
                    current_curve.to_be_merged_right_edge_index = -1;
                    // continue;
                }
                else {
                    //> Check if the edge in the target_curve is either the first two or the last two edges
                    if (b_is_in_first_or_last_two(target_curve.edge_indices, current_curve.to_be_merged_right_edge_index)) {
                            
                        //> Get edges from target curve 
                        std::vector<int> edges_to_add;
                        for (size_t j = 0; j < target_curve.edge_indices.size(); ++j) {
                            // if (j != connect_pos) {
                                edges_to_add.push_back(target_curve.edge_indices[j]);
                            // }
                        }
                        
                        //> Check to see if the target curve should be added to the "beginning" or the "end" of the curve
                        Eigen::Vector3d first_current_edge = connectivity_graph.at( current_curve.edge_indices.front() ).location;
                        Eigen::Vector3d last_current_edge  = connectivity_graph.at( current_curve.edge_indices.back() ).location;
                        Eigen::Vector3d first_target_edge  = connectivity_graph.at( target_curve.edge_indices.front() ).location;
                        Eigen::Vector3d last_target_edge   = connectivity_graph.at( target_curve.edge_indices.back() ).location;
                        std::vector<double> link_dists = { (first_current_edge - last_target_edge).norm(), \
                                                            (first_current_edge - first_target_edge).norm(), \
                                                            (last_current_edge - first_target_edge).norm(), \
                                                            (last_current_edge - last_target_edge).norm() };
                        auto min_it = std::min_element(link_dists.begin(), link_dists.end());
                        //> Get the index corresponding to the minimum value
                        int min_index = std::distance(link_dists.begin(), min_it);
                        switch (min_index) {
                            case 0:
                                //> Add target curve edges to the beginning of current curve
                                current_curve.edge_indices.insert(current_curve.edge_indices.begin(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            case 1:
                                //> Reverse the order of the target curve edges and add them to the beginning of current curve
                                std::reverse(edges_to_add.begin(), edges_to_add.end());
                                current_curve.edge_indices.insert(current_curve.edge_indices.begin(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            case 2:
                                //> Add target curve edges to the end of current curve
                                current_curve.edge_indices.insert(current_curve.edge_indices.end(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            case 3:
                                //> Reverse the order of the target curve edges and add them to the end of current curve
                                std::reverse(edges_to_add.begin(), edges_to_add.end());
                                current_curve.edge_indices.insert(current_curve.edge_indices.end(), edges_to_add.begin(), edges_to_add.end());
                                break;
                            default:
                                LOG_ERROR("Something's wrong here...");
                        }

                        //> Add target curve edges to the end of current curve
                        // current_curve.edge_indices.insert(current_curve.edge_indices.end(), edges_to_add.begin(), edges_to_add.end());
                        
                        //> Inherit target curve's right-side properties for the current curve
                        current_curve.b_loops_back_on_right = target_curve.b_loops_back_on_right;
                        current_curve.to_be_merged_right_edge_index = target_curve.to_be_merged_right_edge_index;
                        current_curve.to_be_merged_right_curve_index = target_curve.to_be_merged_right_curve_index;
                        
                        //> Reset the target curve to default values
                        target_curve.edge_indices.clear();
                        target_curve.index = -1;
                        target_curve.b_loops_back_on_left = false;
                        target_curve.b_loops_back_on_right = false;
                        target_curve.to_be_merged_left_edge_index = -1;
                        target_curve.to_be_merged_right_edge_index = -1;
                        target_curve.to_be_merged_left_curve_index = -1;
                        target_curve.to_be_merged_right_curve_index = -1;
                        
                        b_extensions_occurred = true;
                        //std::cout << "  Extended curve " << i << " with curve " << target_idx << " (right extension, added " << edges_to_add.size() << " edges)" << std::endl;
                    }
                }
            }
        }
        
        if (!b_extensions_occurred) {
            //std::cout << "No more extensions possible. Completed after " << iteration << " iterations." << std::endl;
        }
    }
    
    //> Remove empty curves (those that were absorbed during extensions)
    // curves.erase(std::remove_if(curves.begin(), curves.end(),[](const Curve& curve) { return curve.edge_indices.empty(); }), curves.end());
    
    // Create final vector with only non-empty curves
    std::vector<Curve> final_curves;
    for (const auto& curve : curves) {
        if (!curve.edge_indices.empty()) {
            final_curves.push_back(curve);
        }
    }
    
    std::cout << "Complete tracing 3D edges to form 3D curves" << std::endl;
    return final_curves;
}

void EdgeMapping::findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d> all_R,
                                           const std::vector<Eigen::Vector3d> all_T,
                                           const Eigen::Matrix3d K,
                                           const int Num_Of_Total_Imgs) 
{ 
    // convert 3D-> 2D relationship to 2D->3D relationship
    auto uncorrected_map = map_Uncorrected2DEdge_To_SupportingData();
    // {3D edge 1, 3D edge 2} -> weight (how many 2d edges they have in common)
    
    std::vector<EdgeCurvelet> all_curvelets = read_curvelets();
    std::vector<Eigen::MatrixXd> all_edgels = read_edgels();

    auto graph = build3DEdgeWeightedGraph(uncorrected_map, all_edgels, all_curvelets, all_R, all_T);//change datapath used in this function is not using object 0006
    auto pruned_graph = pruneEdgeGraph_by_3DProximityAndOrientation(graph);
    auto pruned_graph_by_projection = pruneEdgeGraphbyProjections(pruned_graph, all_R, all_T, K, Num_Of_Total_Imgs);

    // crate 3d -> its neighbor data structure
    auto neighbor_map = buildEdgeNodeGraph(pruned_graph_by_projection);

    //> Edge alignment
    align3DEdgesUsingEdgeNodes(neighbor_map);

    //> Construct edge connectivity graph
    createConnectivityGraph(neighbor_map);

    //> Trace edges to form curves using edge connectivity graph
    auto curves = buildCurvesFromConnectivityGraph();

    //> Write curve data to a file
    writeCurvesToFile(curves, "curves_from_connectivity_graph", true);
    writeCurvesToFile(curves, Scene_Name, false);
}

void EdgeMapping::writeConnectivityGraphToFile(const ConnectivityGraph& graph, const std::string& file_name) {
    std::string file_path = "../../outputs/" + file_name + ".txt";
    std::ofstream outfile(file_path);
    if (!outfile.is_open()) return; 
    
    outfile << "# Connectivity Graph\n";
    outfile << "# Format: NodeIndex Location(x,y,z); LeftNeighborIndex LeftOrientation(x,y,z); RightNeighborIndex RightOrientation(x,y,z)\n";
    
    for (const auto& pair : graph) {
        int node_index = pair.first;
        const ConnectivityGraphNode& node = pair.second;
        
        outfile << node_index << " "
                << node.location.x() << " " << node.location.y() << " " << node.location.z() << " ;"
                << node.orientation.x() << " " << node.orientation.y() << " " << node.orientation.z() << " "
                << node.left_neighbor.first << " ";
        
        if (node.left_neighbor.first >= 0) {
            outfile << node.left_neighbor.second.x() << " " << node.left_neighbor.second.y() << " " << node.left_neighbor.second.z() << " ;";
        } else {
            outfile << "0 0 0 ";
        }
        
        outfile << node.right_neighbor.first << " ";
        
        if (node.right_neighbor.first >= 0) {
            outfile << node.right_neighbor.second.x() << " " << node.right_neighbor.second.y() << " " << node.right_neighbor.second.z();
        } else {
            outfile << "0 0 0";
        }
        outfile << "\n";
    }
    outfile.close();
}

void EdgeMapping::writeCurvesToFile(const std::vector<Curve>& curves, 
                                   const std::string& file_name, bool b_write_curve_info) {
    std::string file_path = "../../outputs/" + file_name + ".txt";
    std::ofstream outfile(file_path);
    
    if (!outfile.is_open()) {
        std::string err_msg = "Cannot open " + file_path + " for writing.";
        LOG_ERROR(err_msg);
        return;
    }
    
    if (b_write_curve_info) {
        outfile << "# Curves from Connectivity Graph\n";
        outfile << "# Format: CurveID 3DEdgeIndex Location(x,y,z) Orientation(x,y,z)\n";
    }
    
    for (size_t curve_id = 0; curve_id < curves.size(); ++curve_id) {
        const auto& curve = curves[curve_id];
        
        for (const auto& node_index : curve.edge_indices) {
            const auto& node = connectivity_graph.at(node_index);

            if (b_write_curve_info) {
                outfile << curve_id << " " << node_index << " " << (node.location).transpose() << " " << node.orientation.transpose();
                outfile << " " << curve.b_loops_back_on_left << " " << curve.b_loops_back_on_right;
                outfile << " " << curve.to_be_merged_left_edge_index << " " << curve.to_be_merged_right_edge_index;
                outfile << " " << curve.to_be_merged_left_curve_index << " " << curve.to_be_merged_right_curve_index << "\n";
            }
            else {
                outfile << node.location.x() << " " << node.location.y() << " " << node.location.z() << "\n";
            }
        }
        
        // Add a blank line between curves for better visualization
        if (b_write_curve_info) outfile << "\n";
    }
    
    outfile.close();
    std::cout << "Wrote " << curves.size() << " curves to " << file_path << std::endl;
}

//> given a vector, check if the input num is the first or last two of that vec
bool EdgeMapping::b_is_in_first_or_last_two(const std::vector<int>& vec, int num) {
    if (vec.empty()) {
        return false;
    }

    if (vec.size() == 1) {
        return vec[0] == num; // Only one element, check if it matches
    }

    //> Check first two elements
    if (vec[0] == num || vec[1] == num) {
        return true;
    }

    //> Check last two elements (for vectors with size >= 2)
    //> Ensure there are at least two elements to avoid out-of-bounds access
    if (vec.size() >= 2 && (vec[vec.size() - 1] == num || vec[vec.size() - 2] == num)) {
        return true;
    }

    return false;
}
