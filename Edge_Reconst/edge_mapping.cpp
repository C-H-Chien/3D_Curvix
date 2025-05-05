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
    Eigen::Vector3d target_edge1(0.499087, -0.00374706, 0.556089);
    Eigen::Vector3d target_edge2(0.499348, -0.00377749, 0.554009);

    for (const auto& [edge_3D, support_list] : edge_3D_to_supporting_edges) {

        // bool is_target = false;
        // if (edge_3D.isApprox(target_edge1, 1e-6) || edge_3D.isApprox(target_edge2, 1e-6)) {
        //     is_target = true;
        //     std::cout << "Found target 3D edge: (" << edge_3D.x() << ", " << edge_3D.y() << ", " << edge_3D.z() << ")" << std::endl;
        //     std::cout << "Corresponding 2D edges:" << std::endl;
        // }
        
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

            // if (is_target) {
            //     std::cout << "  Uncorrected 2D edge: (" 
            //               << data.edge_uncorrected.x() << ", " 
            //               << data.edge_uncorrected.y() << ", " 
            //               << data.edge_uncorrected.z() << ")" << std::endl;
            //     std::cout << "  Corrected 2D edge: (" 
            //               << data.edge.x() << ", " 
            //               << data.edge.y() << ")" << std::endl;
            //     std::cout << "  Image number: " << data.image_number << std::endl;
            //     std::cout << "  ---------------------" << std::endl;
            // }

            map[key].push_back(record);
        }
    }
    return map;
}
///////////////////////// convert 3d->2d relationship to 2d uncorrect -> 3d /////////////////////////


////////////////////////////////////////////////// read curvelets and edgels //////////////////////////////////////////////////
std::vector<EdgeCurvelet> EdgeMapping::read_curvelets(int thresh_EDG) {
    std::vector<EdgeCurvelet> all_curvelets;
    
    // Initialize file_reader if needed
    if (!file_reader_ptr) {
        // You'll need to provide appropriate parameters for your dataset
        std::string dataset_path = "/gpfs/data/bkimia/Datasets/";
        std::string dataset_name = "ABC-NEF"; // Replace with your dataset name
        std::string scene_name = "00000006";     // Replace with your scene name
        int num_images = 50;                      // Replace with your number of images
        
        file_reader_ptr = std::make_shared<file_reader>(
            dataset_path, dataset_name, scene_name, num_images);
    }
    
    // Read all curvelets
    file_reader_ptr->read_All_Curvelets(all_curvelets, thresh_EDG);
    
    std::cout << "Read " << all_curvelets.size() << " curvelets" << std::endl;
    return all_curvelets;
}

std::vector<Eigen::MatrixXd> EdgeMapping::read_edgels(int thresh_EDG) {
    std::vector<Eigen::MatrixXd> all_edgels;
    
    // Initialize file_reader if needed
    if (!file_reader_ptr) {
        // You'll need to provide appropriate parameters for your dataset
        std::string dataset_path = "/gpfs/data/bkimia/Datasets/";
        std::string dataset_name = "ABC-NEF"; // Replace with your dataset name
        std::string scene_name = "00000006";     // Replace with your scene name
        int num_images = 50;                      // Replace with your number of images
        
        file_reader_ptr = std::make_shared<file_reader>(
            dataset_path, dataset_name, scene_name, num_images);
    }
    
    // Read all edgels
    file_reader_ptr->read_All_Edgels(all_edgels, thresh_EDG);
    
    std::cout << "Read " << all_edgels.size() << " edgel files" << std::endl;
    
    // // Debug: print information about edgels
    // for (size_t i = 0; i < all_edgels.size() && i < 5; ++i) {
    //     std::cout << "Image " << i << " has " << all_edgels[i].rows() 
    //               << " edges with " << all_edgels[i].cols() << " dimensions per edge" << std::endl;
        
    //     if (all_edgels[i].rows() > 0) {
    //         std::cout << "  First edge: " << all_edgels[i].row(0) << std::endl;
    //     }
    // }
    
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

        //std::cout<<"size of records_all is: "<<records_all.size()<<std::endl;

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

std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::pruneEdgeGraph_by_3DProximityAndOrientation(std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph ){
    //> write the 3D edge graph before pruning
    write_edge_graph( graph, "3D_edge_graph" );

    std::vector<double> distances;
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

        //> Compute the orientation between the two (in degrees)
        angles.push_back( std::acos(fabs(t1.dot(t2)))*180/PI );

        //> compute the distance between the two
        //> (here I find the normal distance rather than the Euclidean distance)
        Eigen::Vector3d closest_point = util->findClosestVectorFromPointToLine(p2, t2, p1);
        distances.push_back( closest_point.norm() );
        
        //tangential distance
        Eigen::Vector3d projected_p2 = p2 + closest_point;
        tangential_distances.push_back((p1 - p2).norm());
        //std::cout<<"p2: "<<p2.transpose()<<

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
    auto [mu3, sigma3] = mean_std(tangential_distances);

    std::cout << "[GRAPH STATS BEFORE PRUNING]\n";
    std::cout << "μ1 (meters): " << mu1 << ", σ1: " << sigma1 << "\n";
    std::cout << "μ3 (meters): " << mu3 << ", σ3: " << sigma3 << "\n";
    std::cout << "μ2 (degrees): " << mu2 << ", σ2: " << sigma2 << "\n";

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
        double tangential_dist_fiff = tangential_distances[index];

        access_index++;
        edge_link_counter++;

        if ((proximity_diff < mu1 + sigma1*PRUNE_3D_EDGE_GRAPH_LAMBDA1 && angle_diff < mu2 + sigma2*PRUNE_3D_EDGE_GRAPH_LAMBDA2 
            && tangential_dist_fiff < mu3 + sigma3*PRUNE_3D_EDGE_GRAPH_LAMBDA3) || weight > 1) { //
            pruned_graph[pair]++;
        }
    }

    //> write the 3D edge graph after pruning
    write_edge_graph( pruned_graph, "3D_edge_pruned_graph" );
    std::cout << "Original graph size: " << graph.size() << std::endl;
    std::cout << "Pruned graph size: " << pruned_graph.size() << std::endl;

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
    Eigen::Vector3d target(0.892454, 0.232254, 0.356584);

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
            
            if (Edge3D_1_Location.isApprox(target, 1e-6) && prune_flag) {
                std::cout<<"location prune, neighbor: ("<<Edge3D_2_Location.transpose()<<") at image #"<<vi<<std::endl;
                std::cout<<"proj_edge_1_location is "<<proj_edge_1_location.transpose()<<"; proj_edge_2_location is "<<proj_edge_2_location.transpose()<<std::endl;
            }

            if (prune_flag) break;

            // std::cout << "Projected tangents: (" << proj_tangent_1(0) << ", " << proj_tangent_1(1) << ", " << proj_tangent_1(2) << "), (" \
            //           << proj_tangent_2(0) << ", " << proj_tangent_2(1) << ", " << proj_tangent_2(2) << ")" << std::endl;

            //> Rule out if the orientation difference is over a threshold
            //> TODO: make the threhold as a Macro
            double abs_dot_prod = fabs(proj_tangent_1(0)*proj_tangent_2(0) + proj_tangent_1(1)*proj_tangent_2(1));
            prune_flag = (abs_dot_prod < ore_diff_threshold) ? (true) : (false);

            if (Edge3D_1_Location.isApprox(target, 1e-6) && prune_flag) {
                std::cout<<"orientation prune, neighbor: ("<<Edge3D_2_Location.transpose()<<") at image #"<<vi<<std::endl;
                std::cout<<"proj_edge_1_location is "<<proj_edge_1_location.transpose()<<"; proj_edge_2_location is "<<proj_edge_2_location.transpose()<<std::endl;
                std::cout<<"proj_tangent_1 is "<<proj_tangent_1.transpose()<<"; proj_tangent_2 is "<<proj_tangent_2.transpose()<<std::endl;
            }

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
void EdgeMapping::align3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes, int iterations, double step_size_force, double step_size_torque) {

    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::ofstream before_out("../../outputs/3D_edges_before_smoothing.txt");


    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        const auto& node = edge_nodes[i];
        before_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
        // if (target_indices.count(i)){
        //     before_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
        // }
    }
    std::cout << "\n";
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

            //> Location aligning
            Eigen::Vector3d sum_force = Eigen::Vector3d::Zero();
            for (const auto& neighbor_pair : node->neighbors) {
                const EdgeNode* neighbor = neighbor_pair.second;
                const Eigen::Vector3d& p = neighbor->location;
                const Eigen::Vector3d& t_neighbor = neighbor->orientation;
                const Eigen::Vector3d& B = node->location;

                Eigen::Vector3d tangential_dist = util->findClosestVectorFromPointToLine(p, t_neighbor, B);
                sum_force += tangential_dist;
            }
            sum_force /= static_cast<double>(node->neighbors.size());
            
            //> Update edge location
            new_locations[i] = node->location + step_size_force * sum_force;

            //> Orientation aligning
            Eigen::Vector3d sum_tangent = Eigen::Vector3d::Zero();
            Eigen::Vector3d sum_euler_angles = Eigen::Vector3d::Zero();
            for (const auto& neighbor_pair : node->neighbors) {
                const EdgeNode* neighbor = neighbor_pair.second;
                Eigen::Vector3d euler_angles = util->getShortestAlignEulerAnglesDegrees(node->orientation, neighbor->orientation);
                sum_euler_angles += euler_angles;
            }
            sum_euler_angles /= static_cast<double>(node->neighbors.size());
            sum_euler_angles *= step_size_torque;
            //> convert from degrees to radians
            sum_euler_angles = sum_euler_angles * M_PI / 180.0;
            Eigen::Matrix3d R_align = util->euler_to_rotation_matrix(sum_euler_angles(0), sum_euler_angles(1), sum_euler_angles(2));

            //> Update edge orientation
            new_orientations[i] = R_align * node->orientation;
        }

        //> Update all edge locations and orientations
        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            edge_nodes[i]->location = new_locations[i];
            edge_nodes[i]->orientation = new_orientations[i];
            const auto& node = edge_nodes[i];

            if(iter == iterations-1){
                after_out << node->location.transpose() << " " << node->orientation.transpose() << "\n";
                // if (target_indices.count(i)){
                //     //after_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
                //     //std::cout << node->location.transpose()<<" "<<node->orientation.transpose()<< "; "<<std::endl;
                // }
            }
        }
        //std::cout << "\n";
    }

    ///////////////////// find neighbor connection //////////////////////////////////
    std::ofstream neighbor_file_after("../../outputs/target_indices_neighbors_after.txt");
    // Eigen::Vector3d target_after_1(0.350545, 0.0919067, 0.34939);
    // Eigen::Vector3d target_after_2(0.349817, 0.0923272, 0.349419);
    // for (auto i : target_indices) {
    //     if (i >= static_cast<int>(edge_nodes.size())) continue; 
    //     const auto& node = edge_nodes[i];
    //     if (node->location.isApprox(target_after_1)) {
    //         std::cout << "aligned index for 1 is: " << i << std::endl;
    //     }
    //     if (node->location.isApprox(target_after_2)) {
    //         std::cout << "aligned index for 2 is: " << i << std::endl;
    //     }
    //     // std::cout << "neighbors of " << node->location.transpose() << " after alignment are: " << std::endl;
    //     // for (const auto& neighbor_pair : node->neighbors) {
    //     //     int neighbor_idx = neighbor_pair.first;
    //     //     const EdgeNode* neighbor = neighbor_pair.second;
    //     //     std::cout << neighbor->location.transpose() << " (Index: " << neighbor_idx << ")" << std::endl;
    //     // }
    // }
    neighbor_file_after.close();
    ///////////////////// find neighbor connection //////////////////////////////////

    //after_out.close();
    std::string msg = "[ALIGNMENT COMPLETE] Aligned edges written to file after " + std::to_string(iterations) + " iterations with force step size " + std::to_string(step_size_force) + " and torque step size " + std::to_string(step_size_torque); 
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

EdgeMapping::ConnectivityGraph EdgeMapping::createConnectivityGraph(EdgeNodeList& edge_nodes) {

    ConnectivityGraph connectivity_graph;

    if (!util) {util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());}
    
    std::cout<<"Creating connectivity graph for " + std::to_string(edge_nodes.size()) + " edge nodes..."<<std::endl;

    Eigen::Vector3d target1(0.355401, 0.0890953, 0.349353);
    Eigen::Vector3d target2(0.354278, 0.0897453, 0.349358);
    Eigen::Vector3d target3(0.35297, 0.0905021, 0.349363); //11681
    
    // for each node in the graph
    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        EdgeNode* node = edge_nodes[i].get();

        ConnectivityGraphNode graph_node;
        graph_node.location = node->location;
        graph_node.orientation = node->orientation;
        graph_node.left_neighbor = std::make_pair(-1, Eigen::Vector3d::Zero());  // Default: no left neighbor
        graph_node.right_neighbor = std::make_pair(-1, Eigen::Vector3d::Zero()); // Default: no right neighbor

        bool is_target1 = node->location.isApprox(target1, 1e-6);
        bool is_target2 = node->location.isApprox(target2, 1e-6);
        bool is_target3 = node->location.isApprox(target3, 1e-6);
        
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
                
                Eigen::Vector3d adjusted_orientation = neighbor->orientation;
                updated_neighbors.push_back(std::make_pair(neighbor_index, adjusted_orientation));
                
                double line_latent_variable = util->getLineVariable(node->location, node->orientation, neighbor->location);
                proj_neighbor.push_back(line_latent_variable);

                neighbor->has_orientation_fixed_in_connectivity_graph = true;
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

            if (is_target1 || is_target2 || is_target3) {

                if(is_target1){std::cout << "Found target1: " << target1.transpose() << " at node index " << i << std::endl;}
                if(is_target2){std::cout << "Found target2: " << target2.transpose() << " at node index " << i << std::endl;}
                if(is_target3){std::cout << "Found target3: " << target3.transpose() << " at node index " << i << std::endl;}

                std::cout<<"orientation is: "<<(graph_node.orientation).transpose()<<std::endl;
                std::cout<<"neighbors location and orientations:  are: "<<std::endl;
                for (const auto& neighbor_pair : node->neighbors) {
                    const int neighbor_index = neighbor_pair.first;
                    const EdgeNode* neighbor = neighbor_pair.second;
                    
                    std::cout<< (edge_nodes[neighbor_index]->location).transpose()<<" ; " <<(edge_nodes[neighbor_index]->orientation).transpose()<<std::endl;
                }
                    
                
                // Print left neighbor info
                if (graph_node.left_neighbor.first >= 0) {
                    int left_idx = graph_node.left_neighbor.first;
                    Eigen::Vector3d left_loc = edge_nodes[left_idx]->location;
                    std::cout << "  Left neighbor (index " << left_idx << "): " 
                              << left_loc.x() << ", " << left_loc.y() << ", " << left_loc.z() << std::endl;
                } else {
                    std::cout << "  No left neighbor" << std::endl;
                }
                
                // Print right neighbor info
                if (graph_node.right_neighbor.first >= 0) {
                    int right_idx = graph_node.right_neighbor.first;
                    Eigen::Vector3d right_loc = edge_nodes[right_idx]->location;
                    std::cout << "  Right neighbor (index " << right_idx << "): " 
                              << right_loc.x() << ", " << right_loc.y() << ", " << right_loc.z() << std::endl;
                } else {
                    std::cout << "  No right neighbor" << std::endl;
                }
                
                std::cout << "------------------------" << std::endl;
            }
        }
        
        // Add the node to the connectivity graph
        connectivity_graph[i] = graph_node;
    }
    
    //writeConnectivityGraphToFile(connectivity_graph, "connectivity_graph");
    
    return connectivity_graph;
}



void EdgeMapping::writeConnectivityGraphToFile(const ConnectivityGraph& graph, const std::string& file_name) {
    std::string file_path = "../../outputs/" + file_name + ".txt";
    std::ofstream outfile(file_path);
    
    if (!outfile.is_open()) {
        return;
    }
    
    outfile << "# Connectivity Graph\n";
   // outfile << "# Format: NodeIndex Location(x,y,z) Orientation(x,y,z) LeftNeighborIndex LeftOrientation(x,y,z) RightNeighborIndex RightOrientation(x,y,z)\n";
    outfile << "# Format: NodeIndex Location(x,y,z); LeftNeighborIndex LeftOrientation(x,y,z); RightNeighborIndex RightOrientation(x,y,z)\n";
    
    for (const auto& pair : graph) {
        int node_index = pair.first;
        const ConnectivityGraphNode& node = pair.second;
        
        outfile << node_index << " "
                << node.location.x() << " " << node.location.y() << " " << node.location.z() << " ;"
                //<< node.orientation.x() << " " << node.orientation.y() << " " << node.orientation.z() << " "
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

// Define a Curve as a sequence of edge node indices
struct Curve {
    std::vector<int> edge_indices;
};

void EdgeMapping::writeCurvesToFile(const std::vector<Curve>& curves, 
                                   const ConnectivityGraph& connectivity_graph, 
                                   const std::string& file_name, bool b_write_curve_info) {
    std::string file_path = "../../outputs/" + file_name + ".txt";
    std::ofstream outfile(file_path);
    
    if (!outfile.is_open()) {
        std::cerr << "Could not open " << file_path << " for writing." << std::endl;
        return;
    }
    
    if (b_write_curve_info) {
        outfile << "# Curves from Connectivity Graph\n";
        outfile << "# Format: CurveID NodeIndex Location(x,y,z) Orientation(x,y,z)\n";
    }
    
    for (size_t curve_id = 0; curve_id < curves.size(); ++curve_id) {
        const auto& curve = curves[curve_id];
        
        for (const auto& node_index : curve.edge_indices) {
            const auto& node = connectivity_graph.at(node_index);
            
            if (b_write_curve_info) {
                outfile << curve_id << " " << node_index << " "
                        << node.location.x() << " " << node.location.y() << " " << node.location.z() << " "
                        << node.orientation.x() << " " << node.orientation.y() << " " << node.orientation.z() << "\n";
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

std::vector<EdgeMapping::Curve> EdgeMapping::buildCurvesFromConnectivityGraph(const ConnectivityGraph& connectivity_graph) {
    std::vector<Curve> curves;
    std::unordered_set<int> assigned_edges;

    int target1_index = 98;    // (0.355401, 0.0890953, 0.349353)
    int target2_index = 3148;  // (0.354278, 0.0897453, 0.349358)

    // Loop until all edges are assigned to a curve
    for (const auto& [node_index, node] : connectivity_graph) {
        // Skip if this edge is already assigned to a curve
        if (assigned_edges.find(node_index) != assigned_edges.end()) {
            continue;
        }
        
        // Create a new curve starting with this edge
        Curve curve;
        curve.edge_indices.push_back(node_index);
        assigned_edges.insert(node_index);

        //bool target_in_curve = (node_index == target1_index || node_index == target2_index);
        if (node_index == target1_index) {
            std::cout << "Starting curve with Target1 (node index " << target1_index << ")" << std::endl;
        }
        if (node_index == target2_index) {
            std::cout << "Starting curve with Target2 (node index " << target2_index << ")" << std::endl;
        }
        
        // Forward step: follow right neighbors until we can't go further
        int current_index = node_index;
        while (true) {
            const auto& current_node = connectivity_graph.at(current_index);
            int right_neighbor_index = current_node.right_neighbor.first;
            
            if (current_index == target1_index) {
                std::cout << " Target 1,  Right neighbor index: " << right_neighbor_index << std::endl;
            }
            if (current_index == target2_index) {
                std::cout << " Target 2,  Right neighbor index: " << right_neighbor_index << std::endl;
            }

            // Stop if no right neighbor or already assigned
            if (right_neighbor_index < 0 || 
                connectivity_graph.find(right_neighbor_index) == connectivity_graph.end() ||
                assigned_edges.find(right_neighbor_index) != assigned_edges.end()) {
                    if (current_index == target1_index) {
                        std::cout<<"checking right neighbor of "<<target1_index<<std::endl;
                        bool condition2 = connectivity_graph.find(right_neighbor_index) == connectivity_graph.end();
                        bool condition3 = assigned_edges.find(right_neighbor_index) != assigned_edges.end();
                        std::cout << " Right neighbor index: " << right_neighbor_index << std::endl;
                        std::cout<<" condition2: "<<condition2<<std::endl;
                        std::cout<<" condition3: "<<condition3<<std::endl;
                    }
                break;
            }
            
            // Add right neighbor to curve and mark as assigned
            curve.edge_indices.push_back(right_neighbor_index);
            assigned_edges.insert(right_neighbor_index);

            if(right_neighbor_index == target2_index){
                std::cout<<" edge "<<current_index <<" claimed right neighbor of "<<target2_index<<std::endl;
            }

            current_index = right_neighbor_index;
        }
        
        // Backward step: follow left neighbors until we can't go further
        current_index = node_index;
        while (true) {
            const auto& current_node = connectivity_graph.at(current_index);
            int left_neighbor_index = current_node.left_neighbor.first;

            if (current_index == target1_index) {
                std::cout << " target 98,  Left neighbor index: " << left_neighbor_index << std::endl;

            }
            if (current_index == target2_index) {
                std::cout << " target 3148  Left neighbor index: " << left_neighbor_index << std::endl;
            }
            
            // Stop if no left neighbor or already assigned
            if (left_neighbor_index < 0 || 
                connectivity_graph.find(left_neighbor_index) == connectivity_graph.end() ||
                assigned_edges.find(left_neighbor_index) != assigned_edges.end()) {
                break;
                if (current_index == target2_index) {
                        std::cout<<"checking left neighbor of "<<target2_index<<std::endl;
                        bool condition2 = connectivity_graph.find(left_neighbor_index) == connectivity_graph.end();
                        bool condition3 = assigned_edges.find(left_neighbor_index) != assigned_edges.end();
                        std::cout << " left neighbor index: " << left_neighbor_index << std::endl;
                        std::cout<<" condition2: "<<condition2<<std::endl;
                        std::cout<<" condition3: "<<condition3<<std::endl;
                    }
            }
            
            // Add left neighbor to beginning of curve and mark as assigned
            curve.edge_indices.insert(curve.edge_indices.begin(), left_neighbor_index);
            assigned_edges.insert(left_neighbor_index);

            if(left_neighbor_index == target2_index){
                std::cout<<" edge "<<current_index <<" claimed left neighbor of "<<target2_index<<std::endl;
            }

            current_index = left_neighbor_index;
            
            
        }
        
        //if (curve.edge_indices.size() >= 10){
        curves.push_back(curve);
        //}
    }
    
    return curves;
}

std::vector<std::vector<EdgeMapping::SupportingEdgeData>> EdgeMapping::findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d> all_R,
                                                                                                const std::vector<Eigen::Vector3d> all_T,
                                                                                                const Eigen::Matrix3d K,
                                                                                                const int Num_Of_Total_Imgs) { 

    int num_iteration = 1000;
    double step_size_force = 0.01;
    double step_size_torque = 0.1;

    // convert 3D-> 2D relationship to 2D->3D relationship
    auto uncorrected_map = map_Uncorrected2DEdge_To_SupportingData();
    // {3D edge 1, 3D edge 2} -> weight (how many 2d edges they have in common)
    
    std::vector<EdgeCurvelet> all_curvelets = read_curvelets(1);
    std::vector<Eigen::MatrixXd> all_edgels = read_edgels(1);

    auto graph = build3DEdgeWeightedGraph(uncorrected_map, all_edgels, all_curvelets, all_R, all_T);//change datapath used in this function is not using object 0006
    auto pruned_graph = pruneEdgeGraph_by_3DProximityAndOrientation(graph);
    auto pruned_graph_by_projection = pruneEdgeGraphbyProjections(pruned_graph, all_R, all_T, K, Num_Of_Total_Imgs);

    // crate 3d -> its neighbor data structure
    auto neighbor_map = buildEdgeNodeGraph(pruned_graph_by_projection);
    align3DEdgesUsingEdgeNodes(neighbor_map, num_iteration, step_size_force, step_size_torque); //call it align
    auto connectivity_graph = createConnectivityGraph(neighbor_map);
    auto curves = buildCurvesFromConnectivityGraph(connectivity_graph);
    writeCurvesToFile(curves, connectivity_graph, "curves_from_connectivity_graph", true);
    writeCurvesToFile(curves, connectivity_graph, "curves_points", false);

    // /////////// TEST: manually create and iteratively smooth 2 test EdgeNodes  /////////// 

    // /////////// Create EdgeNodes from input files ///////////
    // std::string points_file = "../../files/line_noisy_points.txt";
    // std::string tangents_file = "../../files/line_noisy_tangents.txt";
    // std::string connections_file = "../../files/line_connections.txt";
    
    // //EdgeNodeList test_node = createEdgeNodesFromFiles(points_file, tangents_file, connections_file);

    // //align3DEdgesUsingEdgeNodes(test_node, num_iteration); //call it align

    std::vector<std::vector<SupportingEdgeData>> merged_groups;

    // // Projection view's extrinsics
    // Eigen::Matrix3d R_view = all_R[7];
    // Eigen::Vector3d T_view = all_T[7];
    // util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    // std::ofstream proj_outfile("../../outputs/3D_edge_groups_projected.txt");
    // if (!proj_outfile.is_open()) {
    //     std::cerr << "Could not open 3D edge group projection output file!" << std::endl;
    //     return merged_groups;
    // }

    // std::ofstream edge_3d_outfile("../../outputs/3D_edge_groups.txt");
    // if (!edge_3d_outfile.is_open()) {
    //     std::cerr << "Could not open 3D edge projection output file!" << std::endl;
    //     return merged_groups;
    // }

    Eigen::Vector3d target1(0.356127, 0.0903582,  0.350045);
    Eigen::Vector3d target2(0.354024, 0.089324,  0.34985);
    // Eigen::Vector3d target3(0.704316, 0.884627,  0.35257);

    // Set to collect neighbors of target1 and target2
    std::set<NeighborWithOrientation> neighbors_of_target1;
    std::set<NeighborWithOrientation> neighbors_of_target2;
    // // std::set<NeighborWithOrientation> neighbors_of_target3;

    int group_id = 0;
    for (const auto& [pair, weight] : pruned_graph_by_projection) {
    //for (const auto& [pair, weight] : pruned_graph) {
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

            if (a.isApprox(target2, 1e-6)) 
                neighbors_of_target2.insert({b, tangent2});
            else if (b.isApprox(target2, 1e-6)) 
                neighbors_of_target2.insert({a, tangent1});

            // if (a.isApprox(target3, 1e-6)) 
            //     neighbors_of_target3.insert({b, tangent2});
            // else if (b.isApprox(target3, 1e-6)) 
            //     neighbors_of_target3.insert({a, tangent1});
        }
        ///////////// find neighbors of the target point /////////////

    }


    // std::cout<<"neighbors of target 1 & 2 are: "<<std::endl;
    // for (auto& n1 : neighbors_of_target1) {
    //     for (const auto& n2 : neighbors_of_target2) {
    //        if (n1.location.isApprox(n2.location, 1e-6)) {
    //             std::cout << n1.location.transpose() << ";\n";
    //         }
    //    }
    // }
    //     std::cout<<"neighbors of target 1 are: "<<std::endl;
    // for (auto& n1 : neighbors_of_target1) {
    //     //for (const auto& n2 : neighbors_of_target2) {
    //     //    if (n1.isApprox(n2, 1e-6)) {
    //     std::cout << n1.location.transpose() << ";\n";
    //     //    }
    //    // }
    // }
    // std::cout<<"neighbors of target 2 are: "<<std::endl;
    // for (auto& n1 : neighbors_of_target2) {
    //     //for (const auto& n2 : neighbors_of_target2) {
    //     //    if (n1.isApprox(n2, 1e-6)) {
    //     std::cout << n1.location.transpose() << ";\n";
    //     //    }
    //    // }
    // }
    
    return merged_groups;
}
