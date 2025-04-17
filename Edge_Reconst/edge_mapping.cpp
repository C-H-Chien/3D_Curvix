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


void EdgeMapping::add3DToFrameMapping(const Eigen::Vector3d& edge_3D, 
                                      const Eigen::Vector2d& supporting_edge, 
                                      int frame) {
    frame_to_edge_to_3D_map[frame][supporting_edge].push_back(edge_3D);
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

                Eigen::Vector3d target1(0.4459, 0.0341785, 0.352049);
                Eigen::Vector3d target2(0.447857, 0.0352899, 0.348052);
                bool match_direct = (a.isApprox(target1, 1e-6) && b.isApprox(target2, 1e-6));
                bool match_reverse = (a.isApprox(target2, 1e-6) && b.isApprox(target1, 1e-6));

                if (match_direct || match_reverse) {
                    std::cout << "[DEBUG] Found target pair with weight: " << graph[edge_pair] << std::endl;
                }

            }
        }
    }
    return graph;
}
///////////////////////// Create the 3D edge weight graph /////////////////////////

void EdgeMapping::write_edge_graph( 
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
    HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
    std::string file_name ) 
{
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
EdgeMapping::pruneEdgeGraph_by_3DProximityAndOrientation(
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                       HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
    double lambda1, double lambda2)
{
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

        if ((proximity_diff < mu1 + sigma1*lambda1 && angle_diff < mu2 + sigma2*lambda2) || weight > 1) {
            pruned_graph[pair]++;
        }
    }

    //> write the 3D edge graph after pruning
    write_edge_graph( pruned_graph, "3D_edge_pruned_graph" );

    return pruned_graph;
}
///////////////////////// compute weighted graph stats /////////////////////////

///////////////////////// Start of pruning weighted graph by projecting to all views /////////////////////
std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                   HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::pruneEdgeGraphbyProjections(
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                       HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
    double lambda1, double lambda2, 
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
    > pruned_graph;

    int remove_link_counter = 0;

    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    for (const auto& [pair, weight] : graph) {

        //> Find the locations and tangents of the 3D edge pair
        const Eigen::Vector3d& Edge3D_1_Location = pair.first;
        const Eigen::Vector3d& Edge3D_2_Location = pair.second;

        auto it1 = edge_3D_to_supporting_edges.find(Edge3D_1_Location);
        auto it2 = edge_3D_to_supporting_edges.find(Edge3D_2_Location);

        //> Skip if the 3D edge does not exist in the structure edge_3D_to_supporting_edges (is it possible?)
        if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end()) continue;

        Eigen::Vector3d unit_Tangent3D_1 = it1->second.front().tangents_3D_world.normalized();
        Eigen::Vector3d unit_Tangent3D_2 = it2->second.front().tangents_3D_world.normalized();

        bool prune_flag = false;

        double Orientation_Diff_Thresh = 30; //> in degrees
        double ore_diff_threshold = cos(double(Orientation_Diff_Thresh)/180*PI);
        
        //> Project to all views 
        for (unsigned vi = 0; vi < Num_Of_Total_Imgs; vi++) {

            //> Project 3D edge point to the 2D image
            Eigen::Vector3d proj_edge_1_location = K * (All_R[vi] * Edge3D_1_Location + All_T[vi]);
            Eigen::Vector3d proj_edge_2_location = K * (All_R[vi] * Edge3D_2_Location + All_T[vi]);
            proj_edge_1_location = util->getNormalizedProjectedPoint( proj_edge_1_location );
            proj_edge_2_location = util->getNormalizedProjectedPoint( proj_edge_2_location );

            //> Project 3D tangent to the 2D image
            Eigen::Vector3d proj_tangent_1 = util->project_3DTangent_to_Image(All_R[vi], K, unit_Tangent3D_1, proj_edge_1_location);
            Eigen::Vector3d proj_tangent_2 = util->project_3DTangent_to_Image(All_R[vi], K, unit_Tangent3D_2, proj_edge_2_location);

            //> Rule out if the distance between the projected 3D edge points are over a threshold
            //> TODO: make the threhold as a Macro
            prune_flag = ((proj_edge_1_location - proj_edge_2_location).norm() > 3) ? (true) : (false);

            //> Rule out if the orientation difference is over a threshold
            //> TODO: make the threhold as a Macro
            double abs_dot_prod = fabs(proj_tangent_1(0)*proj_tangent_2(0) + proj_tangent_1(1)*proj_tangent_2(1));
            prune_flag = (abs_dot_prod < ore_diff_threshold) ? (true) : (false);

            if (prune_flag) break;
        }

        if (prune_flag) {
            remove_link_counter++;
        }
        else {
            pruned_graph[pair]++;
        }
    }

    return pruned_graph;
}


///////////////////////// End of pruning weighted graph by projecting to all views /////////////////////

///////////////////////// build a map of 3d edges with its neighbors /////////////////////////
EdgeMapping::EdgeNodeList EdgeMapping::buildEdgeNodeGraph(
    const std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int,
                             HashEigenVector3dPair, FuzzyVector3dPairEqual>& pruned_graph) {
    std::unordered_map<Eigen::Vector3d, EdgeNode*, HashEigenVector3d, FuzzyVector3dEqual> node_map;
    EdgeNodeList nodes;

    //> CH: trying to understand this part...
    // Create node objects
    for (const auto& [edge, _] : edge_3D_to_supporting_edges) {
        //auto node = std::make_unique<EdgeNode>();
        std::unique_ptr<EdgeNode> node(new EdgeNode());
        node->value = edge;     //> 3D edge location
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

        Eigen::Vector3d t_a = Eigen::Vector3d::Zero();
        Eigen::Vector3d t_b = Eigen::Vector3d::Zero();

        // const Eigen::Vector3d& p1 = pair.first;
        // const Eigen::Vector3d& p2 = pair.second;

        // auto it1 = edge_3D_to_supporting_edges.find(p1);
        // auto it2 = edge_3D_to_supporting_edges.find(p2);

        // bool valid_tangents = false;
        // if (it1 != edge_3D_to_supporting_edges.end() && !it1->second.empty() &&
        //     it2 != edge_3D_to_supporting_edges.end() && !it2->second.empty()) {
        //     tangent1 = it1->second.front().tangents_3D_world.normalized();
        //     tangent2 = it2->second.front().tangents_3D_world.normalized();
        //     valid_tangents = true;
        // }

        auto it1 = edge_3D_to_supporting_edges.find(pair.first);
        auto it2 = edge_3D_to_supporting_edges.find(pair.second);

        if (it1 != edge_3D_to_supporting_edges.end() && !it1->second.empty() &&
            it2 != edge_3D_to_supporting_edges.end() && !it2->second.empty()) {
            t_a = it1->second.front().tangents_3D_world.normalized();
            t_b = it2->second.front().tangents_3D_world.normalized();
            // valid_tangents = true;
        }

        // for (const auto& s : edge_3D_to_supporting_edges[pair.first]) t_a += s.tangents_3D_world;
        // t_a.normalize();

        // Eigen::Vector3d t_b = Eigen::Vector3d::Zero();
        // for (const auto& s : edge_3D_to_supporting_edges[pair.second]) t_b += s.tangents_3D_world;
        // t_b.normalize();

        a->neighbors.emplace_back(b, std::make_pair(t_a, t_b));
        b->neighbors.emplace_back(a, std::make_pair(t_b, t_a));
    }

    return nodes;
}
///////////////////////// build a map of 3d edges with its neighbors /////////////////////////



///////////////////////// Smoothing 3d edges with its neighbors /////////////////////////
void EdgeMapping::smooth3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes, int iterations) {
    // --- Write BEFORE smoothing ---
    std::ofstream before_out("../../outputs/3D_edges_before_smoothing.txt");
    if (!before_out.is_open()) {
        LOG_ERROR("Could not open before smoothing file.");
        return;
    }
    for (const auto& node : edge_nodes) {
        before_out << node->value.transpose() << "\n";
    }
    before_out.close();

    // --- Smoothing iterations ---
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Eigen::Vector3d> new_values(edge_nodes.size());

        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            const auto& node = edge_nodes[i];
            if (node->neighbors.empty()) {
                new_values[i] = node->value;
                continue;
            }

            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            for (const auto& [neighbor_ptr, _] : node->neighbors) {
                sum += neighbor_ptr->value;
            }
            //new_values[i] = sum / node->neighbors.size();
            Eigen::Vector3d average = sum / node->neighbors.size();
            new_values[i] = 0.5 * node->value + 0.5 * average;

        }

        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            edge_nodes[i]->value = new_values[i];
        }
    }

    // --- Write AFTER smoothing ---
    std::ofstream after_out("../../outputs/3D_edges_after_smoothing.txt");
    if (!after_out.is_open()) {
        std::cerr << "[ERROR] Could not open after smoothing file.\n";
        return;
    }
    for (const auto& node : edge_nodes) {
        after_out << node->value.transpose() << "\n";
    }
    after_out.close();

    // --- Update internal mapping with smoothed values ---
    decltype(edge_3D_to_supporting_edges) new_map;
    for (const auto& node : edge_nodes) {
        auto it = edge_3D_to_supporting_edges.find(node->value);
        if (it != edge_3D_to_supporting_edges.end()) {
            new_map[node->value] = it->second;
        }
    }

    //> Maybe we need a way to restructure this edge_3D_to_supporting_edges
    edge_3D_to_supporting_edges = std::move(new_map);

    std::cout << "[SMOOTHING COMPLETE] Smoothed edges written to file after " << iterations << " iterations.\n";
}
///////////////////////// Smoothing 3d edges with its neighbors /////////////////////////


std::vector<std::vector<EdgeMapping::SupportingEdgeData>> 
EdgeMapping::findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d> all_R,
                                      const std::vector<Eigen::Vector3d> all_T,
                                      const Eigen::Matrix3d K,
                                      const int Num_Of_Total_Imgs) 
{ 
    // convert 3D-> 2D relationship to 2D->3D relationship
    auto uncorrected_map = map_Uncorrected2DEdge_To_SupportingData();
    // {3D edge 1, 3D edge 2} -> weight (how many 2d edges they have in common)
    auto graph = build3DEdgeWeightedGraph(uncorrected_map);
    //pruning
    // auto pruned_graph = computeGraphEdgeDistanceAndAngleStats(graph, 0, 0);
    auto pruned_graph = pruneEdgeGraph_by_3DProximityAndOrientation(graph, 1, 1);
    // auto pruned_graph = pruneEdgeGraphbyProjections(pruned_graph, 0, 0, all_R, all_T, K, Num_Of_Total_Imgs);

    // crate 3d -> its neighbor data structure
    // auto neighbor_map = buildEdgeNodeGraph(pruned_graph);
    // smooth3DEdgesUsingEdgeNodes(neighbor_map, 10);

    // for (const auto& [edge_ptr, neighbor_list] : neighbor_map) {
    //     std::cout << "[EDGE] " << edge_ptr->transpose() << "\n";
    //     for (const auto& [neighbor_ptr, tangents] : neighbor_list) {
    //         std::cout << "  -> Neighbor: " << neighbor_ptr->transpose()
    //                 << " | Self Tangent: " << tangents.first.transpose()
    //                 << " | Neighbor Tangent: " << tangents.second.transpose() << "\n";
    //     }
    // }

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

    int group_id = 0;
    for (const auto& [pair, weight] : pruned_graph) {
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

        // Project both endpoints
        Eigen::Vector3d pc1 = R_view * p1 + T_view;
        Eigen::Vector3d pc2 = R_view * p2 + T_view;

        if (pc1(2) == 0 || pc2(2) == 0) {
            std::string warning_msg = "Z=0 for group" + std::to_string(group_id) + " edge: skipping";
            LOG_WARNING(warning_msg);
            continue;
        }

        Eigen::Vector3d pi1 = K * pc1;
        Eigen::Vector3d pi2 = K * pc2;
        
        Eigen::Vector3d proj_edge_1_location = util->getNormalizedProjectedPoint( pi1 );
        Eigen::Vector3d proj_edge_2_location = util->getNormalizedProjectedPoint( pi2 );

        //> Project 3D tangent to the 2D image
        Eigen::Vector3d proj_tangent_1 = util->project_3DTangent_to_Image(R_view, K, tangent1, proj_edge_1_location);
        Eigen::Vector3d proj_tangent_2 = util->project_3DTangent_to_Image(R_view, K, tangent2, proj_edge_2_location);

        //> write projected edge location and orientation to a file
        proj_outfile << proj_edge_1_location(0) << "\t" << proj_edge_1_location(1) << "\t" \
                     << proj_edge_2_location(0) << "\t" << proj_edge_2_location(1) << "\t" \
                     << proj_tangent_1(0) << "\t" << proj_tangent_1(1) << "\t"
                     << proj_tangent_2(0) << "\t" << proj_tangent_2(1) << "\t" << valid_tangents << "\n";


        // Add tangent angle classification
        // if (valid_tangents) {
        //     Eigen::Vector3d edge_dir_3d = (p2 - p1).normalized();

        //     double angle1_a = std::acos(edge_dir_3d.dot(tangent1));
        //     double angle1_b = std::acos((-edge_dir_3d).dot(tangent1));
        //     double angle1_deg = std::min(angle1_a, angle1_b) * 180.0 / M_PI;
        //     double angle2_a = std::acos(edge_dir_3d.dot(tangent1));
        //     double angle2_b = std::acos((-edge_dir_3d).dot(tangent1));
        //     double angle2_deg = std::min(angle1_a, angle1_b) * 180.0 / M_PI;

        //     if (angle1_deg < 30.0 && angle2_deg < 30.0) {
        //         proj_outfile << " 1 "; // Sequential
        //     } else if (angle1_deg > 75.0 && angle2_deg > 75.0) {
        //         proj_outfile << " 0 "; // Parallel
        //     }else{
        //         proj_outfile << " 2 "; // None
        //     }
        // }


        // proj_outfile << weight; 

        // proj_outfile << "\n";
        // edge_3d_outfile << p1.transpose() << " \t " << p2.transpose() << "\n";
        group_id++;
    }
    proj_outfile.close();
    std::cout << "[INFO] Wrote projected 3D edge groups to 2D view." << std::endl;


    // std::ofstream outfile("../../outputs/3D_edge_groups.txt");
    // if (!outfile.is_open()) {
    //     std::cerr << "Could not open output file!" << std::endl;
    //     return merged_groups;
    // }

    
    // int group_id_2d_uncorrected = 0;

    // for (const auto& [key, records] : uncorrected_map) {
    //     for (const auto& record : records) {
    //         const Eigen::Vector3d& edge3D = record.edge_3D;
    //         outfile << edge3D.transpose() << " " << group_id_2d_uncorrected << "\n";
    //     }
    //     group_id_2d_uncorrected++;
    // }

    // Eigen::Vector3d target1(0.4459, 0.0341785, 0.352049);
    // Eigen::Vector3d target2(0.447857, 0.0352899, 0.348052);

    // for (const auto& [pair, weight] : graph) {
    //     const Eigen::Vector3d& edge_3D_1 = pair.first;
    //     const Eigen::Vector3d& edge_3D_2 = pair.second;

    //     bool match_direct = (edge_3D_1.isApprox(target1, 1e-6) && edge_3D_2.isApprox(target2, 1e-6));
    //     bool match_reverse = (edge_3D_1.isApprox(target2, 1e-6) && edge_3D_2.isApprox(target1, 1e-6));

    //     if (match_direct || match_reverse) {
    //         std::cout << "[DEBUG] Found target pair with weight: " << weight << std::endl;
    //     }

    //     auto it1 = edge_3D_to_supporting_edges.find(edge_3D_1);
    //     auto it2 = edge_3D_to_supporting_edges.find(edge_3D_2);
    //     if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end())
    //         continue;

    //     // Combine the two supporting edge lists into a unique set
    //     std::set<SupportingEdgeData> unique_edges(it1->second.begin(), it1->second.end());
    //     unique_edges.insert(it2->second.begin(), it2->second.end());

    //     // Convert to vector and store in merged_groups
    //     std::vector<SupportingEdgeData> group(unique_edges.begin(), unique_edges.end());
    //     merged_groups.push_back(group);
    // }



    //std::cout << "[SUMMARY] Total Merged Groups from Uncorrected Map: " << group_id << std::endl;
    return merged_groups;
}




// std::vector<std::vector<EdgeMapping::SupportingEdgeData>> EdgeMapping::findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d>& all_R,const std::vector<Eigen::Vector3d>& all_T,int Num_Of_Total_Imgs) { 
    
//     double fx = 1111.11136542426;
//     double fy = 1111.11136542426;
//     double cx = 399.500000000000;
//     double cy = 399.500000000000;
//     Eigen::Matrix3d K;
//     K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

//     auto uncorrected_map = map_Uncorrected2DEdge_To_SupportingData();
//     auto graph = build3DEdgeWeightedGraph(uncorrected_map);
//     computeGraphEdgeDistanceAndAngleStats(graph);
//     // for (const auto& [pair, weight] : graph) {
//     //     std::cout << "Edge: (" << pair.first.transpose() << ") <-> (" 
//     //             << pair.second.transpose() << ") | weight: " << weight << "\n";
//     // }


//     std::vector<std::vector<EdgeMapping::SupportingEdgeData>> merged_groups; 

//     bool visited_CH[edge_3D_to_supporting_edges.size()] = {false};

//     int merged_count = 0;

//     std::cout << "edge_3D_to_supporting_edges size: " << edge_3D_to_supporting_edges.size() << std::endl;

//     std::ofstream outfile("../../outputs/3D_edge_groups.txt");
//     if (!outfile.is_open()) {
//         std::cerr << "Could not open output file!" << std::endl;
//         return merged_groups;
//     }

//     int group_id = 0;
//     int counter_query_3D_edge = 0;


//     for (auto it1 = edge_3D_to_supporting_edges.begin(); it1 != edge_3D_to_supporting_edges.end(); ++it1) {
//         const Eigen::Vector3d& edge_3D_1 = it1->first;
//         auto& support_2D_1 = it1->second;

//         if (visited_CH[counter_query_3D_edge]) {
//             counter_query_3D_edge++;
//             continue;
//         }

//         std::set<EdgeMapping::SupportingEdgeData> unique_edges(support_2D_1.begin(), support_2D_1.end());
//         std::vector<Eigen::Vector3d> merged_3D_edges;
//         merged_3D_edges.push_back(edge_3D_1);
//         visited_CH[counter_query_3D_edge] = true;
        
//         int counter_database_3D_edge = 0;
//         bool is_merged = false;
//         for (auto it2 = edge_3D_to_supporting_edges.begin(); it2 != edge_3D_to_supporting_edges.end(); ++it2) {
//             const Eigen::Vector3d& edge_3D_2 = it2->first;
//             auto& support_2D_2 = it2->second;
            
//             if (edge_3D_1 == edge_3D_2 || visited_CH[counter_database_3D_edge]) {
//                 counter_database_3D_edge++;
//                 continue;
//             }

//             // // Compute Euclidean distance between 3D edges
//             // double distance = (edge_3D_1 - edge_3D_2).norm(); 

//             // // Only merge if the distance is within 10 pixels
//             // if (distance > 0.005) {
//             //     counter_database_3D_edge++;
//             //     continue;
//             // }

//             int common_count = 0;
//             for (const auto& edge1 : support_2D_1) {
//                 for (const auto& edge2 : support_2D_2) {
//                     double orientation_difference = (edge1.tangents_3D_world - edge2.tangents_3D_world).norm();
//                     if (orientation_difference > 0.01) continue;

//                     if (edge1.image_number == edge2.image_number && edge1.edge_uncorrected == edge2.edge_uncorrected) {
//                         // ///////////// check 2d reprojections of 3d edges /////////////
//                         // Eigen::Vector3d point3D_1 = edge_3D_1.transpose();
//                         // Eigen::Vector3d point_camera_1 = edge1.rotation * point3D_1 + edge1.translation;
//                         // Eigen::Vector3d point3D_2 = edge_3D_2.transpose();
//                         // Eigen::Vector3d point_camera_2 = edge2.rotation * point3D_1 + edge2.translation;
//                         // Eigen::Vector3d point_image_1 = K * point_camera_1;
//                         // Eigen::Vector3d point_image_2 = K * point_camera_2;
//                         // Eigen::Vector2d edges2D_1;
//                         // edges2D_1(0) = point_image_1(0) / point_image_1(2);
//                         // edges2D_1(1) = point_image_1(1) / point_image_1(2);
//                         // Eigen::Vector2d edges2D_2;
//                         // edges2D_2(0) = point_image_2(0) / point_image_2(2);
//                         // edges2D_2(1) = point_image_2(1) / point_image_2(2);
//                         // double distance = (edges2D_1 - edges2D_2).norm();
//                         // if (distance > 10) {
//                         //     continue;
//                         // }
//                         // ///////////// check 2d reprojections of 3d edges /////////////
//                         common_count++;
//                     }
//                     if (common_count >= 2) break;
//                 }
//                 if (common_count >= 2) break;
//             }

//             if (common_count >= 2) {
//                 unique_edges.insert(support_2D_2.begin(), support_2D_2.end());
//                 visited_CH[counter_database_3D_edge] = true;
//                 merged_3D_edges.push_back(edge_3D_2);
//                 is_merged = true;
//             }

//             counter_database_3D_edge++;
//         }

//         std::vector<EdgeMapping::SupportingEdgeData> group(unique_edges.begin(), unique_edges.end());
//         merged_groups.push_back(group);

//         // for (const auto& merged_edge : merged_3D_edges) {
//         //     outfile << merged_edge.transpose() << " " << group_id << "\n";
//         // }

//         group_id++;

//         merged_groups.push_back(group);
//         if (is_merged) merged_count++;


//         // if (is_merged) {
//         //     merged_groups.push_back(group);
//         //     merged_count++;
//         // }

//         counter_query_3D_edge++;
//     }

//     std::cout << "[SUMMARY] Merged Groups: " << merged_count << std::endl;
//     return merged_groups;
// }


