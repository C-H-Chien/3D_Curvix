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

        if (!prune_flag) 
            pruned_graph_by_proj[pair]++;
    }

    //> write the 3D edge graph after pruning by projection
    write_edge_graph( pruned_graph_by_proj, "3D_edge_pruned_graph_by_proj" );

    return pruned_graph_by_proj;
}


///////////////////////// End of pruning weighted graph by projecting to all views /////////////////////

///////////////////////// build a map of 3d edges with its neighbors /////////////////////////
EdgeMapping::EdgeNodeList EdgeMapping::buildEdgeNodeGraph(
    const std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int,
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
void EdgeMapping::smooth3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes, int iterations) {
    std::ofstream before_out("../../outputs/3D_edges_before_smoothing.txt");
    for (const auto& node : edge_nodes) {
        before_out << node->location.transpose() << "\n";
    }
    before_out.close();

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

                double a = (B - p).dot(t_neighbor);
                Eigen::Vector3d projected = p + a * t_neighbor;
                sum_force += (projected - B);
            }

            new_locations[i] = node->location + 0.1 * sum_force;

            //orientation aligning
            Eigen::Vector3d sum_tangent = node->orientation;
            for (const auto& neighbor : node->neighbors) {
                sum_tangent += neighbor->orientation;
            }
            new_orientations[i] = sum_tangent.normalized();
        }

        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            edge_nodes[i]->location = new_locations[i];
            edge_nodes[i]->orientation = new_orientations[i];
        }
    }

    std::ofstream after_out("../../outputs/3D_edges_after_smoothing.txt");
    for (const auto& node : edge_nodes) {
        after_out << node->location.transpose() << " " << node->orientation.transpose() << "\n";
    }
    after_out.close();

    std::cout << "[SMOOTHING COMPLETE] Smoothed edges written to file after " << iterations << " iterations.\n";
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
////////////////// test //////////////////

std::vector<std::vector<EdgeMapping::SupportingEdgeData>> 
EdgeMapping::findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d> all_R,
                                      const std::vector<Eigen::Vector3d> all_T,
                                      const Eigen::Matrix3d K,
                                      const int Num_Of_Total_Imgs) 
{ 

    int num_iteration = 10;

    // convert 3D-> 2D relationship to 2D->3D relationship
    auto uncorrected_map = map_Uncorrected2DEdge_To_SupportingData();

    // {3D edge 1, 3D edge 2} -> weight (how many 2d edges they have in common)
    auto graph = build3DEdgeWeightedGraph(uncorrected_map);

    //pruning
    // auto pruned_graph = computeGraphEdgeDistanceAndAngleStats(graph, 0, 0);
    auto pruned_graph = pruneEdgeGraph_by_3DProximityAndOrientation(graph, 0.5, 0.5);

    // auto pruned_graph_by_projection = pruneEdgeGraphbyProjections(pruned_graph, all_R, all_T, K, Num_Of_Total_Imgs);

    // crate 3d -> its neighbor data structure
    auto neighbor_map = buildEdgeNodeGraph(pruned_graph);
    smooth3DEdgesUsingEdgeNodes(neighbor_map, num_iteration); //call it align


    /////////// TEST: manually create and iteratively smooth 2 test EdgeNodes  /////////// 
    EdgeNodeList test_smooth_node;
    std::unique_ptr<EdgeNode> node1(new EdgeNode());
    node1->location = Eigen::Vector3d(0.703582, 0.881398, 0.353133);
    node1->orientation = Eigen::Vector3d(-0.865327, 0.501203, 0.00209428);

    std::unique_ptr<EdgeNode> node2(new EdgeNode());
    node2->location = Eigen::Vector3d(0.710306, 0.877993, 0.353316);
    node2->orientation = Eigen::Vector3d(-0.868224, 0.496146, -0.00518501);

    std::unique_ptr<EdgeNode> node3(new EdgeNode());
    node3->location = Eigen::Vector3d(0.704316, 0.884627, 0.35257);
    node3->orientation = Eigen::Vector3d(0.85584, -0.517224, 0.00410801);
    
    std::unique_ptr<EdgeNode> node4(new EdgeNode());
    node3->location = Eigen::Vector3d(0.710307, 0.881131, 0.352345);
    node3->orientation = Eigen::Vector3d(0.853602,  -0.520643, -0.0171691);


    // Connect neighbors
    node1->neighbors.push_back(node2.get());
    node1->neighbors.push_back(node3.get());
    node1->neighbors.push_back(node4.get());

    node2->neighbors.push_back(node1.get());
    node2->neighbors.push_back(node3.get());
    node2->neighbors.push_back(node4.get());

    node3->neighbors.push_back(node1.get());
    node3->neighbors.push_back(node2.get());
    node4->neighbors.push_back(node4.get());

    node4->neighbors.push_back(node1.get());
    node4->neighbors.push_back(node2.get());
    node4->neighbors.push_back(node3.get());

    test_smooth_node.push_back(std::move(node1));
    test_smooth_node.push_back(std::move(node2));
    test_smooth_node.push_back(std::move(node3));
    test_smooth_node.push_back(std::move(node4));

    for (size_t i = 0; i < test_smooth_node.size(); ++i) {
        const auto& node = test_smooth_node[i];
        std::cout << node->location.transpose()<< "; ";
    }
    std::cout << "\n";

    for (int iter = 0; iter < 100; ++iter) {
        std::vector<Eigen::Vector3d> new_locations(test_smooth_node.size());
        std::vector<Eigen::Vector3d> new_orientations(test_smooth_node.size());

        for (size_t i = 0; i < test_smooth_node.size(); ++i) {
            const auto& node = test_smooth_node[i];
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

                double a = (B - p).dot(t_neighbor);
                Eigen::Vector3d projected = p + a * t_neighbor;
                sum_force += (projected - B);
            }

            new_locations[i] = node->location + 0.1 * sum_force;

            Eigen::Vector3d sum_tangent = node->orientation;
            for (const auto& neighbor : node->neighbors) {
                sum_tangent += neighbor->orientation;
            }
            new_orientations[i] = sum_tangent.normalized();
        }

        for (size_t i = 0; i < test_smooth_node.size(); ++i) {
            test_smooth_node[i]->location = new_locations[i];
            test_smooth_node[i]->orientation = new_orientations[i];
        }

        // std::cout << "[Test Iteration " << iter + 1 << "]\n";
        // for (size_t i = 0; i < test_smooth_node.size(); ++i) {
        //     const auto& node = test_smooth_node[i];
        //     //std::cout << "  Node " << i << " - Location: " << node->location.transpose() << " | Orientation: " << node->orientation.transpose() << "\n";
        //     std::cout << node->location.transpose() << " ; " << node->orientation.transpose() << "\n";
        // }
        for (size_t i = 0; i < test_smooth_node.size(); ++i) {
            const auto& node = test_smooth_node[i];
            std::cout << node->location.transpose()<< "; ";
        }
        std::cout << "\n";
    }
    /////////// TEST: manually create and iteratively smooth 2 test EdgeNodes  /////////// 

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

    Eigen::Vector3d target1(0.703582, 0.881398, 0.353133);
    Eigen::Vector3d target2(0.710306, 0.877993, 0.353316);
    Eigen::Vector3d target3(0.704316, 0.884627,  0.35257);

    // Set to collect neighbors of target1 and target2
    std::set<NeighborWithOrientation> neighbors_of_target1;
    std::set<NeighborWithOrientation> neighbors_of_target2;
    std::set<NeighborWithOrientation> neighbors_of_target3;

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

        ///////////// find neighbors of the target point /////////////
        if (valid_tangents){
            const Eigen::Vector3d& a = pair.first;
            const Eigen::Vector3d& b = pair.second;

            if (a.isApprox(target1, 1e-6)) 
                neighbors_of_target1.insert({b, tangent2});
            else if (b.isApprox(target1, 1e-6)) 
                neighbors_of_target1.insert({a, tangent1});

            if (a.isApprox(target2, 1e-6)) 
                neighbors_of_target2.insert({b, tangent2});
            else if (b.isApprox(target2, 1e-6)) 
                neighbors_of_target2.insert({a, tangent1});

            if (a.isApprox(target3, 1e-6)) 
                neighbors_of_target3.insert({b, tangent2});
            else if (b.isApprox(target3, 1e-6)) 
                neighbors_of_target3.insert({a, tangent1});
        }
        ///////////// find neighbors of the target point /////////////


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
                     << proj_tangent_2(0) << "\t" << proj_tangent_2(1) << "\t";


        // Add tangent angle classification
        if (valid_tangents) {
            Eigen::Vector3d edge_dir_3d = (p2 - p1).normalized();

            double angle1_a = std::acos(edge_dir_3d.dot(tangent1));
            double angle1_b = std::acos((-edge_dir_3d).dot(tangent1));
            double angle1_deg = std::min(angle1_a, angle1_b) * 180.0 / M_PI;
            double angle2_a = std::acos(edge_dir_3d.dot(tangent1));
            double angle2_b = std::acos((-edge_dir_3d).dot(tangent1));
            double angle2_deg = std::min(angle1_a, angle1_b) * 180.0 / M_PI;

            if (angle1_deg < 30.0 && angle2_deg < 30.0) {
                proj_outfile << " 1 "; // Sequential
            } else if (angle1_deg > 75.0 && angle2_deg > 75.0) {
                proj_outfile << " 0 "; // Parallel
            }else{
                proj_outfile << " 2 "; // None
            }
        }


        proj_outfile << weight << "\n"; 

        edge_3d_outfile << p1.transpose() << " \t " << p2.transpose() << "\n";
        group_id++;
    }
    proj_outfile.close();
    edge_3d_outfile.close();

    std::cout << "[INFO] Wrote projected 3D edge groups to 2D view." << std::endl;


    ////////////////////////// write after smoothing data //////////////////////////
    std::ofstream smoothed_proj_out("../../outputs/3D_edge_groups_projected_after_smoothing.txt");
    if (!smoothed_proj_out.is_open()) {
        std::cerr << "Could not open smoothed projection output file!" << std::endl;
        return merged_groups;
    }

    int smoothed_id = 0;
    for (const auto& node : neighbor_map) {
        for (const auto* neighbor : node->neighbors) {
            // Avoid writing duplicate pairs (only write when node < neighbor)
            if (node->location.isApprox(neighbor->location, 1e-10) ||
                (node->location.x() > neighbor->location.x()) ||
                (node->location.x() == neighbor->location.x() && node->location.y() > neighbor->location.y()) ||
                (node->location.x() == neighbor->location.x() && node->location.y() == neighbor->location.y() && node->location.z() > neighbor->location.z())) {
                continue;
            }

            Eigen::Vector3d p1 = node->location;
            Eigen::Vector3d p2 = neighbor->location;

            Eigen::Vector3d t1 = node->orientation.normalized();
            Eigen::Vector3d t2 = neighbor->orientation.normalized();

            Eigen::Vector3d pc1 = R_view * p1 + T_view;
            Eigen::Vector3d pc2 = R_view * p2 + T_view;

            if (pc1(2) == 0 || pc2(2) == 0) continue;

            Eigen::Vector3d pi1 = K * pc1;
            Eigen::Vector3d pi2 = K * pc2;
            
            Eigen::Vector3d proj_edge_1_location = util->getNormalizedProjectedPoint( pi1 );
            Eigen::Vector3d proj_edge_2_location = util->getNormalizedProjectedPoint( pi2 );

            //> Project 3D tangent to the 2D image
            Eigen::Vector3d proj_tangent_1 = util->project_3DTangent_to_Image(R_view, K, t1, proj_edge_1_location);
            Eigen::Vector3d proj_tangent_2 = util->project_3DTangent_to_Image(R_view, K, t2, proj_edge_2_location);

            Eigen::Vector3d edge_dir_3d = (p2 - p1).normalized();

            double angle1_a = std::acos(edge_dir_3d.dot(t1));
            double angle1_b = std::acos((-edge_dir_3d).dot(t1));
            double angle1_deg = std::min(angle1_a, angle1_b) * 180.0 / M_PI;
            double angle2_a = std::acos(edge_dir_3d.dot(t2));
            double angle2_b = std::acos((-edge_dir_3d).dot(t2));
            double angle2_deg = std::min(angle2_a, angle2_b) * 180.0 / M_PI;

            int classification = 2;
            if (angle1_deg < 30.0 && angle2_deg < 30.0) {
                classification = 1; // Sequential
            } else if (angle1_deg > 75.0 && angle2_deg > 75.0) {
                classification = 0; // Parallel
            }

            // smoothed_proj_out << u1 << " " << v1 << " " << u2 << " " << v2 << " "
            //                 << t_img1(0) << " " << t_img1(1) << " "
            //                 << t_img2(0) << " " << t_img2(1) << " "
            //                 << classification << " 1" << "\n";  // weight = 1
            // smoothed_proj_out << u1 << " " << v1 << " " << u2 << " " << v2 << " "
            //       << t_img1(0) << " " << t_img1(1) << " "
            //       << t_img2(0) << " " << t_img2(1) << " "
            //       << classification << " 1 " << "\n";
            //> write projected edge location and orientation to a file
            smoothed_proj_out << proj_edge_1_location(0) << "\t" << proj_edge_1_location(1) << "\t" \
                              << proj_edge_2_location(0) << "\t" << proj_edge_2_location(1) << "\t" \
                              << proj_tangent_1(0) << "\t" << proj_tangent_1(1) << "\t" \
                              << proj_tangent_2(0) << "\t" << proj_tangent_2(1) << "\t" \
                              << classification << " 1 " << "\n";

            smoothed_id++;
        }
    }

    smoothed_proj_out.close();

    std::cout << "[INFO] Wrote smoothed 3D edge groups to 2D view." << std::endl;
    ////////////////////////// write after smoothing data //////////////////////////


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
