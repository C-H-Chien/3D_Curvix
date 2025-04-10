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




///////////////////////// compute weighted graph stats /////////////////////////
template <typename T>
T clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(v, hi));
}

std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                   HashEigenVector3dPair, FuzzyVector3dPairEqual>
EdgeMapping::computeGraphEdgeDistanceAndAngleStats(
    std::unordered_map<std::pair<Eigen::Vector3d, Eigen::Vector3d>, int, 
                       HashEigenVector3dPair, FuzzyVector3dPairEqual>& graph,
    double lambda1, double lambda2)
{
    std::vector<double> distances;
    std::vector<double> angles;
     std::unordered_map<
    std::pair<Eigen::Vector3d, Eigen::Vector3d>,
    int,
    HashEigenVector3dPair,
    FuzzyVector3dPairEqual
    > pruned_graph;


    for (const auto& [pair, weight] : graph) {
        const Eigen::Vector3d& p1 = pair.first;
        const Eigen::Vector3d& p2 = pair.second;

        double dist = (p1 - p2).norm();
        distances.push_back(dist);

        auto it1 = edge_3D_to_supporting_edges.find(p1);
        auto it2 = edge_3D_to_supporting_edges.find(p2);
        if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end()) continue;

        Eigen::Vector3d t1 = Eigen::Vector3d::Zero();
        for (const auto& data : it1->second) t1 += data.tangents_3D_world;
        t1.normalize();

        Eigen::Vector3d t2 = Eigen::Vector3d::Zero();
        for (const auto& data : it2->second) t2 += data.tangents_3D_world;
        t2.normalize();

        double dot = clamp(t1.dot(t2), -1.0, 1.0);
        double angle = std::acos(dot);
        angles.push_back(angle);

        //distance_angle_map[pair] = {dist, angle};
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
    std::cout << "μ1 (distance): " << mu1 << ", σ1: " << sigma1 << "\n";
    std::cout << "μ2 (angle rad): " << mu2 << ", σ2: " << sigma2 << "\n";

    int removed_count = 0;
    for (const auto& [pair, weight] : graph) {
        const Eigen::Vector3d& p1 = pair.first;
        const Eigen::Vector3d& p2 = pair.second;

        double dist = (p1 - p2).norm();

        auto it1 = edge_3D_to_supporting_edges.find(p1);
        auto it2 = edge_3D_to_supporting_edges.find(p2);
        if (it1 == edge_3D_to_supporting_edges.end() || it2 == edge_3D_to_supporting_edges.end()) continue;

        Eigen::Vector3d t1 = Eigen::Vector3d::Zero();
        for (const auto& data : it1->second) t1 += data.tangents_3D_world;
        t1.normalize();

        Eigen::Vector3d t2 = Eigen::Vector3d::Zero();
        for (const auto& data : it2->second) t2 += data.tangents_3D_world;
        t2.normalize();

        double dot = clamp(t1.dot(t2), -1.0, 1.0);
        double angle = std::acos(dot);

        //if ((dist<mu1 + sigma1*lambda1 && angle<mu2+sigma2*lambda2)|| weight > 1){
        if (dist<0.0002 || weight > 1){
            pruned_graph[pair]++;
        }else{
            removed_count++;
        }
    }

    std::cout << "[PRUNING COMPLETE] Edges removed with weight == 1: " << removed_count << "\n";

    return pruned_graph;
}
///////////////////////// compute weighted graph stats /////////////////////////


std::vector<std::vector<EdgeMapping::SupportingEdgeData>> 
EdgeMapping::findMergable2DEdgeGroups(const std::vector<Eigen::Matrix3d>& all_R,
                                      const std::vector<Eigen::Vector3d>& all_T,
                                      int Num_Of_Total_Imgs) 
{ 
    auto uncorrected_map = map_Uncorrected2DEdge_To_SupportingData();
    auto graph = build3DEdgeWeightedGraph(uncorrected_map);
    graph = computeGraphEdgeDistanceAndAngleStats(graph,0, 0);

    std::vector<std::vector<SupportingEdgeData>> merged_groups;

    // Intrinsic matrix
    double fx = 1111.11136542426;
    double fy = 1111.11136542426;
    double cx = 399.5;
    double cy = 399.5;
    Eigen::Matrix3d K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;

    // Projection view's extrinsics
    Eigen::Matrix3d R_view;
    R_view <<  -0.865629,   -0.500686, -4.12948e-08,
              -0.221951,    0.383728,     0.896376,
              0.448803,     -0.77593,    0.443294;
    Eigen::Vector3d T_view(0.683157, -0.529078, -3.90556);

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
    for (const auto& [pair, weight] : graph) {
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


        // Eigen::Vector3d target1(0.4459, 0.0341785, 0.352049);
        // Eigen::Vector3d target2(0.447857, 0.0352899, 0.348052);
        // bool match_direct = (p1.isApprox(target1, 1e-6) && p2.isApprox(target2, 1e-6));
        // bool match_reverse = (p1.isApprox(target2, 1e-6) && p2.isApprox(target1, 1e-6));

        // if (match_direct || match_reverse) {
        //     std::cout << "[DEBUG]findMergable2DEdgeGroups Found target pair with weight: " << weight << std::endl;
        // }


        // Project both endpoints
        Eigen::Vector3d pc1 = R_view * p1 + T_view;
        Eigen::Vector3d pc2 = R_view * p2 + T_view;

        if (pc1(2) == 0 || pc2(2) == 0) {
            std::cout << "[WARNING] Z=0 for group " << group_id << " edge: skipping\n";
            continue;
        }

        Eigen::Vector3d pi1 = K * pc1;
        Eigen::Vector3d pi2 = K * pc2;

        double u1 = pi1(0) / pi1(2);
        double v1 = pi1(1) / pi1(2);
        double u2 = pi2(0) / pi2(2);
        double v2 = pi2(1) / pi2(2);

        // proj_outfile << u1 << " " << v1 << " " << u2 << " " << v2 << " ";

        // project 3d tangentes onto 2d images
        Eigen::Vector3d tangent_camera_1 =  R_view * tangent1;
        Eigen::Vector3d e3(0.0, 0.0, 1.0);
        Eigen::Vector2d gamma1_meters(-(u1 - cx) / fx, -(v1 - cy) / fy);
        Eigen::Vector2d gamma2_meters(-(u2 - cx) / fx, -(v2 - cy) / fy);
        double e3_dot_T1 = e3.dot(tangent_camera_1); 
        Eigen::Vector2d T1_xy(tangent_camera_1(0), tangent_camera_1(1));
        Eigen::Vector2d t_img1 = (T1_xy - e3_dot_T1 * gamma1_meters).normalized();

        // Project tangent2
        Eigen::Vector3d tangent_camera_2 =  R_view * tangent2;
        double e3_dot_T2 = e3.dot(tangent_camera_2); 
        Eigen::Vector2d T2_xy(tangent_camera_2(0), tangent_camera_2(1));
        Eigen::Vector2d t_img2 = (T2_xy - e3_dot_T2 * gamma2_meters).normalized();

        proj_outfile << u1 << " " << v1 << " " << u2 << " " << v2 << " "
             << t_img1(0) << " " << t_img1(1) << " "
             << t_img2(0) << " " << t_img2(1) << " ";


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


        proj_outfile << weight; 

        proj_outfile << "\n";
        edge_3d_outfile << p1.transpose() << " \t " << p2.transpose() << "\n";
        group_id++;
    }

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


