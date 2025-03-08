#include "edge_mapping.hpp"
#include <iostream>
#include <unordered_set>
#include <algorithm>

void EdgeMapping::add3DToSupportingEdgesMapping(const Eigen::Vector3d &edge_3D, 
                                                const Eigen::Vector2d &supporting_edge, 
                                                int image_number, 
                                                const Eigen::Matrix3d &rotation,
                                                const Eigen::Vector3d &translation) {
    if (edge_3D_to_supporting_edges.find(edge_3D) == edge_3D_to_supporting_edges.end()) {
        edge_3D_to_supporting_edges[edge_3D] = {};
    }
    edge_3D_to_supporting_edges[edge_3D].push_back({supporting_edge, image_number, rotation, translation});
}

void EdgeMapping::add3DToFrameMapping(const Eigen::Vector3d& edge_3D, 
                                      const Eigen::Vector2d& supporting_edge, 
                                      int frame) {
    frame_to_edge_to_3D_map[frame][supporting_edge].push_back(edge_3D);
}

void EdgeMapping::printFirst10Edges() { 
    int count = 0;
    for (const auto& pair : edge_3D_to_supporting_edges) {
        if (count >= 10) break;

        std::cout << "3D Edge: (" << pair.first(0) << ", " << pair.first(1) << ", " << pair.first(2) << ")\n";
        std::cout << "Supporting 2D Edges:\n";

        for (const auto& support : pair.second) {
            std::cout << "   View " << support.image_number << ": (" << support.edge(0) << ", " << support.edge(1) << ")\n";
            std::cout << "   Rotation:\n" << support.rotation << "\n";
            std::cout << "   Translation: (" << support.translation.transpose() << ")\n";
        }
        std::cout << "-----------------------------\n";

        count++;
    }
}



std::vector<std::vector<EdgeMapping::SupportingEdgeData>> EdgeMapping::findMergable2DEdgeGroups() { 
    std::vector<std::vector<EdgeMapping::SupportingEdgeData>> all_groups; 
    std::unordered_map<Eigen::Vector3d, bool, HashEigenVector3d> visited;
    int merged_count = 0;
    int unmerged_count = 0;

    std::cout << "edge_3D_to_supporting_edges size: " << edge_3D_to_supporting_edges.size() << std::endl;

    // Mark all 3D edges as unvisited initially
    for (const auto& entry : edge_3D_to_supporting_edges) {
        visited[entry.first] = false;
    }

    // Outer loop: iterate over each unvisited 3D edge
    for (auto it1 = edge_3D_to_supporting_edges.begin(); it1 != edge_3D_to_supporting_edges.end(); ++it1) {
        const Eigen::Vector3d& edge_3D_1 = it1->first;
        auto& support_2D_1 = it1->second;

        if (visited[edge_3D_1]) continue;  // Skip if already visited

        // Start a new group and add all its supporting 2D edges
        std::vector<EdgeMapping::SupportingEdgeData> group(support_2D_1.begin(), support_2D_1.end()); 
        visited[edge_3D_1] = true;
        
        for (auto it2 = edge_3D_to_supporting_edges.begin(); it2 != edge_3D_to_supporting_edges.end(); ++it2) {
            const Eigen::Vector3d& edge_3D_2 = it2->first;
            auto& support_2D_2 = it2->second;

            if (edge_3D_1 == edge_3D_2 || visited[edge_3D_2]) continue; // Skip if same or already merged

            // Count the number of matching 2D edges from the same image
            int common_count = 0;
            for (const auto& edge1 : support_2D_1) {
                for (const auto& edge2 : support_2D_2) {
                    if (edge1.image_number == edge2.image_number &&
                        edge1.edge.isApprox(edge2.edge, 1e-4)) { // Looser precision
                        common_count++;
                    }
                    if (common_count >= 2) break;  // Stop early if already found enough matches
                }
                if (common_count >= 2) break;
            }

            // If at least 2 matching 2D edges exist, merge into the same group
            if (common_count >= 2) {
                group.insert(group.end(), support_2D_2.begin(), support_2D_2.end());
                visited[edge_3D_2] = true;
            }
        }

        // Track merged vs. unmerged groups
        if (group.size() > support_2D_1.size()) {
            merged_count++;
        } else {
            unmerged_count++;
        }

        all_groups.push_back(group);
    }

    // Check if all 3D edges were visited
    int not_visited_count = 0;
    for (const auto& entry : visited) {
        if (!entry.second) {
            not_visited_count++;
        }
    }

    std::cout << "[SUMMARY] Total Groups: " << all_groups.size() << std::endl;
    std::cout << "[SUMMARY] Merged Groups: " << merged_count << std::endl;
    std::cout << "[SUMMARY] Unmerged Groups: " << unmerged_count << std::endl;
    std::cout << "[WARNING] Unaccounted 3D edges: " << not_visited_count << " (should be 0)" << std::endl;

    // std::cout << "\n========== FIRST 5 GROUPS ==========\n";
    // int group_count = 0;
    // for (const auto& group : all_groups) {
    //     if (group.empty()) continue;  // Skip empty groups
    //     if (group_count >= 5) break;  // Limit to 5 groups

    //     std::cout << "Group " << group_count + 1 << " (" << group.size() << " edges):\n";
    //     for (const auto& support : group) {
    //         std::cout << "   [View " << support.image_number << "] Edge: ("
    //                 << support.edge(0) << ", " << support.edge(1) << ")\n";
    //         std::cout << "   Rotation:\n" << support.rotation << "\n";
    //         std::cout << "   Translation: (" << support.translation.transpose() << ")\n";
    //         std::cout << "--------------------------------------\n";
    //     }

    //     group_count++;
    // }
    // std::cout << "=====================================\n";


    return all_groups;
}
