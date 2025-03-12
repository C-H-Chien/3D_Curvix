#include "edge_mapping.hpp"
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <set>
#include <cmath>

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

    bool visited_CH[edge_3D_to_supporting_edges.size()] = {false};

    int merged_count = 0;
    int unmerged_count = 0;

    std::cout << "edge_3D_to_supporting_edges size: " << edge_3D_to_supporting_edges.size() << std::endl;

    // Outer loop: iterate over each unvisited 3D edge
    int counter_query_3D_edge = 0;
    for (auto it1 = edge_3D_to_supporting_edges.begin(); it1 != edge_3D_to_supporting_edges.end(); ++it1) {
        const Eigen::Vector3d& edge_3D_1 = it1->first;
        auto& support_2D_1 = it1->second;

        if (visited_CH[counter_query_3D_edge]) {
            counter_query_3D_edge++;
            continue;
        }

        // Use a set to ensure uniqueness of supporting edges
        std::set<EdgeMapping::SupportingEdgeData> unique_edges(support_2D_1.begin(), support_2D_1.end());
        visited_CH[counter_query_3D_edge] = true;
        
        int counter_database_3D_edge = 0;
        for (auto it2 = edge_3D_to_supporting_edges.begin(); it2 != edge_3D_to_supporting_edges.end(); ++it2) {
            
            const Eigen::Vector3d& edge_3D_2 = it2->first;
            auto& support_2D_2 = it2->second;
            
            if (edge_3D_1 == edge_3D_2 || visited_CH[counter_database_3D_edge]) {
                counter_database_3D_edge++;
                continue; // Skip if same or already merged
            }

            // Compute Euclidean distance between 3D edges
            double distance = (edge_3D_1 - edge_3D_2).norm(); 

            // Only merge if the distance is within 10 pixels
            if (distance > 0.005) {
                counter_database_3D_edge++;
                continue;
            }

            // Count the number of matching 2D edges from the same image
            int common_count = 0;
            for (const auto& edge1 : support_2D_1) {
                for (const auto& edge2 : support_2D_2) {
                    if (edge1.image_number == edge2.image_number &&
                        edge1.edge.isApprox(edge2.edge, 1e-4)) { // Looser precision
                        common_count++;
                        //std::cout<<"distance is: " <<distance <<std::endl;
                    }
                    if (common_count >= 2) break;  // Stop early if already found enough matches (?)
                }
                if (common_count >= 2) break;
            }

            // If at least 2 matching 2D edges exist, merge into the same group
            if (common_count >= 2) {
                unique_edges.insert(support_2D_2.begin(), support_2D_2.end());
                visited_CH[counter_database_3D_edge] = true;
            }

            counter_database_3D_edge++;
        }

        // Convert the unique set to a vector
        std::vector<EdgeMapping::SupportingEdgeData> group(unique_edges.begin(), unique_edges.end());

        // Track merged vs. unmerged groups
        if (group.size() > support_2D_1.size()) {
            merged_count++;
        } else {
            unmerged_count++;
        }

        all_groups.push_back(group);

        counter_query_3D_edge++;
    }

    assert(counter_query_3D_edge == edge_3D_to_supporting_edges.size());

    // Check if all 3D edges were visited
    int not_visited_count = 0;
    for (unsigned i = 0; i < edge_3D_to_supporting_edges.size(); i++) {
        if (!visited_CH[i]) {
            not_visited_count++;
        }
    }

    std::cout << "[SUMMARY] Total Groups: " << all_groups.size() << std::endl;
    std::cout << "[SUMMARY] Merged Groups: " << merged_count << std::endl;
    std::cout << "[SUMMARY] Unmerged Groups: " << unmerged_count << std::endl;
    std::cout << "[WARNING] Unaccounted 3D edges: " << not_visited_count << " (should be 0)" << std::endl;

    return all_groups;
}
