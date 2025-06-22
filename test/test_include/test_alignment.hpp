#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>
#include <cmath>

#include "../Edge_Reconst/definitions.h"
#include "../Edge_Reconst/util.hpp"
#include "test_connectivity_graph.hpp"

void align3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes, int iterations, double location_step_size, double orientation_step_size) {

    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::ofstream before_out("../../outputs/test_3D_edges_before_smoothing.txt");
    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        const auto& node = edge_nodes[i];
        before_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
    }
    before_out.close();
    std::ofstream after_out("../../outputs/test_3D_edges_after_smoothing.txt");
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

            //> Force on the edge location orthogonal to neighbor's tangential direction
            Eigen::Vector3d sum_force = Eigen::Vector3d::Zero();
            for (const auto& neighbor_pair : node->neighbors) {
                const EdgeNode* neighbor = neighbor_pair.second;
                const Eigen::Vector3d& p = neighbor->location;
                const Eigen::Vector3d& t_neighbor = neighbor->orientation;
                const Eigen::Vector3d& B = node->location;

                Eigen::Vector3d tangential_direction = util->findClosestVectorFromPointToLine(p, t_neighbor, B);
                sum_force += tangential_direction;
            }
            sum_force /= static_cast<double>(node->neighbors.size());
            new_locations[i] = node->location + location_step_size * sum_force;

            //> Alining the orientation by Rodrigues' formula but with careful check on geometric lock
            Eigen::Vector3d sum_tangent = Eigen::Vector3d::Zero();
            Eigen::Vector3d sum_euler_angles = Eigen::Vector3d::Zero();
            for (const auto& neighbor_pair : node->neighbors) {
                const EdgeNode* neighbor = neighbor_pair.second;
                //> the returned euler angles are in degrees
                Eigen::Vector3d euler_angles = util->getShortestAlignEulerAnglesDegrees(node->orientation, neighbor->orientation);
                sum_euler_angles += euler_angles;
            }
            sum_euler_angles /= static_cast<double>(node->neighbors.size());
            sum_euler_angles *= orientation_step_size;
            // std::cout << "sum_euler_angles = " << sum_euler_angles.transpose() << std::endl;

            //> Convert from degrees to radians
            sum_euler_angles = sum_euler_angles * M_PI / 180.0;
            Eigen::Matrix3d R_align = util->euler_to_rotation_matrix(sum_euler_angles(0), sum_euler_angles(1), sum_euler_angles(2));
            new_orientations[i] = R_align * node->orientation;
        }

        //> Update all edge locations and orientations
        for (size_t i = 0; i < edge_nodes.size(); ++i) {
            edge_nodes[i]->location = new_locations[i];
            edge_nodes[i]->orientation = new_orientations[i];
            const auto& node = edge_nodes[i];

            if(iter % 10 == 0){
                after_out << node->location.transpose() << " " << node->orientation.transpose() << "\n";
            }
        }
    }

    after_out.close();
    std::string msg = "[ALIGNMENT COMPLETE] Aligned edges written to file after " + std::to_string(iterations) \
                    + " iterations with location step size " + std::to_string(location_step_size) \
                    + " and orientation step size " + std::to_string(orientation_step_size); 
    LOG_GEN_MESG(msg);
}

EdgeNodeList createEdgeNodesFromFiles(const std::string& points_file, 
    const std::string& tangents_file, 
    const std::string& connections_file) 
{
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
                //> Add neighbor to current node
                // current_node->neighbors.push_back(node_ptrs[neighbor_index]);
                current_node->neighbors.push_back(std::make_pair(neighbor_index, node_ptrs[neighbor_index]));
            }
        }
        index++;
    }

    connections_infile.close();

    // Print node list information
    std::cout << std::endl;
    LOG_GEN_MESG("[NODE LIST SUMMARY]");
    std::cout << "Total nodes: " << node_list.size() << std::endl;
    std::cout << "Created " << node_list.size() << " edge nodes with connections from files." << std::endl;
    return node_list;
}

//MARK: TEST ALIGNMENT MAIN
void test_alignment( bool b_test_connectivity_graph = true ) {

    //> [TEST]
    /////////// Create EdgeNodes from input files ///////////
    std::string points_file = "../../files/synthetic_circle_noisy_points.txt";      //> circle_edge_locations, line_noisy_points
    std::string tangents_file = "../../files/synthetic_circle_noisy_tangents.txt";  //> circle_edge_orientations, line_noisy_tangents
    std::string connections_file = "../../files/synthetic_circle_connections.txt";  //> circle_edge_connections_renumbered, line_connections
    
    EdgeNodeList edge_node = createEdgeNodesFromFiles(points_file, tangents_file, connections_file);

    int num_iteration = 1000;
    double location_step_size = 0.1;
    double orientation_step_size = 1;
    align3DEdgesUsingEdgeNodes(edge_node, num_iteration, location_step_size, orientation_step_size);

    //MARK: TEST CONNECTIVITY GRAPH
    if ( b_test_connectivity_graph ) {

        const auto& target_node = edge_node[13];
        int left_edge_index = -1;
        int right_edge_index = -1;
        std::vector<std::pair<int, Eigen::Vector3d>> updated_neighbors;
        updated_neighbors = getConnectivityGraph(target_node.get(), left_edge_index, right_edge_index);

        //> Update the orientation of the neighbors
        for (const auto& updated_neighbor_pair : updated_neighbors) {
            int index = updated_neighbor_pair.first;
            Eigen::Vector3d updated_orientation = updated_neighbor_pair.second;
            const auto& node = edge_node[index];
            node->orientation = updated_orientation;
        }

        if ( left_edge_index >= 0 && right_edge_index >= 0 ) {
            //> If both left and right edges exist
            const auto& left_node = edge_node[left_edge_index];
            const auto& right_node = edge_node[right_edge_index];

            //>**************************************************************************************************
            //> Write to the file for testing
            std::ofstream target_left_right_out("../../outputs/test_3D_edge_left_and_right.txt");
            target_left_right_out << target_node->location.transpose() << "\t" << target_node->orientation.transpose() << "\n";
            target_left_right_out << left_node->location.transpose() << "\t" << left_node->orientation.transpose() << "\n";
            target_left_right_out << right_node->location.transpose() << "\t" << right_node->orientation.transpose() << "\n";
            target_left_right_out.close();
            //>**************************************************************************************************
        }
        else if ( left_edge_index == -1 && right_edge_index >= 0 ) {
            //> If only the right edge exists
            const auto& right_node = edge_node[right_edge_index];

            //>**************************************************************************************************
            //> Write to the file for testing
            std::ofstream target_left_right_out("../../outputs/test_3D_edge_left_and_right.txt");
            target_left_right_out << target_node->location.transpose() << "\t" << target_node->orientation.transpose() << "\n";
            target_left_right_out << right_node->location.transpose() << "\t" << right_node->orientation.transpose() << "\n";
            target_left_right_out << right_node->location.transpose() << "\t" << right_node->orientation.transpose() << "\n";
            target_left_right_out.close();
            //>**************************************************************************************************
        }
        else {
            //> If only the left edge exists
            const auto& left_node = edge_node[left_edge_index];

            //>**************************************************************************************************
            //> Write to the file for testing
            std::ofstream target_left_right_out("../../outputs/test_3D_edge_left_and_right.txt");
            target_left_right_out << target_node->location.transpose() << "\t" << target_node->orientation.transpose() << "\n";
            target_left_right_out << left_node->location.transpose() << "\t" << left_node->orientation.transpose() << "\n";
            target_left_right_out << left_node->location.transpose() << "\t" << left_node->orientation.transpose() << "\n";
            target_left_right_out.close();
            //>**************************************************************************************************
        }
    }
}