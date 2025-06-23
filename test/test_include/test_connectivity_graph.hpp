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

struct EdgeNode {
    Eigen::Vector3d location;  
    Eigen::Vector3d orientation; 
    std::vector<std::pair<int, EdgeNode*>> neighbors;
};

using EdgeNodeList = std::vector<std::unique_ptr<EdgeNode>>;

std::pair<int, int> findConnectedEdges_test(std::vector<double> proj_neighbor) {
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

    return {left_idx, right_idx};
}

std::pair<int, int> findConnectedEdges(std::vector<double> proj_neighbor) {
    if (proj_neighbor.size() < 2) {
        return {-1, -1}; 
    }

    int left_index, right_index;

    //> Split into all positive latent variables and all negative latent variables
    std::vector<double> positive_latent_variables;
    std::vector<double> negative_latent_variables;
    for (size_t i = 0; i < proj_neighbor.size(); ++i) {
        if (proj_neighbor[i] >= 0)
            positive_latent_variables.push_back(proj_neighbor[i]);
        else
            negative_latent_variables.push_back(proj_neighbor[i]);
    }

    //> Check if there exists a left edge and if so, find the min of positive_latent_variables
    if (positive_latent_variables.empty()) {
        left_index = -1;
    }
    else {
        auto min_element_it = std::min_element(positive_latent_variables.begin(), positive_latent_variables.end());
        double min_val = *min_element_it;
        auto it = std::find(proj_neighbor.begin(), proj_neighbor.end(), min_val);
        left_index = std::distance(proj_neighbor.begin(), it);
    }

    //> Check if there exists a right edge and if so, find the max of negative_latent_variables
    if (negative_latent_variables.empty()) {
        right_index = -1;
    }
    else {
        auto max_element_it = std::max_element(negative_latent_variables.begin(), negative_latent_variables.end());
        double max_val = *max_element_it;
        auto it = std::find(proj_neighbor.begin(), proj_neighbor.end(), max_val);
        right_index = std::distance(proj_neighbor.begin(), it);
    }
    // std::cout << "left_index = " << left_index << ", right_index = " << right_index << std::endl;

    return {left_index, right_index};
}

std::vector<std::pair<int, Eigen::Vector3d>> getConnectivityGraph(EdgeNode* node, int& left_index, int& right_index) {

    //>**************************************************************************************************
    //> Write to the file for testing
    std::ofstream before_out("../../outputs/test_3D_edges_before_connectivity_graph.txt");
    before_out << node->location.transpose() << "\t" << node->orientation.transpose() << "\n";
    for (const auto& neighbor_pair : node->neighbors) {
        const EdgeNode* neighbor = neighbor_pair.second;
        before_out << neighbor->location.transpose() << "\t" << neighbor->orientation.transpose() << "\n";
    }
    before_out.close();
    //>**************************************************************************************************

    //> Begin of constructing connectivity graph
    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::vector<std::pair<int, Eigen::Vector3d>> updated_neighbors;
    std::vector<double> proj_neighbor;
    for (const auto& neighbor_pair : node->neighbors) {
        const int neighbor_index = neighbor_pair.first;
        const EdgeNode* neighbor = neighbor_pair.second;

        //> Check if the neighbor edge should flip its orientation
        double factor = (util->checkOrientationConsistency(node->orientation, neighbor->orientation)) ? (1) : (-1);
        updated_neighbors.push_back(std::make_pair(neighbor_index, factor * neighbor->orientation));

        //> Find the latent variable of the line that the node edge is on
        double line_latent_variable = util->getLineVariable(node->location, node->orientation, neighbor->location);
        proj_neighbor.push_back(line_latent_variable);

        // std::cout << neighbor->location.transpose() << "\tline_latent_variable = " << line_latent_variable << std::endl;
    }
    // std::pair<int, int> connected_edge_index = findConnectedEdges( proj_neighbor );
    std::pair<int, int> connected_edge_index = findConnectedEdges_test( proj_neighbor );

    //>**************************************************************************************************
    //> Write updated_neighbors to the file for testing
    std::ofstream after_out("../../outputs/test_3D_edges_after_connectivity_graph.txt");
    after_out << node->location.transpose() << "\t" << node->orientation.transpose() << "\n";
    int neighbor_counter = 0;
    for (const auto& neighbor_pair : node->neighbors) {
        const EdgeNode* neighbor = neighbor_pair.second;
        Eigen::Vector3d updated_orientation = updated_neighbors[neighbor_counter].second;
        after_out << neighbor->location.transpose() << "\t" << updated_orientation.transpose() << "\n";
        neighbor_counter++;
    }
    after_out.close();
    //>**************************************************************************************************

    if (connected_edge_index.first == -1 && connected_edge_index.second == -1) {
        LOG_WARNING("No left nor right edges!");
    }
    else if (connected_edge_index.first == -1) {
        LOG_WARNING("No left edge!");
        std::pair<int, Eigen::Vector3d> right_neighbor = updated_neighbors[connected_edge_index.second];
        right_index = right_neighbor.first;
    }
    else if (connected_edge_index.second == -1) {
        LOG_WARNING("No right edge!");
        std::pair<int, Eigen::Vector3d> left_neighbor = updated_neighbors[connected_edge_index.first];
        left_index = left_neighbor.first;
    }
    else {
        std::pair<int, Eigen::Vector3d> left_neighbor = updated_neighbors[connected_edge_index.first];
        std::pair<int, Eigen::Vector3d> right_neighbor = updated_neighbors[connected_edge_index.second];
        left_index = left_neighbor.first;
        right_index = right_neighbor.first;
    }

    return updated_neighbors;
}