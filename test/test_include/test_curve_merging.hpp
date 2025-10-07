#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>

#include "../../Edge_Reconst/definitions.h"
#include "../../Edge_Reconst/edge_mapping.hpp"

//> helper function to trim leading/trailing spaces
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return ""; //> all whitespace
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

//> Read edge connectivity graph from file
void read_connectivity_graph_from_file(EdgeMapping::ConnectivityGraph& connectivity_graph, \
                                       std::shared_ptr<EdgeMapping> edgeMapping, const std::string file_name)
{
    const std::string file_path = "../../test/test_data/" + file_name;
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        LOG_CANNOT_OPEN_FILE_ERROR(file_path);
        return;
    }

    std::string line;
    int node_index;

    //> Skip the first two lines
    std::getline(infile, line);
    std::getline(infile, line);

    #define graph_node edgeMapping->connectivity_graph_node_member
    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        //> Replace semicolons by spaces
        for (char& c : line) {
            if (c == ';') c = ' ';
        }

        std::stringstream ss(line);

        // edgeMapping->connectivity_graph_node_member graph;

        ss >> node_index \
           >> graph_node.location.x() >> graph_node.location.y() >> graph_node.location.z() \
           >> graph_node.orientation.x() >> graph_node.orientation.y() >> graph_node.orientation.z() \
           >> graph_node.left_neighbor.first \
           >> graph_node.left_neighbor.second.x() >> graph_node.left_neighbor.second.y() >> graph_node.left_neighbor.second.z() \
           >> graph_node.right_neighbor.first \
           >> graph_node.right_neighbor.second.x() >> graph_node.right_neighbor.second.y() >> graph_node.right_neighbor.second.z();

        if (ss) {
            connectivity_graph[node_index] = graph_node;
        } 
        else {
            LOG_WARNING("Warning: failed to parse line");
        }
    }
}

//> Read curves from connectivity graph
void read_curves_from_connectivity_graph(std::vector<EdgeMapping::Curve>& curves, const std::string file_name)
{
    const std::string file_path = "../../test/test_data/" + file_name;
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        LOG_CANNOT_OPEN_FILE_ERROR(file_path);
        return;
    }

    std::string line;
    int curve_index;
    int edge_index;
    int curve_count = 0;
    curves.resize(1000);

    //> Skip the first two lines
    std::getline(infile, line);
    std::getline(infile, line);

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);

        //> switch to the next curve if an empty line is found
        if (trim(line).empty()) {
            curve_count++;
            if (curve_count >= 1000)
                curves.resize(curves.size() + 1000);
            continue;
        }

        ss >> curve_index >> edge_index;
        (curves[curve_index].edge_indices).push_back(edge_index);
    }
}

//> Read curve indices sets
std::vector<std::vector<int>> read_curve_indices_sets(const std::string file_name)
{
    std::vector<std::vector<int>> curve_indices_sets;
    const std::string file_path = "../../test/test_data/" + file_name;
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        LOG_CANNOT_OPEN_FILE_ERROR(file_path);
        return curve_indices_sets;
    }
    
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue; // skip empty lines
        std::stringstream ss(line);
        int val;
        std::vector<int> row;

        while (ss >> val) {
            row.push_back(val);
        }
        curve_indices_sets.push_back(std::move(row));
    }
    return curve_indices_sets;
}

//> Ensure orientation consistency between two curves
std::vector<int> make_curve_orientation_consistent(
    const std::vector<int>& curve1,
    const std::vector<int>& curve2,
    const EdgeMapping::ConnectivityGraph& connectivity_graph)
{
    Eigen::Vector3d d1 = connectivity_graph.at(curve1.back()).location - connectivity_graph.at(curve1.front()).location;
    Eigen::Vector3d d2 = connectivity_graph.at(curve2.back()).location - connectivity_graph.at(curve2.front()).location;

    //> reversed orientation if needed
    return (d1.dot(d2) < 0) ? std::vector<int>(curve2.rbegin(), curve2.rend()) : curve2;
}

//> Merge two curves into one
std::vector<int> merge_curve_pair(
    const std::vector<int>& curve1,
    const std::vector<int>& curve2,
    const EdgeMapping::ConnectivityGraph& connectivity_graph)
{
    std::vector<int> c2 = make_curve_orientation_consistent(curve1, curve2, connectivity_graph);

    //> Compute endpoint distances
    double dAthenB = (connectivity_graph.at(curve1.back()).location - connectivity_graph.at(c2.front()).location).norm();
    double dBthenA = (connectivity_graph.at(c2.back()).location   - connectivity_graph.at(curve1.front()).location).norm();
    double dRevBthenA = (connectivity_graph.at(c2.front()).location - connectivity_graph.at(curve1.front()).location).norm();
    double dAthenRevB = (connectivity_graph.at(curve1.back()).location - connectivity_graph.at(c2.back()).location).norm();

    //> Choose the best option
    double bestDist = dAthenB;
    std::string choice = "AthenB";
    if (dBthenA < bestDist) { bestDist = dBthenA; choice = "BthenA"; }
    if (dRevBthenA < bestDist) { bestDist = dRevBthenA; choice = "revBthenA"; }
    if (dAthenRevB < bestDist) { bestDist = dAthenRevB; choice = "AthenrevB"; }

    std::vector<int> merged;
    if (choice == "AthenB") {
        merged = curve1;
        merged.insert(merged.end(), c2.begin(), c2.end());
    } 
    else if (choice == "BthenA") {
        merged = c2;
        merged.insert(merged.end(), curve1.begin(), curve1.end());
    } 
    else if (choice == "revBthenA") {
        merged.assign(c2.rbegin(), c2.rend());
        merged.insert(merged.end(), curve1.begin(), curve1.end());
    } 
    else { // "AthenrevB"
        merged = curve1;
        merged.insert(merged.end(), c2.rbegin(), c2.rend());
    }

    return merged;
}

//> Merge multiple curves by CH
std::vector<int> merge_multiple_curves(const std::vector<int> curve_indices, \
                                       const std::vector<EdgeMapping::Curve> all_curves, \
                                       const EdgeMapping::ConnectivityGraph& connectivity_graph)
{
    //> Each curve in `curves` is a sequence of edge indices
    std::vector<std::vector<int>> curves;
    curves.resize(curve_indices.size());
    int curve_count = 0;
    for (const auto& crv_idx : curve_indices) {
        // std::cout << crv_idx << " ";
        curves[curve_count] = all_curves[crv_idx].edge_indices;
        curve_count++;
    }
    // std::cout << std::endl;

    std::vector<int> merged = curves.front();
    curves.erase(curves.begin());

    // std::cout << curves.size() << std::endl;

    while (!curves.empty()) {
        int bestIdx = -1;
        double bestDist = std::numeric_limits<double>::max();

        //> find the closest curve by the endpoint distance
        for (size_t i = 0; i < curves.size(); ++i) {
            const auto& c = curves[i];
            double d0 = (connectivity_graph.at(merged.front()).location - connectivity_graph.at(c.front()).location).norm();
            double d1 = (connectivity_graph.at(merged.front()).location - connectivity_graph.at(c.back()).location).norm();
            double d2 = (connectivity_graph.at(merged.back()).location  - connectivity_graph.at(c.front()).location).norm();
            double d3 = (connectivity_graph.at(merged.back()).location  - connectivity_graph.at(c.back()).location).norm();
            double d = std::min({d0, d1, d2, d3});
            if (d < bestDist) {
                bestDist = d;
                bestIdx = (int)i;
            }
        }

        //> put the best curve together
        merged = merge_curve_pair(merged, curves[bestIdx], connectivity_graph);
        curves.erase(curves.begin() + bestIdx);
    }

    return merged;
}

// //> Merge multiple curves
// std::vector<int> merge_multiple_curves(
//     std::vector<std::vector<int>> curves,
//     const std::vector<Eigen::Vector3d>& points)
// {
//     std::vector<int> merged = curves.front();
//     curves.erase(curves.begin());

//     while (!curves.empty()) {
//         int bestIdx = -1;
//         double bestDist = std::numeric_limits<double>::max();

//         //> find the closest curve by the endpoint distance
//         for (size_t i = 0; i < curves.size(); ++i) {
//             const auto& c = curves[i];
//             double d0 = (points[merged.front()] - points[c.front()]).norm();
//             double d1 = (points[merged.front()] - points[c.back()]).norm();
//             double d2 = (points[merged.back()]  - points[c.front()]).norm();
//             double d3 = (points[merged.back()]  - points[c.back()]).norm();
//             double d = std::min({d0, d1, d2, d3});
//             if (d < bestDist) {
//                 bestDist = d;
//                 bestIdx = (int)i;
//             }
//         }

//         //> put the best curve together
//         merged = merge_curve_pair(merged, curves[bestIdx], points);
//         curves.erase(curves.begin() + bestIdx);
//     }

//     return merged;
// }

//> Reorder points of a curve so it follows a smooth path (tangents of the curve is roughly consistent everywhere)
std::vector<int> reorderCurve(
    const std::vector<int>& curve,
    EdgeMapping::ConnectivityGraph& connectivity_graph)
{
    if (curve.size() <= 2) return curve;

    std::vector<int> reordered;
    // std::vector<bool> used(num_of_edges, false);

    //> make start as the leftmost point (smallest x)
    int start = curve[0];
    for (int idx : curve) {
        if (connectivity_graph.at(idx).location.x() < connectivity_graph.at(start).location.x()) 
            start = idx;
    }

    reordered.push_back(start);
    connectivity_graph.at(start).used = true;

    int current = start;
    Eigen::Vector3d prevDir(0,0,0);

    while (reordered.size() < curve.size()) {
        double bestScore = std::numeric_limits<double>::max();
        int bestIdx = -1;

        for (int idx : curve) {
            if ( connectivity_graph.at(idx).used ) continue;
            Eigen::Vector3d dir = connectivity_graph.at(idx).location - connectivity_graph.at(current).location;
            double dist = dir.norm();
            if (dist == 0) continue;

            dir.normalize();
            double anglePenalty = (reordered.size() > 1) ? (1.0 - prevDir.dot(dir)) : 0.0;
            
            //> weighted tradeoff between point pairwise distance and the curve smoothness
            // double score = dist + 0.01 * anglePenalty;
            double score = dist;

            if (score < bestScore) {
                bestScore = score;
                bestIdx = idx;
            }
        }

        if (bestIdx == -1) break;
        Eigen::Vector3d newDir = connectivity_graph.at(bestIdx).location - connectivity_graph.at(current).location;
        if (newDir.norm() > 1e-12) prevDir = newDir.normalized();
        current = bestIdx;
        reordered.push_back(current);
        connectivity_graph.at(current).used = true;
    }

    return reordered;
}

void write_final_curves_to_file(const std::vector<EdgeMapping::Curve>& curves, const std::string& file_name, const EdgeMapping::ConnectivityGraph& connectivity_graph, bool b_write_curve_info = false) 
{
    std::string file_path = OUTPUT_FOLDER_NAME + "/" + file_name + ".txt";
    std::ofstream outfile(file_path);
    
    if (!outfile.is_open()) {
        std::string err_msg = "Cannot open " + file_path + " for writing.";
        LOG_ERROR(err_msg);
        return;
    }
    
    outfile << "# Curves from Connectivity Graph\n";
    outfile << "# Format: CurveID 3DEdgeIndex Location(x,y,z) Orientation(x,y,z)\n";
    
    for (size_t curve_id = 0; curve_id < curves.size(); ++curve_id) {
        const auto& curve = curves[curve_id];
        
        for (const auto& node_index : curve.edge_indices) {
            const auto& node = connectivity_graph.at(node_index);
            outfile << curve_id << " " << node_index << " " << (node.location).transpose() << " " << node.orientation.transpose() << "\n";
        }
        
        //> Add a blank line between curves for better visualization
        outfile << "\n";
    }
    
    outfile.close();
    std::cout << "Wrote " << curves.size() << " curves to " << file_path << std::endl;
}

void f_test_curve_merging() {

    //> we need data structures from the EdgeMapping class
    std::shared_ptr<EdgeMapping> edgeMapping = std::make_shared<EdgeMapping>();

    EdgeMapping::ConnectivityGraph connectivity_graph;
    std::vector<EdgeMapping::Curve> all_curves;

    //> Read the connectivity graph
    std::string graph_file_name = "connectivity_graph.txt";
    read_connectivity_graph_from_file(connectivity_graph, edgeMapping, graph_file_name);

    //> Read curves from the connectivity graph
    std::string curves_from_connectivity_graph_file_name = "curves_from_connectivity_graph.txt";
    read_curves_from_connectivity_graph(all_curves, curves_from_connectivity_graph_file_name);

    //> Read curve indices set for curve merging
    std::string curve_set_file_name = "curve_indices_for_merging.txt";
    std::vector<std::vector<int>> curve_indices_sets = read_curve_indices_sets(curve_set_file_name);

    std::cout << "Edge indices forming a curve indexed by 3:" << std::endl;
    for (const auto& edge_idx : all_curves[3].edge_indices) {
        std::cout << edge_idx << " ";
    }
    std::cout << std::endl;
    
    // std::vector<Eigen::Vector3d> points = {
    //     {0.903229, 0.767576, -0.355823}, // 0
    //     {0.905429, 0.766344, -0.355838}, // 1
    //     {0.903540, 0.767363, -0.355819}, // 2
    //     {0.905217, 0.766591, -0.355850}, // 3
    //     {0.907000, 0.765500, -0.355900}, // 4
    //     {0.909000, 0.764800, -0.355950}  // 5
    // };

    // //> Synthetically define curves represented by indices
    // std::vector<int> curveA = {0, 1};
    // std::vector<int> curveB = {2, 3};
    // std::vector<int> curveC = {4, 5};

    // std::vector<std::vector<int>> fragments = {curveA, curveB, curveC};

    // auto merged = merge_multiple_curves(fragments, points);
    // auto smooth = reorderCurve(merged, points);

    // auto merged_curve = merge_multiple_curves(curve_indices_sets[0], all_curves, connectivity_graph);
    // std::cout << "Complete merge" << std::endl;
    // auto smooth_curve = reorderCurve(merged_curve, connectivity_graph);
    // std::cout << "Complete smooth" << std::endl;

    // for (int idx : merged_curve) {
    //     std::cout << connectivity_graph.at(idx).location.transpose() << "\n";
    // }

    std::vector<EdgeMapping::Curve> final_curves;
    final_curves.resize(curve_indices_sets.size());
    unsigned final_curve_count = 0;
    for (const auto& set_of_curve_ids : curve_indices_sets) {
        auto merged_curve = merge_multiple_curves(set_of_curve_ids, all_curves, connectivity_graph);
        final_curves[final_curve_count].edge_indices = reorderCurve(merged_curve, connectivity_graph);
        final_curve_count++;
    }
    write_final_curves_to_file(final_curves, "test_curves_from_connectivity_graph", connectivity_graph, false);
}
