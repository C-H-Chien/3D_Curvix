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

//> Select the test
#define TEST_EDGE_ALIGNMENT         (true)
#define TEST_CONNECTIVITY_GRAPH     (true)  //> TEST_EDGE_ALIGNMENT has to be true to activate this test
#define TANGENT_COORD_TRANSFORM     (false)
#define ALIGN_VEC_BY_RODRIGUE       (false)
#define TANGENT_PROJECTION          (false)

struct EdgeNode {
    Eigen::Vector3d location;  
    Eigen::Vector3d orientation; 
    std::vector<std::pair<int, EdgeNode*>> neighbors;
};

using EdgeNodeList = std::vector<std::unique_ptr<EdgeNode>>;

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
    std::pair<int, int> connected_edge_index = findConnectedEdges( proj_neighbor );

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

///////////////////////// Smoothing 3d edges with its neighbors /////////////////////////
void align3DEdgesUsingEdgeNodes(EdgeNodeList& edge_nodes, int iterations, double location_step_size, double orientation_step_size) {

    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::ofstream before_out("../../outputs/test_3D_edges_before_smoothing.txt");

    for (size_t i = 0; i < edge_nodes.size(); ++i) {
        const auto& node = edge_nodes[i];
        std::cout << node->location.transpose() << "; ";
        before_out << node->location.transpose() << " " << node->orientation.transpose()<<"\n";
    }
    std::cout << "\n";
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

            // std::cout << node->location.transpose()<< "; ";
            //std::cout << node->orientation.transpose() << ";";

            // if(iter == iterations-1){
                after_out << node->location.transpose() << " " << node->orientation.transpose() << "\n";
                // if (target_indices.count(i)){
                //     //std::cout << node->location.transpose()<<" "<<node->orientation.transpose()<< "; "<<std::endl;
                // }
            // }
        }
        // std::cout << "\n";
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

Eigen::Matrix3d getSkewSymmetric(Eigen::Vector3d T) {
    Eigen::Matrix3d T_x = (Eigen::Matrix3d() << 0.,  -T(2),   T(1), T(2),  0.,  -T(0), -T(1),  T(0),   0.).finished();
    return T_x;
}

Eigen::Matrix3d getRodriguesRotationMatrix(Eigen::Vector3d v1, Eigen::Vector3d v2) {

    //> make sure that the input vectors are unit-vectors
    v1 /= v1.norm();
    v2 /= v2.norm();

    Eigen::Vector3d v1_cross_v2 = v1.cross(v2);
    double s = v1_cross_v2.norm();
    double c = v1.dot(v2);
    double coeff = 1.0 / (1.0 + c);
    Eigen::Matrix3d v_x = getSkewSymmetric(v1_cross_v2);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + v_x + coeff * v_x * v_x;
    return R;
}

void Compute_3D_Tangents(
    const Eigen::MatrixXd& pt_edge_view1,
    const Eigen::MatrixXd& pt_edge_view2,
    const Eigen::Matrix3d& K1,
    const Eigen::Matrix3d& K2,
    const Eigen::Matrix3d& R21,
    const Eigen::Vector3d& T21,
    Eigen::MatrixXd& tangents_3D)
{
    tangents_3D.resize(1, 3);

    Eigen::Vector3d e1  = {1,0,0};
    Eigen::Vector3d e3  = {0,0,1};

    // Normalize edge points
    Eigen::Vector3d Gamma1 = K1.inverse() * Eigen::Vector3d(pt_edge_view1(0), pt_edge_view1(1), 1.0);
    Eigen::Vector3d Gamma2 = K2.inverse() * Eigen::Vector3d(pt_edge_view2(0), pt_edge_view2(1), 1.0);

    Eigen::Vector3d tgt1(cos(pt_edge_view1(2)), sin(pt_edge_view1(2)), 0.0);
    Eigen::Vector3d tgt2(cos(pt_edge_view2(2)), sin(pt_edge_view2(2)), 0.0);
    Eigen::Vector3d tgt1_meters = K1.inverse() * tgt1;
    Eigen::Vector3d tgt2_meters = K2.inverse() * tgt2;

    double rho1 = (double(e1.transpose() * T21) - double(e3.transpose() * T21) * double(e1.transpose() *Gamma2))/(double(e3.transpose() * R21 * Gamma1)* double(e1.transpose() * Gamma2) - double(e1.transpose() * R21 * Gamma1));

    Eigen::Vector3d n1 = tgt1_meters.cross(Gamma1);
    Eigen::Vector3d n2 = R21.transpose() * tgt2_meters.cross(Gamma2);

    Eigen::Vector3d T3D = n1.cross(n2) / (n1.cross(n2) ).norm();
    tangents_3D = T3D;
}

Eigen::Matrix3d euler_to_rotation_matrix(double roll, double pitch, double yaw) {
    //> Create rotation matrices for each axis
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    //> Combine the rotations 
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    
    //> Convert to rotation matrix
    return q.toRotationMatrix();
}

// MARK: MAIN
int main(int argc, char **argv) {

#if TEST_EDGE_ALIGNMENT
    //> [TEST]
    /////////// Create EdgeNodes from input files ///////////
    std::string points_file = "../../files/line_noisy_points.txt";
    std::string tangents_file = "../../files/line_noisy_tangents.txt";
    std::string connections_file = "../../files/line_connections.txt";
    
    EdgeNodeList edge_node = createEdgeNodesFromFiles(points_file, tangents_file, connections_file);

    int num_iteration = 50;
    double location_step_size = 0.1;
    double orientation_step_size = 0.02;
    align3DEdgesUsingEdgeNodes(edge_node, num_iteration, location_step_size, orientation_step_size);

#if TEST_CONNECTIVITY_GRAPH
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
#endif

#endif  //> end of TEST_EDGE_ALIGNMENT

//> ---------------------------------------
#if TANGENT_COORD_TRANSFORM
    //> [TEST] 3D tangents
    // Given Rotation and Translation Matrices
    MultiviewGeometryUtil::multiview_geometry_util util;
    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d T1, T2;

    R1 << 0.229714, 0.973258, -1.29728e-07,
            -0.220415, 0.0520239, 0.974018,
            -0.947971, 0.223746, -0.226472;
    T1 << -0.601486, -0.402814, -3.93395;
    R2 << 0.159322, -0.987227, -3.64782e-08,
            -0.0530916, -0.00856814, 0.998553,
            0.985798, 0.159092, 0.0537785;
    T2 << 0.413953, -0.468447, -3.95085;

    // Camera Intrinsic Matrix (example, adjust if needed)
    Eigen::Matrix3d K_;
    K_ << 2584.93250981950, 0, 249.771375872214,
            0, 2584.79186060577, 278.312679379194,
            0, 0, 1;

    // Point coordinates from two images
    std::vector<Eigen::Vector2d> points_img1 = {
        {520.590800, 428.498217}, {520.581695, 427.998471}, {520.573102, 427.498616},
        {520.564490, 426.998731}, {520.555647, 426.498837}, {520.645733, 430.996517},
        {520.634568, 430.496660}, {520.623192, 429.996899}, {520.611786, 429.497305},
        {520.600795, 428.997801}};
    std::vector<Eigen::Vector2d> points_img2 = {
        {240.855540, 429.000920}, {240.849805, 428.001160}, {240.839928, 427.001388},
        {240.839928, 427.001388}, {240.834057, 426.501440}, {240.885916, 432.001500},
        {240.872839, 431.001501}, {240.866802, 430.501425}, {240.861729, 430.001200},
        {240.855540, 429.000920}};

    // Relative Rotation and Translation
    Eigen::Matrix3d R21 = R2 * R1.transpose();
    Eigen::Vector3d T21 = T2 - R21 * T1;

    for (size_t i = 0; i < points_img1.size(); ++i) {
        // Prepare the corresponding points
        Eigen::Vector2d pt1 = points_img1[i];
        Eigen::Vector2d pt2 = points_img2[i];
        Eigen::Vector3d pt1_h(pt1(0), pt1(1), 1.0);
        Eigen::Vector3d pt2_h(pt2(0), pt2(1), 1.0);

        // Triangulate the 3D point
        std::vector<Eigen::Vector2d> pts_test = {pt1, pt2};
        std::vector<Eigen::Matrix3d> Rs_test = {R21};
        std::vector<Eigen::Vector3d> Ts_test = {T21};

        Eigen::Vector3d edge_pt_3D = util.linearTriangulation(2, pts_test, Rs_test, Ts_test, K_);
        Eigen::Vector3d edge_pt_3D_world = util.transformToWorldCoordinates(edge_pt_3D, R1, T1);

        // Compute the 3D tangent
        Eigen::MatrixXd tangent_3D;
        Compute_3D_Tangents(pt1_h, pt2_h, K_, K_, R21, T21, tangent_3D);
        Eigen::Vector3d tangent_3D_world = R1.transpose() * tangent_3D;

        // Print results
        std::cout << "Point " << i + 1 << ": (" << edge_pt_3D_world(0) << ", " << edge_pt_3D_world(1) << ", " << edge_pt_3D_world(2) << ")" << std::endl;
        std::cout << "Tangent " << i + 1 << ": (" << tangent_3D_world(0) << ", " << tangent_3D_world(1) << ", " << tangent_3D_world(2) << ")" << std::endl;
    }
    std::cout << "----------------------------------------------------------------------" << std::endl;
#endif

#if ALIGN_VEC_BY_RODRIGUE
    std::cout << "[TEST] Verify the correctness of the aligning two unit vectors in 3D by Rodrigues' formula" << std::endl;
    Eigen::Vector3d v1(0.0838570596738639,	0.941037461651904,	0.327744548864806);
    Eigen::Vector3d v2(0.0245471150744476,	0.969469975008114,	0.243978291450876);
    Eigen::Matrix3d R_align_v1_to_v2 = getRodriguesRotationMatrix(v1, v2);
    Eigen::Matrix3d R_align_v2_to_v1 = getRodriguesRotationMatrix(v2, v1);
    // Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(v1, v2);
    // Eigen::Matrix3d R_align_v1_to_v2 = q.toRotationMatrix();


    Eigen::Vector3d aligned_v2 = R_align_v1_to_v2 * v1;
    std::cout << aligned_v2.transpose() << std::endl;
    Eigen::Vector3d eulerAnglesXYZ = R_align_v1_to_v2.eulerAngles(0, 1, 2);
    eulerAnglesXYZ = eulerAnglesXYZ * (180.0 / M_PI);
    std::cout << "Euler angles (in degrees):" << std::endl << eulerAnglesXYZ.transpose() << std::endl;

    eulerAnglesXYZ = R_align_v2_to_v1.eulerAngles(0, 1, 2);
    eulerAnglesXYZ *= -1;
    Eigen::Matrix3d R_test = euler_to_rotation_matrix(eulerAnglesXYZ(0), eulerAnglesXYZ(1), eulerAnglesXYZ(2));
    eulerAnglesXYZ = eulerAnglesXYZ * (180.0 / M_PI);
    std::cout << "Euler angles (in degrees):" << std::endl << eulerAnglesXYZ.transpose() << std::endl;
    

    // R_align_v1_to_v2 = R_align_v2_to_v1.transpose();
    aligned_v2 = R_test * v1;
    std::cout << aligned_v2.transpose() << std::endl;
    eulerAnglesXYZ = R_test.eulerAngles(0, 1, 2);
    eulerAnglesXYZ = eulerAnglesXYZ * (180.0 / M_PI);
    std::cout << "Euler angles (in degrees):" << std::endl << eulerAnglesXYZ.transpose() << std::endl;

    std::cout << "----------------------------------------------------------------------" << std::endl;
#endif

#if TANGENT_PROJECTION
    //> [TEST] Verify the correctness of projecting a 3D unit tangent vector (in world coordinate) to a 2D image
    Eigen::Vector3d Tangent_3D_w(-0.604940217207650, 0.119991911797586, 0.787178045112998);
    Eigen::Matrix3d Rot;
    Rot <<  0.283307133314224,  0.599568603826573,  0.748501541427090, \
           -0.799349013318374, -0.283603601454719,  0.529726488056670, \
            0.529885103697223, -0.748389261378974,  0.398917648559720;

    Eigen::Vector3d point_location(313.128094185699, 221.221179621611, 1.0);
    Eigen::Matrix3d K;
    K << 2584.93250981950,	0,	249.771375872214, \
         0,	2584.79186060577,	278.312679379194, \
         0,	0,	1;
    
    Eigen::Vector3d point_in_meters = K.inverse() * point_location;
    Eigen::Vector3d Tangent_3D_c = Rot * Tangent_3D_w;
    Eigen::Vector3d tangent_2D   = Tangent_3D_c - Tangent_3D_c(2) * point_in_meters;
    tangent_2D.normalize();
    std::cout << "projected tangent is (" << tangent_2D(0) << ", " << tangent_2D(1) << ", " << tangent_2D(2) << ")" << std::endl;
#endif
    return 0;
}