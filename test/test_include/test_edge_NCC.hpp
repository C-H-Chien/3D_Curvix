#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "../Edge_Reconst/file_reader.hpp"
#include "../Edge_Reconst/definitions.h"
#include "../Edge_Reconst/util.hpp"

void f_TEST_EDGE_ORIENTATION_AVG() {

    Eigen::MatrixXd edge_set;
    edge_set.conservativeResize(6, 3);
    edge_set.row(0) << 611.3950, 7.4897, 1.5623;
    edge_set.row(1) << 611.4090, 7.9915, 1.5564;
    edge_set.row(2) << 611.4210, 8.4932, 1.5637;
    edge_set.row(3) << 611.5030, 4.5006, -1.5276;
    edge_set.row(4) << 611.4360, 5.4918, -1.4957;
    edge_set.row(5) << 611.4730, 4.9990, -1.5429;

    //> TODO
    Eigen::Vector2d avg_unit_vec(0.0, 0.0);
    for (int i = 0; i < 6; i++) {
        Eigen::Vector2d mapped_unit_vec(cos(2*edge_set(i,2)), sin(2*edge_set(i,2)));
        avg_unit_vec += mapped_unit_vec;
    }
    avg_unit_vec /= 6.0;


}

bool test_getGTEdgePairsBetweenImages(int hyp01_view_indx, int hyp02_view_indx, \
                                      std::vector<std::pair<int, int>>& gt_edge_pairs, \
                                      std::vector<std::vector<int>> GT_EdgePairs ) 
{    
    gt_edge_pairs.clear();
    
    //> Sanity Check: make sure that GT data is loaded
    if (GT_EdgePairs.empty()) {
        LOG_ERROR("Error: Ground truth edge pairs data not loaded. Check Read_GT_EdgePairs_Data() first.");
        return false;
    }
    
    //> Loop through all ground truth 3D points
    for (const auto& gt_row : GT_EdgePairs) {
        //> Extract edge IDs for the two images
        int edge_id_img1 = gt_row[hyp01_view_indx + 1]-1;
        int edge_id_img2 = gt_row[hyp02_view_indx + 1]-1;
        
        //> if there is a valid pair between edges from the two hypothesis views
        if (edge_id_img1 >= 0 && edge_id_img2 >= 0) {
            gt_edge_pairs.push_back(std::make_pair(edge_id_img1, edge_id_img2));
        }
    }

    if (gt_edge_pairs.size() == 0) 
    {
        LOG_INFOR_MESG("Exiting the program due to zero GT edge correspondences");
        return false;
    }
    return true;
}

template <typename T>
double Bilinear_Interpolation(cv::Mat meshGrid, cv::Point2d P)
{
    //> y2 Q12--------Q22
    //      |          |
    //      |    P     |
    //      |          |
    //  y1 Q11--------Q21
    //      x1         x2
    cv::Point2d Q12(floor(P.x), floor(P.y));
    cv::Point2d Q22(ceil(P.x), floor(P.y));
    cv::Point2d Q11(floor(P.x), ceil(P.y));
    cv::Point2d Q21(ceil(P.x), ceil(P.y));

    if (Q11.x < 0 || Q11.y < 0 || Q21.x >= meshGrid.cols || Q21.y >= meshGrid.rows ||
        Q12.x < 0 || Q12.y < 0 || Q22.x >= meshGrid.cols || Q22.y >= meshGrid.rows)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double f_x_y1 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q11.y, Q11.x) + ((P.x - Q11.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q21.y, Q21.x);
    double f_x_y2 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q12.y, Q12.x) + ((P.x - Q11.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q22.y, Q22.x);
    return ((Q12.y - P.y) / (Q12.y - Q11.y)) * f_x_y1 + ((P.y - Q11.y) / (Q12.y - Q11.y)) * f_x_y2;
}

template<typename T>
T Uniform_Random_Number_Generator(T range_from, T range_to) {
    std::random_device                                          rand_dev;
    std::mt19937                                                rng(rand_dev());
    std::uniform_int_distribution<std::mt19937::result_type>    distr(range_from, range_to);
    return distr(rng);
}

std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points( const Eigen::Vector3d edgel )
{
    double shifted_x1 = edgel(0) + ORTHOGONAL_SHIFT_MAG * (std::sin(edgel(2)));
    double shifted_y1 = edgel(1) + ORTHOGONAL_SHIFT_MAG * (-std::cos(edgel(2)));
    double shifted_x2 = edgel(0) + ORTHOGONAL_SHIFT_MAG * (-std::sin(edgel(2)));
    double shifted_y2 = edgel(1) + ORTHOGONAL_SHIFT_MAG * (std::cos(edgel(2)));

    cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
    cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

    return {shifted_point_plus, shifted_point_minus};
}

void get_patch_on_one_edge_side( cv::Point2d shifted_point, double theta, \
                                 cv::Mat &patch_coord_x, cv::Mat &patch_coord_y, \
                                 cv::Mat &patch_val, const cv::Mat img ) 
{
    int half_patch_size = floor(PATCH_SIZE / 2);
    for (int i = -half_patch_size; i <= half_patch_size; i++) {
        for (int j = -half_patch_size; j <= half_patch_size; j++) {
            //> get the rotated coordinate
            cv::Point2d rotated_point(cos(theta)*(i) - sin(theta)*(j) + shifted_point.x, sin(theta)*(i) + cos(theta)*(j) + shifted_point.y);
            patch_coord_x.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.x;
            patch_coord_y.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.y;

            //> get the image intensity of the rotated coordinate
            double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
            patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
        }
    }
}

double ComputeNCC(const cv::Mat patch_one, const cv::Mat patch_two)
{
    double mean_one = (cv::mean(patch_one))[0];
    double mean_two = (cv::mean(patch_two))[0];
    double sum_of_squared_one = (cv::sum((patch_one - mean_one).mul(patch_one - mean_one))).val[0];
    double sum_of_squared_two = (cv::sum((patch_two - mean_two).mul(patch_two - mean_two))).val[0];

    cv::Mat norm_one = (patch_one - mean_one) / sqrt(sum_of_squared_one);
    cv::Mat norm_two = (patch_two - mean_two) / sqrt(sum_of_squared_two);
    return norm_one.dot(norm_two);
}

double apply_Sigmoid_regularization(double x, const double a, const double b)
{
    return 1.0 / (1.0 + exp(-a*(x - b)));
}

double get_normalized_SSD(const cv::Mat patch_one, const cv::Mat patch_two) 
{
    //> numerator
    cv::Mat patch_element_mul = patch_one.mul(patch_two);
    double sum_of_squared_element_mul = (cv::sum((patch_element_mul).mul(patch_element_mul))).val[0];

    //> denominator
    double normalize_term = sqrt(cv::sum(patch_one.mul(patch_one)).val[0]) * sqrt(cv::sum(patch_two.mul(patch_two)).val[0]);

    return sum_of_squared_element_mul / normalize_term;
    // return apply_Sigmoid_regularization(sum_of_squared_element_mul / normalize_term, 1.0, 10000.0);
}

double get_similarity_CH(const cv::Mat patch_one, const cv::Mat patch_two) 
{
    double numerator_term = cv::sum((patch_one - patch_two).mul(patch_one - patch_two)).val[0];
    double normalize_term = cv::sum(patch_one.mul(patch_one)).val[0] + cv::sum(patch_two.mul(patch_two)).val[0];
    return 1 - numerator_term / normalize_term;
}

void f_TEST_ROTATED_PATCH() 
{
    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::string source_dataset_folder = "/gpfs/data/bkimia/Datasets/";
    std::string dataset_name = "ABC-NEF/";
    std::string object_name = "00000006";
    cv::Mat gray_img;

    file_reader data_loader(source_dataset_folder, dataset_name, object_name, 50);    

    Eigen::MatrixXd edges = data_loader.read_Edgels_Of_a_File(0, 1);              //> Edge_0_t1.txt
    bool b_get_img = data_loader.read_an_image(0, gray_img);
    if (!b_get_img) exit(1);
    
    int rand_edge_idx;
    while (true) {
        rand_edge_idx = Uniform_Random_Number_Generator< int >(0, edges.rows()-1);
        double target_edge_orient = edges(rand_edge_idx, 2); 
        if (fabs(fabs(target_edge_orient) - fabs(M_PI / 4.0)) < 0.02)
            break;
    }

    Eigen::Vector3d target_edge(edges(rand_edge_idx, 0), edges(rand_edge_idx, 1), edges(rand_edge_idx, 2));
    
    std::cout << "Picked edge: (" << target_edge(0) << ", " << target_edge(1) << ", " << target_edge(2) << ")" << std::endl;
    std::cout << "Orientation in degree: " << util->rad_to_deg(target_edge(2)) << std::endl;

    std::pair<cv::Point2d, cv::Point2d> shifted_points = get_Orthogonal_Shifted_Points( target_edge );

    cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

    //> get the patches on the two sides of the edge
    get_patch_on_one_edge_side( shifted_points.first,  target_edge(2), patch_coord_x_plus,  patch_coord_y_plus,  patch_plus,  gray_img );
    get_patch_on_one_edge_side( shifted_points.second, target_edge(2), patch_coord_x_minus, patch_coord_y_minus, patch_minus, gray_img );

    // std::cout << "Shifted point (+) location: (" << shifted_points.first.x << ", " << shifted_points.first.y << ")" << std::endl;
    // std::cout << "Patch (+) coordinates: " << std::endl;
    // std::cout << patch_coord_x_plus << std::endl;
    // std::cout << patch_coord_y_plus << std::endl;

    // std::cout << "Shifted point (-) location: (" << shifted_points.second.x << ", " << shifted_points.second.y << ")" << std::endl;
    // std::cout << "Patch (-) coordinates: " << std::endl;
    // std::cout << patch_coord_x_minus << std::endl;
    // std::cout << patch_coord_y_minus << std::endl;
}

void f_TEST_NCC() 
{
    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::string source_dataset_folder = "/gpfs/data/bkimia/Datasets/";
    std::string dataset_name = "ABC-NEF/";
    std::string object_name = "00000006";
    cv::Mat gray_img_H1, gray_img_H2;
    const int H1_index = 25;
    const int H2_index = 49;

    file_reader data_loader(source_dataset_folder, dataset_name, object_name, 50);

    //> get the GT edge pairs for testing the NCC scores
    std::vector<std::vector<int>> GT_EdgePairs;
    std::vector<std::pair<int, int>> gt_edge_pairs;
    data_loader.readGT_EdgePairs( GT_EdgePairs );
    test_getGTEdgePairsBetweenImages( H1_index, H2_index, gt_edge_pairs, GT_EdgePairs );
    
    //> Read the two images 
    if (!data_loader.read_an_image( H1_index, gray_img_H1 )) exit(1);
    if (!data_loader.read_an_image( H2_index, gray_img_H2 )) exit(1);

    if (gray_img_H1.type() != CV_64F) gray_img_H1.convertTo(gray_img_H1, CV_64F);
    if (gray_img_H2.type() != CV_64F) gray_img_H2.convertTo(gray_img_H2, CV_64F);

    //> Read the third-order edges from the two images
    Eigen::MatrixXd edges_H1 = data_loader.read_Edgels_Of_a_File(H1_index, 1);
    Eigen::MatrixXd edges_H2 = data_loader.read_Edgels_Of_a_File(H2_index, 1);

    //> Randomly select a GT edge pair
    int rand_GT_edge_pair_idx = Uniform_Random_Number_Generator< int >(0, gt_edge_pairs.size()-1);
    std::pair<int, int> edge_pair_index = gt_edge_pairs[ rand_GT_edge_pair_idx ];
    std::cout << "Selected edge pair index = (" << edge_pair_index.first << ", " << edge_pair_index.second << ")" << std::endl;
    Eigen::Vector3d target_edge_H1(edges_H1(edge_pair_index.first, 0), edges_H1(edge_pair_index.first, 1), edges_H1(edge_pair_index.first, 2));
    Eigen::Vector3d target_edge_H2(edges_H2(edge_pair_index.second, 0), edges_H2(edge_pair_index.second, 1), edges_H2(edge_pair_index.second, 2));

    std::cout << "Picked H1 edge: (" << target_edge_H1(0) << ", " << target_edge_H1(1) << ", " << target_edge_H1(2) << ")" << std::endl;
    std::cout << "Orientation in degree: " << util->rad_to_deg(target_edge_H1(2)) << std::endl;
    std::cout << "Picked H2 edge: (" << target_edge_H2(0) << ", " << target_edge_H2(1) << ", " << target_edge_H2(2) << ")" << std::endl;
    std::cout << "Orientation in degree: " << util->rad_to_deg(target_edge_H2(2)) << std::endl;

    std::pair<cv::Point2d, cv::Point2d> shifted_points_H1 = get_Orthogonal_Shifted_Points( target_edge_H1 );
    std::pair<cv::Point2d, cv::Point2d> shifted_points_H2 = get_Orthogonal_Shifted_Points( target_edge_H2 );

    std::cout << "Orthogoanl shifted point for H1 edge: (" << shifted_points_H1.first.x << ", " << shifted_points_H1.second.y << ")" << std::endl;
    std::cout << "Orthogoanl shifted point for H2 edge: (" << shifted_points_H2.first.x << ", " << shifted_points_H2.second.y << ")" << std::endl;

    cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_plus_H1       = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_minus_H1      = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_plus_H2       = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_minus_H2      = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

    //> get the two patches on the two sides of the edge in H1
    get_patch_on_one_edge_side( shifted_points_H1.first,  target_edge_H1(2), patch_coord_x_plus,  patch_coord_y_plus,  patch_plus_H1,  gray_img_H1 );
    get_patch_on_one_edge_side( shifted_points_H1.second, target_edge_H1(2), patch_coord_x_minus, patch_coord_y_minus, patch_minus_H1, gray_img_H1 );

    //> get the two patches on the two sides of the edge in H2
    get_patch_on_one_edge_side( shifted_points_H2.first,  target_edge_H2(2), patch_coord_x_plus,  patch_coord_y_plus,  patch_plus_H2,  gray_img_H2 );
    get_patch_on_one_edge_side( shifted_points_H2.second, target_edge_H2(2), patch_coord_x_minus, patch_coord_y_minus, patch_minus_H2, gray_img_H2 );

    if (patch_plus_H1.type() != CV_32F)     patch_plus_H1.convertTo(patch_plus_H1, CV_32F);
    if (patch_minus_H1.type() != CV_32F)    patch_minus_H1.convertTo(patch_minus_H1, CV_32F);
    if (patch_plus_H2.type() != CV_32F)     patch_plus_H2.convertTo(patch_plus_H2, CV_32F);
    if (patch_minus_H2.type() != CV_32F)    patch_minus_H2.convertTo(patch_minus_H2, CV_32F);

    std::cout << "patch_plus_H1 = \n" << patch_plus_H1 << std::endl;
    std::cout << "patch_minus_H1 = \n" << patch_minus_H1 << std::endl;
    std::cout << "patch_plus_H2 = \n" << patch_plus_H2 << std::endl;
    std::cout << "patch_minus_H2 = \n" << patch_minus_H2 << std::endl;

    //> compare the patches to get the scores of:
    //> (i) NCC scores
    double ncc_pp = ComputeNCC(patch_plus_H1, patch_plus_H2);   //> (A+, B+)
    double ncc_nn = ComputeNCC(patch_minus_H1, patch_minus_H2); //> (A-, B-)
    double ncc_pn = ComputeNCC(patch_plus_H1, patch_minus_H2);  //> (A+, B-)
    double ncc_np = ComputeNCC(patch_minus_H1, patch_plus_H2);  //> (A-, B+) 
    std::cout << "- NCC scores of (pp, nn, pn, np) = (" << ncc_pp << ", " << ncc_nn << ", " << ncc_pn << ", " << ncc_np << ")" << std::endl;
    double final_NCC_score = std::max({ncc_pp, ncc_nn, ncc_pn, ncc_np});
    std::cout << "  Final NCC score = " << final_NCC_score << std::endl;

    //> (ii) NSSD scores
    double nssd_pp = get_normalized_SSD(patch_plus_H1, patch_plus_H2);   //> (A+, B+)
    double nssd_nn = get_normalized_SSD(patch_minus_H1, patch_minus_H2); //> (A-, B-)
    double nssd_pn = get_normalized_SSD(patch_plus_H1, patch_minus_H2);  //> (A+, B-)
    double nssd_np = get_normalized_SSD(patch_minus_H1, patch_plus_H2);  //> (A-, B+) 
    std::cout << "- NSSD scores of (pp, nn, pn, np) = (" << nssd_pp << ", " << nssd_nn << ", " << nssd_pn << ", " << nssd_np << ")" << std::endl;
    double final_NSSD_score = std::max({nssd_pp, nssd_nn, nssd_pn, nssd_np});
    std::cout << "  Final NSSD score = " << final_NSSD_score << std::endl;

    //> (iii) CH's similarity scores
    double sim_pp = get_similarity_CH(patch_plus_H1, patch_plus_H2);   //> (A+, B+)
    double sim_nn = get_similarity_CH(patch_minus_H1, patch_minus_H2); //> (A-, B-)
    double sim_pn = get_similarity_CH(patch_plus_H1, patch_minus_H2);  //> (A+, B-)
    double sim_np = get_similarity_CH(patch_minus_H1, patch_plus_H2);  //> (A-, B+)
    std::cout << "- CH's Similarity scores of (pp, nn, pn, np) = (" << sim_pp << ", " << sim_nn << ", " << sim_pn << ", " << sim_np << ")" << std::endl;
    double final_SIM_score = std::max({sim_pp, sim_nn, sim_pn, sim_np});
    std::cout << "  Final CH's Similarity score = " << final_SIM_score << std::endl;
    


}

