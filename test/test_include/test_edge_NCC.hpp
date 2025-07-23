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
}

double Bilinear_Interpolation(const cv::Mat &meshGrid, cv::Point2d P)
{
    cv::Point2d Q12(floor(P.x), floor(P.y));
    cv::Point2d Q22(ceil(P.x), floor(P.y));
    cv::Point2d Q11(floor(P.x), ceil(P.y));
    cv::Point2d Q21(ceil(P.x), ceil(P.y));

    if (Q11.x < 0 || Q11.y < 0 || Q21.x >= meshGrid.cols || Q21.y >= meshGrid.rows ||
        Q12.x < 0 || Q12.y < 0 || Q22.x >= meshGrid.cols || Q22.y >= meshGrid.rows)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double fQ11 = meshGrid.at<float>(Q11.y, Q11.x);
    double fQ21 = meshGrid.at<float>(Q21.y, Q21.x);
    double fQ12 = meshGrid.at<float>(Q12.y, Q12.x);
    double fQ22 = meshGrid.at<float>(Q22.y, Q22.x);

    double f_x_y1 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ11 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ21;
    double f_x_y2 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ12 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ22;
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
            double interp_val = Bilinear_Interpolation(img, rotated_point);
            patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
        }
    }
}

void f_TEST_NCC() 
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

    std::cout << "Shifted point (+) location: (" << shifted_points.first.x << ", " << shifted_points.first.y << ")" << std::endl;
    std::cout << "Patch (+) coordinates: " << std::endl;
    std::cout << patch_coord_x_plus << std::endl;
    std::cout << patch_coord_y_plus << std::endl;

    std::cout << "Shifted point (-) location: (" << shifted_points.second.x << ", " << shifted_points.second.y << ")" << std::endl;
    std::cout << "Patch (-) coordinates: " << std::endl;
    std::cout << patch_coord_x_minus << std::endl;
    std::cout << patch_coord_y_minus << std::endl;

    if (patch_plus.type() != CV_32F) {
        patch_plus.convertTo(patch_plus, CV_32F);
    }
    if (patch_minus.type() != CV_32F) {
        patch_minus.convertTo(patch_minus, CV_32F);
    }

    //> compare the patches to get NCC scores
}

