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
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "../Edge_Reconst/file_reader.hpp"
#include "../Edge_Reconst/definitions.h"
#include "../Edge_Reconst/util.hpp"

void f_TEST_SIFT_DESCP_ON_EDGES() 
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

    // if (gray_img_H1.type() != CV_64F) gray_img_H1.convertTo(gray_img_H1, CV_64F);
    // if (gray_img_H2.type() != CV_64F) gray_img_H2.convertTo(gray_img_H2, CV_64F);

    //> Read the third-order edges from the two images
    Eigen::MatrixXd edges_H1 = data_loader.read_Edgels_Of_a_File(H1_index, 1);
    Eigen::MatrixXd edges_H2 = data_loader.read_Edgels_Of_a_File(H2_index, 1);

    //> Randomly select a GT edge pair
    int rand_GT_edge_pair_idx = Uniform_Random_Number_Generator< int >(0, gt_edge_pairs.size()-1);
    std::pair<int, int> edge_pair_index = gt_edge_pairs[ rand_GT_edge_pair_idx ];
    std::cout << "Selected edge pair index = (" << edge_pair_index.first << ", " << edge_pair_index.second << ")" << std::endl;

    cv::Point2d target_edge_H1(edges_H1(edge_pair_index.first, 0), edges_H1(edge_pair_index.first, 1));
    cv::Point2d target_edge_H2(edges_H2(edge_pair_index.second, 0), edges_H2(edge_pair_index.second, 1));

    std::cout << "Picked H1 edge: (" << target_edge_H1.x << ", " << target_edge_H1.y << ")" << std::endl;
    std::cout << "Picked H2 edge: (" << target_edge_H2.x << ", " << target_edge_H2.y << ")" << std::endl;

    cv::Mat descriptors_H1, descriptors_H2;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    //> Convert an edge location to a KeyPoint data type
    cv::KeyPoint edge_kpt_H1(target_edge_H1, 1.0f);
    cv::KeyPoint edge_kpt_H2(target_edge_H2, 1.0f);

    std::vector<cv::KeyPoint> edge_kpts_H1;
    edge_kpts_H1.push_back(edge_kpt_H1);
    std::vector<cv::KeyPoint> edge_kpts_H2;
    edge_kpts_H2.push_back(edge_kpt_H2);

    sift->compute(gray_img_H1, edge_kpts_H1, descriptors_H1);
    sift->compute(gray_img_H2, edge_kpts_H2, descriptors_H2);

    cv::Mat normalized_desc_H1, normalized_desc_H2;
    cv::normalize(descriptors_H1, normalized_desc_H1, 1.0, 0.0, cv::NORM_L2); 
    cv::normalize(descriptors_H2, normalized_desc_H2, 1.0, 0.0, cv::NORM_L2); 

    std::cout << "Normalized SIFT Descriptors" << std::endl;
    std::cout << normalized_desc_H1 << std::endl;
    std::cout << normalized_desc_H2 << std::endl;

    double similarity = cv::norm(normalized_desc_H1, normalized_desc_H2, cv::NORM_L2);
    std::cout << "Descriptor-based similarity: " << similarity << std::endl;
}

