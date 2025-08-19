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
#include <set>

#include "../Edge_Reconst/file_reader.hpp"
#include "../Edge_Reconst/definitions.h"
#include "../Edge_Reconst/util.hpp"

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

std::vector<std::pair<int, int>> get_Unique_GT_H1_Edge_Index_Pairs(const std::vector<std::pair<int, int>>& input_vector) {
    std::set<int> unique_first_elements;
    std::vector<std::pair<int, int>> result_pairs;

    // Collect unique first elements
    for (const auto& p : input_vector) {
        unique_first_elements.insert(p.first);
    }

    // Collect corresponding pairs for unique first elements
    for (int unique_val : unique_first_elements) {
        // Find the first occurrence of a pair with this unique_val as its first element
        auto it = std::find_if(input_vector.begin(), input_vector.end(),
                               [unique_val](const std::pair<int, int>& p) {
                                   return p.first == unique_val;
                               });
        if (it != input_vector.end()) {
            result_pairs.push_back(*it);
        }
    }
    return result_pairs;
}

void f_TEST_GT_EDGE_PAIR() 
{
    std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util> util = nullptr;
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());

    std::string source_dataset_folder = "/gpfs/data/bkimia/Datasets/";
    std::string dataset_name = "ABC-NEF/";
    std::string object_name = "00000006";
    const int H1_index = 25;
    const int H2_index = 49;

    file_reader data_loader(source_dataset_folder, dataset_name, object_name, 50);

    //> get the GT edge pairs for testing the NCC scores
    std::vector<std::vector<int>> GT_EdgePairs;
    std::vector<std::pair<int, int>> gt_edge_pairs;
    data_loader.readGT_EdgePairs( GT_EdgePairs );
    test_getGTEdgePairsBetweenImages( H1_index, H2_index, gt_edge_pairs, GT_EdgePairs );
    
    std::cout << "Size of gt_edge_pairs = " << gt_edge_pairs.size() << std::endl;
    std::vector<std::pair<int, int>> unique_GT_edge_pairs = get_Unique_GT_H1_Edge_Index_Pairs(gt_edge_pairs);
   
    std::cout << "Size of the unique GT H1 edge = " << unique_GT_edge_pairs.size() << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << "(" << unique_GT_edge_pairs[i].first << ", " << unique_GT_edge_pairs[i].second << ")" << std::endl;
    }
    
}

