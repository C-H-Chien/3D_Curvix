#ifndef EDGE_CLUSTERER_HPP
#define EDGE_CLUSTERER_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "definitions.h"
// #include "EdgeSketch_Core.hpp"

//> Custom hash function for std::pair<int, int>
    struct PairHash {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
    size_t h1 = std::hash<T1>{}(p.first);
    size_t h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 << 1);
    }
    };

class EdgeClusterer {
public:

    //> Constructor
    EdgeClusterer( int, Eigen::MatrixXd, int );

    // //> Custom hash function for std::pair<int, int>
    // struct PairHash {
    // template <class T1, class T2>
    // size_t operator()(const std::pair<T1, T2>& p) const {
    // size_t h1 = std::hash<T1>{}(p.first);
    // size_t h2 = std::hash<T2>{}(p.second);
    // return h1 ^ (h2 << 1);
    // }
    // };

    Eigen::MatrixXd performClustering( Eigen::MatrixXd HYPO2_idx_raw, Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd edgels_HYPO2_corrected );
    std::vector<int> cluster_labels;
    std::unordered_map<int, double> cluster_avg_orientations;
    double updateAvgOrientation( int label1, int label2 );

    std::vector<std::vector<int> > clusters;
    
    //> For each edge index, store all other edge indices in the same cluster
    std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> H2_Clusters; //<H1 edge index, H2 edge index>, cluster of H2 edges

private:
    int getClusterSize(int label);
    bool areSimilarOrientations(double orient1_deg, double orient2_deg);

    Eigen::Vector2d computeCentroid(const std::vector<int>& indices,
                                    const Eigen::MatrixXd& points);

    double maxIntraClusterDistance(const std::vector<int>& indices,
                                   const Eigen::MatrixXd& points);

    int H1_edge_idx;
    int Num_Of_Epipolar_Corrected_H2_Edges;
    Eigen::MatrixXd Epip_Correct_H2_Edges;

    // //> lambda expressions
    // //> (i) get the size of the cluster
    // std::function<int(int, int)> getClusterSize = [&cluster_labels](int label) -> int {
    //     int size = 0;
    //     for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
    //         if (cluster_labels[i] == label) size++;
    //     }
    //     return size;
    // };

    // //> (ii) Check if two orientations are within threshold in degrees
    // std::function<bool(double, double)> areSimilarOrientations = [](double orient1_deg, double orient2_deg) -> bool {
    //     double diff = std::fabs(orient1_deg - orient2_deg);
    //     return diff < CLUSTER_ORIENT_THRESH;
    // };
    
};

#endif // EDGE_CLUSTERER_HPP
