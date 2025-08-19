#include "EdgeClusterer.hpp"
#include <unordered_map>
#include <numeric>
#include <limits>
#include <cmath>

EdgeClusterer::EdgeClusterer( int N, Eigen::MatrixXd Edges_HYPO2_final, int H1_edge_idx ) 
    : Num_Of_Epipolar_Corrected_H2_Edges(N), Epip_Correct_H2_Edges(Edges_HYPO2_final), H1_edge_idx(H1_edge_idx)
{
    //> Initialize each H2 edge as a single cluster. The cluster label of edge i is i-1, i.e., cluster_labels[i] = i-1
    // std::vector<int> cluster_labels(Num_Of_Epipolar_Corrected_H2_Edges);
    cluster_labels.resize(Num_Of_Epipolar_Corrected_H2_Edges);
    std::iota(cluster_labels.begin(), cluster_labels.end(), 0); // Each point starts in its own cluster

    //> util class object
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());
}

EdgeClusterer::EdgeClusterer( int N, std::vector<Eigen::Vector3d> Edges_HYPO2_final ) : Num_Of_Epipolar_Corrected_H2_Edges(N)
{
    //> Initialize each H2 edge as a single cluster. The cluster label of edge i is i-1, i.e., cluster_labels[i] = i-1
    // std::vector<int> cluster_labels(Num_Of_Epipolar_Corrected_H2_Edges);
    cluster_labels.resize(Num_Of_Epipolar_Corrected_H2_Edges);
    std::iota(cluster_labels.begin(), cluster_labels.end(), 0); // Each point starts in its own cluster

    Epip_Correct_H2_Edges.conservativeResize(Edges_HYPO2_final.size(), 3);
    for (int i = 0; i < Edges_HYPO2_final.size(); i++) {
        Epip_Correct_H2_Edges.row(i) = Edges_HYPO2_final[i];
    }

    //> Don't care about H1_edge_idx
    H1_edge_idx = 0;

    //> util class object
    util = std::shared_ptr<MultiviewGeometryUtil::multiview_geometry_util>(new MultiviewGeometryUtil::multiview_geometry_util());
}

//> If orientation (radians) is less than -90 + threshold/2, add 180 degrees
double EdgeClusterer::normalizeOrientation(double orientation) {

    double wrap_threshold = -90.0 + (CLUSTER_ORIENT_THRESH / 2.0);
    // double orientation_degrees = orientation * (180.0 / M_PI);

    double normalized_orientation = orientation;
    if (util->rad_to_deg(orientation) < wrap_threshold) {
        normalized_orientation += M_PI;
    }

    return normalized_orientation;
}

int EdgeClusterer::getClusterSize(int label) {
    int size = 0;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label) size++;
    }
    return size;
};

std::tuple<double, double, double> EdgeClusterer::computeGaussianAverage( int label1, int label2 ) 
{
    //> If label2 is -1 by default, compute for single cluster (label1 only)
    //> Otherwise, compute for merged cluster (label1 + label2)

    //> CH DOCUMENT:
    //> Given two clusters, find the corresponding H2 edges (with epipolar shifted).
    //  Then consider the two as a single cluster, calculate the centroid edge location and the distance of each edge member to the centroid edge location
    //  Finally, use the distance from the centroid edge location, calculate weighted average of the orientations from all the edge member
    //  The returned weighted average orientation is in degree
    
    // Calculate the geometric mean of the cluster(s)
    double sum_x = 0, sum_y = 0;
    int count = 0;
    
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label1 || (label2 != -1 && cluster_labels[i] == label2)) {
            sum_x += Epip_Correct_H2_Edges(i, 0);
            sum_y += Epip_Correct_H2_Edges(i, 1);
            count++;
        }
    }
    
    if (count == 0) return std::make_tuple(0.0, 0.0, 0.0);
    
    double centroid_x = sum_x / count;
    double centroid_y = sum_y / count;
    
    // Calculate mean shift distance from centroid for this cluster
    double total_shift_from_centroid = 0.0;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label1 || (label2 != -1 && cluster_labels[i] == label2)) {
            double dx = Epip_Correct_H2_Edges(i, 0) - centroid_x;
            double dy = Epip_Correct_H2_Edges(i, 1) - centroid_y;
            double distance_from_centroid = std::sqrt(dx*dx + dy*dy);
            total_shift_from_centroid += distance_from_centroid;
        }
    }
    double mean_shift_from_centroid = total_shift_from_centroid / count;
    
    // Calculate Gaussian-weighted averages for x, y, and orientation
    double sum_weighted_x = 0;
    double sum_weighted_y = 0;
    double sum_weighted_orientation = 0;
    double total_weight = 0;
    
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label1 || (label2 != -1 && cluster_labels[i] == label2)) {
            double dx = Epip_Correct_H2_Edges(i, 0) - centroid_x;
            double dy = Epip_Correct_H2_Edges(i, 1) - centroid_y;
            double distance_from_centroid = std::sqrt(dx*dx + dy*dy);
            double gaussian_weight = std::exp(-0.5 * std::pow((distance_from_centroid - mean_shift_from_centroid) / CLUSTER_ORIENT_GAUSS_SIGMA, 2));

            sum_weighted_x += gaussian_weight * Epip_Correct_H2_Edges(i, 0);
            sum_weighted_y += gaussian_weight * Epip_Correct_H2_Edges(i, 1);
            sum_weighted_orientation += gaussian_weight * Epip_Correct_H2_Edges(i, 2);
            total_weight += gaussian_weight;
        }
    }

    double gaussian_weighted_x = sum_weighted_x / total_weight;
    double gaussian_weighted_y = sum_weighted_y / total_weight;
    double gaussian_weighted_orientation = sum_weighted_orientation / total_weight;
    
    return std::make_tuple(gaussian_weighted_x, gaussian_weighted_y, gaussian_weighted_orientation);
}

Eigen::MatrixXd EdgeClusterer::performClustering( Eigen::MatrixXd HYPO2_idx_raw, Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd edgels_HYPO2_corrected, \
                                                  bool b_use_edge_sketch_H2_index_format, bool b_pair_up_with_H1_edge ) 
{
    //> Track average orientations for each cluster in degrees
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        double normalized_orient = normalizeOrientation(Epip_Correct_H2_Edges(i, 2));
        Epip_Correct_H2_Edges(i, 2) = normalized_orient;
        cluster_avg_orientations[i] = normalized_orient;
    }

    //> Merge clusters starting from closest pairs
    bool merged = true;
    while (merged) {
        merged = false;

        // For each point, find its nearest neighbor and merge if within threshold
        for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int nearest = -1;
            
            // Find the nearest edge to the current edge
            for (int j = 0; j < Num_Of_Epipolar_Corrected_H2_Edges; ++j) {
                if (cluster_labels[i] != cluster_labels[j]) {
                    double dist = (Epip_Correct_H2_Edges.row(i).head<2>() - Epip_Correct_H2_Edges.row(j).head<2>()).norm();
                    //> orient_i and orient_j are both in degrees
                    double orient_i = cluster_avg_orientations[cluster_labels[i]];
                    double orient_j = cluster_avg_orientations[cluster_labels[j]];
                    if (dist < min_dist && dist < CLUSTER_DIST_THRESH && std::abs(orient_i - orient_j) < util->deg_to_rad(CLUSTER_ORIENT_THRESH)) {
                        min_dist = dist;
                        nearest = j;
                    }
                }
            }
            // If found a nearest edge within threshold, merge clusters
            if (nearest != -1) {
                int old_label = cluster_labels[nearest];
                int new_label = cluster_labels[i];
                int size_old = getClusterSize(old_label);
                int size_new = getClusterSize(new_label);
                if (size_old + size_new <= MAX_CLUSTER_SIZE) {
                    //> Calculate new average orientation for the merged cluster
                    std::tuple<double, double, double> result = computeGaussianAverage( old_label, new_label );

                    double merged_orientation = std::get<2>(result);
                    // Update the average orientation of the merged cluster
                    //cluster_avg_orientations[new_label] = merged_orientation;

                    double normalized_merged_orient = normalizeOrientation(merged_orientation);
                    cluster_avg_orientations[new_label] = normalized_merged_orient;

                    // Update all points in the smaller cluster
                    for (int k = 0; k < Num_Of_Epipolar_Corrected_H2_Edges; ++k) {
                        if (cluster_labels[k] == old_label) {
                            cluster_labels[k] = new_label;
                        }
                    }

                    merged = true;
                    break;
                }
            }
        }
    }

    //> Group hypothesis edge indices by their cluster label
    // example: cluster_labels = [0, 0, 1, 2, 1]
    // result: cluster 0: edge 0 and 1; cluster 1: edge 2 and 4; cluster 2: edge 3
    std::map<int, std::vector<int> > label_to_cluster;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        label_to_cluster[cluster_labels[i]].push_back(i); 
    }

    //////////// push to clusters////////////
    
    if (b_pair_up_with_H1_edge) {
        std::unordered_map<int, std::vector<int>> thread_local_clusters;
        // thread_local_clusters.clear(); // Clears the thread-local cluster storage for the current hypothesis 1 edge

        std::map<int, std::vector<int> >::iterator kv_it;

        // label_to_cluster contains cluster labels as keys and vectors of local edge indices as values
        // example: {0: [0,1], 1: [2,4], 2: [3]} 
        for (kv_it = label_to_cluster.begin(); kv_it != label_to_cluster.end(); ++kv_it) {
            std::vector<int> original_indices;
            
            //> CH: In each cluster the H2 edges are epipolar corrected edges,
            //  but here we use HYPO2_idx_raw which comes from the original H2 edges.
            //  
            // Use HYPO2_idx_raw as a lookup table to convert local indices back to original edge indices in Edges_HYPO2 
            for (size_t i = 0; i < kv_it->second.size(); ++i) {
                int local_idx = kv_it->second[i];
                if (local_idx >= 0 && local_idx < HYPO2_idx_raw.rows()) {
                    int original_idx = static_cast<int>(HYPO2_idx_raw(local_idx));
                    if (original_idx >= 0 && original_idx < Edges_HYPO2.rows()) {
                        original_indices.push_back(original_idx);
                    }
                }
            }
            
            // For each edge in the cluster, it stores all edges in that cluster
            for (size_t i = 0; i < original_indices.size(); ++i) {
                int original_idx = original_indices[i];
                thread_local_clusters[original_idx] = original_indices;
            }
        }

        std::unordered_map<int, std::vector<int> >::iterator kv_it_;
        for (kv_it_ = thread_local_clusters.begin(); kv_it_ != thread_local_clusters.end(); ++kv_it_) {
            H2_Clusters[std::make_pair(H1_edge_idx, kv_it_->first)] = kv_it_->second;
        }
    }

    //> CH TODO DOCUMENTATION
    std::map<int, std::vector<int> >::iterator it;
    for (it = label_to_cluster.begin(); it != label_to_cluster.end(); ++it) {
        clusters.push_back(it->second);
    }

    Eigen::MatrixXd HYPO2_idx(Num_Of_Epipolar_Corrected_H2_Edges, 1);

    Num_Of_Clusters = clusters.size();

    //> For each cluster, compute the Gaussian-weighted average edge and update all edges in the cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const std::vector<int>& cluster = clusters[c];
        if (cluster.empty()) continue;

        int cluster_label = cluster_labels[cluster[0]];
        std::tuple<double, double, double> result = computeGaussianAverage( cluster_label );
        double gaussian_average_x = std::get<0>(result);
        double gaussian_average_y = std::get<1>(result);
        double gaussian_average_orientation = std::get<2>(result);
        
        // Create the Gaussian-weighted average edge
        Eigen::RowVector4d gaussian_weighted_avg;
        gaussian_weighted_avg << gaussian_average_x, gaussian_average_y, gaussian_average_orientation, Epip_Correct_H2_Edges(cluster[0], 3);
        
        // Find the edge closest to the Gaussian-weighted average to use as the representative
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = -1;
        for (size_t i = 0; i < cluster.size(); ++i) {
            int idx = cluster[i];
            double dist = (Epip_Correct_H2_Edges.row(idx).head<2>() - gaussian_weighted_avg.head<2>()).norm();
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = idx;
            }
        }
        
        // Update all edges in the cluster with the average edge
        for (size_t i = 0; i < cluster.size(); ++i) {
            int idx = cluster[i];
            Epip_Correct_H2_Edges.row(idx) = gaussian_weighted_avg;
            
            // Preserve the original index for reference
            if (b_use_edge_sketch_H2_index_format) {
                if (edgels_HYPO2_corrected.cols() > 8) {
                    HYPO2_idx(idx, 0) = edgels_HYPO2_corrected(closest_idx, 8);
                } else {
                    HYPO2_idx(idx, 0) = -2;
                }
            }
        }
    }

    return HYPO2_idx;
}

// Eigen::Vector2d EdgeClusterer::computeCentroid(const std::vector<int>& indices,
//                                                const Eigen::MatrixXd& points) {
//     Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
//     for (int idx : indices) {
//         centroid += points.row(idx).head<2>();
//     }
//     centroid /= static_cast<double>(indices.size());
//     return centroid;
// }

double EdgeClusterer::maxIntraClusterDistance(const std::vector<int>& indices,
                                              const Eigen::MatrixXd& points) {
    double max_dist = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = i + 1; j < indices.size(); ++j) {
            double dist = (points.row(indices[i]).head<2>() - points.row(indices[j]).head<2>()).norm();
            max_dist = std::max(max_dist, dist);
        }
    }
    return max_dist;
}
