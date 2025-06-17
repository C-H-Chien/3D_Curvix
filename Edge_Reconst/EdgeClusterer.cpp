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

    //> Track average orientations for each cluster in degrees
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        cluster_avg_orientations[i] = Epip_Correct_H2_Edges(i, 2) * (180.0 / M_PI);
    }
}

int EdgeClusterer::getClusterSize(int label) {
    int size = 0;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label) size++;
    }
    return size;
};

bool EdgeClusterer::areSimilarOrientations(double orient1_deg, double orient2_deg) {
    return (std::fabs(orient1_deg - orient2_deg) < CLUSTER_ORIENT_THRESH) ? (true) : (false);
};

double EdgeClusterer::updateAvgOrientation( int label1, int label2 ) {
    //> CH DOCUMENT:
    //> Given two clusters, find the corresponding H2 edges (with epipolar shifted).
    //  Then consider the two as a single cluster, calculate the centroid edge location and the distance of each edge member to the centroid edge location
    //  Finally, use the distance from the centroid edge location, calculate weighted average of the orientations from all the edge member
    //  The returned weighted average orientation is in degree

    // calculate the mean of the merged cluster
    double sum_x = 0, sum_y = 0;
    int count = 0;
    
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label1 || cluster_labels[i] == label2) {
            sum_x += Epip_Correct_H2_Edges(i, 0);
            sum_y += Epip_Correct_H2_Edges(i, 1);
            count++;
        }
    }
    
    if (count == 0) return 0.0;
    
    double centroid_x = sum_x / count;
    double centroid_y = sum_y / count;
    
    // Calculate mean shift distance from centroid for this cluster
    double total_shift_from_centroid = 0.0;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label1 || cluster_labels[i] == label2) {
            double dx = Epip_Correct_H2_Edges(i, 0) - centroid_x;
            double dy = Epip_Correct_H2_Edges(i, 1) - centroid_y;
            double distance_from_centroid = std::sqrt(dx*dx + dy*dy);
            total_shift_from_centroid += distance_from_centroid;
        }
    }
    double mean_shift_from_centroid = total_shift_from_centroid / count;
    
    // calculate weighted average orientation using Gaussian weights
    double sum_weighted_orin = 0;
    
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
        if (cluster_labels[i] == label1 || cluster_labels[i] == label2) {
            double orientation_deg = Epip_Correct_H2_Edges(i, 2) * (180.0 / M_PI);
            double dx = Epip_Correct_H2_Edges(i, 0) - centroid_x;
            double dy = Epip_Correct_H2_Edges(i, 1) - centroid_y;
            double distance_from_centroid = std::sqrt(dx*dx + dy*dy);
            double weight = std::exp(-0.5 * std::pow((distance_from_centroid - mean_shift_from_centroid) / CLUSTER_ORIENT_GAUSS_SIGMA, 2));

            sum_weighted_orin += weight * orientation_deg;
        }
    }

    return sum_weighted_orin / count;
}



Eigen::Vector2d EdgeClusterer::computeCentroid(const std::vector<int>& indices,
                                               const Eigen::MatrixXd& points) {
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (int idx : indices) {
        centroid += points.row(idx).head<2>();
    }
    centroid /= static_cast<double>(indices.size());
    return centroid;
}

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

Eigen::MatrixXd EdgeClusterer::performClustering( Eigen::MatrixXd HYPO2_idx_raw, Eigen::MatrixXd Edges_HYPO2, Eigen::MatrixXd edgels_HYPO2_corrected ) {
    
    // Merge clusters starting from closest pairs
    bool merged = true;
    while (merged) {
        merged = false;

        // For each point, find its nearest neighbor and merge if within threshold
        for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int nearest = -1;
            
            // Find the nearest edge index "nearest" of the current edge (indexed by i)
            for (int j = 0; j < Num_Of_Epipolar_Corrected_H2_Edges; ++j) {
                if (cluster_labels[i] != cluster_labels[j]) {
                    double dist = (Epip_Correct_H2_Edges.row(i).head<2>() - Epip_Correct_H2_Edges.row(j).head<2>()).norm();
                    double orient_i = cluster_avg_orientations[cluster_labels[i]];
                    double orient_j = cluster_avg_orientations[cluster_labels[j]];
                    if (dist < min_dist && dist < CLUSTER_DIST_THRESH && areSimilarOrientations(orient_i, orient_j)) {
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

                    // Calculate new average orientation for the merged cluster
                    double merged_avg_orientation = updateAvgOrientation(old_label, new_label);

                    //std::cout<<"merged orientation is: "<<merged_avg_orientation <<std::endl;

                    // Update all points in the smaller cluster
                    for (int k = 0; k < Num_Of_Epipolar_Corrected_H2_Edges; ++k) {
                        if (cluster_labels[k] == old_label) {
                            cluster_labels[k] = new_label;
                        }
                    }
                    // Update the average orientation of the merged cluster
                    cluster_avg_orientations[new_label] = merged_avg_orientation;

                    //> Once the two clusters are merged, return to the beginning of the while loop
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

    // std::vector<std::vector<int> > clusters;
    std::map<int, std::vector<int> >::iterator it;
    for (it = label_to_cluster.begin(); it != label_to_cluster.end(); ++it) {
        clusters.push_back(it->second);
    }

    Eigen::MatrixXd HYPO2_idx(Num_Of_Epipolar_Corrected_H2_Edges, 1);

    //> CH TODO: In this part, the orientation is not weighted averaged!
    // For each cluster, compute the average edge and update all edges in the cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const std::vector<int>& cluster = clusters[c];

        // Compute average position and orientation
        Eigen::RowVector4d cluster_avg = Eigen::RowVector4d::Zero();
        for (size_t i = 0; i < cluster.size(); ++i) {
            int idx = cluster[i];
            cluster_avg += Epip_Correct_H2_Edges.row(idx);
        }
        cluster_avg /= static_cast<double>(cluster.size());
        
        // Find the edge closest to the average 
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = -1;
        for (size_t i = 0; i < cluster.size(); ++i) {
            int idx = cluster[i];
            double dist = (Epip_Correct_H2_Edges.row(idx).head<2>() - cluster_avg.head<2>()).norm();
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = idx;
            }
        }
        
        // Update all edges in the cluster with the average edge
        for (size_t i = 0; i < cluster.size(); ++i) {
            int idx = cluster[i];
            Epip_Correct_H2_Edges.row(idx) = cluster_avg;
            // Preserve the original index for reference
            if (edgels_HYPO2_corrected.cols() > 8) {
                HYPO2_idx(idx, 0) = edgels_HYPO2_corrected(closest_idx, 8);
            } else {
                HYPO2_idx(idx, 0) = -2;
            }
        }
    }

    return HYPO2_idx;
}
