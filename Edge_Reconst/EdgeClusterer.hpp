#ifndef EDGE_CLUSTERER_HPP
#define EDGE_CLUSTERER_HPP

#include <vector>
#include <Eigen/Dense>

class EdgeClusterer {
public:
    // Perform clustering with constraints:
    // - Max intra-cluster distance <= 1 pixel
    // - Min centroid distance between clusters > 2 pixels
    void performClustering(const Eigen::MatrixXd& edgePoints,
                           std::vector<int>& clusterLabels,
                           std::vector<std::vector<int>>& clusters);

private:
    Eigen::Vector2d computeCentroid(const std::vector<int>& indices,
                                    const Eigen::MatrixXd& points);

    double maxIntraClusterDistance(const std::vector<int>& indices,
                                   const Eigen::MatrixXd& points);
};

#endif // EDGE_CLUSTERER_HPP
