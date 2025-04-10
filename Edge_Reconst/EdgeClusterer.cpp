#include "EdgeClusterer.hpp"
#include <unordered_map>
#include <numeric>
#include <limits>
#include <cmath>

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

void EdgeClusterer::performClustering(const Eigen::MatrixXd& edgePoints,
                                      std::vector<int>& clusterLabels,
                                      std::vector<std::vector<int>>& clusters) {
    int N = edgePoints.rows();
    clusterLabels.resize(N);
    std::iota(clusterLabels.begin(), clusterLabels.end(), 0);

    bool merged = true;
    while (merged) {
        merged = false;

        std::unordered_map<int, std::vector<int>> labelMap;
        for (int i = 0; i < N; ++i)
            labelMap[clusterLabels[i]].push_back(i);

        std::vector<std::pair<int, std::vector<int>>> clusterList;
        for (const auto& kv : labelMap) {
            clusterList.emplace_back(kv.first, kv.second);
        }


        for (size_t i = 0; i < clusterList.size(); ++i) {
            for (size_t j = i + 1; j < clusterList.size(); ++j) {
                const auto& cluster_i = clusterList[i].second;
                const auto& cluster_j = clusterList[j].second;

                Eigen::Vector2d centroid_i = computeCentroid(cluster_i, edgePoints);
                Eigen::Vector2d centroid_j = computeCentroid(cluster_j, edgePoints);
                double centroid_dist = (centroid_i - centroid_j).norm();

                if (centroid_dist <= 2.0)
                    continue;

                std::vector<int> mergedCluster = cluster_i;
                mergedCluster.insert(mergedCluster.end(), cluster_j.begin(), cluster_j.end());

                double maxIntra = maxIntraClusterDistance(mergedCluster, edgePoints);
                if (maxIntra > 1.0)
                    continue;

                int oldLabel = clusterList[j].first;
                int newLabel = clusterList[i].first;
                for (int k = 0; k < N; ++k) {
                    if (clusterLabels[k] == oldLabel)
                        clusterLabels[k] = newLabel;
                }
                merged = true;
                break;
            }
            if (merged) break;
        }
    }

    clusters.clear();
    std::unordered_map<int, std::vector<int>> finalMap;
    for (int i = 0; i < N; ++i)
        finalMap[clusterLabels[i]].push_back(i);

    for (auto& kv : finalMap)
        clusters.push_back(kv.second);
}
