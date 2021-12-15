import numpy as np
import random

stop_threshold = 1e-4
cluster_threshold = 1e-1

class Mean_Shift:
    def __init__(self, radius):
        self.radius = radius

    def fit(self, points):
        centroids = []
        shifting = [True] * points.shape[0]

        for i in range(len(points)):
            centroids.append(points[i])
        
        while True:
            optimized = True
            pre_centroids = centroids.copy()
            for i in range(len(centroids)):
                if not shifting[i]:
                    continue
                optimized = False
                
                in_bandwidth = []
                centroid = centroids[i]
                for point in points:
                    if np.linalg.norm(point - centroid) < self.radius:
                        in_bandwidth.append(point)

                new_centroid = np.average(np.array(in_bandwidth), axis=0)
                centroids[i] = new_centroid
                distance = np.linalg.norm(new_centroid - centroid)
                shifting[i] = distance > stop_threshold
                
            if optimized:
                break
        
        cluster_ids = self.cluster_points(np.array(centroids).tolist())
        return centroids, cluster_ids
    
    def cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                find_center = False
                for center in cluster_centers:
                    distance = np.linalg.norm(np.array(point) - np.array(center))
                    if(distance < cluster_threshold):
                        cluster_ids.append(cluster_centers.index(center))
                        find_center = True
                        break
                if not find_center:
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return np.array(cluster_ids)