__author__ = "Adriano Fonseca"
__email__ = "a.fonseca@ccom.unh.edu"
__version__ = "1.0.0"

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
import cupy as cp
import logging
from math import tan, radians, ceil

@dataclass
class QuadNode:
    bounds: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    points: np.ndarray
    children: List['QuadNode'] = None
    centroid: np.ndarray = None
    level: int = 0  # Track tree depth for progress

class BathymetryQuadTree:
    def __init__(self, config: dict):
        """Initialize quadtree with configuration."""
        # Clustering mode
        self.mode = config['clustering']['mode']
        if self.mode not in ['adaptive', 'fixed']:
            raise ValueError("Clustering mode must be either 'adaptive' or 'fixed'")
            
        # Fixed mode parameters
        self.points_per_leaf = config['clustering']['points_per_leaf']
        self.min_points = config['clustering']['min_points']
        
        # Adaptive mode parameters
        if self.mode == 'adaptive':
            self.target_cell_size = config['clustering']['target_cell_size']
            self.beam_angle = config['clustering']['beam_angle']
        
        # Common parameters
        self.max_tree_depth = config['clustering']['max_tree_depth']
        self.verbose = config['clustering']['verbose']
        self.current_max_depth = 0
        self.logger = logging.getLogger(__name__)
        
        # GPU settings
        self.gpu_config = config['processing']['gpu']
        self.memory_config = config['processing']['memory_management']
        
        # Progress tracking
        self.total_points = 0
        self.processed_points = 0
        self.pbar = None

    def get_optimal_points(self, points: np.ndarray) -> int:
        """Get optimal number of points based on mode."""
        if self.mode == 'fixed':
            return max(self.points_per_leaf, self.min_points)
        else:  # adaptive mode
            median_depth = np.median(points[:, 2])
            beam_footprint = 2 * abs(median_depth) * tan(radians(self.beam_angle/2))
            density_factor = (beam_footprint / self.target_cell_size) ** 2
            return max(ceil(density_factor), self.min_points)

    def _get_node_stats(self, points: np.ndarray, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate average depth and cell area for a node."""
        avg_depth = np.mean(points[:, 2])
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        cell_area = width * height
        return avg_depth, cell_area

    def _subdivide(self, points: np.ndarray, bounds: Tuple[float, float, float, float], level: int = 0) -> QuadNode:
        """Subdivide points recursively."""
        node = QuadNode(bounds=bounds, points=points, level=level)
        self.current_max_depth = max(self.current_max_depth, level)
        
        # Update progress tracking
        if self.verbose and self.pbar is not None:
            self.processed_points += len(points)
            self.pbar.update(len(points))
            self.pbar.set_postfix({
                "depth": level, 
                "max_depth": self.current_max_depth
            }, refresh=True)
        
        # Get optimal points based on mode
        optimal_points = self.get_optimal_points(points)
        
        # Check stopping criteria
        if len(points) <= optimal_points or level >= self.max_tree_depth:
            node.centroid = points.mean(axis=0)
            return node
        
        # Calculate midpoints for subdivision
        x_mid = (bounds[0] + bounds[2]) / 2
        y_mid = (bounds[1] + bounds[3]) / 2
        
        # Create masks for each quadrant
        q1_mask = (points[:, 0] <= x_mid) & (points[:, 1] > y_mid)
        q2_mask = (points[:, 0] > x_mid) & (points[:, 1] > y_mid)
        q3_mask = (points[:, 0] <= x_mid) & (points[:, 1] <= y_mid)
        q4_mask = (points[:, 0] > x_mid) & (points[:, 1] <= y_mid)
        
        # Check if any quadrant would have fewer than min_points
        quadrant_points = [
            np.sum(q1_mask), np.sum(q2_mask),
            np.sum(q3_mask), np.sum(q4_mask)
        ]
        
        if any(count < self.min_points for count in quadrant_points):
            node.centroid = points.mean(axis=0)
            return node
        
        # If no points, don't subdivide
        if len(points) == 0:
            return None
        
        # Create quadrants
        quads = []
        for i, x_bounds in enumerate([(bounds[0], x_mid), (x_mid, bounds[2])]):
            for j, y_bounds in enumerate([(bounds[1], y_mid), (y_mid, bounds[3])]):
                x_mask = (points[:, 0] >= x_bounds[0]) & (
                    points[:, 0] <= x_bounds[1] if i == 1 else points[:, 0] < x_bounds[1]
                )
                y_mask = (points[:, 1] >= y_bounds[0]) & (
                    points[:, 1] <= y_bounds[1] if j == 1 else points[:, 1] < y_bounds[1]
                )
                mask = x_mask & y_mask
                quad_points = points[mask]
                if len(quad_points) > 0:
                    quad_bounds = (x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1])
                    quad_node = self._subdivide(quad_points, quad_bounds, level + 1)
                    if quad_node is not None:
                        quads.append(quad_node)
        
        if not quads:  # If no valid subdivisions were created
            node.centroid = points.mean(axis=0)
            return node
            
        node.children = quads
        node.points = None  # Free memory
        return node

    def process_partition(self, points: np.ndarray, gpu_id: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process a partition of points using quadtree, optionally on GPU."""
        try:
            n_points = len(points)
            optimal_points = self.get_optimal_points(points)
            
            # Estimate depth based on optimal points for this depth
            estimated_depth = min(
                self.max_tree_depth,
                max(0, int(np.ceil(np.log(n_points / optimal_points) / np.log(4))))
            )
            
            self.total_points = int(n_points * estimated_depth)
            self.processed_points = 0
            
            if gpu_id is not None and cp.cuda.is_available() and self.gpu_config['enable']:
                with cp.cuda.Device(gpu_id):
                    mem_buffer = self.gpu_config['memory_buffer']
                    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
                    self.logger.info(f"Processing {len(points):,} points on GPU {gpu_id}")
                    
                    if self.verbose:
                        self.pbar = tqdm(
                            total=self.total_points,
                            desc=f"Building quadtree (GPU {gpu_id})",
                            unit="points",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
                        )
                    
                    result = self._process_points(points)
                    if self.verbose and self.pbar is not None:
                        self.pbar.close()
                    self.logger.info(f"Created {len(result[0])} clusters")
                    return result
            else:
                if self.verbose:
                    self.pbar = tqdm(
                        total=self.total_points,
                        desc="Building quadtree (CPU)",
                        unit="points"
                    )
                result = self._process_points(points)
                if self.verbose and self.pbar is not None:
                    self.pbar.close()
                return result
                
        except Exception as e:
            self.logger.error(f"Processing failed on {'GPU ' + str(gpu_id) if gpu_id is not None else 'CPU'}: {str(e)}")
            raise

    def _process_points(self, points: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process points and extract clusters and centroids."""
        if self.memory_config['enable_batching'] and len(points) > self.memory_config['max_batch_size']:
            return self._process_in_batches(points)
        
        root = self.build(points)
        clusters = []
        centroids = []
        
        def collect_leaves(node: QuadNode):
            if node.children is None:
                clusters.append(node.points)
                centroids.append(node.centroid)
            else:
                for child in node.children:
                    if child is not None:
                        collect_leaves(child)
        
        collect_leaves(root)
        return clusters, np.array(centroids)

    def _process_in_batches(self, points: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process large point sets in batches."""
        batch_size = self.memory_config['max_batch_size']
        n_batches = int(np.ceil(len(points) / batch_size))
        all_clusters = []
        all_centroids = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(points))
            batch_points = points[start_idx:end_idx]
            
            clusters, centroids = self._process_points(batch_points)
            all_clusters.extend(clusters)
            all_centroids.append(centroids)
        
        return all_clusters, np.vstack(all_centroids)

    def build(self, points: np.ndarray) -> QuadNode:
        """Build quadtree from points."""
        try:
            x_min, y_min = points[:, :2].min(axis=0)
            x_max, y_max = points[:, :2].max(axis=0)
            
            root = self._subdivide(points, (x_min, y_min, x_max, y_max), level=0)
            return root
        finally:
            if self.verbose and self.pbar is not None:
                self.pbar.refresh()

    def set_normalization_params(self, xy_mean: np.ndarray, xy_std: np.ndarray):
        """Set normalization parameters used for X,Y coordinates."""
        self.xy_mean = xy_mean
        self.xy_std = xy_std

    def _get_cell_size_meters(self, bounds: Tuple[float, float, float, float]) -> float:
        """Calculate the cell size in meters, accounting for normalization if used."""
        if self.normalize_xy:
            # Denormalize the bounds to get real-world dimensions
            x_min_real = bounds[0] * self.xy_std[0] + self.xy_mean[0]
            x_max_real = bounds[2] * self.xy_std[0] + self.xy_mean[0]
            y_min_real = bounds[1] * self.xy_std[1] + self.xy_mean[1]
            y_max_real = bounds[3] * self.xy_std[1] + self.xy_mean[1]
            
            width = abs(x_max_real - x_min_real)
            height = abs(y_max_real - y_min_real)
        else:
            # If not normalized, bounds are already in meters
            width = abs(bounds[2] - bounds[0])
            height = abs(bounds[3] - bounds[1])
            
        return max(width, height) 