__author__ = "Adriano Fonseca"
__email__ = "a.fonseca@ccom.unh.edu"
__version__ = "1.0.0"



import numpy as np
import laspy
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import logging
import h5py
import yaml
import time
import matplotlib.pyplot as plt
from scipy import stats

class LASDataLoader:
    """Handles loading and preprocessing of LAS point cloud files."""
    
    def __init__(self, config: dict):
        """Initialize the LAS data loader."""
        self.config = config
        self._validate_config()  # Remove normalization code from here
        self.input_dir = Path(config['data']['input_dir'])
        self.output_dir = Path(config['data']['output_dir'])
        self.logger = logging.getLogger(__name__)
        
        # Compression settings
        self.compression = config['processing']['compression']

    def normalize_coordinates(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize coordinates if configured to do so."""
        if self.config['data']['coordinate_processing']['normalize_xy']:
            # Calculate mean and std for X,Y coordinates
            xy_mean = np.mean(points[:, :2], axis=0)
            xy_std = np.std(points[:, :2], axis=0)
            
            # Normalize X,Y coordinates
            points[:, :2] = (points[:, :2] - xy_mean) / xy_std
            
            return points, xy_mean, xy_std
        return points, None, None

    def load_las_file(self, file_path: str) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Load LAS file and extract X, Y, Z coordinates."""
        try:
            las = laspy.read(file_path)
            points = np.vstack((las.x, las.y, las.z)).T
            z_range = (float(las.header.mins[2]), float(las.header.maxs[2]))
            return points, z_range
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {str(e)}")
            raise
    
    def load_survey_points(self) -> Tuple[np.ndarray, Tuple[float, float], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Load and combine points from all LAS files in the survey."""
        las_files = list(self.input_dir.glob("*.las"))
        if not las_files:
            raise ValueError(f"No LAS files found in {self.input_dir}")
            
        self.logger.info(f"Found {len(las_files)} LAS files in survey")
        all_points = []
        z_min = float('inf')
        z_max = float('-inf')
        
        for las_file in tqdm(las_files, desc="Loading LAS files"):
            points, (file_z_min, file_z_max) = self.load_las_file(las_file)
            all_points.append(points)
            z_min = min(z_min, file_z_min)
            z_max = max(z_max, file_z_max)
            
        combined_points = np.vstack(all_points)
        
        # Normalize coordinates if configured
        normalized_points, xy_mean, xy_std = self.normalize_coordinates(combined_points)
        
        self.logger.info(f"Total points in survey: {len(normalized_points)}")
        return normalized_points, (z_min, z_max), (xy_mean, xy_std) if xy_mean is not None else None

    def save_clusters(self, clusters: List[np.ndarray], input_name: str, z_range: Tuple[float, float]):
        """Save clusters efficiently using HDF5 format and create Z distribution histogram."""
        
        # Create date-based subdirectories with time
        current_datetime = time.strftime("%Y%m%d_%H%M")
        
        # Create output directories with date and time using config structure
        clusters_dir = (self.output_dir / 
                       self.config['output']['directory_structure']['clusters'] /
                       current_datetime /
                       input_name)
        images_dir = (self.output_dir / 
                     self.config['output']['directory_structure']['images'] /
                     current_datetime /
                     input_name)
        
        clusters_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Starting cluster saving process...")
        
        # Get clusters per file from config
        clusters_per_file = self.config['processing']['memory_management']['clusters_per_file']
        total_clusters = len(clusters)
        num_files = (total_clusters + clusters_per_file - 1) // clusters_per_file
        
        cluster_stats = []
        
        # Main progress bar for all clusters
        with tqdm(total=total_clusters, desc="Saving clusters") as pbar:
            for file_idx in range(num_files):
                start_idx = file_idx * clusters_per_file
                end_idx = min((file_idx + 1) * clusters_per_file, total_clusters)
                
                # Create HDF5 file for this batch
                file_suffix = f"_part{file_idx + 1}" if num_files > 1 else ""
                h5_path = clusters_dir / f"clusters{file_suffix}.h5"
                
                with h5py.File(h5_path, 'w') as f:
                    points_group = f.create_group('points')
                    centroids_group = f.create_group('centroids')
                    
                    for i, cluster in enumerate(clusters[start_idx:end_idx], start=start_idx):
                        # Save cluster data with configured compression
                        points_group.create_dataset(
                            f'cluster_{i:04d}',
                            data=cluster,
                            compression=self.compression['method'],
                            compression_opts=self.compression['level'],
                            chunks=True
                        )
                        
                        centroid = cluster.mean(axis=0)
                        centroids_group.create_dataset(f'cluster_{i:04d}', data=centroid)
                        
                        cluster_stats.append({
                            'cluster_id': i,
                            'file_index': file_idx + 1,
                            'num_points': len(cluster),
                            'centroid': centroid.tolist(),
                            'bounds': {
                                'min': cluster.min(axis=0).tolist(),
                                'max': cluster.max(axis=0).tolist()
                            }
                        })
                        
                        # Create histogram at configured intervals
                        if i % self.config['visualization']['histogram']['interval'] == 0:
                            self._create_z_histogram(
                                cluster[:, 2],
                                f'z_distribution_cluster_{i}',
                                images_dir,
                                z_range
                            )
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({'file': f'part{file_idx + 1}' if num_files > 1 else 'single'})
                
                self.logger.info(f"Saved clusters {start_idx:,} to {end_idx:,} in {h5_path}")
        
        self.logger.info("Finished saving clusters, writing metadata...")
        
        # Calculate cluster size statistics
        cluster_sizes = [stat['num_points'] for stat in cluster_stats]
        avg_size = np.mean(cluster_sizes)
        min_size = np.min(cluster_sizes)
        max_size = np.max(cluster_sizes)
        median_size = np.median(cluster_sizes)

        self.logger.info(f"Cluster size statistics:")
        self.logger.info(f"  Average points per cluster: {avg_size:.1f}")
        self.logger.info(f"  Median points per cluster: {median_size:.1f}")
        self.logger.info(f"  Min points per cluster: {min_size}")
        self.logger.info(f"  Max points per cluster: {max_size}")

        # Save metadata
        metadata = {
            'num_clusters': total_clusters,
            'num_files': num_files,
            'clusters_per_file': clusters_per_file,
            'total_points': sum(len(c) for c in clusters),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'input_file': str(input_name),
            'cluster_statistics': cluster_stats,
            'z_range': {'min': z_range[0], 'max': z_range[1]},
            'processing_config': {
                'compression': self.compression,
                'histogram_interval': self.config['visualization']['histogram']['interval']
            },
            'file_structure': {
                'format': 'HDF5',
                'files': [f"clusters_part{i+1}.h5" for i in range(num_files)] if num_files > 1 else ["clusters.h5"],
                'groups': {
                    'points': 'Contains point data for each cluster',
                    'centroids': 'Contains centroid data for each cluster'
                }
            },
            'point_cloud_settings': {
                'las_version': self.config['point_cloud']['las_version'],
                'point_format': self.config['point_cloud']['point_format'],
                'output_formats': self.config['point_cloud']['output_formats']
            },
            'cluster_size_stats': {
                'average': float(avg_size),
                'median': float(median_size),
                'min': int(min_size),
                'max': int(max_size)
            }
        }
        
        metadata_path = clusters_dir / "metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        self.logger.info("Completed saving clusters and metadata.")

        # Save point clouds of centroids
        centroids = np.array([cluster.mean(axis=0) for cluster in clusters])
        self.save_point_clouds(centroids, current_datetime, input_name)

    def _create_z_histogram(self, z_values: np.ndarray, name: str, output_dir: Path, z_range: Optional[Tuple[float, float]] = None) -> None:
        """Create and save a histogram of Z values with KDE overlay."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get histogram and KDE settings from config
            hist_config = self.config['visualization']['histogram']
            kde_config = self.config['visualization']['kde']
            
            # Create figure
            plt.figure(figsize=hist_config['figure_size'])
            
            # Plot histogram
            plt.hist(z_values, bins=hist_config['bins'], density=True,
                    color=hist_config['color'], alpha=hist_config['alpha'])
            
            # Compute and plot KDE
            x_eval, kde_values = self._compute_kde(z_values)
            plt.plot(x_eval, kde_values, color=kde_config['color'],
                    linewidth=kde_config['line_width'])
            
            # Detect and plot peaks if enabled
            peak_info = None
            if kde_config['peak_detection']['enable']:
                from scipy.signal import find_peaks
                
                # Convert relative parameters to absolute values
                min_distance = int(kde_config['peak_detection']['min_distance'] * len(x_eval))
                min_height = kde_config['peak_detection']['min_height'] * max(kde_values)
                prominence = kde_config['peak_detection']['prominence'] * max(kde_values)
                
                # Find peaks
                peaks, _ = find_peaks(kde_values,
                                    height=min_height,
                                    distance=min_distance,
                                    prominence=prominence)
                
                # Plot peaks
                plt.plot(x_eval[peaks], kde_values[peaks], 'o',
                        color=kde_config['peak_detection']['marker_color'],
                        markersize=kde_config['peak_detection']['marker_size'])
                
                # Add peak information to annotations
                peak_depths = [f"{x_eval[p]:.2f}m" for p in peaks]
                peak_info = f"Peaks: {len(peaks)} at {', '.join(peak_depths)}"
            
            # Add statistical annotations
            mean_z = z_values.mean()
            median_z = np.median(z_values)
            std_z = z_values.std()
            min_z, max_z = z_values.min(), z_values.max()
            
            plt.title(f"Z Distribution - {name}")
            plt.xlabel("Depth (m)")
            plt.ylabel("Density")
            
            # Add annotations
            annotations = [
                f"Mean: {mean_z:.2f}m",
                f"Median: {median_z:.2f}m",
                f"Std Dev: {std_z:.2f}m",
                f"Range: [{min_z:.2f}m, {max_z:.2f}m]"
            ]
            
            if peak_info:
                annotations.append(peak_info)
            
            if z_range:
                annotations.append(f"Survey Range: [{z_range[0]:.2f}m, {z_range[1]:.2f}m]")
            
            plt.figtext(0.95, 0.95, '\n'.join(annotations),
                       horizontalalignment='right',
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Save figure
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            output_path = output_dir / f"z_distribution_cluster_{name}_{timestamp}.png"
            plt.savefig(output_path, dpi=hist_config['dpi'], bbox_inches='tight')
            self.logger.info(f"Saved Z distribution histogram with KDE to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create histogram: {str(e)}")
        finally:
            plt.close('all')

    def _compute_kde(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Kernel Density Estimation for the given data using SciPy's gaussian_kde.
        Uses Scott's rule with a minimum bandwidth based on the data range."""
        kde_config = self.config['visualization']['kde']
        num_points = kde_config['num_points']
        max_samples = kde_config['max_samples']
        
        # Use random sampling for large datasets
        if len(data) > max_samples:
            rng = np.random.default_rng()
            data = rng.choice(data, size=max_samples, replace=False)
        
        # Generate evaluation points
        x_min, x_max = np.min(data), np.max(data)
        x_points = np.linspace(x_min, x_max, num_points)
        
        # Calculate minimum bandwidth based on data range
        min_bandwidth = (x_max - x_min) * kde_config['min_bandwidth_factor']
        
        # Compute KDE with Scott's rule
        kde = stats.gaussian_kde(data, bw_method='scott')
        
        # Get Scott's rule bandwidth and ensure it's not smaller than minimum
        scott_bw = kde.factor
        final_bw = max(scott_bw, min_bandwidth)
        
        # Set the final bandwidth
        kde.set_bandwidth(final_bw)
        kde_values = kde(x_points)
        
        return x_points, kde_values

    def _validate_config(self) -> None:
        """Validate the configuration."""
        required_keys = ['data']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if 'features' not in self.config['data']:
            raise ValueError("Missing required config key: data.features")
        
        if not isinstance(self.config['data']['features'], list):
            raise ValueError("Features must be a list")
        
        if not self.config['data']['features']:
            raise ValueError("Features list cannot be empty")

    def save_point_clouds(self, points: np.ndarray, current_datetime: str, input_name: str) -> None:
        """Save point cloud data in configured formats."""
        try:
            # Create output directory using config structure
            point_clouds_dir = (self.output_dir / 
                              self.config['output']['directory_structure']['point_clouds'] /
                              current_datetime /
                              input_name)
            point_clouds_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_formats = self.config['point_cloud']['output_formats']
            
            if 'ply' in output_formats:
                import open3d as o3d
                # Create point cloud for points
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # Save point cloud in PLY format
                ply_path = point_clouds_dir / f"points_{timestamp}.ply"
                o3d.io.write_point_cloud(str(ply_path), pcd)
                self.logger.info(f"Saved point cloud to {ply_path}")
                
            if 'las' in output_formats:
                # Save as LAS file
                las_path = point_clouds_dir / f"points_{timestamp}.las"
                las = laspy.create(
                    file_version=self.config['point_cloud']['las_version'],
                    point_format=self.config['point_cloud']['point_format']
                )
                las.x = points[:, 0]
                las.y = points[:, 1]
                las.z = points[:, 2]
                las.write(las_path)
                self.logger.info(f"Saved point cloud to {las_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save point clouds: {str(e)}") 