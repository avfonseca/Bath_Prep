# Bathymetry Processing Pipeline - Technical Documentation

## Data Loading and Preprocessing:
a. LAS File Processing:
    - Load using laspy
    - Direct extraction of X, Y, Z coordinates with native scaling
    - Preserve original depth values (no normalization)
    - Memory efficient loading with multiple file support
    - Header validation and coordinate range checking
   
b. Coordinate Processing:
- X,Y coordinates: Optional normalization (configurable)
    * μ_xy = mean(points[:, :2])
    * σ_xy = std(points[:, :2])
    * normalized_xy = (points[:, :2] - μ_xy) / σ_xy

- Z coordinate: 
    * Preserved in original units
    * No normalization to maintain true depth values
    * Range extracted from LAS header for validation

## Quadtree Partitioning:

a. Configuration:
- Two available subdivision modes:
  1. Fixed Mode:
     * Uses constant points per leaf
     * Simple and predictable
     * Configured with points_per_leaf parameter
     * Ensures minimum point threshold
     
  2. Adaptive Mode:
     * Dynamic point count based on depth
     * target_cell_size: Desired resolution in meters
     * Uses beam angle and footprint calculations
     * Adapts to depth changes
     
- Common parameters:
  * min_points: Minimum points threshold (512)
  * max_tree_depth: Maximum tree depth
  * verbose: Enable progress tracking

b. Point Count Calculation:
- Dynamic calculation based on depth:
    * beam_footprint = 2 * |depth| * tan(beam_angle/2)
    * density_factor = (beam_footprint/target_cell_size)²
    * optimal_points = max(ceil(density_factor), min_points)
    
- Rationale:
    * Beam footprint increases with depth
    * More points needed for deeper areas
    * Ensures minimum statistical significance
    * Adapts to varying resolution requirements
    
c. Partitioning Process:
- Split criteria:
    * Calculate median depth of current node
    * Determine optimal points for that depth
    * Split if points > optimal_points
    * Stop if max_depth reached
      
- Node structure:
    * Bounds: (x_min, y_min, x_max, y_max)
    * Points: Raw point data or None for non-leaves
    * Children: List of child nodes or None for leaves
    * Centroid: Mean position of points in leaf nodes
    * Level: Current depth in tree

d. Benefits:
- Adaptive to depth changes:
    * Larger cells in deeper water
    * Finer resolution in shallow areas
    * Matches sonar capabilities
- Statistical robustness:
    * Maintains minimum point threshold
    * Accounts for beam footprint
    * Prevents over-splitting

## GPU Processing:
a. Multi-GPU Support:
- Automatic GPU detection and allocation
- Data partitioning for parallel processing
- Memory management with configurable buffer sizes

b. Processing Strategy:
- Split along longest axis for multi-GPU
- Parallel execution with ThreadPoolExecutor
- GPU memory monitoring and cleanup

## Cluster Generation:
   a. Cluster Formation:
      - Each leaf node becomes a cluster
      - Configurable cluster size via quadtree parameters
      - Centroid calculation:
        * Mean of points in leaf (for now)
        * Preserves original depth values
   
   b. Memory Management:
      - Points stored only in leaf nodes
      - Internal nodes free memory after subdivision
      - Batch processing for large datasets

## Output Generation:
   a. Data Storage:
      - HDF5 format for efficient storage of clusters
      - File splitting strategy:
        * Creates new HDF5 file after every N clusters (e.g., 100,000)
        * Prevents performance degradation with large cluster counts
        * File naming convention:
          - clusters_part1.h5
          - clusters_part2.h5
          - etc.
        * Metadata handling for split files:
          - Each part file contains its own metadata including:
            * Start and end cluster indices
            * Number of clusters in the file
            * Point statistics for clusters in this file
          - A master metadata.yaml file contains:
            * Total number of clusters across all files
            * List of all part files
            * Global statistics
            * Processing configuration
      - Hierarchical structure in each file:
        * /points/
          - cluster_0000: First cluster's points
          - cluster_0001: Second cluster's points
          - ...
        * /centroids/
          - cluster_0000: First cluster's centroid
          - cluster_0001: Second cluster's centroid
          - ...
      - Each cluster stored as separate dataset with:
        * Individual compression
        * Chunked storage for efficient access
        * Metadata including point count and bounds
      - Separate storage enables:
        * Loading individual clusters without full file read
        * Efficient memory usage
        * Fast access to specific clusters
        * Better performance through file size management
   
   b. Visualization:
      - Depth distribution histograms:
        * Generated every N clusters (configurable)
        * Natural depth range for each cluster
        * Includes Kernel Density Estimation (KDE) overlay:
          - Uses SciPy's gaussian_kde with bandwidth selection:
            * Primary bandwidth calculation using Scott's rule
            * Minimum bandwidth protection based on data range
              - min_bandwidth = (z_max - z_min) * min_bandwidth_factor
              - min_bandwidth_factor: 0.1 (configurable)
            * Final bandwidth = max(scott_bandwidth, min_bandwidth)
          - Configurable KDE parameters:
            * Number of evaluation points (default: 1000)
            * Line color (default: red [1.0, 0.0, 0.0])
            * Line width (default: 2)
            * Maximum samples for large clusters (default: 10000)
          - Prevents over-fitting in clusters with small depth ranges
          - Provides smooth representation of depth distribution
        * Peak Detection on KDE:
          - Identifies significant peaks in depth distribution
          - Uses scipy.signal.find_peaks with relative thresholds
          - Peak detection parameters:
            * min_height: Minimum peak height relative to max density
            * min_distance: Minimum separation between peaks
            * prominence: Required prominence relative to neighbors
          - Peaks marked with configurable markers:
            * Size and color customizable
            * Default: red dots at peak locations
          - Helps identify distinct depth layers or features
        * Statistical annotations including:
          - Mean, median, std dev
          - Cluster depth range
          - Full survey depth range
          - Number and locations of detected peaks
        * Histogram settings:
          - Figure size: [12, 6]
          - Number of bins: 100
          - Color: blue [0.0, 0.0, 1.0]
          - Alpha: 0.7
          - DPI: 300
      
      - Centroid Export:
        * Point cloud files saved in both PLY and LAS formats
        * PLY format:
          - Standard point cloud format
          - Compatible with various 3D visualization tools
          - Contains denormalized coordinates (original scale)
        * LAS format:
          - Industry standard for bathymetric data
          - Version 1.4, point format 6
          - Preserves original coordinate system
        * Files named with timestamps for tracking
        * Separate directory structure for each survey
      
   c. Directory Structure:
      /Output
        /Clusters
          /YYYYMMDD_HHMM
            /Survey_Name
              clusters.h5       # Compressed point data and centroids
              metadata.yaml     # Processing details and statistics
        /PointClouds
          /YYYYMMDD_HHMM
            /Survey_Name
              centroids_*.ply  # Point cloud files for external viewers
              centroids_*.las  # LAS format point clouds
        /Images
          /YYYYMMDD_HHMM
            /Survey_Name
              - Depth distribution histograms
        /Logs
          /YYYYMMDD_HHMM
            - Processing logs
            - Performance metrics

## Configuration Options:
   a. Data Settings:
      - input_dir: Path to directory containing LAS files
      - output_dir: Base directory for all outputs
      - features: List of point cloud features to process (X, Y, Z)
      - coordinate_processing:
        * normalize_xy: Whether to normalize X,Y coordinates (true/false)
        * preserve_z: Keep original Z values without normalization (true/false)

   b. Clustering Settings:
      - points_per_cluster: Number of points per leaf node (default: 1024)
      - max_tree_depth: Maximum depth of quadtree (default: 20)
      - verbose: Enable detailed progress output (true/false)
      - histogram_interval: Create histogram every N clusters (default: 10000)

   c. Processing Settings:
      - gpu:
        * enable: Enable GPU acceleration (true/false)
        * memory_buffer: Fraction of GPU memory to keep free (default: 0.2)
      - memory_management:
        * max_batch_size: Maximum points to process at once (default: 1000000)
        * enable_batching: Use batch processing for large datasets (true/false)
        * clusters_per_file: Number of clusters per HDF5 file (default: 700000)
      - compression:
        * method: HDF5 compression method (default: "gzip")
        * level: Compression level 1-9 (default: 4)

   d. Visualization Settings:
      - histogram:
        * figure_size: Plot dimensions in inches [width, height]
        * bins: Number of histogram bins (default: 100)
        * color: RGB values for histogram bars [R, G, B]
        * alpha: Transparency of histogram bars (0-1)
        * dpi: Resolution of saved images (default: 300)
      - kde:
        * num_points: Number of points for KDE evaluation (default: 1000)
        * color: RGB values for KDE line [R, G, B]
        * line_width: Width of KDE line (default: 2)
        * bw_method: Bandwidth estimation method ('scott')
        * min_bandwidth_factor: Minimum bandwidth as fraction of data range (default: 0.1)
        * max_samples: Maximum points to use for KDE estimation (default: 10000)
        * peak_detection:
          - enable: Enable peak detection (true/false)
          - min_height: Minimum peak height relative to maximum density (default: 0.05)
          - min_distance: Minimum distance between peaks as fraction of range (default: 0.1)
          - prominence: Minimum peak prominence relative to neighbors (default: 0.1)
          - marker_size: Size of peak markers in points (default: 10)
          - marker_color: RGB values for peak markers [R, G, B]
      - point_cloud:
        * las_version: LAS file version (default: "1.4")
        * point_format: LAS point data format (default: 6)

   e. Output Settings:
      - formats: List of output formats to generate ["hdf5", "yaml", "ply", "las"]
      - directory_structure:
        * clusters: Directory for cluster data (default: "Clusters")
        * images: Directory for histograms (default: "Images")
        * logs: Directory for log files (default: "Logs")
        * point_clouds: Directory for point cloud files (default: "PointClouds")

   f. Logging Settings:
      - level: Logging detail level ("INFO", "DEBUG", etc.)
      - file_logging: Enable logging to file (true/false)
      - console_logging: Enable console output (true/false)



## Notes

- Silvermans method on KDE
- check for edge effects (wrap around kernel or mirror samples)
- R trees
- Verify wether processing linewise or all together