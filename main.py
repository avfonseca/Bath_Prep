__author__ = "Adriano Fonseca"
__email__ = "a.fonseca@ccom.unh.edu"
__version__ = "1.0.0"


import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import concurrent.futures
from src.data_loader import LASDataLoader
from src.quadtree import BathymetryQuadTree, QuadNode
import matplotlib.pyplot as plt
import h5py

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BathymetryProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the bathymetry processor with configuration."""
        # Load config first
        self.config = self._load_config(config_path)
        
        # Get timestamp once for consistent naming
        self.timestamp = time.strftime('%Y%m%d-%H%M%S')
        
        # Initialize all config-dependent attributes
        self.verbose = self.config['clustering']['verbose']
        
        # Setup all directories first
        self.base_dir = Path(self.config['data']['output_dir'])
        self.setup_directories()
        
        # Create configs directory and save current config
        configs_dir = self.base_dir / "Configs" / time.strftime("%Y%m%d_%H%M")
        configs_dir.mkdir(parents=True, exist_ok=True)
        config_backup_path = configs_dir / "config.yaml"
        
        # Save a copy of the config
        with open(config_backup_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Setup logging based on config
        self.setup_logging()
        
        # GPU settings
        self.use_gpu = self.config['processing']['gpu']['enable']
        self.gpu_loggers = {}
        self.num_gpus = 0
        
        # Initialize GPU resources if enabled
        if self.use_gpu:
            self.setup_gpu()
        
        # Initialize components with config
        self.data_loader = LASDataLoader(self.config)        
        # Get survey name from input directory
        self.survey_name = Path(self.config['data']['input_dir']).name
    
    def setup_directories(self):
        """Create output directory structure from config."""
        base_dir = Path(self.config['data']['output_dir'])
        for dir_name in self.config['output']['directory_structure'].values():
            (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.config['logging']
        log_level = getattr(logging, log_config['level'].upper())
        
        # Create date-based log directory with time
        current_datetime = time.strftime("%Y%m%d_%H%M")
        log_dir = self.logs_dir / current_datetime
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handlers = []
        if log_config['console_logging']:
            handlers.append(logging.StreamHandler())
        
        if log_config['file_logging']:
            log_file = log_dir / f"processing_{self.timestamp}.log"
            handlers.append(logging.FileHandler(log_file))
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Ensure we reset any existing handlers
        )
        
        # Get our specific logger
        self.logger = logging.getLogger(__name__)
    
    def process_survey(self):
        """Process the entire survey."""
        try:
            # Load and preprocess points
            points, z_range, norm_params = self.data_loader.load_survey_points()
            
            # Store normalization parameters at class level if they exist
            if norm_params is not None:
                self.xy_mean, self.xy_std = norm_params
            else:
                self.xy_mean = self.xy_std = None
            
            # Process points
            if self.num_gpus > 1:
                clusters, centroids = self._process_multi_gpu(points)
            else:
                clusters, centroids = self._process_single_device(points)
            
            # Save results
            self.data_loader.save_clusters(clusters, self.survey_name, z_range)
            
        except Exception as e:
            self.logger.error(f"Survey processing failed: {str(e)}")
            raise
    
    def _process_multi_gpu(self, points: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process points using multiple GPUs."""
        # Split along longest axis
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        
        if x_range > y_range:
            split_coord = points[:, 0].mean()
            left_mask = points[:, 0] <= split_coord
            axis_name = "X"
        else:
            split_coord = points[:, 1].mean()
            left_mask = points[:, 1] <= split_coord
            axis_name = "Y"
        
        left_points = points[left_mask]
        right_points = points[~left_mask]
        
        self.logger.info(f"Split data along {axis_name} axis:")
        self.logger.info(f"  GPU 0: {len(left_points):,} points")
        self.logger.info(f"  GPU 1: {len(right_points):,} points")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_left = executor.submit(self._process_partition, left_points, 0)
            future_right = executor.submit(self._process_partition, right_points, 1)
            
            left_results = future_left.result()
            right_results = future_right.result()
        
        return (left_results[0] + right_results[0], 
                np.vstack([left_results[1], right_results[1]]))
    
    def _process_single_device(self, points: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process points using single GPU or CPU."""
        return self._process_partition(points, 0 if self.use_gpu else None)
    
    def _process_partition(self, points: np.ndarray, gpu_id: Optional[int]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process a partition of points using quadtree."""
        # Create a new quadtree instance for each partition
        quadtree = BathymetryQuadTree(config=self.config)
        
        # If normalization was applied, set the parameters
        if self.config['data']['coordinate_processing']['normalize_xy']:
            quadtree.set_normalization_params(self.xy_mean, self.xy_std)
            
        return quadtree.process_partition(points, gpu_id)

    def setup_gpu(self) -> None:
        """Initialize GPU resources and logging."""
        self.use_gpu = cp.cuda.is_available()
        if not self.use_gpu:
            self.logger.warning("No GPU available, using CPU")
            self.num_gpus = 0
            return
        
        self.num_gpus = cp.cuda.runtime.getDeviceCount()
        self.logger.info(f"Using {self.num_gpus} GPUs for clustering")
        
        # Setup GPU loggers
        self.gpu_loggers = {}
        for gpu_id in range(self.num_gpus):
            self._setup_gpu_logger(gpu_id)
            self._log_gpu_properties(gpu_id)

    def _setup_gpu_logger(self, gpu_id: int) -> None:
        """Setup logging for a specific GPU."""
        gpu_logger = logging.getLogger(f"GPU_{gpu_id}")
        gpu_logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Create date-based log directory with time
        current_datetime = time.strftime("%Y%m%d_%H%M")
        log_dir = self.logs_dir / current_datetime
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        gpu_file = log_dir / f"gpu_{gpu_id}_{self.timestamp}.log"
        file_handler = logging.FileHandler(gpu_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - [GPU %(name)s] - %(levelname)s - %(message)s')
        )
        gpu_logger.addHandler(file_handler)
        
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(f'[GPU {gpu_id}] %(message)s'))
            gpu_logger.addHandler(console_handler)
        
        self.gpu_loggers[gpu_id] = gpu_logger

    def _log_gpu_properties(self, gpu_id: int) -> None:
        """Log properties of a specific GPU."""
        with cp.cuda.Device(gpu_id):
            props = cp.cuda.runtime.getDeviceProperties(gpu_id)
            total_memory = cp.cuda.Device().mem_info[1]
            self.log_gpu(gpu_id, f"Initialized: {props['name'].decode()}, {total_memory/1e9:.2f}GB total memory")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def log_gpu(self, gpu_id: int, message: str, level: str = "info") -> None:
        """GPU-specific logging with proper formatting."""
        if gpu_id in self.gpu_loggers:
            logger = self.gpu_loggers[gpu_id]
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)

    def log(self, message: str, level: str = "info", always_show: bool = False) -> None:
        """Centralized logging with verbosity control."""
        if always_show or self.verbose:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)

if __name__ == "__main__":
    processor = BathymetryProcessor()
    processor.process_survey()
