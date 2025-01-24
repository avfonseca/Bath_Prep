# Bath-Prep

Bath-Prep is a Python-based tool for processing and clustering bathymetric point cloud data using adaptive quadtree subdivision. It efficiently handles large-scale bathymetric surveys while maintaining IHO standards-compliant data density.

## Features

- Adaptive quadtree clustering based on depth-dependent beam footprint
- GPU acceleration support for large datasets
- Automatic KDE-based depth distribution analysis
- Multi-format output support (HDF5, LAS, PLY)
- Progress tracking and detailed logging
- Configurable processing parameters

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional)

### Dependencies
bash
numpy
cupy-cuda11x # for GPU acceleration
laspy
h5py
matplotlib
scipy
tqdm
pyyaml
open3d


## Installation

1. Clone the repository:
bash
git clone https://github.com/adrianofonseca/bath-prep.git
cd bath-pre

2. Install dependencies:
pip install -r requirements.txt


## Usage

1. Configure processing parameters in `config/config.yaml`
2. Run the processor:
python main.py



## Configuration

The tool can be configured through `config/config.yaml`. Key configuration options include:

- Clustering mode (adaptive/fixed)
- Target cell size
- Minimum points per cluster
- GPU settings
- Output formats
- Visualization parameters

## Output Structure

Output/
├── Clusters/
│ └── YYYYMMDD_HHMM/
│       └── survey_name/
│           ├── clusters.h5
│           └── metadata.yaml
├── Images/
│ └── YYYYMMDD_HHMM/
│       └── survey_name/
│           └── z_distribution_.png
└── PointClouds/
    └── YYYYMMDD_HHMM/
        └── survey_name/
            ├── points_.ply
            └── points_.las


## Author

**Adriano Fonseca**  
Center for Coastal and Ocean Mapping  
University of New Hampshire  
Email: a.fonseca@ccom.unh.edu

## License

MIT License

Copyright (c) 2024 Adriano Fonseca

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.