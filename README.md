# CV Feature Enhancement Processor

A command-line tool for batch processing images with various computer vision feature enhancement techniques. This tool is designed to help calibrate and test different CV preprocessing methods by applying them to a set of source images.

## Features

- Process all images in a source directory with various CV-specific transformations
- Multiple edge detection methods (Canny, Sobel, Laplacian, structured edges)
- Feature enhancement techniques (CLAHE, adaptive thresholding)
- Morphological operations for feature manipulation
- Corner and blob detection enhancement
- Parallel processing support for faster execution
- Progress bar to track processing status
- Organizes processed images in timestamped subfolders for easy tracking and comparison

## Sample Results

Here's a demonstration of various transformations applied to a sample image:

### Original Image
![Sample Image](img_source/sampleimg.png)

### Edge Detection Examples
![Canny Edges](img_output/processed_19700000_000000/sampleimg_canny_edges.png)
![Structured Edges](img_output/processed_19700000_000000/sampleimg_structured_edges.png)

### Feature Enhancement
![CLAHE Enhanced](img_output/processed_19700000_000000/sampleimg_clahe_enhance.png)
![Adaptive Threshold](img_output/processed_19700000_000000/sampleimg_adaptive_threshold.png)

### Feature Detection
![Corner Detection](img_output/processed_19700000_000000/sampleimg_corner_enhance.png)
![Blob Detection](img_output/processed_19700000_000000/sampleimg_blob_enhance.png)

## Installation

1. Clone this repository
2. Create and activate a conda environment (recommended):
```bash
conda create -n cv_imgprocessor python=3.8
conda activate cv_imgprocessor
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Place your source images in the `img_source` directory and run:

```bash
python img_processor.py
```

This will apply all available transformations to each image and save the results in a timestamped subfolder within the `img_output` directory (e.g., `img_output/processed_20230619_142530/`).

### Command Line Options

```bash
python img_processor.py [-h] [-s SOURCE] [-o OUTPUT] [-t TRANSFORM [TRANSFORM ...]]
                        [-c COMBO [COMBO ...]] [-w WORKERS] [-l]
```

Arguments:
- `-h, --help`: Show help message
- `-s, --source`: Source directory (default: img_source)
- `-o, --output`: Output directory (default: img_output)
- `-t, --transform`: Apply specific transformations
- `-c, --combo`: Apply specific combination transformations
- `-w, --workers`: Number of worker processes (default: 1)
- `-l, --list`: List available transformations and exit

### Examples

Apply all transformations to all images:
```bash
python img_processor.py
```

Apply specific edge detection methods:
```bash
python img_processor.py --transform canny_edges structured_edges
```

Apply feature extraction combination:
```bash
python img_processor.py --combo feature_extraction
```

List available transformations:
```bash
python img_processor.py --list
```

Process with multiple worker processes:
```bash
python img_processor.py --workers 4
```

## Available Transformations

### Edge Detection Methods
- `canny_edges`: Standard Canny edge detection
- `canny_edges_tight`: Canny with tighter thresholds for stronger edges
- `canny_edges_loose`: Canny with looser thresholds for weaker edges
- `sobel_edges`: Sobel gradient-based edge detection
- `laplacian_edges`: Laplacian edge detection
- `structured_edges`: Advanced edge detection combining multiple methods
- `ridge_detection`: Ridge detection using Hessian matrix eigenvalues

### Feature Enhancement
- `clahe_enhance`: Contrast Limited Adaptive Histogram Equalization
- `adaptive_threshold`: Adaptive Gaussian thresholding
- `otsu_threshold`: Otsu's automatic thresholding

### Morphological Operations
- `dilate_features`: Dilation to expand features
- `erode_features`: Erosion to shrink features
- `open_features`: Opening operation (erosion followed by dilation)
- `close_features`: Closing operation (dilation followed by erosion)
- `tophat_features`: Top-hat transformation for bright features on dark background

### Feature Detection
- `corner_enhance`: Harris corner detection enhancement
- `blob_enhance`: Blob detection and enhancement

### Combination Transformations
- `feature_extraction`: Combines CLAHE and structured edges for optimal feature extraction
- `blob_detection`: Enhances and detects blob-like structures
- `corner_detection`: Enhances and detects corner features
- `edge_analysis`: Multi-step edge detection and enhancement
- `texture_analysis`: Enhances texture features using multiple methods 