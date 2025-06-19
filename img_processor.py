#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Callable, Optional, Union
from datetime import datetime


class ImageProcessor:
    """
    A class to process images with various CV-specific feature enhancement transformations.
    """
    
    def __init__(self, source_dir: str = "img_source", output_dir: str = "img_output"):
        """
        Initialize the image processor with source and output directories.
        
        Args:
            source_dir: Directory containing source images
            output_dir: Directory to save processed images
        """
        self.source_dir = Path(source_dir)
        
        # Create timestamped subfolder within output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"processed_{timestamp}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Define supported image extensions
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Define transformation functions with their parameters
        self.transformations = {
            # Edge Detection Methods
            "canny_edges": lambda img: self._apply_canny(img, 50, 150),
            "canny_edges_tight": lambda img: self._apply_canny(img, 75, 200),
            "canny_edges_loose": lambda img: self._apply_canny(img, 25, 100),
            "sobel_edges": lambda img: self._apply_sobel(img),
            "laplacian_edges": lambda img: self._apply_laplacian(img),
            
            # Feature Enhancement
            "clahe_enhance": lambda img: self._apply_clahe(img),
            "adaptive_threshold": lambda img: self._apply_adaptive_threshold(img),
            "otsu_threshold": lambda img: self._apply_otsu_threshold(img),
            
            # Morphological Operations
            "dilate_features": lambda img: self._apply_morphology(img, cv2.MORPH_DILATE),
            "erode_features": lambda img: self._apply_morphology(img, cv2.MORPH_ERODE),
            "open_features": lambda img: self._apply_morphology(img, cv2.MORPH_OPEN),
            "close_features": lambda img: self._apply_morphology(img, cv2.MORPH_CLOSE),
            "tophat_features": lambda img: self._apply_morphology(img, cv2.MORPH_TOPHAT),
            
            # Corner and Blob Enhancement
            "corner_enhance": lambda img: self._enhance_corners(img),
            "blob_enhance": lambda img: self._enhance_blobs(img),
            
            # Advanced Edge Detection
            "structured_edges": lambda img: self._structured_edge_detection(img),
            "ridge_detection": lambda img: self._detect_ridges(img),
        }
        
        # Define combinations of transformations for specific CV tasks
        self.combinations = {
            "feature_extraction": ["clahe_enhance", "structured_edges"],
            "blob_detection": ["clahe_enhance", "blob_enhance"],
            "corner_detection": ["clahe_enhance", "corner_enhance"],
            "edge_analysis": ["clahe_enhance", "canny_edges", "dilate_features"],
            "texture_analysis": ["clahe_enhance", "laplacian_edges", "adaptive_threshold"],
        }
        
        print(f"Output directory: {self.output_dir}")

    def _apply_canny(self, image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
        """Apply Canny edge detection with specified thresholds."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Convert back to 3 channels for consistency
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def _apply_sobel(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Sobel in x and y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine x and y gradients
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255 range
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        # Convert back to 3 channels
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
    
    def _apply_laplacian(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian edge detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Convert to absolute values and scale
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Convert back to 3 channels
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Apply CLAHE to L channel
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to 3 channels
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Convert back to 3 channels
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _apply_morphology(self, image: np.ndarray, operation: int) -> np.ndarray:
        """Apply morphological operation."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create kernel
        kernel = np.ones((5,5), np.uint8)
        
        # Apply morphological operation
        result = cv2.morphologyEx(gray, operation, kernel)
        
        # Convert back to 3 channels
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    def _enhance_corners(self, image: np.ndarray) -> np.ndarray:
        """Enhance corners using Harris corner detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect corners
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        
        # Dilate corner detections
        corners = cv2.dilate(corners, None)
        
        # Create output image
        result = image.copy()
        result[corners > 0.01 * corners.max()] = [0, 0, 255]  # Mark corners in red
        
        return result
    
    def _enhance_blobs(self, image: np.ndarray) -> np.ndarray:
        """Enhance blob-like structures."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Set up blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 100
        
        # Create blob detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Draw blobs
        result = cv2.drawKeypoints(image, keypoints, np.array([]), 
                                 (0,0,255), 
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return result
    
    def _structured_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhanced edge detection combining multiple methods."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Compute gradients using Sobel
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Non-maximum suppression
        magnitude = cv2.convertScaleAbs(magnitude)
        
        # Apply double thresholding
        high_thresh = np.percentile(magnitude, 90)
        low_thresh = high_thresh * 0.5
        
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Convert back to 3 channels
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def _detect_ridges(self, image: np.ndarray) -> np.ndarray:
        """Detect ridge-like structures using eigenvalues of Hessian matrix."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Compute Hessian matrix components
        sobelxx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
        sobelyy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
        sobelxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        
        # Compute eigenvalues
        lambda1 = 0.5 * (sobelxx + sobelyy + np.sqrt((sobelxx - sobelyy)**2 + 4*sobelxy**2))
        lambda2 = 0.5 * (sobelxx + sobelyy - np.sqrt((sobelxx - sobelyy)**2 + 4*sobelxy**2))
        
        # Ridge measure
        ridgeness = np.abs(lambda2)
        
        # Normalize to 0-255 range
        ridgeness = np.uint8(255 * ridgeness / np.max(ridgeness))
        
        # Convert back to 3 channels
        return cv2.cvtColor(ridgeness, cv2.COLOR_GRAY2BGR)
    
    def get_source_images(self) -> List[Path]:
        """Get all valid image files from the source directory."""
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(self.source_dir.glob(f"*{ext}"))
            image_files.extend(self.source_dir.glob(f"*{ext.upper()}"))
        return image_files
    
    def process_image(self, image_path: Path, transformations: List[str]) -> None:
        """Process a single image with the specified transformations."""
        try:
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return
            
            # Apply transformations in sequence
            processed_image = image.copy()
            for transform_name in transformations:
                if transform_name in self.transformations:
                    processed_image = self.transformations[transform_name](processed_image)
                elif transform_name in self.combinations:
                    # Apply combination of transformations
                    for sub_transform in self.combinations[transform_name]:
                        processed_image = self.transformations[sub_transform](processed_image)
                else:
                    print(f"Warning: Unknown transformation '{transform_name}'")
            
            # Generate output filename
            stem = image_path.stem
            suffix = image_path.suffix
            transform_suffix = "_".join(transformations)
            output_filename = f"{stem}_{transform_suffix}{suffix}"
            output_path = self.output_dir / output_filename
            
            # Save the processed image
            cv2.imwrite(str(output_path), processed_image)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    def process_all(self, transformations: List[str] = None, combinations: List[str] = None, 
                   workers: int = 1) -> None:
        """Process all images with specified transformations."""
        image_files = self.get_source_images()
        if not image_files:
            print(f"No images found in {self.source_dir}")
            return
        
        # Determine which transformations to apply
        transforms_to_apply = []
        
        # Add individual transformations
        if transformations:
            for t in transformations:
                if t in self.transformations:
                    transforms_to_apply.append([t])
                else:
                    print(f"Warning: Unknown transformation '{t}'")
        
        # Add combinations
        if combinations:
            for c in combinations:
                if c in self.combinations:
                    transforms_to_apply.append([c])
                else:
                    print(f"Warning: Unknown combination '{c}'")
        
        # If no transformations specified, apply all individual transformations
        if not transforms_to_apply:
            transforms_to_apply = [[t] for t in self.transformations.keys()]
        
        # Create tasks list
        tasks = []
        for img_path in image_files:
            for transform_list in transforms_to_apply:
                tasks.append((img_path, transform_list))
        
        # Process images with progress bar
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = []
                    for img_path, transform_list in tasks:
                        future = executor.submit(self.process_image, img_path, transform_list)
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        pbar.update(1)
            else:
                for img_path, transform_list in tasks:
                    self.process_image(img_path, transform_list)
                    pbar.update(1)
    
    def list_transformations(self) -> None:
        """Print all available transformations and combinations."""
        print("\nAvailable individual transformations:")
        
        # Group transformations by category
        categories = {
            "Edge Detection": ["canny_edges", "canny_edges_tight", "canny_edges_loose", 
                             "sobel_edges", "laplacian_edges", "structured_edges", "ridge_detection"],
            "Feature Enhancement": ["clahe_enhance", "adaptive_threshold", "otsu_threshold"],
            "Morphological Operations": ["dilate_features", "erode_features", "open_features", 
                                       "close_features", "tophat_features"],
            "Feature Detection": ["corner_enhance", "blob_enhance"]
        }
        
        for category, transforms in categories.items():
            print(f"\n{category}:")
            for name in sorted(transforms):
                if name in self.transformations:
                    print(f"  - {name}")
        
        print("\nAvailable combination transformations:")
        for name, transforms in sorted(self.combinations.items()):
            print(f"  - {name}: {', '.join(transforms)}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Process images with CV-specific feature enhancement transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply all transformations to all images
  python img_processor.py
  
  # Apply specific transformations
  python img_processor.py --transform canny_edges clahe_enhance
  
  # Apply a combination
  python img_processor.py --combo feature_extraction
  
  # List available transformations
  python img_processor.py --list
  
  # Process with multiple worker processes
  python img_processor.py --workers 4
"""
    )
    
    parser.add_argument("-s", "--source", default="img_source",
                        help="Source directory containing images (default: img_source)")
    parser.add_argument("-o", "--output", default="img_output",
                        help="Output directory for processed images (default: img_output)")
    parser.add_argument("-t", "--transform", nargs="+", metavar="TRANSFORM",
                        help="Apply specific transformations")
    parser.add_argument("-c", "--combo", nargs="+", metavar="COMBO",
                        help="Apply specific combination transformations")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List available transformations and exit")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ImageProcessor(args.source, args.output)
    
    # List transformations if requested
    if args.list:
        processor.list_transformations()
        return
    
    # Process images
    processor.process_all(args.transform, args.combo, args.workers)


if __name__ == "__main__":
    main() 