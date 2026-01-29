UAS-DTU Search and Rescue Image Processing System


📋 Project Overview
An autonomous image segmentation and feature detection system for Search and Rescue operations. This project processes aerial images from UAVs to detect casualties and assign them to rescue camps based on priority and distance constraints.

🎯 Objective
To develop an intelligent system that:

Detects ocean/land regions in SAR images

Identifies casualties.

Locates rescue camps.

Assigns casualties to optimal rescue camps based on priority rules

Generates comprehensive outputs for rescue coordination


🛠️ Installation

numpy==1.24.3
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
matplotlib==3.7.2
scipy==1.11.4
pip (Python package manager)


📊 Usage Instructions

1. Single Image Processing : 
python
from optimized_sar import OptimizedSARProcessor

processor = OptimizedSARProcessor(debug=True)
results = processor.process_single_image("path/to/image.png")

2. Batch Processing : 
python
from optimized_sar import OptimizedSARProcessor

processor = OptimizedSARProcessor(debug=True)
processor.process_batch("C:/Users/user/Documents/uas_dtu_project/images/task_images")

3. Command Line Execution :
python optimized_sar.py


📈 Output Specifications
For Each Image:
1. Visual Output Image (image_output.png):

Ocean (blue overlay) and land (green overlay) segmentation

Casualties marked with shapes and colors

Rescue camps with labels

Assignment lines showing casualty-to-camp connections

2. Statistical Outputs:
Casualty counts per camp: [blue, pink, grey]
Casualty details: [[shape_priority, condition_priority], ...]
Camp priorities: [blue_total, pink_total, grey_total]
Rescue ratio (average priority): XX.XX

3. Text Summary (image_summary.txt):

Detailed assignment information

Priority calculations

Detection statistics



⚙️ Configuration

Color Ranges (Customizable)
# Adjust these in the OptimizedSARProcessor __init__ method
COLOR_RANGES = {
    'ocean': ([90, 60, 60], [130, 255, 255]),
    'land': ([15, 40, 60], [40, 255, 255]),
    'red': ([0, 100, 100], [10, 255, 255]),
    # ... etc
}

Priority Rules
SHAPE_PRIORITY = {'star': 3, 'triangle': 2, 'square': 1}
CONDITION_PRIORITY = {'red': 3, 'yellow': 2, 'green': 1}
CAMP_CAPACITIES = {'blue': 4, 'pink': 3, 'grey': 2}


🔧 Algorithm Details
1. Image Segmentation
Uses HSV color space for better color separation

Applies morphological operations to clean masks

Creates distinct overlays for ocean (blue) and land (green)

2. Casualty Detection
Color-based filtering: Detects red, yellow, green objects

Shape classification:

Contour approximation for basic shapes

Hu moments for advanced shape recognition

Template matching as fallback

Priority calculation: Shape priority × Condition priority

3. Camp Detection
Color-based camp identification

Hough Circle Transform as backup

Circularity verification to ensure correct detection

4. Assignment Algorithm
1. Sort casualties by total priority (descending)

2. For each casualty:

Calculate distance to all non-full camps

Compute score: (priority × 100) / (distance + 1)

Assign to camp with highest score

3. Respect camp capacity constraints

4. Optimize for maximum total priority across all camps


Common Issues & Solutions
1. No casualties detected

Enable debug mode: debug=True

Check color ranges in COLOR_RANGES

Adjust area thresholds in detection methods

2. Poor segmentation

Modify HSV ranges for ocean/land

Adjust morphological operation kernels

Check input image quality

3. Assignment errors

Verify camp capacities match requirements

Check distance calculations

Review priority scoring formula

4. Performance issues

Reduce image size in preprocessing

Disable debug mode for batch processing

Optimize contour processing thresholds


Debug Mode
Enable debug mode to see intermediate processing steps:
python
processor = OptimizedSARProcessor(debug=True)


📊 Sample Output Format
python
# For each image:
{
    "image_name": "1.png",
    "casualty_counts": [3, 2, 1],  # [blue, pink, grey]
    "casualty_details": [
        [[3,3], [1,2], [2,1]],  # Blue camp details
        [[3,2], [2,2]],         # Pink camp details
        [[1,1]]                 # Grey camp details
    ],
    "camp_priorities": [24, 15, 8],  # [blue, pink, grey]
    "total_priority": 47,
    "average_priority": 7.83
}


Final Summary:
All images ranked by rescue ratio (descending)

Comprehensive comparison across all processed images

JSON format for programmatic access