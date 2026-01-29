import cv2
import numpy as np
import os
import json
import math
from datetime import datetime
from collections import defaultdict

# ==================== CONSTANTS ====================
# Priority mappings
SHAPE_PRIORITY = {
    'star': 3,      # Children (highest) - ⭐
    'triangle': 2,  # Elderly - 🔺
    'square': 1     # Adults (lowest) - ⬛
}

EMERGENCY_PRIORITY = {
    'red': 3,       # Severe (highest) - 🔴
    'yellow': 2,    # Mild - 🟡
    'green': 1      # Safe (lowest) - 🟢
}

CAMP_CAPACITY = {
    'pink': 3,
    'blue': 4,
    'grey': 2
}

# BGR colors for display
CAMP_COLORS_BGR = {
    'pink': (180, 105, 255),    # Pink in BGR
    'blue': (255, 0, 0),        # Blue in BGR
    'grey': (128, 128, 128)     # Grey in BGR
}

EMERGENCY_COLORS_BGR = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0)
}

# ==================== IMAGE SEGMENTATION ====================
def segment_ocean_land(image):
    """Segment ocean (blue) and land (brown/green) regions."""
    print("  Segmenting ocean and land...")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # HSV ranges for ocean (blue)
    ocean_lower = np.array([90, 60, 60])
    ocean_upper = np.array([130, 255, 255])
    
    # HSV ranges for land (brown/green)
    land_lower = np.array([15, 40, 40])
    land_upper = np.array([35, 255, 255])
    
    # Create masks
    ocean_mask = cv2.inRange(hsv, ocean_lower, ocean_upper)
    land_mask = cv2.inRange(hsv, land_lower, land_upper)
    
    # Clean up masks with morphology
    kernel = np.ones((3, 3), np.uint8)
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel)
    land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create segmented overlay
    segmented = image.copy()
    
    # Color ocean with semi-transparent blue
    ocean_overlay = np.zeros_like(image)
    ocean_overlay[ocean_mask > 0] = (255, 0, 0)  # Blue in BGR
    cv2.addWeighted(segmented, 0.7, ocean_overlay, 0.3, 0, segmented)
    
    # Color land with semi-transparent green
    land_overlay = np.zeros_like(image)
    land_overlay[land_mask > 0] = (0, 255, 0)    # Green in BGR
    cv2.addWeighted(segmented, 0.7, land_overlay, 0.3, 0, segmented)
    
    return segmented, ocean_mask, land_mask

# ==================== CASUALTY DETECTION ====================
def detect_casualties(image):
    """Detect and classify casualties (shapes with emergency colors)."""
    print("  Detecting casualties...")
    
    # Make a copy for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding for better shape detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    casualties = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 50 or area > 5000:  # Filter by size
            continue
        
        # Get centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Classify shape
        shape = classify_shape(contour)
        if shape == 'unknown':
            continue
        
        # Detect emergency color
        emergency = detect_emergency_color(image, cx, cy)
        
        # Calculate priority
        priority = SHAPE_PRIORITY.get(shape, 0) * EMERGENCY_PRIORITY.get(emergency, 0)
        
        casualties.append({
            'id': i,
            'position': (cx, cy),
            'shape': shape,
            'emergency': emergency,
            'priority': priority,
            'shape_priority': SHAPE_PRIORITY.get(shape, 0),
            'emergency_priority': EMERGENCY_PRIORITY.get(emergency, 0)
        })
    
    print(f"  Found {len(casualties)} casualties")
    return casualties

def classify_shape(contour):
    """Classify shape as star, triangle, or square."""
    # Approximate contour
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # Calculate shape features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 'unknown'
    
    # Circularity (4*pi*area/perimeter^2)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    # Classification logic
    if vertices == 3:
        return 'triangle'
    elif vertices == 4 and 0.8 < aspect_ratio < 1.2:
        return 'square'
    elif vertices >= 8 and circularity < 0.7:
        return 'star'
    
    return 'unknown'

def detect_emergency_color(image, x, y, radius=10):
    """Detect the emergency color at a specific location."""
    h, w = image.shape[:2]
    
    # Define ROI around the point
    x1 = max(0, x - radius)
    x2 = min(w, x + radius)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius)
    
    if x2 <= x1 or y2 <= y1:
        return 'unknown'
    
    roi = image[y1:y2, x1:x2]
    
    # Convert to HSV for color detection
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    color_ranges = {
        'red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255]))
        ],
        'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
        'green': (np.array([40, 100, 100]), np.array([80, 255, 255]))
    }
    
    # Count pixels for each color
    max_count = 0
    detected_color = 'unknown'
    
    for color_name, ranges in color_ranges.items():
        if color_name == 'red':
            # Red has two ranges (0-10 and 170-180)
            mask1 = cv2.inRange(hsv_roi, ranges[0][0], ranges[0][1])
            mask2 = cv2.inRange(hsv_roi, ranges[1][0], ranges[1][1])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_roi, ranges[0], ranges[1])
        
        count = np.sum(mask > 0)
        
        if count > max_count:
            max_count = count
            detected_color = color_name
    
    return detected_color

# ==================== CAMP DETECTION ====================
def detect_rescue_camps(image):
    """Detect rescue camps (colored circles)."""
    print("  Detecting rescue camps...")
    
    camps = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=15,
        maxRadius=80
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        
        for i, circle in enumerate(circles):
            x, y, r = circle
            
            # Detect camp color
            camp_color = detect_camp_color(image, x, y, r)
            
            if camp_color in CAMP_CAPACITY:
                camps.append({
                    'id': i,
                    'position': (x, y),
                    'radius': r,
                    'color': camp_color,
                    'capacity': CAMP_CAPACITY[camp_color],
                    'assigned': []  # Will hold assigned casualties
                })
    
    print(f"  Found {len(camps)} rescue camps")
    return camps

def detect_camp_color(image, x, y, radius):
    """Detect the color of a rescue camp."""
    h, w = image.shape[:2]
    
    # Create mask for the circle region
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    
    # Extract the circle region
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Get the mean color in the circle
    mean_color = cv2.mean(image, mask=mask)[:3]
    b, g, r = mean_color
    
    # Classify based on BGR values
    # Note: Adjust these thresholds based on your actual camp colors
    
    if r > 200 and b > 150 and g < 150:  # Pinkish
        return 'pink'
    elif b > 200 and r < 150 and g < 150:  # Blue
        return 'blue'
    elif abs(r - g) < 30 and abs(r - b) < 30:  # Grey
        return 'grey'
    
    return 'unknown'

# ==================== ASSIGNMENT ALGORITHM ====================
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def assign_casualties_to_camps(casualties, camps):
    """Assign casualties to camps based on priority and distance."""
    print("  Assigning casualties to camps...")
    
    # Reset assignments
    for camp in camps:
        camp['assigned'] = []
        camp['current_capacity'] = 0
    
    # Sort casualties by priority (highest first)
    # Tie-breaker: higher emergency priority
    casualties_sorted = sorted(
        casualties,
        key=lambda c: (c['priority'], c['emergency_priority']),
        reverse=True
    )
    
    # Assignment loop
    for casualty in casualties_sorted:
        best_camp = None
        best_score = -float('inf')
        
        for camp in camps:
            # Check capacity
            if camp['current_capacity'] >= camp['capacity']:
                continue
            
            # Calculate distance
            distance = calculate_distance(casualty['position'], camp['position'])
            
            # SCORING FORMULA (Adjust weights as needed)
            # Priority contributes 70%, distance contributes 30%
            priority_score = casualty['priority'] * 70
            distance_score = (1000 - distance) * 0.3  # Normalized distance
            emergency_bonus = casualty['emergency_priority'] * 5
            
            total_score = priority_score + distance_score + emergency_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_camp = camp
        
        # Assign to best camp
        if best_camp is not None:
            best_camp['assigned'].append(casualty)
            best_camp['current_capacity'] += 1
    
    return camps

# ==================== STATISTICS CALCULATION ====================
def calculate_statistics(camps):
    """Calculate statistics for the image."""
    print("  Calculating statistics...")
    
    # Initialize stats for each camp
    stats = {
        'blue': {'count': 0, 'total_priority': 0, 'casualties': []},
        'pink': {'count': 0, 'total_priority': 0, 'casualties': []},
        'grey': {'count': 0, 'total_priority': 0, 'casualties': []}
    }
    
    total_casualties = 0
    total_priority = 0
    
    for camp in camps:
        camp_color = camp['color']
        if camp_color not in stats:
            continue
            
        assigned = camp['assigned']
        stats[camp_color]['count'] = len(assigned)
        
        for casualty in assigned:
            # Convert to required format: [shape_code, emergency_code]
            shape_code = SHAPE_PRIORITY.get(casualty['shape'], 0)
            emergency_code = EMERGENCY_PRIORITY.get(casualty['emergency'], 0)
            
            stats[camp_color]['casualties'].append([shape_code, emergency_code])
            stats[camp_color]['total_priority'] += casualty['priority']
            
            total_casualties += 1
            total_priority += casualty['priority']
    
    # Calculate average priority (rescue ratio)
    avg_priority = total_priority / total_casualties if total_casualties > 0 else 0
    
    stats['total_casualties'] = total_casualties
    stats['total_priority'] = total_priority
    stats['avg_priority'] = avg_priority
    
    return stats

# ==================== VISUALIZATION ====================
def create_visualization(original_image, segmented_image, casualties, camps, output_path):
    """Create visualization image with assignments."""
    print("  Creating visualization...")
    
    # Start with original image
    visualization = original_image.copy()
    
    # Draw camps with labels
    for camp in camps:
        x, y = camp['position']
        r = camp['radius']
        color = CAMP_COLORS_BGR.get(camp['color'], (255, 255, 255))
        
        # Draw camp circle
        cv2.circle(visualization, (x, y), r, color, 3)
        
        # Draw camp label
        label = f"{camp['color']} ({len(camp['assigned'])}/{camp['capacity']})"
        cv2.putText(visualization, label, (x - 40, y - r - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw casualties and assignment lines
    for camp in camps:
        for casualty in camp['assigned']:
            cx, cy = casualty['position']
            
            # Draw line from casualty to camp
            cv2.line(visualization, (cx, cy), camp['position'], (0, 255, 255), 2)
            
            # Draw casualty based on shape and emergency
            emergency_color = EMERGENCY_COLORS_BGR.get(casualty['emergency'], (255, 255, 255))
            
            if casualty['shape'] == 'star':
                # Draw star (simplified as pentagon)
                pts = []
                for i in range(5):
                    angle = 2 * np.pi * i / 5 - np.pi / 2
                    px = cx + int(15 * np.cos(angle))
                    py = cy + int(15 * np.sin(angle))
                    pts.append((px, py))
                cv2.polylines(visualization, [np.array(pts)], True, emergency_color, 2)
                
            elif casualty['shape'] == 'triangle':
                # Draw triangle
                pts = np.array([
                    (cx, cy - 15),
                    (cx - 13, cy + 10),
                    (cx + 13, cy + 10)
                ])
                cv2.polylines(visualization, [pts], True, emergency_color, 2)
                
            elif casualty['shape'] == 'square':
                # Draw square
                cv2.rectangle(visualization, 
                            (cx - 12, cy - 12),
                            (cx + 12, cy + 12),
                            emergency_color, 2)
            
            # Add priority label
            priority_label = f"P:{casualty['priority']}"
            cv2.putText(visualization, priority_label, (cx - 10, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add statistics overlay
    y_offset = 30
    cv2.putText(visualization, "RESCUE ASSIGNMENTS", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    for camp_color in ['blue', 'pink', 'grey']:
        # Count casualties by shape for this camp
        camp = next((c for c in camps if c['color'] == camp_color), None)
        if camp:
            star_count = sum(1 for c in camp['assigned'] if c['shape'] == 'star')
            triangle_count = sum(1 for c in camp['assigned'] if c['shape'] == 'triangle')
            square_count = sum(1 for c in camp['assigned'] if c['shape'] == 'square')
            
            text = f"{camp_color}: {len(camp['assigned'])} casualties "
            text += f"(⭐{star_count} 🔺{triangle_count} ⬛{square_count})"
            
            cv2.putText(visualization, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAMP_COLORS_BGR.get(camp_color, (255, 255, 255)), 1)
            y_offset += 20
    
    # Save the visualization
    cv2.imwrite(output_path, visualization)
    print(f"  Visualization saved: {output_path}")
    
    return visualization

# ==================== SINGLE IMAGE PROCESSING ====================
def process_single_image(image_path, output_dir):
    """Process a single image and generate all outputs."""
    image_name = os.path.basename(image_path)
    print(f"\nProcessing image: {image_name}")
    print("-" * 50)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Create base name for output files
    base_name = os.path.splitext(image_name)[0]
    
    # 1. Segment ocean and land
    segmented_image, ocean_mask, land_mask = segment_ocean_land(image)
    
    # Save segmented image
    segmented_path = os.path.join(output_dir, 'segmented', f"{base_name}_segmented.png")
    cv2.imwrite(segmented_path, segmented_image)
    
    # 2. Detect casualties
    casualties = detect_casualties(image)
    
    # 3. Detect rescue camps
    camps = detect_rescue_camps(image)
    
    # 4. Assign casualties to camps
    camps = assign_casualties_to_camps(casualties, camps)
    
    # 5. Calculate statistics
    stats = calculate_statistics(camps)
    
    # 6. Create visualization
    visualization_path = os.path.join(output_dir, 'results', f"{base_name}_results.png")
    visualization = create_visualization(image, segmented_image, casualties, camps, visualization_path)
    
    # 7. Prepare output data
    output_data = {
        'image_name': image_name,
        'segmented_image': segmented_path,
        'result_image': visualization_path,
        'casualties_per_camp': {
            'blue': stats['blue']['casualties'],
            'pink': stats['pink']['casualties'],
            'grey': stats['grey']['casualties']
        },
        'count_per_camp': {
            'blue': stats['blue']['count'],
            'pink': stats['pink']['count'],
            'grey': stats['grey']['count']
        },
        'priority_per_camp': {
            'blue': stats['blue']['total_priority'],
            'pink': stats['pink']['total_priority'],
            'grey': stats['grey']['total_priority']
        },
        'total_priority': stats['total_priority'],
        'average_priority': stats['avg_priority'],
        'total_casualties': stats['total_casualties']
    }
    
    print(f"✓ Processing complete for {image_name}")
    print(f"  Total casualties: {stats['total_casualties']}")
    print(f"  Average priority: {stats['avg_priority']:.2f}")
    
    return output_data

# ==================== BATCH PROCESSING ====================
def process_all_images(image_folder):
    """Process all images in a folder and generate comprehensive outputs."""
    print("=" * 60)
    print("UAS-DTU SEARCH AND RESCUE IMAGE PROCESSING SYSTEM")
    print("=" * 60)
    
    # Create output directory structure
    output_dir = os.path.join(os.path.dirname(image_folder), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "segmented"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Find all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = []
    
    for file in os.listdir(image_folder):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    print(f"Found {len(image_files)} images to process")
    print("-" * 60)
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] ", end="")
        result = process_single_image(image_path, output_dir)
        if result:
            all_results.append(result)
    
    # Generate summary reports
    generate_reports(all_results, output_dir)
    
    return all_results

def generate_reports(all_results, output_dir):
    """Generate comprehensive reports in required formats."""
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)
    
    # Sort images by average priority (rescue ratio)
    all_results_sorted = sorted(all_results, key=lambda x: x['average_priority'], reverse=True)
    
    # 1. Generate detailed JSON report
    detailed_report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_images': len(all_results),
        'images': all_results_sorted,
        'summary_by_priority': [
            {
                'image_name': result['image_name'],
                'average_priority': result['average_priority'],
                'total_casualties': result['total_casualties']
            }
            for result in all_results_sorted
        ]
    }
    
    json_report_path = os.path.join(output_dir, "reports", "detailed_report.json")
    with open(json_report_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)
    print(f"✓ Detailed JSON report saved: {json_report_path}")
    
    # 2. Generate text summary report (required format)
    text_report_path = os.path.join(output_dir, "reports", "summary_report.txt")
    with open(text_report_path, 'w') as f:
        f.write("UAS-DTU Search and Rescue Task - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("IMAGES SORTED BY RESCUE RATIO (Highest to Lowest):\n")
        f.write("-" * 60 + "\n")
        for i, result in enumerate(all_results_sorted, 1):
            f.write(f"{i}. {result['image_name']}\n")
            f.write(f"   Average Priority (Rescue Ratio): {result['average_priority']:.2f}\n")
            f.write(f"   Total Casualties: {result['total_casualties']}\n")
            f.write(f"   Total Priority: {result['total_priority']}\n\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("DETAILED RESULTS FOR EACH IMAGE:\n")
        f.write("=" * 60 + "\n\n")
        
        for result in all_results:
            f.write(f"Image: {result['image_name']}\n")
            f.write("-" * 40 + "\n")
            
            f.write("Casualties assigned to each camp:\n")
            f.write(f"  Blue camp: {result['casualties_per_camp']['blue']}\n")
            f.write(f"  Pink camp: {result['casualties_per_camp']['pink']}\n")
            f.write(f"  Grey camp: {result['casualties_per_camp']['grey']}\n\n")
            
            f.write("Count of casualties per camp:\n")
            f.write(f"  Blue: {result['count_per_camp']['blue']}\n")
            f.write(f"  Pink: {result['count_per_camp']['pink']}\n")
            f.write(f"  Grey: {result['count_per_camp']['grey']}\n\n")
            
            f.write("Total priority per camp:\n")
            f.write(f"  Blue: {result['priority_per_camp']['blue']}\n")
            f.write(f"  Pink: {result['priority_per_camp']['pink']}\n")
            f.write(f"  Grey: {result['priority_per_camp']['grey']}\n\n")
            
            f.write(f"Total priority: {result['total_priority']}\n")
            f.write(f"Average priority: {result['average_priority']:.2f}\n")
            f.write("\n" + "=" * 40 + "\n\n")
    
    print(f"✓ Text summary report saved: {text_report_path}")
    
    # 3. Generate CSV report for easy analysis
    csv_report_path = os.path.join(output_dir, "reports", "data_analysis.csv")
    with open(csv_report_path, 'w') as f:
        f.write("Image,AvgPriority,TotalCasualties,BlueCount,PinkCount,GreyCount,BluePriority,PinkPriority,GreyPriority\n")
        for result in all_results_sorted:
            f.write(f"{result['image_name']},{result['average_priority']:.2f},")
            f.write(f"{result['total_casualties']},{result['count_per_camp']['blue']},")
            f.write(f"{result['count_per_camp']['pink']},{result['count_per_camp']['grey']},")
            f.write(f"{result['priority_per_camp']['blue']},{result['priority_per_camp']['pink']},")
            f.write(f"{result['priority_per_camp']['grey']}\n")
    
    print(f"✓ CSV data analysis saved: {csv_report_path}")
    
    # 4. Print final summary to console
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("Images ranked by rescue ratio (highest to lowest):")
    print("-" * 60)
    
    for i, result in enumerate(all_results_sorted, 1):
        print(f"{i:2d}. {result['image_name']:20s} | "
              f"Avg Priority: {result['average_priority']:6.2f} | "
              f"Casualties: {result['total_casualties']:2d}")
    
    print("\n" + "=" * 60)
    print("ALL OUTPUTS GENERATED SUCCESSFULLY!")
    print(f"Check the '{output_dir}' folder for:")
    print("  1. Segmented images (ocean/land)")
    print("  2. Result images (with assignments)")
    print("  3. Detailed reports (JSON, TXT, CSV)")
    print("=" * 60)

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("UAS-DTU Search and Rescue Task - Image Processing")
    print("=" * 60)
    
    # Set your image folder path here
    IMAGE_FOLDER = r"C:\Users\user\Documents\uas_dtu_project\images\task images"
    
    # Check if folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' does not exist!")
        print("Please check the path and try again.")
    else:
        # Process all images
        results = process_all_images(IMAGE_FOLDER)
        
        # Optional: Display sample results
        if results:
            print("\nWould you like to view a sample result? (y/n): ", end="")
            choice = input().strip().lower()
            
            if choice == 'y' or choice == 'yes':
                # Display the first result image
                sample_result = results[0]['result_image']
                if os.path.exists(sample_result):
                    img = cv2.imread(sample_result)
                    cv2.imshow("Sample Result - Press any key to close", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Could not load sample image for display.")
    
    print("\nProcessing complete! Good luck with your UAS-DTU task! 🚁")