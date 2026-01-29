"""
UAS-DTU Search and Rescue - Enhanced Image Processor
"""

import cv2
import numpy as np
import math
import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from collections import defaultdict

class EnhancedSARProcessor:
    """Enhanced SAR Image Processor with better detection algorithms"""
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        
        # Enhanced color ranges in HSV
        self.color_ranges = {
            # Ocean: blue water
            'ocean_lower': np.array([90, 60, 60]),
            'ocean_upper': np.array([130, 255, 255]),
            
            # Land: brown/green
            'land_lower': np.array([15, 40, 40]),
            'land_upper': np.array([35, 255, 255]),
            
            # Casualty colors
            'red_lower1': np.array([0, 100, 100]),
            'red_upper1': np.array([10, 255, 255]),
            'red_lower2': np.array([160, 100, 100]),  # Red wraps around in HSV
            'red_upper2': np.array([180, 255, 255]),
            
            'yellow_lower': np.array([20, 100, 100]),
            'yellow_upper': np.array([30, 255, 255]),
            
            'green_lower': np.array([40, 100, 100]),
            'green_upper': np.array([80, 255, 255]),
            
            # Camp colors
            'pink_lower': np.array([140, 50, 50]),
            'pink_upper': np.array([170, 255, 255]),
            
            'blue_lower': np.array([100, 100, 100]),
            'blue_upper': np.array([130, 255, 255]),
            
            'grey_lower': np.array([0, 0, 50]),
            'grey_upper': np.array([180, 50, 200])
        }
        
        # Priority mappings
        self.shape_priority = {'star': 3, 'triangle': 2, 'square': 1}
        self.condition_priority = {'red': 3, 'yellow': 2, 'green': 1}
        
        # Camp capacities
        self.camp_capacity = {
            'blue': 4,
            'pink': 3, 
            'grey': 2
        }

    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Resize if too large for display
        height, width = img.shape[:2]
        if height > 800 or width > 1200:
            scale = min(800/height, 1200/width)
            new_size = (int(width * scale), int(height * scale))
            img = cv2.resize(img, new_size)
        
        return img

    def segment_regions(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment ocean and land regions"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create ocean mask
        ocean_mask = cv2.inRange(hsv, self.color_ranges['ocean_lower'], 
                                self.color_ranges['ocean_upper'])
        
        # Create land mask  
        land_mask = cv2.inRange(hsv, self.color_ranges['land_lower'], 
                               self.color_ranges['land_upper'])
        
        # Apply morphological operations to clean masks
        kernel = np.ones((5, 5), np.uint8)
        ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel)
        land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create overlay
        overlay = img.copy()
        overlay[ocean_mask > 0] = [255, 0, 0]  # Blue for ocean
        overlay[land_mask > 0] = [0, 255, 0]   # Green for land
        
        # Combine masks for segmented image
        segmented = np.zeros_like(img)
        segmented[ocean_mask > 0] = [255, 0, 0]
        segmented[land_mask > 0] = [0, 255, 0]
        
        if self.debug_mode:
            self.show_debug_images([img, ocean_mask, land_mask, overlay], 
                                 ['Original', 'Ocean Mask', 'Land Mask', 'Overlay'])
        
        return segmented, overlay

    def detect_casualties_enhanced(self, img: np.ndarray) -> List[Dict]:
        """Enhanced casualty detection using multiple techniques"""
        casualties = []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect each color of casualty
        for color_name, color_key in [('red', ['red_lower1', 'red_upper1', 'red_lower2', 'red_upper2']),
                                     ('yellow', ['yellow_lower', 'yellow_upper']),
                                     ('green', ['green_lower', 'green_upper'])]:
            
            if color_name == 'red':
                # Red has two ranges
                mask1 = cv2.inRange(hsv, self.color_ranges['red_lower1'], 
                                   self.color_ranges['red_upper1'])
                mask2 = cv2.inRange(hsv, self.color_ranges['red_lower2'], 
                                   self.color_ranges['red_upper2'])
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower = self.color_ranges[f'{color_name}_lower']
                upper = self.color_ranges[f'{color_name}_upper']
                color_mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in the color mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50 or area > 5000:  # Filter by reasonable size
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip if too small
                if w < 10 or h < 10:
                    continue
                
                # Get center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Classify shape
                shape_type = self.classify_shape_enhanced(contour)
                if not shape_type:
                    continue
                
                # Calculate priority scores
                shape_prio = self.shape_priority.get(shape_type, 0)
                condition_prio = self.condition_priority.get(color_name, 0)
                total_prio = shape_prio * condition_prio
                
                casualty = {
                    'id': len(casualties),
                    'center': (center_x, center_y),
                    'shape': shape_type,
                    'color': color_name,
                    'shape_priority': shape_prio,
                    'condition_priority': condition_prio,
                    'total_priority': total_prio,
                    'bbox': (x, y, w, h),
                    'contour': contour
                }
                
                casualties.append(casualty)
        
        if self.debug_mode:
            debug_img = img.copy()
            for c in casualties:
                x, y, w, h = c['bbox']
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"{c['shape']} {c['color']}", 
                          (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.show_debug_images([debug_img], ['Detected Casualties'])
        
        print(f"Detected {len(casualties)} casualties")
        return casualties

    def classify_shape_enhanced(self, contour: np.ndarray) -> str:
        """Enhanced shape classification"""
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Calculate shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity for star detection
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Detect star (many vertices, low circularity)
        if num_vertices >= 8 and circularity < 0.5:
            return 'star'
        
        # Detect triangle (3 vertices)
        elif num_vertices == 3:
            return 'triangle'
        
        # Detect square/rectangle (4 vertices)
        elif num_vertices == 4:
            # Check if it's close to a square
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.7 < aspect_ratio < 1.3:
                return 'square'
        
        # Try convexity defects for star detection
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            
            if defects is not None and len(defects) > 3:
                return 'star'
        except:
            pass
        
        return None

    def detect_rescue_camps_enhanced(self, img: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """Enhanced camp detection"""
        camps = {}
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect each camp color
        camp_colors = {
            'blue': ('blue_lower', 'blue_upper'),
            'pink': ('pink_lower', 'pink_upper'),
            'grey': ('grey_lower', 'grey_upper')
        }
        
        for camp_name, (lower_key, upper_key) in camp_colors.items():
            mask = cv2.inRange(hsv, self.color_ranges[lower_key], 
                              self.color_ranges[upper_key])
            
            # Clean mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Take the largest contour for each camp
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Check if it's circular enough
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.7:  # Circle has circularity close to 1
                        M = cv2.moments(largest_contour)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            camps[camp_name] = (cx, cy)
        
        # If circle detection fails, try HoughCircles as backup
        if len(camps) < 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                      param1=100, param2=30, minRadius=20, maxRadius=60)
            
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                for i, (cx, cy, r) in enumerate(circles[:3]):  # Take up to 3 circles
                    if i == 0 and 'blue' not in camps:
                        camps['blue'] = (cx, cy)
                    elif i == 1 and 'pink' not in camps:
                        camps['pink'] = (cx, cy)
                    elif i == 2 and 'grey' not in camps:
                        camps['grey'] = (cx, cy)
        
        if self.debug_mode:
            debug_img = img.copy()
            for camp_name, (cx, cy) in camps.items():
                cv2.circle(debug_img, (cx, cy), 30, (0, 0, 255), 3)
                cv2.putText(debug_img, camp_name, (cx-20, cy-40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.show_debug_images([debug_img], ['Detected Rescue Camps'])
        
        print(f"Detected {len(camps)} rescue camps: {list(camps.keys())}")
        return camps

    def assign_casualties_priority_based(self, casualties: List[Dict], 
                                        camps: Dict[str, Tuple[int, int]]) -> Dict[str, List[Dict]]:
        """
        Assign casualties to camps based on priority rules
        Returns assignments with camps in order: blue, pink, grey
        """
        # Initialize assignments
        assignments = {
            'blue': {'casualties': [], 'total_priority': 0},
            'pink': {'casualties': [], 'total_priority': 0},
            'grey': {'casualties': [], 'total_priority': 0}
        }
        
        # Sort casualties by priority (highest first)
        # First by total priority, then by emergency priority for ties
        sorted_casualties = sorted(casualties, 
                                 key=lambda c: (c['total_priority'], c['condition_priority']), 
                                 reverse=True)
        
        # Process each casualty
        for casualty in sorted_casualties:
            best_camp = None
            best_score = -float('inf')
            
            cx, cy = casualty['center']
            
            for camp_name, camp_pos in camps.items():
                # Check camp capacity
                if len(assignments[camp_name]['casualties']) >= self.camp_capacity[camp_name]:
                    continue
                
                # Calculate distance
                camp_x, camp_y = camp_pos
                distance = math.sqrt((cx - camp_x)**2 + (cy - camp_y)**2)
                
                # Calculate score: higher priority should get higher score
                # We want to balance priority and distance
                # Score = Priority * Weight / (distance + 1)
                # Where weight is 100 to keep scores in reasonable range
                score = (casualty['total_priority'] * 100) / (distance + 1)
                
                if score > best_score:
                    best_score = score
                    best_camp = camp_name
            
            # Assign to best camp
            if best_camp:
                assignments[best_camp]['casualties'].append(casualty)
                assignments[best_camp]['total_priority'] += casualty['total_priority']
                casualty['assigned_to'] = best_camp
        
        return assignments

    def calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def generate_outputs(self, img: np.ndarray, assignments: Dict[str, List[Dict]], 
                        camps: Dict[str, Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
        """Generate all required outputs"""
        # Create output image with assignments
        output_img = img.copy()
        
        # Draw camps
        camp_colors = {'blue': (255, 0, 0), 'pink': (255, 0, 255), 'grey': (128, 128, 128)}
        
        for camp_name, camp_pos in camps.items():
            color = camp_colors.get(camp_name, (0, 255, 0))
            cv2.circle(output_img, camp_pos, 25, color, 3)
            cv2.putText(output_img, camp_name.upper(), 
                       (camp_pos[0]-20, camp_pos[1]-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw casualties and assignments
        shape_colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
        
        for camp_name, camp_data in assignments.items():
            for casualty in camp_data['casualties']:
                # Draw line from camp to casualty
                camp_pos = camps[camp_name]
                casualty_pos = casualty['center']
                color = camp_colors.get(camp_name, (0, 255, 0))
                cv2.line(output_img, camp_pos, casualty_pos, color, 2)
                
                # Draw casualty shape
                shape = casualty['shape']
                color_cas = shape_colors.get(casualty['color'], (255, 255, 255))
                x, y = casualty_pos
                
                if shape == 'star':
                    self.draw_star(output_img, (x, y), 15, color_cas)
                elif shape == 'triangle':
                    self.draw_triangle(output_img, (x, y), 15, color_cas)
                elif shape == 'square':
                    cv2.rectangle(output_img, (x-10, y-10), (x+10, y+10), color_cas, -1)
        
        # Calculate statistics
        stats = self.calculate_statistics(assignments)
        
        return output_img, stats

    def draw_star(self, img: np.ndarray, center: Tuple[int, int], size: int, color: Tuple[int, int, int]):
        """Draw a star shape"""
        points = []
        for i in range(10):
            angle = i * 36 * np.pi / 180
            r = size if i % 2 == 0 else size // 2
            x = center[0] + int(r * np.cos(angle))
            y = center[1] + int(r * np.sin(angle))
            points.append((x, y))
        
        cv2.fillPoly(img, [np.array(points)], color)

    def draw_triangle(self, img: np.ndarray, center: Tuple[int, int], size: int, color: Tuple[int, int, int]):
        """Draw a triangle shape"""
        height = int(size * np.sqrt(3))
        points = [
            (center[0], center[1] - height//2),
            (center[0] - size, center[1] + height//2),
            (center[0] + size, center[1] + height//2)
        ]
        cv2.fillPoly(img, [np.array(points)], color)

    def calculate_statistics(self, assignments: Dict[str, List[Dict]]) -> Dict:
        """Calculate required statistics"""
        # Camp order as specified: blue, pink, grey
        camp_order = ['blue', 'pink', 'grey']
        
        # 2a. Count casualties per camp
        casualty_counts = [len(assignments[camp]['casualties']) for camp in camp_order]
        
        # 2b. Casualty details [shape_priority, condition_priority]
        casualty_details = []
        for camp in camp_order:
            camp_details = []
            for casualty in assignments[camp]['casualties']:
                camp_details.append([casualty['shape_priority'], casualty['condition_priority']])
            casualty_details.append(camp_details)
        
        # 3. Total priority per camp and average priority
        camp_priorities = [assignments[camp]['total_priority'] for camp in camp_order]
        
        total_casualties = sum(casualty_counts)
        total_priority = sum(camp_priorities)
        avg_priority = total_priority / total_casualties if total_casualties > 0 else 0
        
        return {
            'casualty_counts': casualty_counts,
            'casualty_details': casualty_details,
            'camp_priorities': camp_priorities,
            'total_priority': total_priority,
            'average_priority': avg_priority
        }

    def show_debug_images(self, images: List[np.ndarray], titles: List[str]):
        """Show debug images"""
        plt.figure(figsize=(15, 5))
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), i+1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def process_image_complete(self, image_path: str) -> Dict:
        """Complete processing pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            # 1. Load and preprocess
            img = self.load_and_preprocess(image_path)
            
            # 2. Segment ocean and land
            segmented, overlay = self.segment_regions(img)
            
            # 3. Detect casualties
            casualties = self.detect_casualties_enhanced(img)
            
            # 4. Detect rescue camps
            camps = self.detect_rescue_camps_enhanced(img)
            
            # 5. Assign casualties to camps
            assignments = self.assign_casualties_priority_based(casualties, camps)
            
            # 6. Generate outputs
            output_img, stats = self.generate_outputs(img, assignments, camps)
            
            # Prepare final results
            results = {
                'image_name': os.path.basename(image_path),
                'original_image': img,
                'segmented_image': segmented,
                'overlay_image': overlay,
                'output_image': output_img,
                'casualties_detected': len(casualties),
                'camps_detected': camps,
                'assignments': assignments,
                'statistics': stats
            }
            
            return results
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_results(self, results: Dict, output_dir: str = "enhanced_output"):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        img_name = results['image_name']
        base_name = os.path.splitext(img_name)[0]
        
        # Save images
        cv2.imwrite(f"{output_dir}/{base_name}_original.png", results['original_image'])
        cv2.imwrite(f"{output_dir}/{base_name}_segmented.png", results['segmented_image'])
        cv2.imwrite(f"{output_dir}/{base_name}_overlay.png", results['overlay_image'])
        cv2.imwrite(f"{output_dir}/{base_name}_output.png", results['output_image'])
        
        # Save statistics
        stats = results['statistics']
        with open(f"{output_dir}/{base_name}_results.txt", 'w') as f:
            f.write(f"Image: {img_name}\n")
            f.write(f"Total casualties detected: {results['casualties_detected']}\n")
            f.write(f"Camps detected: {list(results['camps_detected'].keys())}\n\n")
            
            f.write("CASUALTY ASSIGNMENTS:\n")
            f.write("-" * 50 + "\n")
            
            # Camp order: blue, pink, grey
            camp_order = ['blue', 'pink', 'grey']
            for i, camp in enumerate(camp_order):
                count = stats['casualty_counts'][i]
                f.write(f"\n{camp.upper()} Camp:\n")
                f.write(f"  Casualties assigned: {count}\n")
                f.write(f"  Total priority: {stats['camp_priorities'][i]}\n")
                f.write(f"  Casualty details: {stats['casualty_details'][i]}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("SUMMARY STATISTICS:\n")
            f.write("="*50 + "\n")
            f.write(f"Total priority: {stats['total_priority']}\n")
            f.write(f"Average priority (Rescue Ratio): {stats['average_priority']:.2f}\n")
        
        print(f"\nResults saved to {output_dir}/")
        print(f"  - Original: {base_name}_original.png")
        print(f"  - Segmented: {base_name}_segmented.png")
        print(f"  - Output: {base_name}_output.png")
        print(f"  - Results: {base_name}_results.txt")


def main():
    """Main execution function"""
    print("UAS-DTU Search and Rescue - Enhanced Image Processor")
    print("="*60)
    
    # Initialize processor with debug mode ON
    processor = EnhancedSARProcessor(debug_mode=True)
    
    # Your image path
    image_path = r"C:\Users\user\Documents\uas_dtu_project\images\task images\1.png"
    
    # Process the image
    results = processor.process_image_complete(image_path)
    
    if results:
        # Display results
        stats = results['statistics']
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        print("\n1. Casualty counts per camp [blue, pink, grey]:")
        print(f"   Blue: {stats['casualty_counts'][0]}")
        print(f"   Pink: {stats['casualty_counts'][1]}")
        print(f"   Grey: {stats['casualty_counts'][2]}")
        
        print("\n2. Casualty details per camp [blue, pink, grey]:")
        for i, camp in enumerate(['blue', 'pink', 'grey']):
            print(f"   {camp}: {stats['casualty_details'][i]}")
        
        print("\n3. Camp priorities [blue, pink, grey]:")
        print(f"   {stats['camp_priorities']}")
        
        print(f"\n4. Total priority: {stats['total_priority']}")
        print(f"   Average priority (Rescue Ratio): {stats['average_priority']:.2f}")
        
        # Save results
        processor.save_results(results)
        
        # Show output image
        cv2.imshow('SAR Output - Casualty Assignments', results['output_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to process image. Check the error messages above.")

if __name__ == "__main__":
    main()