"""
UAS-DTU Search and Rescue - OPTIMIZED Image Processor
Enhanced with better detection algorithms and priority-based assignment
"""

import cv2
import numpy as np
import math
import os
import json
import glob
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Casualty:
    """Data class for casualty information"""
    id: int
    center: Tuple[int, int]
    shape: str
    color: str
    shape_priority: int
    condition_priority: int
    total_priority: int
    bbox: Tuple[int, int, int, int]
    contour: np.ndarray

@dataclass  
class Camp:
    """Data class for rescue camp information"""
    name: str
    center: Tuple[int, int]
    capacity: int
    assigned_casualties: List[Casualty]
    total_priority: int = 0

class OptimizedSARProcessor:
    """Optimized SAR image processor with improved detection"""
    
    def __init__(self, debug: bool = True):
        self.debug = debug
        
        # Color ranges in HSV (optimized for detection)
        self.COLOR_RANGES = {
            # Ocean (Blue water)
            'ocean': ([90, 60, 60], [130, 255, 255]),
            
            # Land (Brown/Green)
            'land': ([15, 40, 60], [40, 255, 255]),
            
            # Casualty colors
            'red': ([0, 100, 100], [10, 255, 255]),
            'red2': ([170, 100, 100], [180, 255, 255]),  # Red wraps around
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            
            # Camp colors
            'camp_blue': ([100, 100, 50], [130, 255, 255]),
            'camp_pink': ([140, 50, 50], [170, 255, 255]),
            'camp_grey': ([0, 0, 50], [180, 30, 180])
        }
        
        # Priority mappings
        self.SHAPE_PRIORITY = {'star': 3, 'triangle': 2, 'square': 1}
        self.CONDITION_PRIORITY = {'red': 3, 'yellow': 2, 'green': 1}
        
        # Camp capacities
        self.CAMP_CAPACITIES = {'blue': 4, 'pink': 3, 'grey': 2}
        
        # Output directories
        self.OUTPUT_DIRS = {
            'base': 'optimized_output',
            'images': 'images',
            'results': 'results',
            'debug': 'debug'
        }
        
        self.create_directories()
    
    def create_directories(self):
        """Create output directories"""
        for dir_path in self.OUTPUT_DIRS.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def load_images(self, folder_path: str) -> List[str]:
        """Load all images from folder"""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
        image_paths.sort()
        print(f"✓ Found {len(image_paths)} images")
        return image_paths
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Improve contrast in value channel
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        
        # Apply Gaussian blur
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def segment_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment ocean and land regions"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Ocean mask
        ocean_lower, ocean_upper = self.COLOR_RANGES['ocean']
        ocean_mask = cv2.inRange(hsv, np.array(ocean_lower), np.array(ocean_upper))
        
        # Land mask
        land_lower, land_upper = self.COLOR_RANGES['land']
        land_mask = cv2.inRange(hsv, np.array(land_lower), np.array(land_upper))
        
        # Clean masks
        kernel = np.ones((7, 7), np.uint8)
        ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel)
        land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create overlay
        overlay = img.copy()
        overlay[ocean_mask > 0] = [255, 0, 0]  # Blue
        overlay[land_mask > 0] = [0, 255, 0]   # Green
        
        # Segmented image
        segmented = np.zeros_like(img)
        segmented[ocean_mask > 0] = [255, 0, 0]
        segmented[land_mask > 0] = [0, 255, 0]
        
        return segmented, overlay
    
    def detect_casualties_optimized(self, img: np.ndarray) -> List[Casualty]:
        """Optimized casualty detection with multi-stage approach"""
        casualties = []
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Step 1: Create combined mask for all casualty colors
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        color_definitions = [
            ('red', self.COLOR_RANGES['red']),
            ('red2', self.COLOR_RANGES['red2']),
            ('yellow', self.COLOR_RANGES['yellow']),
            ('green', self.COLOR_RANGES['green'])
        ]
        
        for color_name, (lower, upper) in color_definitions:
            if color_name.startswith('red'):
                color_key = 'red'
            else:
                color_key = color_name
            
            mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Step 2: Clean the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Step 3: Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 4: Process each contour
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 50 or area > 2000:  # Reasonable size range
                continue
            
            # Get bounding box and center
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            # Extract ROI for color and shape analysis
            roi = img[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            
            # Determine color
            color = self._detect_color(roi)
            if not color:
                continue
            
            # Determine shape
            shape = self._detect_shape(contour, roi)
            if not shape:
                continue
            
            # Calculate priorities
            shape_prio = self.SHAPE_PRIORITY.get(shape, 0)
            condition_prio = self.CONDITION_PRIORITY.get(color, 0)
            total_prio = shape_prio * condition_prio
            
            # Create Casualty object
            casualty = Casualty(
                id=i,
                center=center,
                shape=shape,
                color=color,
                shape_priority=shape_prio,
                condition_priority=condition_prio,
                total_priority=total_prio,
                bbox=(x, y, w, h),
                contour=contour
            )
            
            casualties.append(casualty)
        
        if self.debug:
            self._debug_casualties(img, casualties)
        
        return casualties
    
    def _detect_color(self, roi: np.ndarray) -> str:
        """Detect dominant color in ROI"""
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Test each color
        color_scores = {}
        
        # Red (two ranges)
        red_mask1 = cv2.inRange(hsv_roi, 
                               np.array(self.COLOR_RANGES['red'][0]), 
                               np.array(self.COLOR_RANGES['red'][1]))
        red_mask2 = cv2.inRange(hsv_roi,
                               np.array(self.COLOR_RANGES['red2'][0]),
                               np.array(self.COLOR_RANGES['red2'][1]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        color_scores['red'] = np.sum(red_mask > 0)
        
        # Yellow
        yellow_mask = cv2.inRange(hsv_roi,
                                 np.array(self.COLOR_RANGES['yellow'][0]),
                                 np.array(self.COLOR_RANGES['yellow'][1]))
        color_scores['yellow'] = np.sum(yellow_mask > 0)
        
        # Green
        green_mask = cv2.inRange(hsv_roi,
                                np.array(self.COLOR_RANGES['green'][0]),
                                np.array(self.COLOR_RANGES['green'][1]))
        color_scores['green'] = np.sum(green_mask > 0)
        
        # Find dominant color
        max_color = max(color_scores, key=color_scores.get)
        if color_scores[max_color] > 10:  # Minimum threshold
            return max_color
        
        return None
    
    def _detect_shape(self, contour: np.ndarray, roi: np.ndarray) -> str:
        """Detect shape using multiple methods"""
        # Method 1: Contour approximation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Method 2: Calculate shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Method 3: Hu moments for shape matching
        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        
        # Shape classification logic
        # Star detection: many vertices and low circularity
        if vertices >= 8 and circularity < 0.6:
            return 'star'
        
        # Triangle detection: 3 vertices
        if vertices == 3:
            return 'triangle'
        
        # Square detection: 4 vertices and nearly equal sides
        if vertices == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.2:
                return 'square'
        
        # Fallback: Template matching for shapes
        shape = self._template_match_shape(roi)
        if shape:
            return shape
        
        return None
    
    def _template_match_shape(self, roi: np.ndarray) -> str:
        """Template matching for shape recognition"""
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Create simple templates
        size = 30
        templates = {
            'triangle': self._create_triangle_template(size),
            'square': self._create_square_template(size),
            'star': self._create_star_template(size)
        }
        
        best_match = None
        best_score = 0
        
        for shape_name, template in templates.items():
            # Resize template to match ROI size
            resized_template = cv2.resize(template, (roi.shape[1], roi.shape[0]))
            
            # Match template
            result = cv2.matchTemplate(gray_roi, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score and max_val > 0.5:
                best_score = max_val
                best_match = shape_name
        
        return best_match
    
    def _create_triangle_template(self, size: int) -> np.ndarray:
        """Create triangle template"""
        template = np.zeros((size, size), dtype=np.uint8)
        pts = np.array([[size//2, 0], [0, size-1], [size-1, size-1]], np.int32)
        cv2.fillPoly(template, [pts], 255)
        return template
    
    def _create_square_template(self, size: int) -> np.ndarray:
        """Create square template"""
        template = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(template, (5, 5), (size-5, size-5), 255, -1)
        return template
    
    def _create_star_template(self, size: int) -> np.ndarray:
        """Create star template"""
        template = np.zeros((size, size), dtype=np.uint8)
        points = []
        for i in range(10):
            angle = i * 36 * np.pi / 180
            radius = size//2 if i % 2 == 0 else size//4
            x = size//2 + int(radius * np.cos(angle))
            y = size//2 + int(radius * np.sin(angle))
            points.append((x, y))
        
        cv2.fillPoly(template, [np.array(points, np.int32)], 255)
        return template
    
    def detect_camps_optimized(self, img: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """Optimized camp detection using color and shape"""
        camps = {}
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Camp color definitions
        camp_colors = {
            'blue': self.COLOR_RANGES['camp_blue'],
            'pink': self.COLOR_RANGES['camp_pink'],
            'grey': self.COLOR_RANGES['camp_grey']
        }
        
        for camp_name, (lower, upper) in camp_colors.items():
            # Color detection
            mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
            
            # Clean mask
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (most likely the camp)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 100:  # Minimum area for camp
                    # Check circularity
                    perimeter = cv2.arcLength(largest_contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.7:  # Should be circular
                            M = cv2.moments(largest_contour)
                            if M['m00'] > 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                                camps[camp_name] = (cx, cy)
                                continue
        
        # Backup: Hough Circle Transform
        if len(camps) < 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=100, param2=30, minRadius=20, maxRadius=50
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                
                # Assign colors based on position or pre-defined order
                if len(circles) >= 3:
                    # Sort circles by x-coordinate (left to right)
                    circles = sorted(circles, key=lambda c: c[0])
                    
                    # Assign in order: blue, pink, grey
                    if 'blue' not in camps:
                        camps['blue'] = (circles[0][0], circles[0][1])
                    if 'pink' not in camps:
                        camps['pink'] = (circles[1][0], circles[1][1])
                    if 'grey' not in camps:
                        camps['grey'] = (circles[2][0], circles[2][1])
        
        if self.debug:
            self._debug_camps(img, camps)
        
        return camps
    
    def assign_casualties_priority_based(
        self, 
        casualties: List[Casualty], 
        camps: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Camp]:
        """
        Priority-based assignment with optimization
        Follows rules: Star(3) > Triangle(2) > Square(1)
        Red(3) > Yellow(2) > Green(1)
        """
        # Initialize camps
        assigned_camps = {}
        for camp_name, center in camps.items():
            assigned_camps[camp_name] = Camp(
                name=camp_name,
                center=center,
                capacity=self.CAMP_CAPACITIES[camp_name],
                assigned_casualties=[]
            )
        
        # Sort casualties by priority (highest first)
        # First by total priority, then by emergency priority for ties
        sorted_casualties = sorted(
            casualties,
            key=lambda c: (c.total_priority, c.condition_priority),
            reverse=True
        )
        
        # Greedy assignment with optimization
        for casualty in sorted_casualties:
            best_camp = None
            best_score = -float('inf')
            
            cx, cy = casualty.center
            
            for camp_name, camp in assigned_camps.items():
                # Skip if camp is full
                if len(camp.assigned_casualties) >= camp.capacity:
                    continue
                
                # Calculate distance
                camp_x, camp_y = camp.center
                distance = math.sqrt((cx - camp_x)**2 + (cy - camp_y)**2)
                
                # Calculate score
                # Formula: Priority * 100 / (distance + 1)
                # This gives higher score to higher priority and closer distances
                score = (casualty.total_priority * 100) / (distance + 1)
                
                # Bonus for matching emergency with camp type
                # (e.g., severe casualties to closer camps)
                if distance < 100:  # Within reasonable distance
                    score *= 1.2
                
                if score > best_score:
                    best_score = score
                    best_camp = camp_name
            
            # Assign to best camp
            if best_camp:
                assigned_camps[best_camp].assigned_casualties.append(casualty)
                assigned_camps[best_camp].total_priority += casualty.total_priority
                casualty.assigned_to = best_camp
        
        return assigned_camps
    
    def calculate_statistics(self, assigned_camps: Dict[str, Camp]) -> Dict:
        """Calculate all required statistics"""
        # Camp order: blue, pink, grey
        camp_order = ['blue', 'pink', 'grey']
        
        # Initialize results
        casualty_counts = []
        casualty_details = []
        camp_priorities = []
        total_priority = 0
        total_casualties = 0
        
        for camp_name in camp_order:
            camp = assigned_camps.get(camp_name)
            
            if camp:
                count = len(camp.assigned_casualties)
                priority = camp.total_priority
                details = []
                
                for casualty in camp.assigned_casualties:
                    details.append([casualty.shape_priority, casualty.condition_priority])
                    total_priority += casualty.total_priority
                    total_casualties += 1
                
                casualty_counts.append(count)
                casualty_details.append(details)
                camp_priorities.append(priority)
            else:
                casualty_counts.append(0)
                casualty_details.append([])
                camp_priorities.append(0)
        
        # Calculate average priority (rescue ratio)
        avg_priority = total_priority / total_casualties if total_casualties > 0 else 0
        
        return {
            'casualty_counts': casualty_counts,  # [blue, pink, grey]
            'casualty_details': casualty_details,  # [[], [], []]
            'camp_priorities': camp_priorities,  # [blue, pink, grey]
            'total_priority': total_priority,
            'average_priority': avg_priority
        }
    
    def create_output_image(
        self,
        img: np.ndarray,
        overlay: np.ndarray,
        casualties: List[Casualty],
        camps: Dict[str, Camp]
    ) -> np.ndarray:
        """Create final output image with visual assignments"""
        # Create base image
        output_img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Camp colors
        camp_colors = {
            'blue': (255, 0, 0),      # Blue
            'pink': (255, 0, 255),    # Pink
            'grey': (128, 128, 128)   # Grey
        }
        
        # Casualty colors
        casualty_colors = {
            'red': (0, 0, 255),      # Red
            'yellow': (0, 255, 255),  # Yellow
            'green': (0, 255, 0)      # Green
        }
        
        # Draw camps
        for camp_name, camp in camps.items():
            if camp.center:
                color = camp_colors.get(camp_name, (0, 255, 0))
                cx, cy = camp.center
                
                # Draw camp circle
                cv2.circle(output_img, (cx, cy), 25, color, 3)
                
                # Draw camp name
                cv2.putText(output_img, camp_name.upper(),
                          (cx - 20, cy - 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw assignment lines
                for casualty in camp.assigned_casualties:
                    # Line from camp to casualty
                    cv2.line(output_img, (cx, cy), casualty.center,
                            color, 2, cv2.LINE_AA)
                    
                    # Draw casualty
                    self._draw_casualty(output_img, casualty, casualty_colors)
        
        # Draw unassigned casualties (if any)
        for casualty in casualties:
            if not hasattr(casualty, 'assigned_to'):
                self._draw_casualty(output_img, casualty, casualty_colors, unassigned=True)
        
        return output_img
    
    def _draw_casualty(self, img: np.ndarray, casualty: Casualty, 
                      colors: Dict, unassigned: bool = False):
        """Draw casualty on image"""
        color = colors.get(casualty.color, (255, 255, 255))
        x, y = casualty.center
        
        # Draw shape
        if casualty.shape == 'star':
            self._draw_star(img, (x, y), 15, color)
        elif casualty.shape == 'triangle':
            self._draw_triangle(img, (x, y), 15, color)
        elif casualty.shape == 'square':
            cv2.rectangle(img, (x-10, y-10), (x+10, y+10), color, -1)
        
        # Draw priority text
        text = f"{casualty.total_priority}"
        if unassigned:
            text += " (U)"
            color = (255, 255, 255)  # White for unassigned
        
        cv2.putText(img, text, (x-10, y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_star(self, img: np.ndarray, center: Tuple[int, int], 
                  size: int, color: Tuple[int, int, int]):
        """Draw star shape"""
        points = []
        for i in range(10):
            angle = i * 36 * np.pi / 180
            radius = size if i % 2 == 0 else size // 2
            x = center[0] + int(radius * np.cos(angle))
            y = center[1] + int(radius * np.sin(angle))
            points.append((x, y))
        
        cv2.fillPoly(img, [np.array(points, np.int32)], color)
    
    def _draw_triangle(self, img: np.ndarray, center: Tuple[int, int], 
                      size: int, color: Tuple[int, int, int]):
        """Draw triangle shape"""
        height = int(size * np.sqrt(3))
        points = [
            (center[0], center[1] - height//2),
            (center[0] - size, center[1] + height//2),
            (center[0] + size, center[1] + height//2)
        ]
        cv2.fillPoly(img, [np.array(points, np.int32)], color)
    
    def _debug_casualties(self, img: np.ndarray, casualties: List[Casualty]):
        """Debug: Show detected casualties"""
        debug_img = img.copy()
        for c in casualties:
            x, y, w, h = c.bbox
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{c.shape} {c.color} P:{c.total_priority}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(f"{self.OUTPUT_DIRS['debug']}/casualties_debug.png", debug_img)
    
    def _debug_camps(self, img: np.ndarray, camps: Dict[str, Tuple[int, int]]):
        """Debug: Show detected camps"""
        debug_img = img.copy()
        for camp_name, (cx, cy) in camps.items():
            cv2.circle(debug_img, (cx, cy), 30, (0, 0, 255), 3)
            cv2.putText(debug_img, camp_name, (cx-20, cy-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite(f"{self.OUTPUT_DIRS['debug']}/camps_debug.png", debug_img)
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Preprocess
            processed_img = self.preprocess_image(img)
            
            # Segment ocean and land
            segmented, overlay = self.segment_image(processed_img)
            
            # Detect casualties
            casualties = self.detect_casualties_optimized(processed_img)
            print(f"✓ Detected {len(casualties)} casualties")
            
            # Detect camps
            camps_dict = self.detect_camps_optimized(processed_img)
            print(f"✓ Detected {len(camps_dict)} camps: {list(camps_dict.keys())}")
            
            # Assign casualties
            camps = self.assign_casualties_priority_based(casualties, camps_dict)
            
            # Calculate statistics
            stats = self.calculate_statistics(camps)
            
            # Create output image
            output_img = self.create_output_image(img, overlay, casualties, camps)
            
            # Prepare results
            results = {
                'image_name': os.path.basename(image_path),
                'casualties': casualties,
                'camps': camps,
                'statistics': stats,
                'output_image': output_img,
                'segmented': segmented,
                'overlay': overlay
            }
            
            print(f"✓ Processing completed")
            print(f"  Rescue Ratio: {stats['average_priority']:.2f}")
            
            return results
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, results: Dict[str, Any]):
        """Save results for a single image"""
        if not results:
            return
        
        img_name = results['image_name']
        base_name = os.path.splitext(img_name)[0]
        
        # Save images
        cv2.imwrite(f"{self.OUTPUT_DIRS['images']}/{base_name}_output.png",
                   results['output_image'])
        cv2.imwrite(f"{self.OUTPUT_DIRS['images']}/{base_name}_segmented.png",
                   results['segmented'])
        
        # Save statistics
        stats = results['statistics']
        with open(f"{self.OUTPUT_DIRS['results']}/{base_name}_results.json", 'w') as f:
            json.dump({
                'image': img_name,
                'casualty_counts': stats['casualty_counts'],
                'casualty_details': stats['casualty_details'],
                'camp_priorities': stats['camp_priorities'],
                'total_priority': stats['total_priority'],
                'average_priority': stats['average_priority']
            }, f, indent=2)
        
        # Save text summary
        self._save_text_summary(results, base_name)
    
    def _save_text_summary(self, results: Dict[str, Any], base_name: str):
        """Save human-readable summary"""
        stats = results['statistics']
        
        with open(f"{self.OUTPUT_DIRS['results']}/{base_name}_summary.txt", 'w') as f:
            f.write(f"Image: {results['image_name']}\n")
            f.write(f"Casualties Detected: {len(results['casualties'])}\n")
            f.write(f"Camps Detected: {len(results['camps'])}\n\n")
            
            f.write("ASSIGNMENT RESULTS\n")
            f.write("-" * 50 + "\n\n")
            
            # Camp order: blue, pink, grey
            camp_names = ['blue', 'pink', 'grey']
            for i, camp_name in enumerate(camp_names):
                f.write(f"{camp_name.upper()} Camp:\n")
                f.write(f"  Casualties: {stats['casualty_counts'][i]}\n")
                f.write(f"  Total Priority: {stats['camp_priorities'][i]}\n")
                f.write(f"  Details: {stats['casualty_details'][i]}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Priority: {stats['total_priority']}\n")
            f.write(f"Average Priority (Rescue Ratio): {stats['average_priority']:.2f}\n")
    
    def process_batch(self, folder_path: str):
        """Process all images in folder"""
        print("\n" + "="*60)
        print("OPTIMIZED BATCH PROCESSING - UAS-DTU SAR")
        print("="*60)
        
        # Get all images
        image_paths = self.load_images(folder_path)
        
        if not image_paths:
            print("✗ No images found!")
            return
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        all_results = []
        
        # Process each image
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] ", end="")
            results = self.process_single_image(img_path)
            
            if results:
                self.save_results(results)
                all_results.append(results)
        
        # Generate final summary
        if all_results:
            self._generate_final_summary(all_results)
    
    def _generate_final_summary(self, all_results: List[Dict[str, Any]]):
        """Generate final summary sorted by rescue ratio"""
        # Sort by average priority (rescue ratio)
        sorted_results = sorted(all_results,
                              key=lambda x: x['statistics']['average_priority'],
                              reverse=True)
        
        # Save final summary
        with open(f"{self.OUTPUT_DIRS['base']}/final_summary.txt", 'w') as f:
            f.write("FINAL SUMMARY - Images Sorted by Rescue Ratio\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"{'Rank':<6} {'Image':<25} {'Rescue Ratio':<15} "
                   f"{'Casualties':<12} {'Total Priority':<15}\n")
            f.write("-"*70 + "\n")
            
            for i, result in enumerate(sorted_results, 1):
                stats = result['statistics']
                f.write(f"{i:<6} {result['image_name']:<25} "
                       f"{stats['average_priority']:<15.2f} "
                       f"{len(result['casualties']):<12} "
                       f"{stats['total_priority']:<15}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("DETAILED RESULTS BY CAMP ORDER [BLUE, PINK, GREY]\n")
            f.write("="*70 + "\n\n")
            
            for i, result in enumerate(sorted_results, 1):
                stats = result['statistics']
                f.write(f"\n{i}. {result['image_name']}\n")
                f.write(f"   Rescue Ratio: {stats['average_priority']:.2f}\n")
                f.write(f"   Casualty Counts: {stats['casualty_counts']}\n")
                f.write(f"   Camp Priorities: {stats['camp_priorities']}\n")
                f.write(f"   Casualty Details:\n")
                
                camp_names = ['blue', 'pink', 'grey']
                for j, camp in enumerate(camp_names):
                    f.write(f"     {camp}: {stats['casualty_details'][j]}\n")
        
        # Display summary
        print("\n" + "="*60)
        print("FINAL RANKING (by Rescue Ratio)")
        print("="*60)
        
        for i, result in enumerate(sorted_results, 1):
            stats = result['statistics']
            print(f"{i:2d}. {result['image_name']:20s} "
                  f"Ratio: {stats['average_priority']:6.2f} "
                  f"Casualties: {len(result['casualties']):2d}")
        
        print("\n✓ All results saved in 'optimized_output/' folder")


def main():
    """Main function"""
    # Initialize processor
    processor = OptimizedSARProcessor(debug=True)
    
    # Your image folder path
    folder_path = r"C:\Users\user\Documents\uas_dtu_project\images\task images"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return
    
    # Process all images
    processor.process_batch(folder_path)


if __name__ == "__main__":
    main()