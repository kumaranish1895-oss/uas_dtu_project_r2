import cv2
import numpy as np
from scipy.spatial import distance

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.image.shape[:2]
    
    def segment_ocean_land(self):
        """
        Segment ocean (blue) and land (brown/green) regions
        Returns segmented image with ocean as blue and land as green
        """
        # Define color ranges for ocean (blue) and land (brown/green)
        # Ocean color range (blue)
        lower_ocean = np.array([90, 50, 50])  # Lower bound for blue
        upper_ocean = np.array([130, 255, 255])  # Upper bound for blue
        
        # Land color range (brown/green)
        lower_land1 = np.array([10, 50, 50])  # Lower bound for green
        upper_land1 = np.array([40, 255, 255])  # Upper bound for green
        lower_land2 = np.array([0, 50, 20])  # Lower bound for brown
        upper_land2 = np.array([20, 255, 200])  # Upper bound for brown
        
        # Create masks
        ocean_mask = cv2.inRange(self.image_hsv, lower_ocean, upper_ocean)
        
        land_mask1 = cv2.inRange(self.image_hsv, lower_land1, upper_land1)
        land_mask2 = cv2.inRange(self.image_hsv, lower_land2, upper_land2)
        land_mask = cv2.bitwise_or(land_mask1, land_mask2)
        
        # Apply morphological operations to clean masks
        kernel = np.ones((5, 5), np.uint8)
        ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel)
        ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_OPEN, kernel)
        land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)
        land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, kernel)
        
        # Create segmented image
        segmented = self.image.copy()
        
        # Color ocean as bright blue (BGR)
        segmented[ocean_mask > 0] = [255, 0, 0]  # Blue in BGR
        
        # Color land as bright green (BGR)
        segmented[land_mask > 0] = [0, 255, 0]  # Green in BGR
        
        return segmented, ocean_mask, land_mask
    
    def detect_casualties(self):
        """
        Detect casualties (stars, squares, triangles) with colors
        Returns list of casualty dictionaries
        """
        casualties = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find shapes
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define color ranges for emergency levels
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),  # Severe
            'yellow': ([20, 50, 50], [40, 255, 255]),  # Mild
            'green': ([40, 50, 50], [80, 255, 255])  # Safe
        }
        
        emergency_priority = {'red': 3, 'yellow': 2, 'green': 1}
        shape_priority = {'star': 3, 'triangle': 2, 'square': 1}
        
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < 100 or area > 5000:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Get shape type based on number of vertices
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            if vertices == 3:
                shape = 'triangle'
            elif vertices == 4:
                shape = 'square'
            elif vertices >= 8:  # Star has many vertices
                shape = 'star'
            else:
                continue
            
            # Get dominant color in the contour region
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Get average color in BGR
            mean_color = cv2.mean(self.image, mask=mask)[:3]
            
            # Convert to HSV for color classification
            mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Determine emergency color
            emergency = 'green'  # Default
            max_match = 0
            
            for color_name, (lower, upper) in color_ranges.items():
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                if np.all(mean_color_hsv >= lower_np) and np.all(mean_color_hsv <= upper_np):
                    emergency = color_name
                    break
            
            # Create casualty dictionary
            casualty = {
                'id': len(casualties),
                'center': (center_x, center_y),
                'shape': shape,
                'emergency': emergency,
                'shape_priority': shape_priority[shape],
                'emergency_priority': emergency_priority[emergency],
                'priority_score': shape_priority[shape] * emergency_priority[emergency],
                'contour': contour
            }
            
            casualties.append(casualty)
        
        return casualties
    
    def detect_rescue_pads(self):
        """
        Detect rescue pads (circles)
        Returns dictionary of rescue pads with their capacities
        """
        rescue_pads = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Define camp capacities
            capacities = {'blue': 4, 'pink': 3, 'grey': 2}
            colors = ['blue', 'pink', 'grey']
            
            for i, circle in enumerate(circles[0, :]):
                if i < 3:  # Only first 3 circles (we expect exactly 3)
                    x, y, r = circle
                    
                    # Get color at center to identify camp type
                    center_color_bgr = self.image[y, x]
                    center_color_hsv = cv2.cvtColor(np.uint8([[center_color_bgr]]), 
                                                   cv2.COLOR_BGR2HSV)[0][0]
                    
                    # Simple color matching (can be improved)
                    pad_id = colors[i] if i < len(colors) else f'pad_{i}'
                    
                    rescue_pads[pad_id] = {
                        'center': (x, y),
                        'radius': r,
                        'capacity': capacities.get(pad_id, 2),
                        'assigned': []
                    }
        
        return rescue_pads
    
    def create_visualization(self, segmented_img, casualties, rescue_pads, assignments):
        """
        Create visualization with assignments
        """
        vis_img = segmented_img.copy()
        
        # Draw casualties with colors based on emergency
        emergency_colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
        
        for casualty in casualties:
            color = emergency_colors.get(casualty['emergency'], (255, 255, 255))
            center = tuple(map(int, casualty['center']))
            cv2.drawContours(vis_img, [casualty['contour']], -1, color, 3)
            cv2.circle(vis_img, center, 5, color, -1)
            
            # Add label
            label = f"{casualty['shape'][0]}:{casualty['emergency'][0]}"
            cv2.putText(vis_img, label, (center[0]+10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw rescue pads
        camp_colors = {'blue': (255, 0, 0), 'pink': (180, 105, 255), 'grey': (128, 128, 128)}
        
        for pad_id, pad_info in rescue_pads.items():
            color = camp_colors.get(pad_id, (255, 255, 255))
            center = tuple(map(int, pad_info['center']))
            radius = int(pad_info['radius'])
            
            cv2.circle(vis_img, center, radius, color, 3)
            cv2.circle(vis_img, center, 5, color, -1)
            
            # Add label
            label = f"{pad_id} (cap:{pad_info['capacity']})"
            cv2.putText(vis_img, label, (center[0]-30, center[1]-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw assignment lines
        for pad_id, assigned_casualties in assignments.items():
            if pad_id in rescue_pads:
                pad_center = rescue_pads[pad_id]['center']
                for casualty in assigned_casualties:
                    casualty_center = tuple(map(int, casualty['center']))
                    color = camp_colors.get(pad_id, (255, 255, 255))
                    cv2.line(vis_img, pad_center, casualty_center, color, 2)
        
        return vis_img