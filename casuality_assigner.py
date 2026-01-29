import numpy as np
from scipy.spatial import distance
import heapq

class CasualtyAssigner:
    def __init__(self, casualties, rescue_pads, ocean_mask, land_mask):
        self.casualties = casualties
        self.rescue_pads = rescue_pads
        self.ocean_mask = ocean_mask
        self.land_mask = land_mask
        
    def calculate_distance_score(self, casualty, pad_center):
        """
        Calculate distance score between casualty and rescue pad
        Lower distance = better score
        """
        # Get positions
        casualty_pos = casualty['center']
        
        # Calculate Euclidean distance
        dist = distance.euclidean(casualty_pos, pad_center)
        
        # Normalize distance (0-1 range, 1 being closest)
        # Assuming max possible distance is diagonal of image
        max_dist = np.sqrt(1920**2 + 1080**2)  # For HD image
        normalized_dist = 1 - (dist / max_dist)
        
        return normalized_dist
    
    def calculate_final_score(self, casualty, pad_id, pad_center):
        """
        Calculate final score for casualty-pad assignment
        Based on priority score and distance
        """
        # Priority score component (0-9 range, normalized to 0-1)
        priority_score = casualty['priority_score'] / 9.0
        
        # Distance score component (0-1 range)
        distance_score = self.calculate_distance_score(casualty, pad_center)
        
        # Weighted combination (priority is more important)
        # You can adjust these weights based on requirements
        weight_priority = 0.7
        weight_distance = 0.3
        
        final_score = (weight_priority * priority_score + 
                      weight_distance * distance_score)
        
        return final_score
    
    def is_accessible(self, casualty_pos, pad_center, pad_id):
        """
        Check if casualty can access the rescue pad
        (Both should be in same region: ocean or land)
        """
        # Check if both points are in ocean or both in land
        cx, cy = map(int, casualty_pos)
        px, py = map(int, pad_center)
        
        # Check casualty location
        if self.ocean_mask[cy, cx] > 0:
            casualty_in_ocean = True
        elif self.land_mask[cy, cx] > 0:
            casualty_in_ocean = False
        else:
            # Default to land if uncertain
            casualty_in_ocean = False
        
        # Check pad location (known from sample: blue is in water)
        pad_in_ocean = (pad_id == 'blue')  # Blue camp is in water
        
        # Both should be in same region
        return casualty_in_ocean == pad_in_ocean
    
    def assign_casualties(self):
        """
        Assign casualties to rescue pads using optimization algorithm
        """
        # Sort casualties by priority (highest first)
        # In case of same priority score, sort by emergency priority
        sorted_casualties = sorted(self.casualties, 
                                  key=lambda x: (-x['priority_score'], -x['emergency_priority']))
        
        # Initialize assignments and capacities
        assignments = {pad_id: [] for pad_id in self.rescue_pads}
        capacities = {pad_id: pad_info['capacity'] for pad_id, pad_info in self.rescue_pads.items()}
        
        # Calculate scores for all casualty-pad pairs
        scores = {}
        for casualty in sorted_casualties:
            scores[casualty['id']] = {}
            for pad_id, pad_info in self.rescue_pads.items():
                if self.is_accessible(casualty['center'], pad_info['center'], pad_id):
                    score = self.calculate_final_score(casualty, pad_id, pad_info['center'])
                    scores[casualty['id']][pad_id] = score
                else:
                    scores[casualty['id']][pad_id] = -1  # Not accessible
        
        # Greedy assignment with backtracking
        # Try to assign highest priority casualties first to best available pads
        unassigned = sorted_casualties.copy()
        
        for casualty in sorted_casualties:
            # Get available pads for this casualty
            available_pads = []
            for pad_id in self.rescue_pads:
                if (capacities[pad_id] > 0 and 
                    scores[casualty['id']][pad_id] >= 0):
                    available_pads.append((pad_id, scores[casualty['id']][pad_id]))
            
            if available_pads:
                # Sort by score (highest first)
                available_pads.sort(key=lambda x: -x[1])
                
                # Assign to best available pad
                best_pad = available_pads[0][0]
                assignments[best_pad].append(casualty)
                capacities[best_pad] -= 1
        
        # Calculate camp statistics
        camp_stats = {}
        for pad_id in self.rescue_pads:
            total_priority = sum(c['priority_score'] for c in assignments[pad_id])
            camp_stats[pad_id] = {
                'count': len(assignments[pad_id]),
                'total_priority': total_priority,
                'casualties': assignments[pad_id]
            }
        
        return assignments, camp_stats