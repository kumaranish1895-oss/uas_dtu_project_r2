import cv2
import numpy as np
import os
import json
from glob import glob
import matplotlib.pyplot as plt
from image_processor import ImageProcessor
from casualty_assigner import CasualtyAssigner
from utils import calculate_priority_ratio, sort_images_by_rescue_ratio

def process_all_images(input_dir="images", output_dir="outputs"):
    """
    Process all images in the input directory
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Get all image files
    image_files = sorted(glob(os.path.join(input_dir, "*.jpg")) + 
                        glob(os.path.join(input_dir, "*.png")) + 
                        glob(os.path.join(input_dir, "*.jpeg")))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return []
    
    results = []
    
    for img_path in image_files:
        print(f"Processing {img_path}...")
        
        # Extract image name
        img_name = os.path.basename(img_path).split('.')[0]
        
        # Process image
        processor = ImageProcessor(img_path)
        
        # 1. Segment ocean and land
        segmented_img, ocean_mask, land_mask = processor.segment_ocean_land()
        
        # 2. Detect casualties
        casualties = processor.detect_casualties()
        
        # 3. Detect rescue pads
        rescue_pads = processor.detect_rescue_pads()
        
        # 4. Assign casualties to camps
        assigner = CasualtyAssigner(casualties, rescue_pads, ocean_mask, land_mask)
        assignments, camp_stats = assigner.assign_casualties()
        
        # 5. Create visualization
        visualization = processor.create_visualization(
            segmented_img, casualties, rescue_pads, assignments
        )
        
        # 6. Save outputs
        # Save segmented image
        seg_output_path = os.path.join(output_dir, f"{img_name}_segmented.jpg")
        cv2.imwrite(seg_output_path, segmented_img)
        
        # Save visualization
        vis_output_path = os.path.join(output_dir, f"{img_name}_assigned.jpg")
        cv2.imwrite(vis_output_path, visualization)
        
        # Calculate priority ratio
        total_priority = sum(camp_stats[pad_id]['total_priority'] 
                           for pad_id in rescue_pads)
        total_casualties = sum(camp_stats[pad_id]['count'] 
                             for pad_id in rescue_pads)
        priority_ratio = total_priority / total_casualties if total_casualties > 0 else 0
        
        # Store results
        result = {
            'image_name': img_name,
            'casualties_count': len(casualties),
            'camp_assignments': {},
            'camp_priorities': {},
            'priority_ratio': priority_ratio
        }
        
        for pad_id, pad_info in rescue_pads.items():
            if pad_id in camp_stats:
                result['camp_assignments'][pad_id] = camp_stats[pad_id]['casualties']
                result['camp_priorities'][pad_id] = camp_stats[pad_id]['total_priority']
        
        results.append(result)
        
        # Print current results
        print(f"  Priority Ratio: {priority_ratio:.2f}")
        print(f"  Casualties: {len(casualties)}")
        for pad_id in rescue_pads:
            count = camp_stats[pad_id]['count'] if pad_id in camp_stats else 0
            print(f"  Camp {pad_id}: {count} casualties")
    
    # Save all results to JSON
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Sort images by rescue ratio
    sorted_images = sort_images_by_rescue_ratio(results)
    
    return results, sorted_images

def print_summary(results, sorted_images):
    """
    Print formatted summary of results
    """
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("\nImages sorted by rescue ratio (descending):")
    for i, img_info in enumerate(sorted_images, 1):
        print(f"{i}. {img_info['image_name']} - Priority Ratio: {img_info['priority_ratio']:.2f}")
    
    print("\n" + "="*60)
    print("DETAILED RESULTS FOR EACH IMAGE")
    print("="*60)
    
    for result in results:
        print(f"\nImage: {result['image_name']}")
        print(f"Total Casualties: {result['casualties_count']}")
        print(f"Priority Ratio: {result['priority_ratio']:.2f}")
        
        for camp_id in ['blue', 'pink', 'grey']:
            if camp_id in result['camp_assignments']:
                casualties = result['camp_assignments'][camp_id]
                priority = result['camp_priorities'][camp_id]
                print(f"\n  Camp {camp_id.upper()}:")
                print(f"    Total Priority: {priority}")
                print(f"    Casualty Count: {len(casualties)}")
                print(f"    Casualties: {casualties}")
            else:
                print(f"\n  Camp {camp_id.upper()}: No casualties assigned")

if __name__ == "__main__":
    # Process all images
    results, sorted_images = process_all_images()
    
    # Print summary
    print_summary(results, sorted_images)
    
    print("\n" + "="*60)
    print("OUTPUT FORMAT (for submission)")
    print("="*60)
    
    # Format output as requested
    print("\n1. Camp assignments for each image:")
    for result in results:
        assignments = []
        for camp_id in ['blue', 'pink', 'grey']:
            if camp_id in result['camp_assignments']:
                camp_casualties = []
                for casualty in result['camp_assignments'][camp_id]:
                    # Format: [shape_priority, emergency_priority]
                    camp_casualties.append([casualty['shape_priority'], casualty['emergency_priority']])
                assignments.append(camp_casualties)
            else:
                assignments.append([])
        print(f"{result['image_name']}: {assignments}")
    
    print("\n2. Camp priorities for each image:")
    for result in results:
        priorities = []
        for camp_id in ['blue', 'pink', 'grey']:
            if camp_id in result['camp_priorities']:
                priorities.append(result['camp_priorities'][camp_id])
            else:
                priorities.append(0)
        print(f"{result['image_name']}: {priorities}")
    
    print("\n3. Priority ratios for each image:")
    for result in results:
        print(f"{result['image_name']}: {result['priority_ratio']:.2f}")
    
    print("\n4. Images sorted by rescue ratio (descending):")
    sorted_names = [img['image_name'] for img in sorted_images]
    print(sorted_names)