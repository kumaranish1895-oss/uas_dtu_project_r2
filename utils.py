import numpy as np
import json

def calculate_priority_ratio(camp_stats):
    """
    Calculate priority ratio for an image
    """
    total_priority = sum(stats['total_priority'] for stats in camp_stats.values())
    total_casualties = sum(stats['count'] for stats in camp_stats.values())
    
    if total_casualties == 0:
        return 0
    
    return total_priority / total_casualties

def sort_images_by_rescue_ratio(results):
    """
    Sort images by rescue ratio (priority ratio) in descending order
    """
    return sorted(results, key=lambda x: x['priority_ratio'], reverse=True)

def save_results_to_csv(results, filename='results/summary.csv'):
    """
    Save results to CSV file
    """
    import pandas as pd
    
    data = []
    for result in results:
        row = {
            'image_name': result['image_name'],
            'total_casualties': result['casualties_count'],
            'priority_ratio': result['priority_ratio']
        }
        
        for camp_id in ['blue', 'pink', 'grey']:
            if camp_id in result['camp_assignments']:
                row[f'{camp_id}_count'] = len(result['camp_assignments'][camp_id])
                row[f'{camp_id}_priority'] = result['camp_priorities'][camp_id]
            else:
                row[f'{camp_id}_count'] = 0
                row[f'{camp_id}_priority'] = 0
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df