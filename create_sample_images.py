import cv2
import numpy as np
import os

def create_sample_image(image_num, num_casualties=10):
    """
    Create a sample test image with ocean, land, casualties, and rescue pads
    """
    # Create blank image
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    # Create ocean (blue area) - top half
    cv2.rectangle(img, (0, 0), (800, 300), (200, 100, 0), -1)  # Blue in BGR

    # Create land (brown/green area) - bottom half
    cv2.rectangle(img, (0, 300), (800, 600), (0, 100, 50), -1)  # Green in BGR

    # Add some texture
    for i in range(0, 800, 20):
        cv2.line(img, (i, 0), (i, 300), (220, 120, 0), 1)
        cv2.line(img, (i, 300), (i, 600), (0, 120, 60), 1)

    # Create rescue pads
    # Blue camp in water
    cv2.circle(img, (200, 150), 40, (255, 0, 0), -1)  # Blue
    cv2.putText(img, "Blue", (170, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Pink camp on land
    cv2.circle(img, (400, 450), 35, (180, 105, 255), -1)  # Pink
    cv2.putText(img, "Pink", (370, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Grey camp on land
    cv2.circle(img, (600, 450), 30, (128, 128, 128), -1)  # Grey
    cv2.putText(img, "Grey", (570, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Create casualties with different shapes and colors
    shapes = ['triangle', 'square', 'star']
    colors = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}
    emergencies = ['red', 'yellow', 'green']
    
    np.random.seed(image_num)  # For reproducibility
    
    for i in range(num_casualties):
        # Random position
        if i % 3 == 0:
            # Some in water
            x = np.random.randint(50, 750)
            y = np.random.randint(50, 250)
        else:
            # Most on land
            x = np.random.randint(50, 750)
            y = np.random.randint(350, 550)
        
        # Random shape and emergency
        shape = shapes[np.random.randint(0, 3)]
        emergency = emergencies[np.random.randint(0, 3)]
        color = colors[emergency]
        
        # Draw shape
        if shape == 'triangle':
            pts = np.array([[x, y-20], [x-15, y+10], [x+15, y+10]], np.int32)
            cv2.drawContours(img, [pts], 0, color, -1)
        elif shape == 'square':
            cv2.rectangle(img, (x-15, y-15), (x+15, y+15), color, -1)
        else:  # star
            # Simple star approximation
            pts = []
            for j in range(5):
                angle = np.pi/2 + j*2*np.pi/5
                pts.append([x + 20*np.cos(angle), y + 20*np.sin(angle)])
                angle += np.pi/5
                pts.append([x + 10*np.cos(angle), y + 10*np.sin(angle)])
            
            pts = np.array(pts, np.int32)
            cv2.drawContours(img, [pts], 0, color, -1)
    
    # Save image
    os.makedirs("images", exist_ok=True)
    filename = f"images/image{image_num}.jpg"
    cv2.imwrite(filename, img)
    print(f"Created {filename}")
    
    return img

if __name__ == "__main__":
    # Create 10 sample images
    for i in range(1, 11):
        create_sample_image(i, num_casualties=np.random.randint(8, 15))
    
    print("\nSample images created in 'images/' folder")