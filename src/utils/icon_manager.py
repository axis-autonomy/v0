import cv2
import os
import numpy as np

class IconManager:
    def __init__(self, icons_path="assets/icons", icon_size=(50, 50)):
        """Load and manage detection icons"""
        self.icons = {}
        self.icon_size = icon_size
        
        # Map YOLO/COCO classes to icon files
        self.class_to_icon = {
            # People
            'person': 'person.png',
            
            # Vehicles
            'car': 'car.png',
            'truck': 'truck.png',
            'bus': 'truck.png',
            'motorcycle': 'car.png',
            'bicycle': 'person.png',
            
            # Livestock
            'cow': 'cow.png',
            'horse': 'cow.png',
            'sheep': 'cow.png',
            
            # Wildlife
            'dog': 'deer.png',
            'cat': 'deer.png',
            'bear': 'deer.png',
            'deer': 'deer.png',
            
            # Default fallback
            'default': 'hazard.png'
        }
        
        # Load all unique icons
        unique_icons = set(self.class_to_icon.values())
        for filename in unique_icons:
            path = os.path.join(icons_path, filename)
            if os.path.exists(path):
                icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon = cv2.resize(icon, icon_size)
                    self.icons[filename] = icon
                    print(f"✅ Loaded: {filename}")
    
    def get_icon(self, class_name):
        """Get icon for a detection class"""
        icon_file = self.class_to_icon.get(class_name, self.class_to_icon['default'])
        return self.icons.get(icon_file)
    
    def draw_detection_with_icon(self, frame, bbox, class_name, confidence, color=(255, 50, 50)):
        """Draw bounding box + icon badge for detection"""
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        
        # 1. Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 2. Get and draw icon
        icon = self.get_icon(class_name)
        if icon is not None:
            icon_h, icon_w = icon.shape[:2]
            
            # Position above bbox
            icon_x = center_x - icon_w // 2
            icon_y = y1 - icon_h - 15
            
            # Keep in frame
            icon_x = max(0, min(icon_x, frame.shape[1] - icon_w))
            icon_y = max(10, icon_y)
            
            # White circle background
            circle_center = (center_x, icon_y + icon_h // 2)
            cv2.circle(frame, circle_center, 30, (255, 255, 255), -1)
            cv2.circle(frame, circle_center, 30, color, 3)
            
            # Overlay icon
            self._overlay_icon(frame, icon, icon_x, icon_y)
            
            # Connector line
            cv2.line(frame, (center_x, icon_y + icon_h), (center_x, y1), color, 2)
        
        # 3. Draw label
        label = f"{class_name.upper()}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(35, y1 - 5)
        
        # Label background
        cv2.rectangle(frame, 
                     (x1, label_y - label_size[1] - 10),
                     (x1 + label_size[0] + 10, label_y),
                     color, -1)
        
        # Label text
        cv2.putText(frame, label, (x1 + 5, label_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _overlay_icon(self, background, icon, x, y):
        """Overlay icon with transparency"""
        h, w = icon.shape[:2]
        
        # Bounds check
        if x + w > background.shape[1] or y + h > background.shape[0]:
            return
        if x < 0 or y < 0:
            return
        
        if icon.shape[2] == 4:  # Has alpha
            alpha = icon[:, :, 3] / 255.0
            icon_rgb = icon[:, :, :3]
            
            roi = background[y:y+h, x:x+w]
            for c in range(3):
                roi[:, :, c] = (alpha * icon_rgb[:, :, c] + 
                               (1 - alpha) * roi[:, :, c])
        else:
            background[y:y+h, x:x+w] = icon[:, :, :3]
