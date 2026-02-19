# src/utils/icon_manager.py
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class IconManager:
    def __init__(self, icons_path="assets/icons", icon_size=(50, 50)):
        """Load and manage detection icons"""
        if icons_path is None:
            
            here = os.path.dirname(os.path.abspath(__file__))
            icons_path = os.path.join(here, "..", "..", "static", "assets", "icons")
            icons_path = os.path.abspath(icons_path)
            
        self.icons_path = icons_path
        self.icons = {}
        self.icon_size = icon_size
        
        self.class_to_icon = {
            'person': 'person.png',
            'car': 'car.png',
            'truck': 'truck.png',
            'bus': 'truck.png',
            'cow': 'cow.png',
            'horse': 'cow.png',
            'dog': 'deer.png',
            'deer': 'deer.png',
            'default': 'hazard.png'
        }
        
        unique_icons = set(self.class_to_icon.values())
        for filename in unique_icons:
            path = os.path.join(icons_path, filename)
            if os.path.exists(path):
                icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon = cv2.resize(icon, icon_size)
                    self.icons[filename] = icon
    
    def get_icon(self, class_name):
        icon_file = self.class_to_icon.get(class_name, self.class_to_icon['default'])
        return self.icons.get(icon_file)
    
    def draw_detection_with_icon(self, frame, bbox, class_name, confidence, color=(0, 0, 255)):
        """Draw SEXY Ultralytics-style detection"""
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        
        # ═══ 1. THIN BOUNDING BOX (2px, not 3+) ═══
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # ═══ 2. ICON BADGE (Circle + Icon, NO TEXT) ═══
        icon = self.get_icon(class_name)
        if icon is not None:
            icon_h, icon_w = icon.shape[:2]
            
            # Position above bbox
            badge_y = max(35, y1 - 60)  # Space for circle
            badge_x = center_x
            
            # Draw RED CIRCLE background (like Ultralytics)
            circle_radius = 28
            cv2.circle(frame, (badge_x, badge_y), circle_radius, color, -1)  # Filled
            cv2.circle(frame, (badge_x, badge_y), circle_radius, (255, 255, 255), 2)  # White border
            
            # Place icon in center of circle
            icon_x = badge_x - icon_w // 2
            icon_y = badge_y - icon_h // 2
            
            # Ensure in bounds
            icon_x = max(0, min(icon_x, frame.shape[1] - icon_w))
            icon_y = max(0, min(icon_y, frame.shape[0] - icon_h))
            
            # Overlay icon with transparency
            self._overlay_icon(frame, icon, icon_x, icon_y)
            
            # Thin connector line
            cv2.line(frame, (center_x, badge_y + circle_radius), 
                    (center_x, y1), color, 2)
        
       # ═══ 3. SEXY ARIAL LABEL (Replaces cv2.putText) ═══
        if confidence < 0.6:
            label = f"{class_name.upper()}"
            
            # 1. Convert OpenCV frame (BGR) to PIL Image (RGB)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 2. Load Arial (Adjust path if on Linux/Mac)
            # Use 14-18 for that "Ultralytics Nano" look
            try:
                font = ImageFont.truetype("arial.ttf", 16) 
            except:
                font = ImageFont.load_default() # Fallback if arial.ttf isn't found
            
            # 3. Calculate text size for the background "pill"
            bbox_text = draw.textbbox((0, 0), label, font=font)
            label_w = bbox_text[2] - bbox_text[0]
            label_h = bbox_text[3] - bbox_text[1]
            
            label_x = x1
            label_y = max(15, y1 - 12)
            
            # 4. Draw the background pill (using CV2 for speed or PIL for rounded)
            # Let's stay in PIL for the whole label for consistency
            draw.rectangle(
                [label_x, label_y - label_h - 6, label_x + label_w + 10, label_y + 4],
                fill=tuple(reversed(color)) # Convert BGR to RGB for PIL
            )
            
            # 5. Draw the smooth Arial text
            draw.text((label_x + 5, label_y - label_h - 4), label, font=font, fill=(255, 255, 255))
            
            # 6. Convert back to OpenCV BGR
            frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        return frame
    
    def _overlay_icon(self, background, icon, x, y):
        """Overlay icon with alpha transparency"""
        h, w = icon.shape[:2]
        
        if x + w > background.shape[1] or y + h > background.shape[0]:
            return
        if x < 0 or y < 0:
            return
        
        if icon.shape[2] == 4:  # Has alpha
            alpha = icon[:, :, 3] / 255.0
            icon_rgb = icon[:, :, :3]
            
            # Convert icon to white (for visibility on red circle)
            icon_white = np.ones_like(icon_rgb) * 255
            icon_white = (icon_white * 0.9).astype(np.uint8)  # Slightly off-white
            
            roi = background[y:y+h, x:x+w]
            for c in range(3):
                roi[:, :, c] = (alpha * icon_white[:, :, c] + 
                               (1 - alpha) * roi[:, :, c])
        else:
            background[y:y+h, x:x+w] = icon[:, :, :3]