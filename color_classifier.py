import cv2
import numpy as np

# Farveområder i HSV (Terræner)
COLOR_RANGES = {
    "grass": [(35, 50, 50), (85, 255, 255)],       
    "wheat": [(20, 50, 50), (35, 255, 255)],      
    "water": [(90, 50, 50), (130, 255, 255)],     
    "forest": [(25, 40, 40), (40, 255, 120)],     
    "swamp": [(5, 40, 40), (20, 255, 100)],      
    "mine": [(0, 0, 0), (180, 255, 50)]          
}

def classify_color(hsv_pixel):
    """ Klassificér en HSV-pixel baseret på farveområder """
    hsv_pixel = np.array(hsv_pixel, dtype=np.uint8).reshape(1, 1, 3)
    for terrain, (low, high) in COLOR_RANGES.items():
        if cv2.inRange(hsv_pixel, np.array(low, np.uint8), np.array(high, np.uint8))[0, 0]:
            return terrain
    return "unknown"

# Test alle terræntyper med midtpunktet i deres farveområder
print("\n🔍 Test af terræntyper:\n")
for terrain, (low, high) in COLOR_RANGES.items():
    hsv_value = tuple((np.array(low) + np.array(high)) // 2)  # Midtpunkt
    result = classify_color(hsv_value)
    print(f"Forventet: {terrain}, Klassificeret: {result}, HSV: {hsv_value}, {'✅' if result == terrain else '❌'}")



