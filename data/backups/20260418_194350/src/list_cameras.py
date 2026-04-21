"""Utility to list available cameras and their capabilities."""

import cv2

def list_cameras(max_devices: int = 10) -> None:
    """Enumerate available video capture devices."""
    print("Available cameras on this system:\n")
    
    found_any = False
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found_any = True
            # Try to get camera name/info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            print(f"  Camera {idx}: {width}x{height} @ {fps}fps")
            cap.release()
    
    if not found_any:
        print("  No cameras detected!")
        return
    
    print("\n💡 Usage: Pass --camera-index to select a camera:")
    print("   python src/collect_data.py --label hamburger --camera-index 1")
    print("   python src/app.py --camera-index 1")

if __name__ == "__main__":
    list_cameras()
