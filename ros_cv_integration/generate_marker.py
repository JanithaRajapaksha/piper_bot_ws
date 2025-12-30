import cv2
import numpy as np

def main():
    # 1. Define the dictionary (Must match the one in test_aruco.py)
    # Your node uses DICT_ARUCO_ORIGINAL
    dictionary_id = cv2.aruco.DICT_ARUCO_ORIGINAL
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    # 2. Configuration
    marker_id = 0       # The ID you want to generate (e.g., 0)
    marker_size = 500   # Resolution in pixels (e.g., 500x500)
    border_size = 100   # White padding size in pixels

    print(f"Generating ArUco Marker ID: {marker_id} (DICT_ARUCO_ORIGINAL)")

    # 3. Generate the marker image
    # generateImageMarker(dictionary, id, sidePixels, borderBits)
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

    # 4. Add a white border (Crucial for detection against dark backgrounds)
    marker_with_border = cv2.copyMakeBorder(
        marker_image, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )

    # 5. Save and Show
    filename = f"aruco_original_id_{marker_id}.png"
    cv2.imwrite(filename, marker_with_border)
    print(f"Saved marker to: {filename}")

    cv2.imshow("Generated Marker", marker_with_border)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()