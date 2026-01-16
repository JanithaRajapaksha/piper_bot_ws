import cv2

# --- Get video feed from camera ---
# The number is the camera index, starting from 0 for the default camera
cap = cv2.VideoCapture(2)

# Verify if the camera is opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Crop the frame to only show the left camera
    width = frame.shape[1]
    left_frame = frame[:, :width // 2]

    # Display the resulting frame
    cv2.imshow('frame', left_frame)
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
