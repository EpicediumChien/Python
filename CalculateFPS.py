import cv2

video_path = r'C:\WIN_WB5023.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get expected FPS from metadata
expected_fps = cap.get(cv2.CAP_PROP_FPS)
if expected_fps <= 0:
    expected_fps = 30

speedup_factor = 3
frame_idx = 0

while True:
    # Skip frames to speed up
    for _ in range(speedup_factor - 1):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    frame_idx += speedup_factor

    # Get current video time (seconds)
    current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    current_sec = current_msec / 1000.0

    # Calculate actual FPS
    actual_fps = frame_idx / current_sec if current_sec > 0 else 0

    # Overlay expected vs. actual FPS
    cv2.putText(frame, f"Expected FPS: {expected_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Actual FPS: {actual_fps:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Time: {current_sec:.2f}s", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Print to console
    print(f"Expected FPS: {expected_fps:.2f}, Actual FPS: {actual_fps:.2f}, Time: {current_sec:.2f}s")

    # Show frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
