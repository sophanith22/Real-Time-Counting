# test_rtsp_connection.py

import cv2

RTSP_URL = "rtsp://admin:Hik123456@192.168.88.119:554/ch1/main/av_stream"  # Replace with RTSP URL, username, and password if needed

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Could not connect to the camera. Check the RTSP URL, username, password, and network connection.")
else:
    print("Connected successfully!")

    while True:
        # Skip a couple extra buffered frames each loop, so we always
        # grab the LATEST frame, not old ones stuck in the queue.
        # This keeps delay from building up over time.
        for _ in range(2):
            cap.grab()

        ret, frame = cap.retrieve()
        if not ret:
            print("Lost connection or stream ended.")
            break




        frame = cv2.resize(frame, (1280,720))  # Resize to 980x720
        cv2.imshow("RTSP Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()