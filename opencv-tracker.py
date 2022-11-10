import cv2

tracker = cv2.legacy.TrackerKCF_create()

# Get the video file and read it
video = cv2.VideoCapture("videos/video1.mp4")
ret, frame = video.read()
print(f'read successful?: {ret}')

frame_height, frame_width = frame.shape[:2]

bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)
# Start tracking
while True:
    ret, frame = video.read()
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    cv2.putText(frame, "MOSSE Tracker", (100,20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
        
video.release()
cv2.destroyAllWindows()