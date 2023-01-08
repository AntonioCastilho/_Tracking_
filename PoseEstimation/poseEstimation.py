import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpPose = mp.solutions.pose
pose = mpPose.Pose()


cap = cv2.VideoCapture(r"PoseEstimation\dance_.mp4")
# cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second: {0}".format(fps))

pTime = 0
while True:
    ret, img = cap.read()
    # img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mpPose.POSE_CONNECTIONS, 
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c=img.shape
            # print(id, lm)
            cx, cy=int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime=time.time()
    tTime=(cTime - pTime)
    # if tTime != 0:
    fps=1/(cTime - pTime)
    pTime=cTime

    cv2.putText(img, "fps: "+str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 180, 0), 3)
    cv2.imshow("Video", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
