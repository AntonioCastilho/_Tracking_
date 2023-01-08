import cv2
import mediapipe as mp
import time
import PoseEstimationModule as pe


pTime = 0
cTime = 0
cap = cv2.VideoCapture(dance_.mp)
detector = pe.poseDetector()
while True:
    ret, img_BGR = cap.read()
    img_BGR = cv2.flip(img_BGR, 1)
    img = detector.findPose(img_BGR)
    lmList = detector.getPosition(img_BGR, draw=False)
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()  # current time
    fps = 1/(cTime-pTime)  # inverse of the period = frequency
    pTime = cTime

    cv2.putText(img_BGR, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow("that's a hand", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
