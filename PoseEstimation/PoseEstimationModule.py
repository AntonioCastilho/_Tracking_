import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self,
                 mode=False,
                 complexity=1,
                 smooth=True,
                 segmentation=True,
                 detectionCon=0.5,
                 trackingCon=0.5
                 ):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.complexity,
            self.smooth,
            self.segmentation,
            self.detectionCon,
            self.trackingCon
        )

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(r"PoseEstimation\dance_.mp4")
    # cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second: {0}".format(fps))
    pTime = 0
    detector = poseDetector()
    while True:
        ret, img = cap.read()
        # img = cv2.flip(img, 1)
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        print(lmList)
        cv2.circle(img, (lmList[22] [1], lmList[22][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        tTime = (cTime - pTime)
        # if tTime != 0:
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, "fps: "+str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 180, 0), 3)
        cv2.imshow("Video", img)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
