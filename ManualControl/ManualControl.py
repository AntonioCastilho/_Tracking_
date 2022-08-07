from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import cv2
import mediapipe as mp
import time
import HandTrackingModulo as htm
import math
import numpy as np

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        lenght = math.hypot((x2 - x1), (y2 - y1))
        # print(lenght)
        # range index finger with thumb: min = 50 and max is 200
        # volume range is: min -65 and max is 0
        vol = np.interp(lenght, [50, 200], [minVol, maxVol])
        volBar = np.interp(lenght, [50, 200], [400, 150])
        volPer = np.interp(lenght, [50, 200], [0, 100])
        #print(int(lenght), vol)
        volume.SetMasterVolumeLevel(vol, None)
        if lenght < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        elif lenght > 200:
            cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 250, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 250, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 35),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("hand control", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
