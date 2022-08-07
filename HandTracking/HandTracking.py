import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    ret, img_BGR = cap.read()
    img_BGR = cv2.flip(img_BGR, 1)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # location of each point on the fingers
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img_BGR.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                # if id == 0:
                # cv2.circle(img_BGR, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                # cv2.circle(img_BGR, (cx, cy), 25,
                #           (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img_BGR, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img_BGR, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()  # current time
    fps = 1/(cTime-pTime)  # inverse of the period = frequency
    pTime = cTime

    cv2.putText(img_BGR, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow("that's a hand", img_BGR)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
