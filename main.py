import os
import cv2
import handTrackingModule as htm
import numpy as np

# prams
width, height = 1920, 1080
folderPath = "Presentation"
gestureThreshold = 500

# camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detector = htm.HandDetector(detectionCon=0.8, maxHands=1)

# variables
imageNum = 0
btnDelay = 20
btnCount = 0
buttonPressed = False
hs, ws = int(120 * 1), int(213 * 1)
annotations = [[]]
annotationNumber = -1
annotationStart = False
smoothening = 4
prevX, prevY = 0, 0
curX, curY = 0, 0

# list of slides
pathSlides = sorted(os.listdir(folderPath), key=len)

while True:
    # Get Image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    fullPathSlides = os.path.join(folderPath, pathSlides[imageNum])
    currSlide = cv2.imread(fullPathSlides)

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, draw=True)
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 225, 0), 10)

    # If hand is detected
    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 3, width // 1.8], [0, width]))
        yVal = int(np.interp(lmList[8][1], [200, height // 2], [0, height]))

        curX = int(prevX + (xVal - prevX) / smoothening)
        curY = int(prevY + (yVal - prevY) / smoothening)
        indexFinger = curX, curY

        if cy <= gestureThreshold:

            # gesture previous
            if fingers == [1, 0, 0, 0, 0]:
                if imageNum > 0:
                    buttonPressed = True
                    imageNum -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

            # gesture next
            if fingers == [0, 0, 0, 0, 1]:
                if imageNum < len(pathSlides) - 1:
                    buttonPressed = True
                    imageNum += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        # gesture pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(currSlide, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            prevX, prevY = curX, curY
        # gesture draw
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(currSlide, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            prevX, prevY = curX, curY
        else:
            annotationStart = False
        # gesture undo
        if fingers == [0, 0, 1, 1, 1]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    # handle button
    if buttonPressed:
        btnCount += 1
        if btnCount > btnDelay:
            btnCount = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(currSlide, annotation[j - 1], annotation[j], (0, 0, 200), 2)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = currSlide.shape
    currSlide[0:hs, w - ws:w] = imgSmall

    # cv2.imshow("Image", img)
    cv2.imshow("Slides", currSlide)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('Q'):
        break
