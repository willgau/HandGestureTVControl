import cv2
from HandPoseEstimation import PoseEstiamtion
from TVControl import TVControl

from HandDectection import HandDetector


def main():

    #tv = TVControl()
    #tv.TurnOffTV()
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.5, trackCon=0.5)
    pe = PoseEstiamtion()


    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        Landmark = detector.findPosition(img)
        if len(Landmark) != 0:
            #print(pe.isThumbOpen(lmList))
            pe.GetPose(Landmark)
            pe.SwipeTick()
            action = pe.GetAction(Landmark)
            if not (action == "None"):
                print(action)

            pe.printfinger(Landmark)
            #tv.DoAction(action)

        cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if k == 27:
            break



if __name__ == "__main__":
    main()

