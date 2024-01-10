"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import math

gaze = GazeTracking()
#webcam = cv2.VideoCapture(0)

color = (0, 255, 0)  # BGR颜色格式，这里是绿色
color0 = (147, 58, 31)
while True:
    # We get a new frame from the webcam
    #_, frame = webcam.read()
    frame = cv2.imread('w2.jpg')
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)
    print(f"gaze.eye_left.lefteys:{gaze.eye_left.lefteys}")
    print(f"gaze.eye_right.righteyes:{gaze.eye_right.righteyes}")
    
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, color), 1)   
    print(f"process successful!!")
    #cv2.imwrite('result.jpg',frame)

    for i in range(6):
        pt_pos1 = (gaze.eye_left.lefteys[i][0],gaze.eye_left.lefteys[i][1])
        cv2.circle(frame, pt_pos1, 1, (255, 0, 0), 2)
        pt_pos2 = (gaze.eye_right.righteyes[i][0],gaze.eye_right.righteyes[i][1])
        cv2.circle(frame, pt_pos2, 1, (255, 0, 0), 2)
    

    # caculate left diagonal middle
    dia0_x = (gaze.eye_left.lefteys[1][0] + gaze.eye_left.lefteys[4][0] + 
    gaze.eye_left.lefteys[2][0] + gaze.eye_left.lefteys[5][0] )/4

    dia0_y = (gaze.eye_left.lefteys[1][1] + gaze.eye_left.lefteys[4][1] + 
    gaze.eye_left.lefteys[2][1] + gaze.eye_left.lefteys[5][1] )/4

    # caculate <iris,middle>
    distance0 = math.sqrt((left_pupil[0] - dia0_x) ** 2 + (left_pupil[1] - dia0_y) ** 2)
    print(f"left offset:{distance0}")

    # caculate right diagonal middle
    dia1_x = (gaze.eye_right.righteyes[1][0] + gaze.eye_right.righteyes[4][0] + 
    gaze.eye_right.righteyes[2][0] + gaze.eye_right.righteyes[5][0] )/4

    dia1_y = (gaze.eye_right.righteyes[1][1] + gaze.eye_right.righteyes[4][1] + 
    gaze.eye_right.righteyes[2][1] + gaze.eye_right.righteyes[5][1] )/4

    # caculate <iris,middle>
    distance1 = math.sqrt((right_pupil[0] - dia1_x) ** 2 + (right_pupil[1] - dia1_y) ** 2)
    print(f"left offset:{distance1}")

    rounded_distance0 = round(distance0, 2)
    rounded_distance1 = round(distance1, 2)
    cv2.putText(frame, "Left pupil offset :  " + str(rounded_distance0), (90, 190), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)
    cv2.putText(frame, "Right pupil offset : " + str(rounded_distance1), (90, 215), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)  

    cv2.imwrite('result.jpg',frame)

    #cv2.imshow("Demo", frame)
    break
    if cv2.waitKey(1) == 27:
        break
   
#webcam.release()
cv2.destroyAllWindows()
