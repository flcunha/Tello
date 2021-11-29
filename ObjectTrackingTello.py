from djitellopy import Tello
import cv2
import numpy as np


######################################################################
width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
deadZone = 50
deadZone_vely = 10
deadZone_velx = 15
deadZone_velf = 6
detected=0
not_detected=0
area_detected=0
area_deadzone=2000
min_times_detected=100
min_times_not_detected=50
######################################################################

startCounter = 0

# CONNECT TO TELLO
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0



print(me.get_battery())

me.streamoff()
me.streamon()
######################## 

frameWidth = width
frameHeight = height

global imgContour
global dirx
global diry
def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",20,179,empty)
cv2.createTrackbar("HUE Max","HSV",40,179,empty)
cv2.createTrackbar("SAT Min","HSV",148,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",89,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",166,255,empty)
cv2.createTrackbar("Threshold2","Parameters",171,255,empty)
cv2.createTrackbar("Area","Parameters",1000,30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img,imgContour):
    _,contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dirx = 0
    diry = 0
    dirf = 0
    global detected
    global not_detected
    global area_detected
    if len(contours)==0 or sum([cv2.contourArea(cnt)>cv2.getTrackbarPos("Area", "Parameters") for cnt in contours])>1:
        not_detected+=1
        if not_detected==min_times_not_detected:
            detected=0
            area_detected=0
    else:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areaMin = cv2.getTrackbarPos("Area", "Parameters")
            if area > areaMin:
                if detected!=min_times_detected:
                    detected+=1
                    area_detected=(area_detected*(detected-1)+area)/detected
                    if detected==min_times_detected:
                        not_detected = 0
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                #print(len(approx))
                x , y , w, h = cv2.boundingRect(approx)
                cx = int(x + (w / 2))  # CENTER X OF THE OBJECT
                cy = int(y + (h / 2))  # CENTER X OF THE OBJECT
                if detected==min_times_detected:
                    if area>area_detected:
                        dirf = area/area_detected-1
                    else:
                        dirf = -area_detected/area-1
                dirx=round((cx-int(frameWidth/2))/deadZone)
                diry=round((cy-int(frameHeight/2))/deadZone)
                if abs(dirx)+abs(diry)>0:
                    cv2.putText(imgContour, " GO HERE " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                    if dirx==0:
                        cv2.rectangle(imgContour,
                                      (int(frameWidth / 2 - deadZone), int(frameHeight / 2 + diry * deadZone)), (
                                      int(frameWidth / 2 + deadZone),
                                      int(frameHeight / 2 + diry * (deadZone + diry / abs(diry)))), (0, 0, 255), cv2.FILLED)
                    elif diry==0:
                        cv2.rectangle(imgContour,
                                      (int(frameWidth / 2 + dirx * deadZone), int(frameHeight / 2 - deadZone)), (
                                      int(frameWidth / 2 + dirx * (deadZone + dirx / abs(dirx))),
                                      int(frameHeight / 2 + deadZone)), (0, 0, 255), cv2.FILLED)
                    else:
                        cv2.rectangle(imgContour,(int(frameWidth/2+dirx*deadZone),int(frameHeight/2+diry*deadZone)),(int(frameWidth/2+(dirx+dirx/abs(dirx))*(deadZone)),int(frameHeight/2+(diry+diry/abs(diry))*(deadZone))),(0,0,255),cv2.FILLED)

                cv2.line(imgContour, (int(frameWidth/2),int(frameHeight/2)), (cx,cy),(0, 0, 255), 3)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2)
                cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2)
                cv2.putText(imgContour, "Area D: " + str(int(area_detected)), (x + w + 20, y + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2)
                cv2.putText(imgContour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,0.7,(0, 255, 0), 2)
    return dirx,diry,dirf

def display(img):
    for i in np.arange(deadZone,int(frameWidth/2),deadZone):
        cv2.line(img,(int(frameWidth/2)-i,0),(int(frameWidth/2)-i,frameHeight),(255,255,0),3) #Linha vertical 1
        cv2.line(img,(int(frameWidth/2)+i,0),(int(frameWidth/2)+i,frameHeight),(255,255,0),3) #Linha vertical 2
    cv2.circle(img,(int(frameWidth/2),int(frameHeight/2)),5,(0,0,255),5) #Circulo pequeno vermelho no centro
    for i in np.arange(deadZone,int(frameHeight/2),deadZone):
        cv2.line(img, (0,int(frameHeight / 2) - i), (frameWidth,int(frameHeight / 2) - i), (255, 255, 0), 3) #Linha horizontal 1
        cv2.line(img, (0, int(frameHeight / 2) + i), (frameWidth, int(frameHeight / 2) + i), (255, 255, 0), 3) #Linha horizontal 2

while True:

    # GET THE IMAGE FROM TELLO
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))
    imgContour = img.copy()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")


    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    dirx,diry,dirf=getContours(imgDil, imgContour)
    display(imgContour)

    ################# FLIGHT
    if startCounter == 0:
       me.takeoff()
       startCounter = 1

    me.up_down_velocity=int(-diry*deadZone_vely)
    me.yaw_velocity=int(dirx*deadZone_velx)
    me.for_back_velocity = int(-dirf*deadZone_velf)

    me.left_right_velocity = 0
    me.for_back_velocity = 0;me.up_down_velocity = 0; me.yaw_velocity = 0
    # SEND VELOCITY VALUES TO TELLO
    if me.send_rc_control:
       me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)

    stack = stackImages(0.9, ([img, result], [imgDil, imgContour]))
    cv2.imshow('Horizontal Stacking', stack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

# cap.release()
cv2.destroyAllWindows()
