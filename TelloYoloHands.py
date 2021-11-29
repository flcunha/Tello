from djitellopy import Tello
import cv2
import numpy as np
import time
#quão confiante a rede neuronal tem de estar no facto de ser uma mão para nós aceitarmos
lim_confidence=0.3
lim_threshold=0.05
#tamanho da imagem a dar à rede neuranal (maior = mais accurate mas mais lento)
size=256
#o tipo de objectos que a rede neuronal vai detectar
labels=["hand"]

# filename="yolov2-tiny-one-class"
# filename="cross-hands-yolov4-tiny"
#Ficheiro de weights/configuration
filename="cross-hands"
net = cv2.dnn.readNetFromDarknet(filename+".cfg",filename+".weights")

def detect_hand(image):

    #Prepara a imagem para ser corrida pela rede neuronal
    command=False
    start = time.time()
    ih, iw = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (size,size), swapRB=True, crop=False)
    net.setInput(blob)
    #Corre a rede neuronal
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            # na última posição é q esta a detection score de hand ou não
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > lim_confidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([iw, ih, iw, ih])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, lim_confidence, lim_threshold)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            id = classIDs[i]
            confidence = confidences[i]

            results.append((id, labels[id], confidence, x, y, w, h))
    width=iw
    height=ih

    output = []

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', width, height)
    num_hands=len(results)
    w=0
    h=0
    cx=0
    cy=0
    hands=[]
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX,
                    0.75, color, 1)
    end = time.time()
    inference_time = end - start
    #escreve os fps no canto superior esquerdo com base no tempo que a rede neuronal demorou a correr
    cv2.putText(image,str(int(1/inference_time))+"fps",(15,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0, 0, 255),1)
    return image, results

#cria o objecto Tello
me = Tello()
time.sleep(1)
#connecta
me.connect()
time.sleep(1)
#printa bateria
print(me.get_battery())
time.sleep(1)
#desliga stream (só para garantir que não está on)
me.streamoff()
time.sleep(1)
#liga stream
me.streamon()
time.sleep(5)
#takeoff
me.takeoff()
time.sleep(2)


width = 480
height = 360

#tamanho da área default da mão
area_detected=0
#se já detectou uma mão
detected_final=False
#quantas frames detectou uma mão
fps_detected=0
#quantas frames não detectou uma mão
fps_not_detected=0
#quantas frames precisa de detectar uma mão para fixar a área
lim_detected=10

#baseline de velocidade (em cm/s acho eu). Velocidade default por cada pixel da imagem da mão de distância para o centro da imagem
up_down_baseline=0.3
left_right_baseline=0.3
#baseline de velocidade frente/trás
front_back_baseline=6
detected_two_hands=[0,0]
flying=True
while flying:
    #obtem imagem
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))

    #à partida velocidades todas a 0
    me.left_right_velocity = 0
    me.for_back_velocity = 0
    me.up_down_velocity = 0
    me.yaw_velocity = 0
    try:
        # detecta mãos na imagem
        image,results=detect_hand(img)
        num_hands=len(results)
        if num_hands==0:
            # se não houver mãos, manda as velocidades a 0
            if me.send_rc_control:
                me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
            # se não houver mãos, manda as velocidades a 0
            detected_two_hands = [0, 0]
            fps_not_detected+=1
            if fps_not_detected==lim_detected:
                detected_final=False
                fps_detected=0
                fps_not_detected=0
        elif num_hands==1:
            #se detectar uma mão, obtem dados do centro e área
            _, _, _, x, y, w, h=results[0]
            cx = x + (w / 2)
            cy = y + (h / 2)

            area=w*h
            detected_two_hands = [0, 0]
            fps_not_detected=0
            #se a área default ainda não tiver sido definida actualiza counters e área default
            if not detected_final:
                fps_detected += 1
                area_detected = (area_detected * (fps_detected - 1) + area) / fps_detected
                if fps_detected == lim_detected:
                    detected_final = True
            if detected_final:
                # vê se o drone deve ir para frente ou trás consoante a área desta imagem seja menor ou maior que a default, respectivamente
                if area > area_detected:
                    me.for_back_velocity = int(-(area / area_detected - 1)*front_back_baseline)
                else:
                    me.for_back_velocity = int((area_detected / area - 1)*front_back_baseline)
            # calcula velocidade de cima/baixo consoante y actual do centro da mão
            me.up_down_velocity=-int((cy-height/2)*up_down_baseline)
            # calcula velocidade de rotação consoante x actual do centro da mão
            me.yaw_velocity=int((cx-width/2)*left_right_baseline)
            # manda velocidades para o drone
            if me.send_rc_control:
                me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
        elif num_hands==2:
            #se detectar duas mãos manda velocidades a zero
            if me.send_rc_control:
                me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
            _, _, _, x1, y1, w1, h1=results[0]
            _, _, _, x2, y2, w2, h2=results[1]
            #se detectar duas mãos manda horizontas (diferenças de y's pequenas comparadas com x's), aumenta o counter
            if abs(y1-y2)<1/2*abs(x1-x2):
                detected_two_hands[0] += 1
            #se detectar duas mãos manda verticais (diferenças de x's pequenas comparadas com y's), aumenta o counter
            if abs(x1 - x2) < 1 / 2 * abs(y1 - y2):
                detected_two_hands[1] += 1
            #se o counter de mãos horizontais tiver chegado a 5, dá o flip e faz reset dos counters
            if detected_two_hands[0]>5:
                me.flip("f")
                detected_two_hands = [0, 0]
            #se o counter de mãos verticais tiver chegado a 5, aterra, e fecha streams
            if detected_two_hands[1]>5:
                me.streamoff()
                me.land()
                cv2.destroyAllWindows()
                flying=False
                detected_two_hands = [0, 0]
            else:
                if me.send_rc_control:
                    me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity,
                                       me.yaw_velocity)

        cv2.imshow("image", image)
    except:
        pass
    # SEND VELOCITY VALUES TO TELLO
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

