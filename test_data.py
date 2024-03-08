from getkeys import key_check
from grabscreen import grab_screen
import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
import time
import datetime

print('we import everithing')



WIDTH = 80
HEIGHT = 30
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    if delay(5) == True:
        PressKey(W)
    PressKey(A)
    ReleaseKey(D)

def right():
    if delay(5) == True:
        PressKey(W)
    PressKey(D)
    ReleaseKey(A)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def showImage(img):

    cv2.imshow('test',img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def get_specified_square_screen(x1, y1, x2, y2):
    screen = grab_screen(region=(x1, y1, x2, y2))
    if screen is not None:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    else:
        print("Error: Screen capture failed.")

    center = (screen.shape[1] // 2, screen.shape[0] // 2)  # Center of the image
    mask = np.zeros_like(screen)
    cv2.circle(mask, (105 ,235), 70, (255, 0, 0), -1)
    result = cv2.addWeighted(screen, 1, mask, 1, 0)

    return result

def get_specified_circle_screen(x1, y1, x2, y2, radius):
    screen = grab_screen(region=(x1, y1, x2, y2))
    if screen is not None:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    else:
        print("Error: Screen capture failed.")

    center = (screen.shape[1] // 2, screen.shape[0] // 2)  # Center of the image
    mask = np.zeros_like(screen)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    result = cv2.bitwise_and(screen, mask)

    return result

lastActionFw = datetime.datetime.now()

def delay(sec):

    global lastActionFw
    now = datetime.datetime.now()

    print(now.second)
    print(lastActionFw.second)

    if lastActionFw.second + sec < 60 and sec < (now.second - lastActionFw.second):
        lastActionFw = datetime.datetime.now()
        print('delay true')
        return True
    elif lastActionFw.second + sec > 60  and (60 - lastActionFw.second + now.second) > sec:
        lastActionFw = datetime.datetime.now()
        print('delay true')
        return True
    else:
        print('delay false')
        return False


def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while True:
        
        if not paused:
            # 800x600 windowed mode
            # screen =  np.array(ImageGrab.grab(bbox=(0,340,800,640)))
            screen = get_specified_square_screen(0, 300, 800, 600)
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.resize(screen, (80,30))
            moves = list(np.around(model.predict([screen.reshape(80,30,1)])[0]))
            if moves == [1,0,0]:
                left()
            elif moves == [0,1,0]:
                print('Going fwd')
                straight()
            elif moves == [0,0,1]:
                right()
   
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()