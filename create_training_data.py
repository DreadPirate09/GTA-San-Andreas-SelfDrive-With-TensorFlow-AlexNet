import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def keys_to_output(keys):
    output = [0,0,0]
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:
        output[1] = 1
    return output

def showImage(img):

    cv2.imshow('test',img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def get_specified_square_screen_no_map(x1, y1, x2, y2):
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

def get_specified_square_screen(x1, y1, x2, y2):
    screen = grab_screen(region=(x1, y1, x2, y2))
    if screen is not None:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    else:
        print("Error: Screen capture failed.")
    return screen

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
    

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    while(True):
        # 800x600 windowed mode
        print('getting the frame with number: '+str(len(training_data)))
        # screen = get_specified_circle_screen(100, 1100, 300, 1300, 100)
        screen = get_specified_square_screen(0, 300, 800, 600);
        showImage(screen)
        last_time = time.time()
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (80,30))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        np_training_data = np.array(training_data, dtype=object)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,np_training_data)

main()