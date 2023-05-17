import cv2 as cv
import numpy as np
from Utilities import *

img = np.zeros(([512, 512, 1]), np.uint8)
img.fill(255)
# For drawing Rectangle
drawing = False
ix,iy = -1,-1
rec = cvRectangle()

def createWhiteBackground():
    # For creating White Background
    global img
    img = np.zeros(([512, 512, 1]), np.uint8)
    img.fill(255)
    print('image shape: ',img.shape)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
            cv.imshow('image', img)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        rec.w = abs(x-ix)
        rec.h = abs(y-iy)
        rec.cx = (ix+x)/2
        rec.cy = (iy+y)/2
        rec.angle = 0
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
        cv.imshow('image', img)

def draw_text_on_rectangle(img, rec, text):
    x = int(rec.cx/2)
    y = int(rec.cy/2)

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 2

    # Viết chữ lên hình chữ nhật
    j = 0
    for i in range(y,y+len(text)*20,20):
        if (j<len(text)):
            cv.putText(img, text[j], (x, i), font, font_scale, color, thickness)
            print(x,i)
            j += 1
        else:
            break

def main():

    global ix,iy,img,rec
    cv.namedWindow("image",0)

    while (True):
        print('Press 1 for creating white background')
        print('Press 2 for drawing a rectangle')
        print('Press 3 for translate a rectangle')
        print('Press 4 for rotate a rectangle')
        print('Press 5 for scaling a rectangle')
        n = input('Your input: ')

        if (n=='1'):
            createWhiteBackground()
        elif (n=='2'):
            print("Just draw one rectangle")
            createWhiteBackground()
            ix = -1
            iy = -1
            cv.setMouseCallback("image", draw_rectangle,[img,rec])
            while True:
                cv.imshow("image", img)
                if cv.waitKey(10) == 27:
                    break
            print(rec.cx, rec.cy, rec.w, rec.h)
        elif (n=='3'):
            dx = int(input("Add coordinate x="))
            dy = int(input("Add coordinate y="))

            # Dịch chuyển hình chữ nhật
            rec.translate((dx,dy))

            # Vẽ lại hình chữ nhật trên ảnh
            img = np.zeros(([512, 512, 1]), np.uint8)
            img.fill(255)
            drawRectangle(rec,img)
            print(rec.cx, rec.cy, rec.w, rec.h)
            print(img.shape)
        elif (n=='4'):
            rotate = (int(input("Rotation angle = ")))
            rec.angle += rotate
            img = np.zeros(([512, 512, 1]), np.uint8)
            img.fill(255)
            drawRectangle(rec,img)
            print(rec.cx, rec.cy, rec.w, rec.h, rec.angle)
        elif (n == '5'):
            print()
            print("Scale rectangle by height = h * sy / width = w * xx")
            sx = float(input("sx="))
            sy = float(input("sy="))

            # Scale image
            #img_scaled = cv.resize(img, None, fx=sx, fy=sy, interpolation=cv.INTER_LINEAR)
            #rec.scale((sx,sy))
            rec.w *= sx
            rec.h *= sy
            print(rec.cx,rec.cy,rec.w,rec.h)
            img = np.zeros(([512, 512, 1]), np.uint8)
            img.fill(255)
            drawRectangle(rec,img)
        else:
            txt = ['Group 1','Pham Huynh Quy An - SE171139','Nguyen Le Bao Xuyen - SE170455','Nguyen Ngoc Chien - SE173133',
                   'Nguyen Vo Thanh Duy - SE171668','Phung Hai Dang - SE171364','Pham Huynh Anh Tu - SE171273','Tran Trong Tin - SE171668']

            draw_text_on_rectangle(img,rec,txt)
            while True:
                cv.imshow("image", img)
                if cv.waitKey(10) == 27:
                    break
            break
        cv.imshow("image", img)
        cv.waitKey(20)


if __name__ == '__main__':
    main()