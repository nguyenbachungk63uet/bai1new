
import numpy as np
import argparse
import time
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\Users\nguye\AppData\Local\tesseract.exe'
input_text = 'NGUYEN BA CHUNG'
image = cv2.imread('image/Chung8.jpg')
# img2 = cv2.resize(img2, (int(img2.shape[1]/7), int(img2.shape[0]/7)))
temp_face = image.copy()

width = image.shape[0]
height = image.shape[1]
my_img = np.zeros((width, height, 3), dtype = np.uint8)
# my_img[:,:,0] = 0
# my_img[:,:,1] = 0
# my_img[:,:,2] = 0

temp_my_img = my_img.copy()
font = cv2.FONT_HERSHEY_SIMPLEX
org_1 = (0,30)
org_2 = (int(height/2), int(width/2))

font_scale = 1
color =  (255, 159, 0)
thickness = 1

image = cv2.putText(image, input_text, org_1, font, font_scale, color, thickness, cv2.LINE_AA)
image = cv2.putText(image, input_text, org_2, font, font_scale, color, thickness, cv2.LINE_AA)

my_img = cv2.putText(my_img, input_text, org_1, font, font_scale, color, thickness, cv2.LINE_AA)
my_img = cv2.putText(my_img, input_text, org_2, font, font_scale, color, thickness, cv2.LINE_AA)

def draw_text(img):
    h = img.shape[1]
    w = img.shape[0]
    # img = cv2.putText(img, input_text, (0, 30), font, 1, color, thickness, cv2.LINE_AA)
    img = cv2.putText(img, input_text, (int(h/2), int(w/2)), font, 1, color, thickness, cv2.LINE_AA)
    return img

def text_detect(orig, img):

    sub = orig - img
    sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(sub, 5, 255, cv2.THRESH_BINARY)
    sub = thresh

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(sub, config=custom_config, lang = 'eng')
    h, w= sub.shape
    boxes = pytesseract.image_to_boxes(sub) 
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

 
    return text, boxes, img

text, boxes, result_1 = text_detect(temp_face, image)
print('In my face image: ')
print(text)
print(text.lower())
print('==================')
my_img_text, my_img_boxes, my_img_result = text_detect(temp_my_img, my_img)
print('Im my hand-made image:')
print(my_img_text)
print(my_img_text.lower())
print('==================')

cv2.imshow('Img 2', my_img_result)
cv2.imshow('Img 1', result_1)
cv2.waitKey()