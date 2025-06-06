import cv2

color_img = cv2.imread('input.jpg')

gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('output_gray.jpg', gray_img)