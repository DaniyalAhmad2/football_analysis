import cv2

image = cv2.imread('input_videos/field.jpg')
cv2.imshow('Football Field', image)
cv2.waitKey(0)
cv2.destroyAllWindows()