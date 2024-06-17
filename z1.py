import cv2

image = cv2.imread('variant-7.jpg')

flipped_horizontally = cv2.flip(image, 1)

flipped_vertically = cv2.flip(flipped_horizontally, 0)

cv2.imwrite('transformed_image.jpg', flipped_vertically)

cv2.imshow('Transformed Image', flipped_vertically)
cv2.waitKey(0)
cv2.destroyAllWindows()
