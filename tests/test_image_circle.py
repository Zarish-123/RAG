import cv2

image = cv2.imread("data/images.jpg")

cv2.circle(
    image,
    (200, 150),   # center (x, y)
    50,           # radius
    (255, 0, 0),  # color: Blue
    0            # thickness
)

cv2.imshow("Circle", image)

cv2.waitKey(0)
cv2.destroyAllWindows()