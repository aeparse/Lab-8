import cv2
import numpy as np


def detect_circle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=40)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            return (pt[0], pt[1]), pt[2]
    return None, None


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 < y2 and x1 < x2:
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])


fly_image = cv2.imread('fly64.png', -1)
fly_alpha_mask = fly_image[:, :, 3] / 255.0
fly_image = fly_image[:, :, :3]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    circle_center, radius = detect_circle(frame)
    if circle_center is not None:
        cv2.circle(frame, circle_center, 1, (0, 255, 0), 3)
        cv2.circle(frame, circle_center, radius, (0, 255, 0), 3)

        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        distance = np.sqrt((circle_center[0] - frame_center[0]) ** 2 + (circle_center[1] - frame_center[1]) ** 2)
        cv2.putText(frame, f"Distance: {distance:.2f}, press q to escape", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        fly_pos = (circle_center[0] - fly_image.shape[1] // 2,
                   circle_center[1] - fly_image.shape[0] // 2)
        overlay_image_alpha(frame, fly_image, fly_pos, fly_alpha_mask)

    cv2.imshow("Frame with Fly", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
