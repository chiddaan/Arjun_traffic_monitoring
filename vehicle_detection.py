import cv2


# cap = cv2.VideoCapture('videos/nov30/Burnett_Perpendicular_150ft.mp4')
# cap = cv2.VideoCapture('videos/nov30/Burnett_Intersection_200ft.mp4')
# cap = cv2.VideoCapture('videos/D-1/DJI_0004.mp4')
# cap = cv2.VideoCapture('videos/I-75 june19/DJI_0005_37_RS.mp4')
# cap = cv2.VideoCapture('videos/scooters.mp4')
# cap = cv2.VideoCapture('videos/media5.avi')
cap = cv2.VideoCapture('/media/arjun/backup_ubuntu/videos/Mitchell/april18/DJI_0002.mp4')

# cap = cv2.VideoCapture(0)
reference_frame = None
image_area = None
ctr = 1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))


# cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    else:
        if reference_frame is None:
            reference_frame = frame
            reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
            image_area = reference_frame.shape[0] * reference_frame.shape[1]
            continue
        frame1 = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(reference_frame, gray)
        # blur = cv2.medianBlur(difference, 13)

        # blur = cv2.GaussianBlur(difference, (11, 11), 0)
        blur = cv2.blur(difference, (15, 15))

        f, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilation = threshold
        # dilation = cv2.dilate(dilation, kernel, iterations=6)
        # dilation = cv2.dilate(dilation, kernel, iterations=8)
        dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            contour_area = cv2.contourArea(i)
            if (contour_area > 0.0007 * image_area) and (contour_area < 0.3 * image_area):
                (x, y, w, h) = cv2.boundingRect(i)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("frames", frame)
        cv2.imshow("dlation", dilation)
        reference_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

         k = cv2.waitKey(1 & 0xff)
        # when user presses SPACE button to pause/play
        if k == 32:
            while True:
                kk = cv2.waitKey(1 & 0xff)
                if kk == 32:
                    break
        # when user presses ESC button to quit execution
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()