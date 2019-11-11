from darkflow.net.build import TFNet
import cv2

class object:
    def __init__(self, name, tracker, id_counter):
        self.name = name
        self.tracker = tracker
        self.fail_counter = 0
        self.speed = 0
        self.id = id_counter
        self.bbox = [150, 1, 1, 1]
        self.direction = "west"
        self.last_five_bbox = [150, 1, 1, 1]
        self.bbox_counter = 1
        # self.printa()

    def calculate_direction(self):
        print("insdisaassaasaswe printB", self.name)


# funtions
def get_overlap(box_a, box_b):
    # box_a --> x1a, y1a, x2a, y2a
    x1a = box_a[0]
    y1a = box_a[1]
    x2a = box_a[2]
    y2a = box_a[3]
    # box_b --> x1b, y1b, x2b, y2b
    x1b = box_b[0]
    y1b = box_b[1]
    x2b = box_b[2]
    y2b = box_b[3]

    # ---------- A inside B or B inside A
    if (x1b > x1a and x2b < x2a) or (x1a > x1b and x2a < x2b):
        return 1
    # ----------

    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    overlap = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return overlap


# capture = cv2.VideoCapture('videos/nov30/Burnett_Intersection_200ft.mp4')
# capture = cv2.VideoCapture('videos/nov30/Burnett_Perpendicular_150ft.mp4')
# capture = cv2.VideoCapture('videos/D-1/DJI_0004.mp4')
# capture = cv2.VideoCapture('videos/D-1/DJI_0003.mp4')
# capture = cv2.VideoCapture('videos/I-75 june19/DJI_0002.MP4')
# capture = cv2.VideoCapture('videos/I-75 june19/8seconds.mp4')
# capture = cv2.VideoCapture('videos/I-75 june19/DJI_0005_37_RS.mp4')
capture = cv2.VideoCapture('./videos/I-75 june19/DJI_0002_14_LS.mp4')
# capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture('videos/Cars-1900.mp4')


ctr = 0
detection_counter = 0  # counter to enable detection every #frames
tracker_counter = 0
car_counter = 0
truck_counter = 0
total_counter = 0

# control variables
detect_for_frames = 1
max_fail_counter = 5
min_overlap = 0.4
bezel_boundary = 20
# should be separate logic to handle left bezel and right bezel to cancel the tracker
# for now, both are same bezel of 150 pixels
options = {'model': './cfg/yolo.cfg', 'load': './weights/yolo.weights', 'threshold': 0.1, 'gpu': 0.9}

# dummy entry to initialize the object as dictioary, DO NOT DELETE
trackers_dict = {0: object("first", cv2.TrackerMedianFlow_create(), 0)}
tfnet = TFNet(options)


while True:
    ok, image = capture.read()
    # run yolo to detect objects in this frame
    result = tfnet.return_predict(image)

    # delete the trackers which have been inactive for last 10 frames
    for k in list(trackers_dict.keys()):
        if trackers_dict[k].fail_counter >= max_fail_counter:
            del trackers_dict[k]
    tracker_counter = len(trackers_dict)

    for item in result:

        bounding_box = (item['topleft']['x'], item['topleft']['y'], (item['bottomright']['x'] - item['topleft']['x']),
                        (item['bottomright']['y'] - item['topleft']['y']))

        # calculate max overlap of this bounding box with all active trackers
        # change current_overlap to 0 everytime before comparing with active trackers
        current_overlap = 0
        for v in trackers_dict.values():
            temp_overlap = get_overlap(bounding_box, v.bbox)
            if temp_overlap > current_overlap:
                current_overlap = temp_overlap

        # if overlap is greater than min_overlap, it means that object is already being tracked
        if current_overlap >= min_overlap:
            break

        # if it's a truck, create new tracker for it.
        if item['label'] == 'truck':
            truck_counter += 1
            total_counter += 1
            trackers_dict[tracker_counter] = object(item['label'], cv2.TrackerMedianFlow_create(), truck_counter)
            ok = trackers_dict[tracker_counter].tracker.init(image, bounding_box)
            tracker_counter += 1

        # if it's a car, create new tracker for it.
        if item['label'] == 'car':
            car_counter += 1
            total_counter += 1
            trackers_dict[tracker_counter] = object(item['label'], cv2.TrackerMedianFlow_create(), car_counter)
            ok = trackers_dict[tracker_counter].tracker.init(image, bounding_box)
            tracker_counter += 1
    detection_counter = 0

    # track the current trackers for "detect_for_frames" times and then run YOLO again.
    while detection_counter <= detect_for_frames:
        _, image = capture.read()

        # failsafe to delete trackers that go to edge in first and last "bezel_boundary" pixels or this
        # tracker has failed for "max_fail_counter" frames
        for k in list(trackers_dict.keys()):
            if trackers_dict[k].fail_counter >= max_fail_counter or trackers_dict[k].bbox[0] < bezel_boundary or \
                            trackers_dict[k].bbox[0] > image.shape[1] - bezel_boundary:
                del trackers_dict[k]
        tracker_counter = len(trackers_dict)

        # updating each tracker
        for id, obj in trackers_dict.items():
            # updating and printing the tracker only if it's not been failing for last 10 frames
            if obj.fail_counter < max_fail_counter:
                ok, bbox = obj.tracker.update(image)
                if ok:
                    obj.bbox = bbox
                    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                                  (255, 153, 51), 4)
                    cv2.putText(image, obj.name + " " + str(obj.id), (int(bbox[0]), int(bbox[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                else:
                    obj.fail_counter += 1
        # cv2.putText(image, "Total vehicle counted: " + str(total_counter) + "    Vehicles on screen: " +
        #             str(tracker_counter), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 3)
        cv2.putText(image, "Total vehicle: " + str(total_counter) + "   Vehicles on screen: " +
                    str(tracker_counter) + "     cars: " + str(car_counter) + "   trucks: " + str(truck_counter),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 3)
        cv2.imshow("feed", image)
        k = cv2.waitKey(1 & 0xff)
        print("Total vehicle: " + str(total_counter) + "   Vehicles on screen: " + str(tracker_counter) + "     cars: " + str(car_counter) + "   trucks: " + str(truck_counter))
        # when user presses SPACE button to pause
        if k == 32:
            while True:
                kk = cv2.waitKey(1 & 0xff)
                if kk == 32:
                    break

        # when user presses ESC button to quit execution
        if k == 27:
            capture.release()
            cv2.destroyAllWindows()
            exit()

        # when user presses 'r' button to reenforce YOLO object detection
        if k == ord('r'):
            break
        detection_counter += 1
