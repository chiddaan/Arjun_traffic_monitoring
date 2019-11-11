from darkflow.net.build import TFNet
import cv2
import copy
import math
import matplotlib.pyplot as plt



class object:
    def __init__(self, name, tracker, id_counter):
        self.name = name  # label like car, truck or bus
        self.tracker = tracker  # tracker object to track movements
        self.fail_counter = 0  # number of frames the tracker has been failing
        self.speed = 0  # speed in mph of the object on screen
        self.id = id_counter
        self.bbox = [150, 1, 1, 1]  # co-ordinates of the bounding box
        self.bbox_color = (255, 153, 51)
        # use next 4 params to calculate total distance covered by obj in total time, thus calculate speed Et Voila
        self.start_bbox = [150, 1, 1, 1]
        self.end_bbox = [150, 1, 1, 1]
        self.total_distance_in_pixel = 0
        self.start_frame = 0
        self.end_frame = 0
        self.total_time = 0
        self.avg_speed = 0
        self.directional_bbox_counter = 1
        self.direction = "dir"  # direction in which the object is moving
        self.last_ten_bbox = [[150, 1, 1, 1]]  # coordinates of last five locations of the object to find direction
        self.bbox_counter = 1  # counter to state how many bounding boxes have been recorded; this value will be
        #                        controlled from 1 to 5 using modular 5
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

    # If there's no intersection between box A and B, then return 0.1
    if not (((x1a < x1b < x2a) and (y1a < y1b < y2a)) or ((x1a < x2b < x2a) and (y1a < y2b < y2a))) or \
            (((x1a < x2b < x2a) and (y1a < y1b < y2a)) or ((x1a < x1b < x2a) and (y1a < y2b < y2a))):
        return 0.1
    # ---------- A inside B or B inside A
    # if (x1b > x1a and x2b < x2a) or (x1a > x1b and x2a < x2b):
    # if (x1b > x1a and x2b < x2a and y1b > y1a and y2b < y2a) or (x1a > x1b and x2a < x2b and y1a > y1b and y2a > y2b):
    #     return 1
    # ----------

    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(x1a, x1b)
    y_a = max(y1a, y1b)
    x_b = min(x2a, x2b)
    y_b = min(y2a, y2b)

    # compute the area of intersection rectangle
    inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = (x2a - x1a + 1) * (y2a - y1a + 1)
    box_b_area = (x2b - x1b + 1) * (y2b - y1b + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    overlap = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    # if overlap > 0:
    #     print(overlap)
    return overlap


def count_parameters():
    global last_total_counter, summation_distance, summation_time
    quarter_count = (total_counter - last_total_counter)

    flow_rate = quarter_count * 4 * 60

    average_travel_speed_in_pixel = summation_distance / summation_time
    average_car_length_in_pixel = (summation_car_length_in_pixel / car_counter)
    average_travel_speed_meters = (average_travel_speed_in_pixel / average_car_length_in_pixel) * length_of_car_in_meter
    average_travel_speed = (average_travel_speed_meters * 3600) / (1609.344)
    density = flow_rate / average_travel_speed

    density_list.append(density)
    average_travel_speed_list.append(average_travel_speed)
    flow_rate_list.append(flow_rate)

    print('After ' + str(frame_counter / fps) + ' seconds:')
    # print('Average Travel Speed in pixel = ' + str(average_travel_speed_in_pixel) + ' pixels/second')
    print('Quarter count: ' + str(quarter_count))
    print('Flow rate: ' + str(flow_rate))
    print('Average Travel Speed = ' + str(round(average_travel_speed, 2)) + ' miles/hour')
    print('Density: ' + str(round(density, 2)) + 'veh/mi')
    print('')

    print('Average car length in pixel: ' + str(summation_car_length_in_pixel / car_counter))
    print(
        'Total vehicle: ' + str(total_counter) + '     cars: ' + str(car_counter) + '   trucks: ' + str(truck_counter))
    print('--------------------------------------------------------')
    summation_time = summation_distance = 1
    last_total_counter = total_counter


# capture = cv2.VideoCapture('videos/nov30/Burnett_Intersection_200ft.mp4')
# capture = cv2.VideoCapture('videos/nov30/Burnett_Perpendicular_150ft.mp4')
# capture = cv2.VideoCapture('videos/D-1/DJI_0007.mp4')
# capture = cv2.VideoCapture('/home/arjun/Documents/virtualEnvs/imagepp3/src/darkflow/darkflow/videos/Traffic_Video/2/2_08_perpendicular_zoomed_out.mp4')
# capture = cv2.VideoCapture('videos/Traffic_Video/2/2_10_perpendicular_zoomed_out.mp4')
# capture = cv2.VideoCapture('videos/Traffic_Video/2/2_07_right_slant_zoomed_in.mp4')
capture = cv2.VideoCapture('/media/arjun/backup_ubuntu/videosa/DJI_250ft_25mm_Jan26.MOV')
# capture = cv2.VideoCapture('/media/arjun/backup_ubuntu/videosa/DJI_150ft_25mm_Jan26.MOV')
# capture = cv2.VideoCapture('videos/D-1/DJI_0003.mp4')
# capture = cv2.VideoCapture('videos/I-75 june19/DJI_0002.MP4')
# capture = cv2.VideoCapture('videos/I-75 june19/8seconds.mp4')
# capture = cv2.VideoCapture('videos/I-75 june19/DJI_0005_37_RS.mp4')
# capture = cv2.VideoCapture('videos/I-75 june19/DJI_0002_14_LS.mp4')
# capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture('videos/Cars-1900.mp4')

# variables

ctr = 0
detection_counter = 0  # counter to enable detection every #frames
tracker_counter = 0
car_counter = 0
truck_counter = 0
total_counter = 0
right_traffic_counter = 0
left_traffic_counter = 0
left_car_counter = 0
left_truck_counter = 0
right_car_counter = 0
right_truck_counter = 0
frame_counter = 0
flow_rate = 0
quarter_count = 0
average_travel_speed_list = list()
flow_rate_list = list()
density_list = list()
last_total_counter = 0
summation_car_length_in_pixel = 0
summation_time = 0
summation_distance = 0


# control variables
detect_for_frames = 20
max_fail_counter = 10
min_overlap = 0.5
fps = 30
length_of_car_in_meter = 4.8
bezel_boundary = 150
# should be separate logic to handle left bezel and right bezel to cancel the tracker
# for now, both are same bezel of 150 pixels
options = {'model': 'cfg/yolo.cfg', 'load': 'weights/yolo.weights', 'threshold': 0.25, 'gpu': 1.0}

# dummy entry to initialize the object as dictioary, DO NOT DELETE
trackers_dict = dict() #{0: object("first", cv2.TrackerMedianFlow_create(), 0)}
tfnet = TFNet(options)

print('--------------------------------------------------------')
while True:
    ok, image = capture.read()
    if not ok:
        print('video error')
        break
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame_counter += 1

    # run yolo to detect objects in this frame
    result = tfnet.return_predict(image)
    image2 = copy.deepcopy(image)
    for item in result:
        cv2.rectangle(image2, (item['topleft']['x'], item['topleft']['y']),
                      ((item['bottomright']['x']), (item['bottomright']['y'])), (0, 0, 255), 5)

    for item in result:
        # to consider only trucks and cars
        if item['label'] != 'truck' and item['label'] != 'car':
            break
        bounding_box = (item['topleft']['x'], item['topleft']['y'], (item['bottomright']['x'] - item['topleft']['x']),
                        (item['bottomright']['y'] - item['topleft']['y']))
        # calculate max overlap of this bounding box with all active trackers
        max_overlap = 0
        for v in trackers_dict.values():
            bbox_param1 = (bounding_box[0], bounding_box[1], (bounding_box[0] + bounding_box[2]), (bounding_box[1] + bounding_box[3]))
            bbox_param2 = (v.bbox[0], v.bbox[1], (v.bbox[0] + v.bbox[2]), (v.bbox[1] + v.bbox[3]))
            # print(bbox_param1)
            # print(bbox_param2)
            temp_overlap = get_overlap(bbox_param1, bbox_param2)
            if temp_overlap > max_overlap:
                max_overlap = temp_overlap
            # print(max_overlap)
            # print()
        # if overlap is
        if max_overlap >= min_overlap and frame_counter > 1:
            break

        # if it's a truck, create new tracker for it.
        if item['label'] == 'truck':

            trackers_dict[tracker_counter] = object(item['label'], cv2.TrackerMedianFlow_create(), truck_counter)
            ok = trackers_dict[tracker_counter].tracker.init(image, bounding_box)
            if ok:
                trackers_dict[tracker_counter].start_frame = frame_counter
                trackers_dict[tracker_counter].start_bbox = bounding_box
                trackers_dict[tracker_counter].bbox_color = (89, 247, 71)
                tracker_counter += 1
                truck_counter += 1
                total_counter += 1

        # if it's a car, create new tracker for it.
        if item['label'] == 'car':
            trackers_dict[tracker_counter] = object(item['label'], cv2.TrackerMedianFlow_create(), car_counter)
            ok = trackers_dict[tracker_counter].tracker.init(image, bounding_box)
            if ok:
                trackers_dict[tracker_counter].start_frame = frame_counter
                trackers_dict[tracker_counter].start_bbox = bounding_box
                trackers_dict[tracker_counter].bbox_color = (255, 153, 51)
                summation_car_length_in_pixel += trackers_dict[tracker_counter].start_bbox[2]
                tracker_counter += 1
                car_counter += 1
                total_counter += 1
    detection_counter = 0

    # track the current trackers for "detect_for_frames" times and then run YOLO again.
    while detection_counter <= detect_for_frames:

        # failsafe to delete trackers that go to edge in first and last "bezel_boundary" pixels or this
        # tracker has failed for "max_fail_counter" frames
        for k in list(trackers_dict.keys()):
            if trackers_dict[k].fail_counter >= max_fail_counter or trackers_dict[k].bbox[0] < bezel_boundary or \
                            trackers_dict[k].bbox[0] > image.shape[1] - bezel_boundary:
                trackers_dict[k].end_frame = frame_counter
                trackers_dict[k].end_bbox = trackers_dict[k].bbox
                trackers_dict[k].total_time = (trackers_dict[k].end_frame - trackers_dict[k].start_frame) / fps
                summation_time += trackers_dict[k].total_time

                x1 = trackers_dict[k].start_bbox[0]
                y1 = trackers_dict[k].start_bbox[1]
                x2 = trackers_dict[k].end_bbox[0]
                y2 = trackers_dict[k].end_bbox[1]
                # distance between coordinates of first and last bbox
                trackers_dict[k].total_distance_in_pixel = math.hypot(x2 - x1, y2 - y1)
                summation_distance += trackers_dict[k].total_distance_in_pixel # calculate total_distance

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
                                  obj.bbox_color, 5)
                    cv2.putText(image, obj.name + " " + str(obj.id) + " " + obj.direction, (int(bbox[0]), int(bbox[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # -----------------
                    # direction finding code

                    if obj.directional_bbox_counter == 6:
                        if obj.last_ten_bbox[1][0] - obj.last_ten_bbox[5][0] < 0:
                            obj.direction = "right"
                            right_traffic_counter += 1
                            if obj.name == 'car':
                                right_car_counter += 1
                            else:
                                right_truck_counter += 1
                        else:
                            obj.direction = "left"
                            left_traffic_counter += 1
                            if obj.name == 'car':
                                left_car_counter += 1
                            else:
                                left_truck_counter += 1
                    else:
                        obj.last_ten_bbox.append(obj.bbox)
                    obj.directional_bbox_counter += 1
                    # -----------------
                else:
                    obj.fail_counter += 1

        # code to find number of seconds after timer_start and at, mod 15*fps (because, print something
        if frame_counter % (15*fps) == 0:
            count_parameters()



        cv2.putText(image, "Total vehicle detected: " + str(total_counter) + "   Vehicles on screen: " +
                    str(tracker_counter) + "     cars: " + str(car_counter) + "   trucks: " + str(truck_counter),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Left total count: " + str(left_traffic_counter) + "   cars: " +
                    str(left_car_counter) + " trucks: " + str(left_truck_counter),
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Right total count: " + str(right_traffic_counter) + "   cars: " +
                    str(right_car_counter) + " trucks: " + str(right_truck_counter),
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Total vehicles: " + str(left_traffic_counter + right_traffic_counter),
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # image = cv2.resize(image, None, fx= 0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("feed", image)
        cv2.imshow("feed2", image2)
        k = cv2.waitKey(1 & 0xff)

        # when user presses SPACE button to pause
        if k == 32:
            while True:
                kk = cv2.waitKey(1 & 0xff)
                if kk == 32:
                    break

        # when user presses 'r' button to reenforce YOLO object detection
        if k == ord('r'):
            break

        # when user presses ESC button to quit execution
        if k == 27:
            capture.release()
            cv2.destroyAllWindows()
            exit()
        detection_counter += 1

        ok, image = capture.read()
        if not ok:
            break
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame_counter += 1

capture.release()
cv2.destroyAllWindows()

# -------------------------
# to count the last values after
count_parameters()
# -------------------------

print("Total vehicle: " + str(total_counter) + " left: " + str(left_traffic_counter) + " right: " +
                    str(right_traffic_counter) +"     cars: " + str(car_counter) + "   trucks: " + str(truck_counter))

# print('{}\n{}\n{}'.format(density_list ,average_travel_speed_list, flow_rate_list))

plt.figure()
plt.subplot(2, 2, 1)
plt.xlabel('Density (veh/mile)')
plt.ylabel('Avg Travel Speed (mi/h)')
plt.xlim(0)
plt.ylim(0)
plt.plot(density_list, average_travel_speed_list, 'go')
plt.title('Speed vs Density')

plt.subplot(2, 2, 2)
plt.xlabel('Flow-rate (veh/h)')
plt.ylabel('Avg Travel Speed (mi/h)')
plt.xlim(0)
plt.ylim(0)
plt.plot(flow_rate_list, average_travel_speed_list, 'go')
plt.title('Flow-rate vs Speed ')

plt.subplot(2, 2, 3)
plt.xlabel('Density (veh/mile)')
plt.ylabel('Flow-rate (veh/h)')
plt.xlim(0)
plt.ylim(0)
plt.plot(density_list, flow_rate_list, 'go')
plt.title('Density vs Flow-rate')

plt.show()