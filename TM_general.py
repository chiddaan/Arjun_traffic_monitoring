from darkflow.net.build import TFNet
import cv2
import copy
import math
import matplotlib.pyplot as plt
import csv
import time
import datetime
import sys
from sklearn.cluster import KMeans
import numpy as np
import collections
import os
import pysrt as py
import configparser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class DetectedObject:
    def __init__(self, name, tracker, bbox_color, max_overlap):
        self.name = name  # label like car, truck or bus
        self.tracker = tracker  # tracker object to track movements
        self.fail_counter = 0  # number of frames the tracker has been failing
        self.speed = 0  # speed in mph of the object on screen
        self.id = -1  # id_counter
        self.original_bbox = [250, 1, 1, 1]
        self.bbox = [250, 1, 1, 1]  # co-ordinates of the bounding box
        self.bbox_color = bbox_color
        self.start_bbox = [250, 1, 1, 1]
        self.end_bbox = [250, 1, 1, 1]
        self.total_distance_in_pixel = 0
        self.start_frame = 0
        self.end_frame = 0
        self.total_time = 0
        self.directional_bbox_counter = 1
        self.overlap = max_overlap
        self.direction = "dir"  # direction in which the object is moving
        self.last_few_bboxes = [[450, 1, 1, 1]]  # coordinates of last few locations of the object to find direction
        self.bbox_counter = 1  # counter to state how many bounding boxes have been recorded; this value will be
        #                        controlled from 1 to 5 using modular 5


# funtions
def get_overlap(box_a, box_b):
    # box_a --> x1a, y1a, x2a, y2a
    x1a, y1a, x2a, y2a = box_a[0], box_a[1], box_a[2], box_a[3]
    # box_b --> x1b, y1b, x2b, y2b
    x1b, y1b, x2b, y2b = box_b[0], box_b[1], box_b[2], box_b[3]
    overlap_1 = -1
    # If there's no intersection between box A and B, then return 0
    if not (((((x1a < x1b < x2a) and (y1a < y1b < y2a)) or ((x1a < x2b < x2a) and (y1a < y2b < y2a))) or
                 (((x1a < x2b < x2a) and (y1a < y1b < y2a)) or ((x1a < x1b < x2a) and (y1a < y2b < y2a)))) or
                ((((x1b < x1a < x2b) and (y1b < y1a < y2b)) or ((x1b < x2a < x2b) and (y1b < y2a < y2b))) or
                     (((x1b < x2a < x2b) and (y1b < y1a < y2b)) or ((x1b < x1a < x2b) and (y1b < y2a < y2b))))):
        overlap_1 = -1

    # ---------- B inside A or A inside B
    if (x1b >= x1a and x2b <= x2a and y1b >= y1a and y2b <= y2a) or \
            (x1a >= x1b and x2a <= x2b and y1a >= y1b and y2a >= y2b):
        return 1
    # ----------

    x_a = max(x1a, x1b)
    y_a = max(y1a, y1b)
    x_b = min(x2a, x2b)
    y_b = min(y2a, y2b)
    inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)
    box_a_area = (x2a - x1a + 1) * (y2a - y1a + 1)
    box_b_area = (x2b - x1b + 1) * (y2b - y1b + 1)
    overlap_2 = inter_area / float(box_a_area + box_b_area - inter_area)
    return max(overlap_1, overlap_2)


def read_new_image():
    ok, new_image = capture.read()
    if ok:
        new_image = cv2.resize(new_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return ok, new_image


def count_parameters():
    global left_summation_distance, right_summation_distance, \
        left_summation_time, right_summation_time, left_last_total_counter, right_last_total_counter, fps, \
        current_timestamp, old_ramp_count, old_interstate_count, extract_parameters_after_this_seconds
    # in the future, make sifferent functions for left side and right side.
    current_seconds = frame_counter / fps
    extrapolation_rate = 3600 / extract_parameters_after_this_seconds

    # Left side parameters count
    left_quarter_count = left_traffic_counter - left_last_total_counter
    left_flow_rate = left_quarter_count * extrapolation_rate
    left_density = 0
    left_average_travel_speed = 0
    # To avoid divide by zero exception
    if left_summation_time > 0 and left_average_car_lw[0] > 0:
        left_average_travel_speed_in_pixel = left_summation_distance / left_summation_time
        left_average_travel_speed_meters = (left_average_travel_speed_in_pixel / left_average_car_lw[0]) * \
                                           length_of_car_in_meter
        left_average_travel_speed = left_average_travel_speed_meters * 2.237
        left_density = left_flow_rate / left_average_travel_speed

    # Right side parameters count
    right_quarter_count = right_traffic_counter - right_last_total_counter
    right_flow_rate = right_quarter_count * extrapolation_rate
    right_density = 0
    right_average_travel_speed = 0
    if right_summation_time > 0 and right_average_car_lw[0] > 0:
        right_average_travel_speed_in_pixel = right_summation_distance / right_summation_time
        right_average_travel_speed_meters = (right_average_travel_speed_in_pixel / right_average_car_lw[0]) * \
                                            length_of_car_in_meter
        right_average_travel_speed = right_average_travel_speed_meters * 2.237
        right_density = right_flow_rate / right_average_travel_speed

    # for writing to CSV
    traffic_parameters.append((file_name, current_timestamp, left_car_counter, left_truck_counter, left_traffic_counter,
                               right_car_counter, right_truck_counter, right_traffic_counter,
                               left_flow_rate, round(left_average_travel_speed), round(left_density),
                               right_flow_rate, round(right_average_travel_speed), round(right_density)))
    print('--------------------------------------------------------')
    print('After {} seconds: '.format(current_seconds))
    print('Left Flow rate                : {} veh/hr'.format(left_flow_rate))
    print('Left Average Travel Speed     : {}'.format(round(left_average_travel_speed, 2)))
    print('Left Density                  : {} veh/mi'.format(round(left_density, 2)))
    print('Right Flow rate               : {} veh/hr'.format(right_flow_rate))
    print('Right Average Travel Speed    : {}'.format(round(right_average_travel_speed, 2)))
    print('Right Density                 : {} veh/mi'.format(round(right_density, 2)))

    left_summation_time = 1
    left_summation_distance = 1

    right_summation_time = 1
    right_summation_distance = 1

    left_last_total_counter = left_traffic_counter
    right_last_total_counter = right_traffic_counter


def create_new_trackers(results):
    for item in results:
        # to consider only trucks and cars
        if item['label'] != 'truck' and item['label'] != 'car' and \
                (item['topleft']['x'] < bezel_boundary or item['topleft']['x'] > image.shape[1] - bezel_boundary):
            continue
        original_bounding_box = (
            item['topleft']['x'], item['topleft']['y'], (item['bottomright']['x'] - item['topleft']['x']),
            (item['bottomright']['y'] - item['topleft']['y']))
        bbox_w = (item['bottomright']['x'] - item['topleft']['x'])
        bbox_h = (item['bottomright']['y'] - item['topleft']['y'])
        x_off = bbox_w * 0.135
        y_off = bbox_h * 0.08
        w_off = bbox_w * 0.29
        h_off = bbox_h * 0.2

        bounding_box = (item['topleft']['x'] + x_off, item['topleft']['y'] + y_off,
                        (item['bottomright']['x'] - item['topleft']['x'] - w_off),
                        (item['bottomright']['y'] - item['topleft']['y'] - h_off))
        # calculate max overlap of this bounding box with all active trackers
        max_overlap = 0
        for v in active_trackers_list:
            bbox_param1 = (
                bounding_box[0], bounding_box[1], (bounding_box[0] + bounding_box[2]),
                (bounding_box[1] + bounding_box[3]))
            bbox_param2 = (v.bbox[0], v.bbox[1], (v.bbox[0] + v.bbox[2]), (v.bbox[1] + v.bbox[3]))
            temp_overlap = get_overlap(bbox_param1, bbox_param2)
            if temp_overlap > max_overlap:
                max_overlap = temp_overlap
        # if overlap is greater than threshold, don't create new object
        if max_overlap >= min_overlap and frame_counter > 1:
            continue

        # if it's a truck, create new tracker for it.
        if item['label'] == 'truck':
            active_trackers_list.append(DetectedObject(item['label'], cv2.TrackerMedianFlow_create(),
                                                       (0, 0, 255), max_overlap))
            active_trackers_list[-1].tracker.init(image, bounding_box)
            active_trackers_list[-1].start_bbox = bounding_box
            active_trackers_list[-1].start_frame = frame_counter
            active_trackers_list[-1].original_bbox = original_bounding_box

        # if it's a car, create new tracker for it.
        if item['label'] == 'car':
            active_trackers_list.append(DetectedObject(item['label'], cv2.TrackerMedianFlow_create(),
                                                       (0, 0, 255), max_overlap))
            active_trackers_list[-1].tracker.init(image, bounding_box)
            active_trackers_list[-1].start_bbox = bounding_box
            active_trackers_list[-1].start_frame = frame_counter
            active_trackers_list[-1].original_bbox = original_bounding_box


def update_trackers():
    global right_traffic_counter, right_car_counter, right_truck_counter, left_traffic_counter, left_truck_counter, \
        left_car_counter, left_summation_car_length_in_pixel, right_summation_car_length_in_pixel, Coords, \
        left_summation_car_width_in_pixel, right_summation_car_width_in_pixel, left_summation_truck_length_in_pixel, \
        right_summation_truck_length_in_pixel, left_summation_truck_width_in_pixel, right_summation_truck_width_in_pixel, bezel_boundary
    for obj in active_trackers_list:
        # updating and printing the tracker only if it's not been failing for max_fail_counter frames
        if obj.fail_counter < max_fail_counter:
            ok, bbox = obj.tracker.update(image)
            if ok:
                obj.bbox = bbox
                cv2.line(image, (bezel_boundary, 0), (bezel_boundary, image.shape[0]), (0, 0, 255), 5)
                cv2.line(image, (image.shape[1] - bezel_boundary, 0), (image.shape[1] - bezel_boundary, image.shape[0]),
                         (0, 0, 255), 5)
                # -------------------------------------------------
                # Right average car size box in the right bottom corner
                cv2.rectangle(image, (
                int(image.shape[1] - right_average_car_lw[0]), int(image.shape[0] - right_average_car_lw[1])),
                              (image.shape[1], image.shape[0]),
                              (255, 153, 51), -1)
                cv2.putText(image, "R AvgCar", (
                    int(image.shape[1] - right_average_car_lw[0]) + 20,
                    int(image.shape[0] - right_average_car_lw[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, "{} x {}".format(int(right_average_car_lw[0]), int(right_average_car_lw[1])),
                            (int(image.shape[1] - right_average_car_lw[0]) + 20,
                             int(image.shape[0] - right_average_car_lw[1]) + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # -------------------------------------------------

                # # Left average car size box in the right bottom corner
                cv2.rectangle(image, (int(image.shape[1] - left_average_car_lw[0]), 0),
                              (image.shape[1], int(left_average_car_lw[1])), (255, 153, 51), -1)
                cv2.putText(image, "L AvgCar",
                            (int(image.shape[1] - left_average_car_lw[0] + 20), 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, "{} x {}".format(int(left_average_car_lw[0]), int(left_average_car_lw[1])),
                            (int(image.shape[1] - left_average_car_lw[0] + 20), 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # -------------------------------------------------

                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                              obj.bbox_color, 5)

                cv2.putText(image,
                            '{} {} {} {} mph'.format(obj.direction[0].upper(), obj.name.capitalize(), obj.id,
                                                     round(obj.speed)), (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # -----------------

                # code to find direction
                if obj.directional_bbox_counter == 5:
                    # if vehicle is going left
                    if obj.last_few_bboxes[1][0] - obj.last_few_bboxes[4][0] > 0:
                        obj.direction = "left"
                        left_traffic_counter += 1
                        # if vehicle going left is car
                        if obj.name == 'car':
                            left_car_counter += 1
                            obj.id = left_car_counter
                            obj.bbox_color = (255, 153, 51)
                            left_summation_car_length_in_pixel += obj.start_bbox[2]
                            left_summation_car_width_in_pixel += obj.start_bbox[3]
                        # if vehicle going left is truck
                        else:
                            left_truck_counter += 1
                            obj.id = left_truck_counter
                            obj.bbox_color = (89, 247, 71)
                            left_summation_truck_length_in_pixel += obj.start_bbox[2]
                            left_summation_truck_width_in_pixel += obj.start_bbox[3]
                    # if vehicle is going right
                    else:
                        obj.direction = "right"
                        right_traffic_counter += 1
                        # if vehicle going right is car
                        if obj.name == 'car':
                            right_car_counter += 1
                            obj.id = right_car_counter
                            obj.bbox_color = (255, 153, 51)
                            right_summation_car_length_in_pixel += obj.start_bbox[2]
                            right_summation_car_width_in_pixel += obj.start_bbox[3]
                        # if vehicle going right is truck
                        else:
                            right_truck_counter += 1
                            obj.id = right_truck_counter
                            obj.bbox_color = (89, 247, 71)
                            right_summation_truck_length_in_pixel += obj.start_bbox[2]
                            right_summation_truck_width_in_pixel += obj.start_bbox[3]

                    if right_car_counter > 0:
                        right_average_car_lw[0] = right_summation_car_length_in_pixel / right_car_counter
                        right_average_car_lw[1] = right_summation_car_width_in_pixel / right_car_counter

                    if left_car_counter > 0:
                        left_average_car_lw[0] = left_summation_car_length_in_pixel / left_car_counter
                        left_average_car_lw[1] = left_summation_car_width_in_pixel / left_car_counter
                else:
                    obj.last_few_bboxes.append(obj.bbox)
                obj.directional_bbox_counter += 1
                # -----------------
            else:
                obj.fail_counter += 1


def _delete_this_tracker(k):
    global left_summation_time, right_summation_time, left_summation_distance, right_summation_distance, fps
    k.end_bbox = k.bbox
    k.end_frame = frame_counter
    k.total_time = (k.end_frame - k.start_frame) / fps

    if k.direction == 'left':
        left_summation_time += k.total_time
    if k.direction == 'right':
        right_summation_time += k.total_time

    x1 = k.start_bbox[0]
    y1 = k.start_bbox[1]
    x2 = k.end_bbox[0]
    y2 = k.end_bbox[1]
    # distance between coordinates of first and last bbox
    k.total_distance_in_pixel = math.hypot(x2 - x1, y2 - y1)
    if k.direction == 'left':
        left_summation_distance += k.total_distance_in_pixel  # calculate total_distance
    if k.direction == 'right':
        right_summation_distance += k.total_distance_in_pixel  # calculate total_distance
    active_trackers_list.remove(k)


def read_caption_file():
    global first_time_stamp
    try:
        subs = py.open(caption_file_path)
    except Exception:
        print("[ERROR] Cannot read caption/subtitle file. Exiting.")
        exit()

    subs_text = subs[0].text
    subs_list = subs_text.split()
    d = subs_list[1]
    t = subs_list[2]
    string_datetime = d + '-' + t
    return (datetime.datetime.strptime(string_datetime, '%Y.%m.%d-%H:%M:%S'))


def delete_trackers():
    # failsafe to delete trackers that go to edge in first and last "bezel_boundary" pixels or this
    # tracker has failed for "max_fail_counter" frames
    global left_summation_time, right_summation_time, left_summation_distance, right_summation_distance, \
        right_car_counter, right_traffic_counter, fps, right_truck_counter, right_summation_car_length_in_pixel, \
        right_summation_car_width_in_pixel, left_traffic_counter, left_summation_car_length_in_pixel, \
        left_summation_car_width_in_pixel, left_truck_counter, left_summation_truck_length_in_pixel, \
        left_summation_truck_width_in_pixel, right_summation_truck_length_in_pixel, \
        right_summation_truck_width_in_pixel, left_car_counter

    for k in active_trackers_list:
        # delete case 1
        if k.fail_counter >= max_fail_counter or k.bbox[0] < bezel_boundary or \
                        k.bbox[0] > image.shape[1] - bezel_boundary:
            _delete_this_tracker(k)
            continue
        # delete case 2
        if int(frame_counter % (0.5 * fps)) == 0:
            if k.name == "car" and k.direction == "right" and (
                                (k.bbox[2] < (1 - 0.5) * right_average_car_lw[0]) or
                                (k.bbox[2] > (1 + outlier_threshold) * right_average_car_lw[0]) or
                            (k.bbox[3] < (1 - outlier_threshold) * right_average_car_lw[1]) or
                        (k.bbox[3] > (1 + outlier_threshold) * right_average_car_lw[1])):
                if k.id is not -1:
                    print("[LOG] " + get_timestamp() + " : -------------Deletion Right------------")
                    print("[LOG] " + get_timestamp() + " : Car {} going {} was deleted because its".format(k.id, k.direction))
                    if (k.bbox[2] < (1 - outlier_threshold) * right_average_car_lw[0]):
                        print("length became shorter than the threshold")
                    if (k.bbox[2] > (1 + outlier_threshold) * right_average_car_lw[0]):
                        print("length became greater than the threshold")
                    if (k.bbox[3] < (1 - outlier_threshold) * right_average_car_lw[1]):
                        print("height became shorter than the threshold")
                    if (k.bbox[3] > (1 + outlier_threshold) * right_average_car_lw[1]):
                        print("height became greater than the threshold")

                    print("[LOG] " + get_timestamp() + " : avg L:H :: {}:{}".format(int(right_average_car_lw[0]), int(right_average_car_lw[1])))
                    print("[LOG] " + get_timestamp() + " : veh L:H :: {}:{}".format(int(k.bbox[2]), int(k.bbox[3])))

                right_car_counter -= 1
                right_traffic_counter -= 1
                right_summation_car_length_in_pixel = right_summation_car_length_in_pixel - k.start_bbox[2]
                right_summation_car_width_in_pixel = right_summation_car_width_in_pixel - k.start_bbox[3]

                if k in active_trackers_list:
                    _delete_this_tracker(k)

            if k.name == "car" and k.direction == "left" and (
                                (k.bbox[2] < (1 - 0.5) * left_average_car_lw[0]) or
                                (k.bbox[2] > (1 + outlier_threshold) * left_average_car_lw[0]) or
                            (k.bbox[3] < (1 - outlier_threshold) * left_average_car_lw[1]) or
                        (k.bbox[3] > (1 + outlier_threshold) * left_average_car_lw[1])):
                if k.id is not -1:
                    print("[LOG] " + get_timestamp() + " : -------------Deletion Left------------")
                    print("[LOG] " + get_timestamp() + " : Car {} going {} was deleted because its".format(k.id, k.direction))
                    if (k.bbox[2] < (1 - outlier_threshold) * left_average_car_lw[0]):
                        print("length became shorter than the threshold")
                    if (k.bbox[2] > (1 + outlier_threshold) * left_average_car_lw[0]):
                        print("length became greater than the threshold")
                    if (k.bbox[3] < (1 - outlier_threshold) * left_average_car_lw[1]):
                        print("height became shorter than the threshold")
                    if (k.bbox[3] > (1 + outlier_threshold) * left_average_car_lw[1]):
                        print("height became greater than the threshold")

                    print("[LOG] " + get_timestamp() + " : avg L:H :: {}:{}".format(int(left_average_car_lw[0]), int(left_average_car_lw[1])))
                    print("[LOG] " + get_timestamp() + " : veh L:H :: {}:{}".format(int(k.bbox[2]), int(k.bbox[3])))

                left_car_counter -= 1
                left_traffic_counter -= 1
                left_summation_car_length_in_pixel = left_summation_car_length_in_pixel - k.start_bbox[2]
                left_summation_car_width_in_pixel = left_summation_car_width_in_pixel - k.start_bbox[3]

                if k in active_trackers_list:
                    _delete_this_tracker(k)


def write_traffic_parameters_to_csv():
    if os.path.exists(traffic_parameters_csv_filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(traffic_parameters_csv_filename, append_write) as csvfile:
        writer = csv.writer(csvfile)
        if append_write == 'w':
            writer.writerow(traffic_parameters_columns)
        writer.writerows(traffic_parameters)


def display_counters_on_screen():
    cv2.putText(image, 'Left total count: {} | Cars: {} | Trucks: {}'.
                format(left_traffic_counter, left_car_counter, left_truck_counter, right_car_counter),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(image, 'Right total count: {} | Cars: {} | Trucks: {}'.
                format(right_traffic_counter, right_car_counter, right_truck_counter),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, 'Total vehicles: {}'.format(left_traffic_counter + right_traffic_counter),
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def check_for_user_input():
    k = cv2.waitKey(1 & 0xff)
    # when user presses SPACE button to pause/play
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


def display_or_write_image():
    cv2.imshow("feed", image)
    if record:
        out.write(image)


def calculate_speed():
    global active_trackers_list, frame_counter, length_of_car_in_meter, right_car_counter, left_car_counter, fps

    for obj in active_trackers_list:
        x1 = obj.start_bbox[0]
        y1 = obj.start_bbox[1]
        x2 = obj.bbox[0]
        y2 = obj.bbox[1]
        dist_pixel = (math.hypot(x2 - x1, y2 - y1))
        if obj.direction == 'left' and left_car_counter > 1:
            dist_meter = dist_pixel / (left_summation_car_length_in_pixel / left_car_counter) * length_of_car_in_meter
            time = (frame_counter - obj.start_frame) / fps
            obj.speed = ((dist_meter / time) * 3600) / 1609.344
        if right_car_counter > 1 and obj.direction == 'right':
            dist_meter = dist_pixel / (right_summation_car_length_in_pixel / right_car_counter) * length_of_car_in_meter
            time = (frame_counter - obj.start_frame) / fps
            obj.speed = ((dist_meter / time) * 3600) / 1609.344


def read_configuration_file():
    config = configparser.ConfigParser()
    config.read('config.ini')
    detect_for_frames = int(config['DEFAULT']['detect_for_frames'])
    max_fail_counter = int(config['DEFAULT']['max_fail_counter'])
    min_overlap = float(config['DEFAULT']['min_overlap'])
    length_of_car_in_meter = float(config['DEFAULT']['length_of_car_in_meter'])
    outlier_threshold = float(config['DEFAULT']['outlier_threshold'])
    extract_parameters_after_this_seconds = int(config['DEFAULT']['extract_parameters_after_this_seconds'])
    model = config['DEFAULT']['model']
    load = config['DEFAULT']['load']
    threshold = float(config['DEFAULT']['threshold'])
    gpu = float(config['DEFAULT']['gpu'])
    vid_record = bool(config['DEFAULT']['record'])
    return detect_for_frames, max_fail_counter, min_overlap, length_of_car_in_meter, outlier_threshold, \
           extract_parameters_after_this_seconds, model, load, threshold, gpu, vid_record


def generate_report():
    leftCoords = np.array([])
    rightCoords = np.array([])
    file_handler = open(report_filename, "w+")

    with open(traffic_parameters_csv_filename, newline='') as myFile:
        reader = csv.reader(myFile)
        counter = 0
        for row in reader:
            if counter == 0:
                counter = 1
                continue
            leftCoords = np.append(leftCoords, np.array([row[10], row[9]]))
            rightCoords = np.append(rightCoords, np.array([row[13], row[12]]))

    left_points = leftCoords.reshape(int(leftCoords.size / 2), 2)
    right_points = rightCoords.reshape(int(rightCoords.size / 2), 2)
    if len(left_points) <= 1 or len(right_points) <= 1:
        file_handler.write("Not enough data points. There should be at least two data points")
        file_handler.close()
        exit()
    left_kmeans = KMeans(n_clusters=2, random_state=0).fit(left_points)
    right_kmeans = KMeans(n_clusters=2, random_state=0).fit(right_points)
    left_clusters = collections.Counter(left_kmeans.labels_)
    right_clusters = collections.Counter(right_kmeans.labels_)
    left_cluster_centers = left_kmeans.cluster_centers_
    right_cluster_centers = right_kmeans.cluster_centers_
    if left_cluster_centers[0][0] < left_cluster_centers[1][0]:
        left_undersaturated_label = 0
        left_oversaturated_label = 1
    else:
        left_undersaturated_label = 1
        left_oversaturated_label = 0
    left_undersaturated = left_clusters[left_undersaturated_label]
    left_oversaturated = left_clusters[left_oversaturated_label]
    if right_cluster_centers[0][0] < right_cluster_centers[1][0]:
        right_undersaturated_label = 0
        right_oversaturated_label = 1
    else:
        right_undersaturated_label = 1
        right_oversaturated_label = 0
    right_undersaturated = right_clusters[right_undersaturated_label]
    right_oversaturated = right_clusters[right_oversaturated_label]
    left_undersaturated_percentage = round(left_undersaturated / (left_undersaturated + left_oversaturated) * 100)
    left_oversaturated_percentage = round(left_oversaturated / (left_undersaturated + left_oversaturated) * 100)
    right_undersaturated_percentage = round(right_undersaturated / (right_undersaturated + right_oversaturated) * 100)
    right_oversaturated_percentage = round(right_oversaturated / (right_undersaturated + right_oversaturated) * 100)
    file_handler.write("From the traffic going left, \nTotal Time processed : {} minutes \nUndersaturated traffic : {} "
                       "minutes ({}%)\nOversaturated traffic : {} minutes "
                       "({}%)".format(left_undersaturated + left_oversaturated,
                                      left_undersaturated,
                                      left_undersaturated_percentage,
                                      left_oversaturated,
                                      left_oversaturated_percentage))
    file_handler.write(
        "\n\nFrom the traffic going right, \nTotal Time processed : {} minutes \nUndersaturated traffic : {} "
        "minutes ({}%) \nOversaturated traffic : {} minutes ({}%)".format(right_undersaturated + right_oversaturated,
                                                                          right_undersaturated,
                                                                          right_undersaturated_percentage,
                                                                          right_oversaturated,
                                                                          right_oversaturated_percentage))
    file_handler.close()


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


current_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
dir_path = sys.argv[1]
# dir_path = '/media/arjun/backup_ubuntu/videos/Mitchell/jan26/'
input_video_file = dir_path + sys.argv[2]
# input_video_file = dir_path + "sample1.mp4"

if not os.path.exists(dir_path + "out/"):
    os.makedirs(dir_path + "out/")
output_file_name = dir_path + 'out/' + "out_" + \
                   input_video_file.split("/")[-1].split(".")[0] + ".avi"
file_name = input_video_file.split("/")[-1].split(".")[0]
traffic_parameters_csv_filename = dir_path + dir_path.split("/")[-2] + '.csv'
report_filename = dir_path + dir_path.split("/")[-2] + '.txt'
caption_file_path = dir_path + input_video_file.split("/")[-1].split(".")[0] + ".SRT"

out = ''

detection_counter = 0  # counter to enable detection every #frames
frame_counter = 0

lane_count = dict()
left_traffic_counter = 0
left_car_counter = 0
left_truck_counter = 0
left_quarter_count = 0
left_last_total_counter = 0
left_summation_car_length_in_pixel = 0
left_summation_car_width_in_pixel = 0
left_summation_truck_length_in_pixel = 0
left_summation_truck_width_in_pixel = 0
left_summation_time = 0
left_summation_distance = 0

right_traffic_counter = 0
right_car_counter = 0
right_truck_counter = 0
right_quarter_count = 0
right_last_total_counter = 0
right_summation_car_length_in_pixel = 0
right_summation_car_width_in_pixel = 0
right_summation_truck_length_in_pixel = 0
right_summation_truck_width_in_pixel = 0
right_summation_time = 0
right_summation_distance = 0
right_average_car_lw = [0, 0]  # initialised with dummy values
left_average_car_lw = [0, 0]  # initialised with dummy values

active_trackers_list = list()
traffic_parameters = list()
traffic_parameters_columns = list()
traffic_parameters_columns.extend(
    ("File_name", "Time", "Left Cars", "Left Trucks", "Left Total", "Right Cars", "Right Trucks", "Right Total",
     'Left_flow_rate', 'Left_average_travel_speed', 'Left_density', 'Right_flow_rate', 'Right_average_travel_speed',
     'Right_density'))

# control variables extracted from config.ini file
detect_for_frames, max_fail_counter, min_overlap, length_of_car_in_meter, outlier_threshold, \
extract_parameters_after_this_seconds, model, load, threshold, gpu, vid_record = read_configuration_file()

current_timestamp = read_caption_file()
capture = cv2.VideoCapture(input_video_file)
fps = int(capture.get(cv2.CAP_PROP_FPS))
record = vid_record

options = {'model': model, 'load': load, 'threshold': threshold, 'gpu': gpu}
yolo_object_detection = TFNet(options)
ok = True
print('--------------------------------------------------------')
while ok:
    ok, image = read_new_image()
    if not ok:
        print("[LOG] Video file processing complete or Error reading the video file")
        break
    frame_counter += 1
    if frame_counter == 1 and record:
        out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (image.shape[1], image.shape[0]))
    if frame_counter == 1:
        bezel_boundary = round(0.073 * image.shape[1])

    # run yolo to detect objects in this frame
    try:
        result = yolo_object_detection.return_predict(image)
    except Exception:
        print("[ERROR] " + get_timestamp() + " : Error in yolo object detection")
    try:
        create_new_trackers(result)
    except Exception:
        print("[ERROR] " + get_timestamp() + " : Error in create_new_trackers")
    detection_counter = 0
    while detection_counter < detect_for_frames:
        if frame_counter > 1:
            try:
                delete_trackers()
            except Exception:
                print("[ERROR] " + get_timestamp() + " : Error in delete_trackers")
            try:
                update_trackers()
            except Exception:
                print("[ERROR] Error in update_trackers")

        # code to extract parameters after said seconds
        if frame_counter % (extract_parameters_after_this_seconds * fps) == 0:
            try:
                count_parameters()
                # increment current timestamp by "extract_parameters_after_this_seconds" seconds
                current_timestamp = current_timestamp + datetime.timedelta(0, extract_parameters_after_this_seconds)
            except Exception:
                print("[ERROR] " + get_timestamp() + " : Error in count_parameters")
        try:
            display_counters_on_screen()
        except Exception:
            print("[ERROR] " + get_timestamp() + " : Error in display_counters_on_screen")
        if frame_counter > 4:
            try:
                calculate_speed()
            except Exception as e:
                print("[ERROR] " + get_timestamp() + " : Error in calculate_speed  = {}".format(e))
        try:
            display_or_write_image()
        except Exception:
            print("[ERROR] " + get_timestamp() + " : Error in display_image")
        try:
            check_for_user_input()
        except Exception:
            print("[ERROR] " + get_timestamp() + " : Error in check_for_user_input")
        detection_counter += 1
        if detection_counter < detect_for_frames:
            ok, image = read_new_image()
            if not ok:
                break
            frame_counter += 1
try:
    capture.release()
    if record:
        out.release()
    cv2.destroyAllWindows()
except Exception as e:
    print("[ERROR] " + get_timestamp() + " : Error in releases" + e)
print("-----------------------FINAL LOG------------------------")
print('Total vehicle: {} \nLeft total: {} | Left cars: {}| Left trucks: {} \n'
      'Right total: {} | Right cars: {} | Right trucks: {}'
      .format(left_traffic_counter + right_traffic_counter, left_traffic_counter,
              left_car_counter, left_truck_counter, right_traffic_counter, right_car_counter, right_truck_counter))
print("--------------------------------------------------------")
try:
    write_traffic_parameters_to_csv()
except Exception:
    print("[ERROR] " + get_timestamp() + " : Error in CSV_writer")
try:
    generate_report()
except Exception as e:
    print("[ERROR] " + get_timestamp() + " : Error in Report Generator : {}".format(e))
