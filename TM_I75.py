from darkflow.net.build import TFNet
import cv2
import copy
import math
import matplotlib.pyplot as plt
import csv
import time
import datetime
import sys
import numpy as np
from sklearn.cluster import KMeans
import collections
import os
import pysrt as py

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
        # self.end_bbox = [250, 1, 1, 1]
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
        # self.calculate_direction()

    def calculate_direction(self):
        print("insdisaassaasaswe printB", self.name)


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
    # to invert colors
    # new_image = cv2.bitwise_not(new_image)
    # if ok:
    #     new_image = cv2.resize(new_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return ok, new_image


def plot_data():
    max_left_density = max(left_density_list)
    max_right_density = max(right_density_list)
    max_left_flow_rate = max(left_flow_rate_list)
    max_right_flow_rate = max(right_flow_rate_list)
    max_left_speed = max(left_average_travel_speed_list)
    max_right_speed = max(right_average_travel_speed_list)
    max_speed = max(max_left_speed, max_right_speed)
    max_density = max(max_left_density, max_right_density)
    max_flowrate = max(max_left_flow_rate, max_right_flow_rate)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.xlabel('Density (veh/mile)')
    plt.ylabel('Avg Travel Speed (mi/h)')
    plt.xlim(0, max_density + 20)
    plt.ylim(0, max_speed + 15)
    plt.plot(left_density_list, left_average_travel_speed_list, 'go')
    plt.title('Speed vs Density')

    plt.subplot(2, 2, 2)
    plt.xlabel('Flow-rate (veh/h)')
    plt.ylabel('Avg Travel Speed (mi/h)')
    plt.xlim(0, max_flowrate + 1000)
    plt.ylim(0, max_speed + 15)
    plt.plot(left_flow_rate_list, left_average_travel_speed_list, 'go')
    plt.title('Flow-rate vs Speed ')

    plt.subplot(2, 2, 3)
    plt.xlabel('Density (veh/mile)')
    plt.ylabel('Flow-rate (veh/h)')
    plt.xlim(0, max_density + 20)
    plt.ylim(0, max_flowrate + 1000)
    plt.plot(left_density_list, left_flow_rate_list, 'go')
    plt.title('Density vs Flow-rate')

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.xlabel('Density (veh/mile)')
    plt.ylabel('Avg Travel Speed (mi/h)')
    plt.xlim(0, max_density + 20)
    plt.ylim(0, max_speed + 15)
    plt.plot(right_density_list, right_average_travel_speed_list, 'ro')
    plt.title('Speed vs Density')

    plt.subplot(2, 2, 2)
    plt.xlabel('Flow-rate (veh/h)')
    plt.ylabel('Avg Travel Speed (mi/h)')
    plt.xlim(0, max_flowrate + 1000)
    plt.ylim(0, max_speed + 15)
    plt.plot(right_flow_rate_list, right_average_travel_speed_list, 'ro')
    plt.title('Flow-rate vs Speed ')

    plt.subplot(2, 2, 3)
    plt.xlabel('Density (veh/mile)')
    plt.ylabel('Flow-rate (veh/h)')
    plt.xlim(0, max_density + 20)
    plt.ylim(0, max_flowrate + 1000)
    plt.plot(right_density_list, right_flow_rate_list, 'ro')
    plt.title('Density vs Flow-rate')

    plt.show()
def count_parameters_for_left():
    global left_summation_distance, right_summation_distance, \
        left_summation_time, right_summation_time, left_last_total_counter, right_last_total_counter, fps, cluster, \
        current_timestamp, old_ramp_count, old_interstate_count
    # in the future, make sifferent functions for left side and right side.
    current_seconds = frame_counter / fps

    interstate_count = 0
    ramp_count = 0

    # Left side parameters count
    # left_quarter_count = left_traffic_counter - left_last_total_counter
    # left_flow_rate = left_quarter_count * 4 * 60
    # left_average_travel_speed_in_pixel = left_summation_distance / left_summation_time
    # left_average_car_length_in_pixel = (left_summation_car_length_in_pixel / left_car_counter)
    # left_average_travel_speed_meters = (left_average_travel_speed_in_pixel / left_average_car_length_in_pixel) * \
    #                                    length_of_car_in_meter
    # left_average_travel_speed = (left_average_travel_speed_meters * 3600) / (1609.344)
    # left_density = left_flow_rate / left_average_travel_speed

    # parameters lists for left bound traffic
    # left_density_list.append(left_density)
    # left_average_travel_speed_list.append(left_average_travel_speed)
    # left_flow_rate_list.append(left_flow_rate)

    # Right side parameters count
    right_quarter_count = right_traffic_counter - right_last_total_counter
    right_flow_rate = right_quarter_count * 60

    right_average_travel_speed_in_pixel = right_summation_distance / right_summation_time
    right_average_car_length_in_pixel = (right_summation_car_length_in_pixel / right_car_counter)
    right_average_travel_speed_meters = (right_average_travel_speed_in_pixel / right_average_car_length_in_pixel) * \
                                        length_of_car_in_meter
    right_average_travel_speed = (right_average_travel_speed_meters * 3600) / (1609.344)
    right_density = right_flow_rate / right_average_travel_speed

    # parameters lists for right bound traffic
    right_density_list.append(right_density)
    right_average_travel_speed_list.append(right_average_travel_speed)
    right_flow_rate_list.append(right_flow_rate)

    print('After {} seconds: '.format(current_seconds))
    print('Flow rate               : {} veh/hr'.format(right_flow_rate))
    print('Average Travel Speed    : {}'.format(round(right_average_travel_speed, 2)))
    print('Density                 : {} veh/mi'.format(round(right_density, 2)))


    if cluster[2]== "Interstate":
        interstate_count = cluster [0]
        ramp_count = cluster[1]
    elif cluster[2] == "Ramp":
        interstate_count = cluster[1]
        ramp_count = cluster[0]
    # timestamp = get_timestamp(first_time_stamp)
    current_ramp_count = ramp_count - old_ramp_count
    current_interstate_count = interstate_count - old_interstate_count

    # for writing to CSV

    traffic_parameters.append((file_name, current_ramp_count, current_interstate_count,
                               current_interstate_count + current_ramp_count, current_timestamp, right_flow_rate,
                               round(right_average_travel_speed), round(right_density)))



    # traffic_parameters.append((file_name, current_timestamp, ramp_count, interstate_count,
    #                            interstate_count + ramp_count, right_flow_rate,
    #                            round(right_average_travel_speed), round(right_density)))

    # traffic_parameters.append((current_seconds, left_average_travel_speed, left_flow_rate, left_density,
    #                            right_average_travel_speed, right_flow_rate, right_density))

    # print('After {} seconds: '.format(current_seconds))
    # print('Average Travel Speed in pixel (pixels/second) | '
    #       'Left: {} | Right{}'.format(left_average_travel_speed_in_pixel, right_average_travel_speed_in_pixel))
    # print('Quarter count Left: {} | Right: {}'.format(left_quarter_count, right_quarter_count))
    # print('Flow rate (veh/h) Left: {} | Right: {}'.format(left_flow_rate, right_flow_rate))
    # print('Average Travel Speed (mi/h) Left: {} | Right: {}'.format(round(left_average_travel_speed, 2),
    #                                                                 round(right_average_travel_speed, 2)))
    # print('Density (veh/mi): Left: {} | Right: {}'.format(round(left_density, 2), round(right_density, 2)))
    # print('')
    # print('Average car length in pixel Left: {} | Right: {}'.format(left_average_car_length_in_pixel,
    #                                                                 right_average_car_length_in_pixel))
    # print('Total vehicle: {} \nLeft total: {} Left cars: {}| Left trucks: {} \n'
    #       'Right total: {} | Right cars: {} | Right trucks: {}'
    #       .format(left_traffic_counter + right_traffic_counter, left_traffic_counter,
    #               left_car_counter, left_truck_counter, right_traffic_counter,
    #               right_car_counter, right_truck_counter))
    print('--------------------------------------------------------')
    # left_summation_time = 1
    # left_summation_distance = 1
    right_summation_time = 1
    right_summation_distance = 1
    # left_last_total_counter = left_traffic_counter
    right_last_total_counter = right_traffic_counter
    old_ramp_count = ramp_count
    old_interstate_count = interstate_count


def count_parameters():
    global left_summation_distance, right_summation_distance, \
        left_summation_time, right_summation_time, left_last_total_counter, right_last_total_counter, fps, cluster, \
        current_timestamp, old_ramp_count, old_interstate_count
    # in the future, make sifferent functions for left side and right side.
    current_seconds = frame_counter / fps

    interstate_count = 0
    ramp_count = 0

    # Left side parameters count
    # left_quarter_count = left_traffic_counter - left_last_total_counter
    # left_flow_rate = left_quarter_count * 4 * 60
    # left_average_travel_speed_in_pixel = left_summation_distance / left_summation_time
    # left_average_car_length_in_pixel = (left_summation_car_length_in_pixel / left_car_counter)
    # left_average_travel_speed_meters = (left_average_travel_speed_in_pixel / left_average_car_length_in_pixel) * \
    #                                    length_of_car_in_meter
    # left_average_travel_speed = (left_average_travel_speed_meters * 3600) / (1609.344)
    # left_density = left_flow_rate / left_average_travel_speed

    # parameters lists for left bound traffic
    # left_density_list.append(left_density)
    # left_average_travel_speed_list.append(left_average_travel_speed)
    # left_flow_rate_list.append(left_flow_rate)

    # Right side parameters count
    right_quarter_count = right_traffic_counter - right_last_total_counter
    right_flow_rate = right_quarter_count * 60

    right_average_travel_speed_in_pixel = right_summation_distance / right_summation_time
    right_average_car_length_in_pixel = (right_summation_car_length_in_pixel / right_car_counter)
    right_average_travel_speed_meters = (right_average_travel_speed_in_pixel / right_average_car_length_in_pixel) * \
                                        length_of_car_in_meter
    right_average_travel_speed = (right_average_travel_speed_meters * 3600) / (1609.344)
    right_density = right_flow_rate / right_average_travel_speed

    # parameters lists for right bound traffic
    right_density_list.append(right_density)
    right_average_travel_speed_list.append(right_average_travel_speed)
    right_flow_rate_list.append(right_flow_rate)

    print('After {} seconds: '.format(current_seconds))
    print('Flow rate               : {} veh/hr'.format(right_flow_rate))
    print('Average Travel Speed    : {}'.format(round(right_average_travel_speed, 2)))
    print('Density                 : {} veh/mi'.format(round(right_density, 2)))


    if cluster[2]== "Interstate":
        interstate_count = cluster [0]
        ramp_count = cluster[1]
    elif cluster[2] == "Ramp":
        interstate_count = cluster[1]
        ramp_count = cluster[0]
    # timestamp = get_timestamp(first_time_stamp)
    current_ramp_count = ramp_count - old_ramp_count
    current_interstate_count = interstate_count - old_interstate_count

    # for writing to CSV

    traffic_parameters.append((file_name, current_ramp_count, current_interstate_count,
                               current_interstate_count + current_ramp_count, current_timestamp, right_flow_rate,
                               round(right_average_travel_speed), round(right_density)))
    # traffic_parameters.append((file_name, current_timestamp, ramp_count, interstate_count,
    #                            interstate_count + ramp_count, right_flow_rate,
    #                            round(right_average_travel_speed), round(right_density)))

    # traffic_parameters.append((current_seconds, left_average_travel_speed, left_flow_rate, left_density,
    #                            right_average_travel_speed, right_flow_rate, right_density))

    # print('After {} seconds: '.format(current_seconds))
    # print('Average Travel Speed in pixel (pixels/second) | '
    #       'Left: {} | Right{}'.format(left_average_travel_speed_in_pixel, right_average_travel_speed_in_pixel))
    # print('Quarter count Left: {} | Right: {}'.format(left_quarter_count, right_quarter_count))
    # print('Flow rate (veh/h) Left: {} | Right: {}'.format(left_flow_rate, right_flow_rate))
    # print('Average Travel Speed (mi/h) Left: {} | Right: {}'.format(round(left_average_travel_speed, 2),
    #                                                                 round(right_average_travel_speed, 2)))
    # print('Density (veh/mi): Left: {} | Right: {}'.format(round(left_density, 2), round(right_density, 2)))
    # print('')
    # print('Average car length in pixel Left: {} | Right: {}'.format(left_average_car_length_in_pixel,
    #                                                                 right_average_car_length_in_pixel))
    # print('Total vehicle: {} \nLeft total: {} Left cars: {}| Left trucks: {} \n'
    #       'Right total: {} | Right cars: {} | Right trucks: {}'
    #       .format(left_traffic_counter + right_traffic_counter, left_traffic_counter,
    #               left_car_counter, left_truck_counter, right_traffic_counter,
    #               right_car_counter, right_truck_counter))
    print('--------------------------------------------------------')
    # left_summation_time = 1
    # left_summation_distance = 1
    right_summation_time = 1
    right_summation_distance = 1
    # left_last_total_counter = left_traffic_counter
    right_last_total_counter = right_traffic_counter
    old_ramp_count = ramp_count
    old_interstate_count = interstate_count

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
        right_summation_truck_length_in_pixel, left_summation_truck_width_in_pixel, right_summation_truck_width_in_pixel,bezel_boundary
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
                # average car size box in the right bottom corner
                cv2.rectangle(image, (int(image.shape[1] - average_car_lw[0]), int(image.shape[0] - average_car_lw[1])),
                              (image.shape[1], image.shape[0]),
                              (255, 153, 51), -1)
                cv2.putText(image, "Avg car", (
                int(image.shape[1] - average_car_lw[0]) + 20, int(image.shape[0] - average_car_lw[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, "{} x {}".format(int(average_car_lw[0]), int(average_car_lw[1])),
                            (int(image.shape[1] - average_car_lw[0]) + 20,
                             int(image.shape[0] - average_car_lw[1]) + 50),
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
                            # for lane wise counts
                        all_trackers_list.append(obj)

                    if right_car_counter > 0:
                        average_car_lw[0] = right_summation_car_length_in_pixel / right_car_counter
                        average_car_lw[1] = right_summation_car_width_in_pixel / right_car_counter

                    if right_truck_counter > 0:
                        average_truck_lw[0] = right_summation_truck_length_in_pixel / right_truck_counter
                        average_truck_lw[1] = right_summation_truck_width_in_pixel / right_truck_counter
                        # if left_car_counter > 0 or right_car_counter > 0:
                        #     average_car_lw[0] = (left_summation_car_length_in_pixel + right_summation_car_length_in_pixel) \
                        #                         / (left_car_counter + right_car_counter)
                        #     average_car_lw[1] = (left_summation_car_width_in_pixel + right_summation_car_width_in_pixel) \
                        #                         / (left_car_counter + right_car_counter)
                        #
                        # if left_truck_counter > 0 or right_truck_counter > 0:
                        #     average_truck_lw[0] = (
                        #                           left_summation_truck_length_in_pixel + right_summation_truck_length_in_pixel) \
                        #                           / (left_truck_counter + right_truck_counter)
                        #     average_truck_lw[1] = (
                        #                           left_summation_truck_width_in_pixel + right_summation_truck_width_in_pixel) \
                        #                           / (left_truck_counter + right_truck_counter)
                        # # for lane wise counts
                        # all_trackers_list.append(obj)
                else:
                    obj.last_few_bboxes.append(obj.bbox)
                obj.directional_bbox_counter += 1
                # -----------------
            else:
                obj.fail_counter += 1


def lane_wise_count():
    global cluster, lane_count, cluster0, cluster1
    Coords = np.array([])
    cluster_red = (0, 0, 255)
    cluster_green = (0, 255, 0)
    cluster[2] = 'Interstate'
    cluster[3] = 'Ramp'
    for tracker in all_trackers_list:
        Coords = np.append(Coords, np.array([100, tracker.start_bbox[1]]))
    cluster_these_coords = Coords.reshape(int(Coords.size / 2), 2)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_these_coords)
    lane_count = collections.Counter(kmeans.labels_)
    cluster[0], cluster[1] = lane_count[0], lane_count[1]

    if kmeans.cluster_centers_[0][1] > kmeans.cluster_centers_[1][1]:
        cluster_green = (0, 0, 255)
        cluster_red = (0, 255, 0)
        cluster[3] = 'Interstate'
        cluster[2] = 'Ramp'

    # cluster[kmeans.predict([[0, 0]])[0]] -= 1
    # print("Cluster1: {}  |  Cluster2: {}".format(cluster[0], cluster[1]))
    for i in range(0, len(cluster_these_coords)):
        if kmeans.labels_[i] == 0:
            cv2.circle(image, (int(cluster_these_coords[i][0]), int(cluster_these_coords[i][1])), 2, cluster_red, -1)
            cv2.circle(image, (int((kmeans.cluster_centers_[0][0])), int((kmeans.cluster_centers_[0][1]))), 5,
                       cluster_red, -1)
        else:
            cv2.circle(image, (int(cluster_these_coords[i][0]), int(cluster_these_coords[i][1])), 2, cluster_green, -1)
            cv2.circle(image, (int((kmeans.cluster_centers_[1][0])), int((kmeans.cluster_centers_[1][1]))), 5,
                       cluster_green, -1)
    return cluster


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
        print("[ERROR] : Cannot read caption/subtitle file. Exiting.")
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
            if k.name == "car" and (
                                (k.bbox[2] < (1 - 0.5) * average_car_lw[0]) or
                                (k.bbox[2] > (1 + outlier_threshold) * average_car_lw[0]) or
                            (k.bbox[3] < (1 - outlier_threshold) * average_car_lw[1]) or
                        (k.bbox[3] > (1 + outlier_threshold) * average_car_lw[1])):
                if k.id is not -1 and k.direction == "right":
                    print("-------------Deletion------------")
                    print("avg L:W :: {}:{}".format(int(average_car_lw[0]), int(average_car_lw[1])))
                    print("veh L:W :: {}:{}".format(int(k.bbox[2]), int(k.bbox[3])))
                    print("{} {} {} {}".format((k.bbox[2] < (1 - outlier_threshold) * average_car_lw[0]),
                                               (k.bbox[2] > (1 + outlier_threshold) * average_car_lw[0]),
                                               (k.bbox[3] < (1 - outlier_threshold) * average_car_lw[1]),
                                               (k.bbox[3] > (1 + outlier_threshold) * average_car_lw[1])))

                    # if (k.bbox[2] < (1 - outlier_threshold) * average_car_lw[0]):
                    #     print("")
                    # if (k.bbox[2] > (1 + outlier_threshold) * average_car_lw[0]):
                    #     print("")
                    # if (k.bbox[3] < (1 - outlier_threshold) * average_car_lw[1]):
                    #     print("")
                    # if (k.bbox[3] > (1 + outlier_threshold) * average_car_lw[1]):
                    #     print("")


                    print('car {} {}'.format(k.id, k.direction))
                if k.direction == "right":
                    right_car_counter -= 1
                    right_traffic_counter -= 1
                    right_summation_car_length_in_pixel = right_summation_car_length_in_pixel - k.start_bbox[2]
                    right_summation_car_width_in_pixel = right_summation_car_width_in_pixel - k.start_bbox[3]

                if k.direction == "left":
                    left_car_counter -= 1
                    left_traffic_counter -= 1
                    left_summation_car_length_in_pixel = left_summation_car_length_in_pixel - k.start_bbox[2]
                    left_summation_car_width_in_pixel = left_summation_car_width_in_pixel - k.start_bbox[3]

                if k in active_trackers_list:
                    _delete_this_tracker(k)
                if k in all_trackers_list:
                    all_trackers_list.remove(k)


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
    if left_traffic_counter + right_traffic_counter > 1:
        cv2.putText(image, '{} : {} | {} : {}'.format(cluster[2], cluster[0], cluster[3], cluster[1]),
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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

def generate_report():
    # read csv and cluster speed vs density and calculate % of each cluster, print average speed,flowrate and density
    leftCoords = np.array([])
    rightCoords = np.array([])
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
    file_handler = open(report_filename, "w+")
    file_handler.write("From the traffic going left, \nTotal Time processed : {} minutes \nUndersaturated traffic : {} "
                       "minutes ({}%)\nOversaturated traffic : {} minutes ({}%)".format(
        left_undersaturated + left_oversaturated,
        left_undersaturated, left_undersaturated_percentage, left_oversaturated, left_oversaturated_percentage))
    file_handler.write(
        "\n\nFrom the traffic going right, \nTotal Time processed : {} minutes \nUndersaturated traffic : {} "
        "minutes ({}%) \nOversaturated traffic : {} minutes ({}%)".format(right_undersaturated + right_oversaturated,
                                                                          right_undersaturated,
                                                                          right_undersaturated_percentage,
                                                                          right_oversaturated,
                                                                          right_oversaturated_percentage))
    file_handler.close()


current_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
# dir_path = sys.argv[1]
dir_path = '/media/arjun/backup_ubuntu/videos/Mitchell/refine/'
# input_video_file = dir_path + sys.argv[2]
input_video_file = dir_path + "DJI_0001-1.mp4"

if not os.path.exists(dir_path + "out/"):
    os.makedirs(dir_path + "out/")
output_file_name = dir_path + 'out/' + "out_" + \
                   input_video_file.split("/")[-1].split(".")[0] + ".avi"

file_name = input_video_file.split("/")[-1].split(".")[0]

traffic_parameters_csv_filename = dir_path + dir_path.split("/")[-2] + '.csv'
report_filename = dir_path + dir_path.split("/")[-2] + '.txt'

caption_file_path = dir_path + input_video_file.split("/")[-1].split(".")[0] + ".SRT"

out = ''
# record = False
record = True

capture = cv2.VideoCapture(input_video_file)
detection_counter = 0  # counter to enable detection every #frames
frame_counter = 0

cluster = [0, 0, 0, 0]
lane_count = dict()
left_traffic_counter = 0
left_car_counter = 0
left_truck_counter = 0
left_quarter_count = 0
left_average_travel_speed_list = list()
left_flow_rate_list = list()
left_density_list = list()
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
right_average_travel_speed_list = list()
right_flow_rate_list = list()
right_density_list = list()
right_flow_rate_list_15min = list()
right_density_list_15min = list()
right_last_total_counter = 0
right_summation_car_length_in_pixel = 0
right_summation_car_width_in_pixel = 0
right_summation_truck_length_in_pixel = 0
right_summation_truck_width_in_pixel = 0
right_summation_time = 0
right_summation_distance = 0
old_interstate_count = 0
old_ramp_count = 0
average_car_lw = [0, 0]  # initialised with dummy values
average_truck_lw = [0, 0]  # initialised with dummy values
traffic_parameters = list()
traffic_parameters_columns = list()
traffic_parameters_columns.extend(
    ('file_name', "Ramp", "Interstate", "Total", 'Time', 'right_flow_rate',
     'right_average_travel_speed', 'right_density'))
# traffic_parameters_columns.extend(('interval', 'left_average_travel_speed', 'left_flow_rate', 'left_density',
#                                    'right_average_travel_speed', 'right_flow_rate', 'right_density'))
current_timestamp = read_caption_file()

# control variables
detect_for_frames = 10
max_fail_counter = 50
min_overlap = 0.3
fps = int(capture.get(cv2.CAP_PROP_FPS))
length_of_car_in_meter = 3.41
outlier_threshold = 0.65
extract_parameters_after_this_seconds = 60
options = {'model': 'cfg/yolo.cfg', 'load': 'weights/yolo.weights', 'threshold': 0.3, 'gpu': 1.0}
active_trackers_list = list()
all_trackers_list = list()
yolo_object_detection = TFNet(options)

print('--------------------------------------------------------')
while capture.isOpened():
    ok, image = read_new_image()
    if not ok:
        print("Could not read file or this is EOF")
        break
    frame_counter += 1
    if frame_counter == 1 and record:
        out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (image.shape[1], image.shape[0]))
    if frame_counter == 1:
        bezel_boundary = round(0.073 * image.shape[1])
        # print(bezel_boundary)
        # exit()
    # run yolo to detect objects in this frame
    try:
        result = yolo_object_detection.return_predict(image)
    except Exception:
        print("Error in yolo object detection")
    try:
        create_new_trackers(result)
    except Exception:
        print("Error in create_new_trackers")
    detection_counter = 0
    while detection_counter < detect_for_frames:
        if frame_counter > 1:
            try:
                delete_trackers()
            except Exception:
                print("Error in delete_trackers")
            try:
                update_trackers()
            except Exception:
                print("Error in update_trackers")
        if left_traffic_counter + right_traffic_counter > 2:
            try:
                lane_wise_count()
            except Exception:
                print("Error in lane_wise_count")
        # code to find number of seconds after timer_start and at, mod 15*fps (because, print something
        if frame_counter % (extract_parameters_after_this_seconds * fps) == 0:
            try:
                count_parameters()
                # increment current timestamp by "extract_parameters_after_this_seconds" seconds
                current_timestamp = current_timestamp + datetime.timedelta(0, extract_parameters_after_this_seconds)
            except Exception:
                print("Error in count_parameters")
        try:
            display_counters_on_screen()
        except Exception:
            print("Error in display_counters_on_screen")
        if frame_counter > 4:
            try:
                calculate_speed()
            except Exception as e:
                print("Error in calculate_speed  = {}".format(e))
        try:
            display_or_write_image()
        except Exception:
            print("Error in display_image")
        try:
            check_for_user_input()
        except Exception:
            print("Error in check_for_user_input")
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
    print("Error in releases" + e)
print("--------------------------------------------------------")
print('Total vehicle: {} \nLeft total: {} | Left cars: {}| Left trucks: {} \n'
      'Right total: {} | Right cars: {} | Right trucks: {}'
      .format(left_traffic_counter + right_traffic_counter, left_traffic_counter,
              left_car_counter, left_truck_counter, right_traffic_counter, right_car_counter, right_truck_counter))
print("--------------------------------------------------------")
print('{}: {} | {}:{}'.format(cluster[2], cluster[0], cluster[3], cluster[1]))
print("--------------------------------------------------------")
try:
    write_traffic_parameters_to_csv()
except Exception:
    print("Error in CSV_writer")
try:
    generate_report()
except Exception:
    print("Error in CSV_writer")
# plot_data()