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

# class variables:
#     Coords = 0
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


def intersection(box_a, box_b):
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b

    # Complete overlap
    if (((x1a <= x1b <= x2a) and (y1a <= y1b <= y2a)) and ((x1a <= x2b <= x2a) and (y1a <= y2b <= y2a))) or \
            (((x1b <= x1a <= x2b) and (y1b <= y1a <= y2b)) and ((x1b <= x2a <= x2b) and (y1b <= y2a <= y2b))):
        return 1

    # No overlap
    if not (((((x1a <= x1b <= x2a) and (y1a <= y1b <= y2a)) or ((x1a <= x2b <= x2a) and (y1a <= y2b <= y2a))) or
                 (((x1a <= x2b <= x2a) and (y1a <= y1b <= y2a)) or ((x1a <= x1b <= x2a) and (y1a <= y2b <= y2a)))) or
                ((((x1b <= x1a <= x2b) and (y1b <= y1a <= y2b)) or ((x1b <= x2a <= x2b) and (y1b <= y2a <= y2b))) or
                     (((x1b <= x2a <= x2b) and (y1b <= y1a <= y2b)) or ((x1b <= x1a <= x2b) and (y1b <= y2a <= y2b))))):
        return -1

    x_a = max(x1a, x1b)
    y_a = max(y1a, y1b)
    x_b = min(x2a, x2b)
    y_b = min(y2a, y2b)

    inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)

    box_a_area = (x2a - x1a + 1) * (y2a - y1a + 1)
    box_b_area = (x2b - x1b + 1) * (y2b - y1b + 1)
    return inter_area / float(box_a_area + box_b_area - inter_area)


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


def count_parameters():
    global left_summation_distance, right_summation_distance, \
        left_summation_time, right_summation_time, left_last_total_counter, right_last_total_counter, fps
    current_seconds = frame_counter / fps

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
    right_flow_rate_15min = right_quarter_count * 15
    right_average_travel_speed_in_pixel = right_summation_distance / right_summation_time
    right_average_car_length_in_pixel = (right_summation_car_length_in_pixel / right_traffic_counter)
    right_average_travel_speed_meters = (right_average_travel_speed_in_pixel / right_average_car_length_in_pixel) * \
                                        length_of_car_in_meter
    right_average_travel_speed = (right_average_travel_speed_meters * 3600) / (1609.344)
    right_density = right_flow_rate / right_average_travel_speed
    right_density_15min = right_flow_rate_15min / right_average_travel_speed

    # parameters lists for right bound traffic
    right_density_list.append(right_density)
    right_average_travel_speed_list.append(right_average_travel_speed)
    right_flow_rate_list.append(right_flow_rate)

    right_density_list_15min.append(right_density_15min)
    right_flow_rate_list_15min.append(right_flow_rate_15min)


    print('After {} seconds: '.format(current_seconds))
    print('Flow rate {} veh/15min'.format(right_flow_rate_15min))
    # print('Density {} veh/mi/15min | {} veh/mi'.format(round(right_density_15min, 2), round(right_density, 2)))
    # print('Flow rate {} veh/15min | {}veh/hour'.format(right_flow_rate_15min, right_flow_rate))
    print('Average Travel Speed (mi/h) Right: {}'.format(round(right_average_travel_speed, 2)))
    # print('Density {} veh/mi/15min | {} veh/mi'.format(round(right_density_15min, 2), round(right_density, 2)))


    # for writing to CSV
    traffic_parameters.append((file_name, current_seconds, right_flow_rate, right_average_travel_speed, right_density))
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


def create_new_trackers(results):
    for item in results:
        # to consider only trucks and cars
        # if item['label'] != 'truck' and item['label'] != 'car' and \
        #         (item['topleft']['x'] < bezel_boundary or item['topleft']['x'] > image.shape[1] - bezel_boundary):
        #     continue
        original_bounding_box = (item['topleft']['x'], item['topleft']['y'], (item['bottomright']['x'] - item['topleft']['x']),
                        (item['bottomright']['y'] - item['topleft']['y']))

        bbox_l = (item['bottomright']['x'] - item['topleft']['x'])
        bbox_w = (item['bottomright']['y'] - item['topleft']['y'])
        x_off = bbox_l * 0.135
        y_off = bbox_w * 0.08
        w_off = bbox_l * 0.29
        h_off = bbox_w * 0.2

        altered_bounding_box = (item['topleft']['x'] + x_off, item['topleft']['y'] + y_off, (item['bottomright']['x'] - item['topleft']['x'] - w_off),
                        (item['bottomright']['y'] - item['topleft']['y'] - h_off))

        # calculate max overlap of this bounding box with all active trackers
        max_overlap = 0
        for v in active_trackers_list:
            bbox_param1 = (
                altered_bounding_box[0], altered_bounding_box[1], (altered_bounding_box[0] + altered_bounding_box[2]),
                (altered_bounding_box[1] + altered_bounding_box[3]))
            bbox_param2 = (v.bbox[0], v.bbox[1], (v.bbox[0] + v.bbox[2]), (v.bbox[1] + v.bbox[3]))
            temp_overlap = get_overlap(bbox_param1, bbox_param2)
            if temp_overlap > max_overlap:
                max_overlap = temp_overlap
        # if overlap is greater than threshold, don't create new object
        if max_overlap >= min_overlap and frame_counter > 1:
            continue
        active_trackers_list.append(DetectedObject("Vehicle", cv2.TrackerMedianFlow_create(), (0, 0, 255), max_overlap))
        active_trackers_list[-1].tracker.init(image, altered_bounding_box)
        active_trackers_list[-1].start_bbox = altered_bounding_box
        active_trackers_list[-1].start_frame = frame_counter
        active_trackers_list[-1].original_bbox = original_bounding_box


def update_trackers():
    global right_traffic_counter, left_traffic_counter,\
        left_summation_car_length_in_pixel, right_summation_car_length_in_pixel, Coords, \
        left_summation_car_width_in_pixel, right_summation_car_width_in_pixel, left_summation_truck_length_in_pixel, \
        right_summation_truck_length_in_pixel, left_summation_truck_width_in_pixel, right_summation_truck_width_in_pixel
    for obj in active_trackers_list:
        # updating and printing the tracker only if it's not been failing for max_fail_counter frames
        if obj.fail_counter < max_fail_counter:
            ok, bbox = obj.tracker.update(image)
            if ok:
                obj.bbox = bbox
                cv2.line(image, (bezel_boundary, 0), (bezel_boundary, image.shape[0]), (0, 0, 255), 5)
                cv2.line(image, (image.shape[1] - bezel_boundary, 0), (image.shape[1] - bezel_boundary, image.shape[0]), (0, 0, 255), 5)

                # average car size box in the right bottom corner
                cv2.rectangle(image, (int(image.shape[1]-average_veh_lw[0]), int(image.shape[0]-average_veh_lw[1])),
                              (image.shape[1], image.shape[0]), (0, 200, 0), -1)
                cv2.putText(image, "Average Car Size", (int(image.shape[1]-average_veh_lw[0])+20,
                                                        int(image.shape[0]-average_veh_lw[1])+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, "{} x {}".format(int(average_veh_lw[0]), int(average_veh_lw[1])), (int(image.shape[1] - average_veh_lw[0])+20,
                                                        int(image.shape[0] - average_veh_lw[1])+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255, 255, 255), 1)
                # -------------------------------------------------

                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                              obj.bbox_color, 5)
                cv2.putText(image,
                            '{} {} {} {} mph'.format(obj.direction[0].upper(), obj.name, obj.id,
                                                     round(obj.speed)), (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # cv2.putText(image,
                #             '{} {} {}'.format(obj.direction[0].upper(), obj.name.capitalize(), obj.id,),
                #             (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # -----------------
                # code to find direction
                if obj.directional_bbox_counter == 5:
                    # if vehicle is going left
                    if obj.last_few_bboxes[1][0] - obj.last_few_bboxes[4][0] > 0:
                        obj.direction = "left"
                        left_traffic_counter += 1
                        obj.id = left_traffic_counter
                        obj.bbox_color = (255, 153, 51)
                        left_summation_car_length_in_pixel += obj.start_bbox[2]
                        left_summation_car_width_in_pixel += obj.start_bbox[3]
                    # if vehicle is going right
                    else:
                        obj.direction = "right"
                        right_traffic_counter += 1
                        obj.id = right_traffic_counter
                        obj.bbox_color = (255, 153, 51)
                        right_summation_car_length_in_pixel += obj.start_bbox[2]
                        right_summation_car_width_in_pixel += obj.start_bbox[3]
                    # for lane wise counts
                    all_trackers_list.append(obj)

                    if left_traffic_counter > 0 or right_traffic_counter > 0:
                        average_veh_lw[0] = int((left_summation_car_length_in_pixel + right_summation_car_length_in_pixel) \
                                            / (left_traffic_counter + right_traffic_counter))
                        average_veh_lw[1] = int((left_summation_car_width_in_pixel + right_summation_car_width_in_pixel) \
                                            / (left_traffic_counter + right_traffic_counter))
                        # average_veh_lw[0] = average_veh_lw[0] * 0.575
                        # average_veh_lw[1] = average_veh_lw[1] * 0.72

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


def _delete_this_tracker(obj):
    global left_summation_time, right_summation_time, left_summation_distance, right_summation_distance, fps
    obj.end_bbox = obj.bbox
    obj.end_frame = frame_counter
    obj.total_time = (obj.end_frame - obj.start_frame) / fps

    if obj.direction == 'left':
        left_summation_time += obj.total_time
    if obj.direction == 'right':
        right_summation_time += obj.total_time

    x1 = obj.start_bbox[0]
    y1 = obj.start_bbox[1]
    x2 = obj.end_bbox[0]
    y2 = obj.end_bbox[1]
    # distance between coordinates of first and last bbox
    obj.total_distance_in_pixel = math.hypot(x2 - x1, y2 - y1)
    if obj.direction == 'left':
        left_summation_distance += obj.total_distance_in_pixel  # calculate total_distance
    if obj.direction == 'right':
        right_summation_distance += obj.total_distance_in_pixel  # calculate total_distance
    active_trackers_list.remove(obj)


def delete_trackers():
    # failsafe to delete trackers that go to edge in first and last "bezel_boundary" pixels or this
    # tracker has failed for "max_fail_counter" frames
    global left_summation_time, right_summation_time, left_summation_distance, right_summation_distance, \
        right_traffic_counter, fps, left_traffic_counter, left_summation_car_length_in_pixel, \
        left_summation_car_width_in_pixel, right_summation_car_length_in_pixel, right_summation_car_width_in_pixel

    for obj in active_trackers_list:

        #delete case 1
        if obj.fail_counter >= max_fail_counter or obj.bbox[0] < bezel_boundary or \
                        obj.bbox[0] > image.shape[1] - bezel_boundary:
            print("-------------Deletion------------")
            print('case1 Vehicle {} {} {}'.format(obj.id, obj.direction, obj.bbox[0]))
            _delete_this_tracker(obj)
            continue
        # delete case 2
        # print(fps)
        # print(int(frame_counter % (0.4 * fps)))
        if int(frame_counter % (0.5 * fps)) == 0:
            if (
               (obj.bbox[2] < (1 - outlier_threshold) * average_veh_lw[0]) or
               (obj.bbox[2] > (1 + outlier_threshold) * average_veh_lw[0]) or
               (obj.bbox[3] < (1 - outlier_threshold) * average_veh_lw[1]) or
               (obj.bbox[3] > (1 + outlier_threshold) * average_veh_lw[1])):

                print("-------------Deletion------------")
                print("avg L:W :: {}:{}".format(average_veh_lw[0], average_veh_lw[1]))
                print("veh L:W:: {}:{}".format(int(obj.bbox[2]), int(obj.bbox[3])))
                print("{} {} {} {}".format((obj.bbox[2] < (1 - outlier_threshold) * average_veh_lw[0]),
                (obj.bbox[2] > (1 + outlier_threshold) * average_veh_lw[0]),
                (obj.bbox[3] < (1 - outlier_threshold) * average_veh_lw[1]),
                (obj.bbox[3] > (1 + outlier_threshold) * average_veh_lw[1])))

                if obj.direction == "right":
                    right_traffic_counter -= 1
                    right_summation_car_length_in_pixel = right_summation_car_length_in_pixel - obj.start_bbox[2]
                    right_summation_car_width_in_pixel = right_summation_car_width_in_pixel - obj.bbox[3]
                if obj.direction == "left":
                    left_traffic_counter -= 1
                    left_summation_car_length_in_pixel = left_summation_car_length_in_pixel - obj.start_bbox[2]
                    left_summation_car_width_in_pixel = left_summation_car_width_in_pixel - obj.bbox[3]
                print('case2 Vehicle {} {}'.format(obj.id, obj.direction))
                if obj in active_trackers_list:
                    _delete_this_tracker(obj)
                if obj in all_trackers_list:
                    all_trackers_list.remove(obj)

        # delete case 3 based on speed of each vehicle


# def write_vehicle_parameters_to_csv():
#     with open(vehicle_parameters_csv_filename, 'w') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(vehicle_parameters_columns)
#         writer.writerows(vehicle_parameters)

def write_traffic_parameters_to_csv():
    if os.path.exists(traffic_parameters_csv_filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(traffic_parameters_csv_filename, append_write) as csvfile:
        writer = csv.writer(csvfile)
        if append_write =='w':
            writer.writerow(traffic_parameters_columns)
        writer.writerows(traffic_parameters)


def display_counters_on_screen():
    if left_traffic_counter + right_traffic_counter >1:
        cv2.putText(image, 'Left total count: {}'.format(left_traffic_counter),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, 'Right total count: {}'.format(right_traffic_counter),
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


def display_image():
    cv2.imshow("feed", image)
    if record:
        out.write(image)


def calculate_speed():
    global active_trackers_list, frame_counter, length_of_car_in_meter, right_traffic_counter, left_traffic_counter, fps,left_summation_car_length_in_pixel
    if right_traffic_counter > 1 or left_traffic_counter > 1:
        for obj in active_trackers_list:
            x1 = obj.start_bbox[0]
            y1 = obj.start_bbox[1]
            x2 = obj.bbox[0]
            y2 = obj.bbox[1]
            dist_pixel = (math.hypot(x2 - x1, y2 - y1))
            if obj.direction == 'left':
                dist_meter = dist_pixel / (left_summation_car_length_in_pixel / left_traffic_counter) * length_of_car_in_meter
                time = (frame_counter - obj.start_frame) / fps
                obj.speed = ((dist_meter/time)*3600)/1609.344

            if obj.direction == 'right':
                dist_meter = dist_pixel / (right_summation_car_length_in_pixel / right_traffic_counter) * length_of_car_in_meter
                time = (frame_counter - obj.start_frame) / fps
                obj.speed = ((dist_meter/time)*3600)/1609.344


current_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
# input_video_file = '/media/arjun/backup_ubuntu/videos/Mitchell/thermal/DJI_0002.mp4'
input_video_file = '/media/arjun/backup_ubuntu/videos/thermal_palettes/process/BlackHot_GainHigh.mp4'
dir_path = '/media/arjun/backup_ubuntu/videos/thermal_palettes/'
# input_video_file = dir_path + 'process/' + sys.argv[1]
output_file_name = dir_path + 'out/' + "out_" + \
                   input_video_file.split("/")[-1].split(".")[0] + ".avi"
file_name = input_video_file.split("/")[-1].split(".")[0]
traffic_parameters_csv_filename = dir_path + input_video_file.split("/")[-3] + '.csv'
# vehicle_parameters_csv_filename = '12_vehicle_parameters.csv'

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

average_veh_lw = [0, 0]  # initialised with dummy variable
traffic_parameters = list()
traffic_parameters_columns = list()
traffic_parameters_columns.extend(('file_name','interval', 'right_flow_rate', 'right_average_travel_speed', 'right_density'))

# traffic_parameters_columns.extend(('interval', 'left_average_travel_speed', 'left_flow_rate', 'left_density',
#                                    'right_average_travel_speed', 'right_flow_rate', 'right_density'))
# control variables
detect_for_frames = 7
max_fail_counter = 50
min_overlap = 0.3
fps = int(capture.get(cv2.CAP_PROP_FPS))
# if fps == 0 :
#     exit
length_of_car_in_meter = 3.79
bezel_boundary = 85
outlier_threshold = 0.45
options = {'model': 'cfg/yolo.cfg', 'load': 'weights/yolo.weights', 'threshold': 0.1, 'gpu': 1.0}
active_trackers_list = list()
all_trackers_list = list()
yolo_object_detection = TFNet(options)

print('--------------------------------------------------------')
while capture.isOpened():
    ok, image = read_new_image()
    if not ok:
        print("Could not read file")
        break
    frame_counter += 1
    if frame_counter == 1 and record:
        out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (image.shape[1], image.shape[0]))
    # run yolo to detect objects in this frame
    try:
        result = yolo_object_detection.return_predict(image)
    except Exception:
        print("Error in yolo object detection")
    # showing output of yolo only in new window
    # image2 = copy.deepcopy(image)
    # for temp_item in result:
    #     cv2.rectangle(image2, (temp_item['topleft']['x'], temp_item['topleft']['y']),
    #                   ((temp_item['bottomright']['x']), (temp_item['bottomright']['y'])), (0, 0, 255), 5)
    # cv2.imshow("feed", image2)
    # -----------------------------------------
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
        # if left_traffic_counter + right_traffic_counter > 2:
        #     try:
        #         lane_wise_count()
        #     except Exception:
        #         print("Error in lane_wise_count")
        # code to find number of seconds after timer_start and at, mod 15*fps (because, print something
        # if frame_counter % (60 * fps) == 0:
        #     try:
        #         count_parameters()
        #     except Exception:
        #         print("Error in count_parameters")
        try:
            display_counters_on_screen()
        except Exception:
            print("Error in display_counters_on_screen")
        if frame_counter > 4:
            try:
                calculate_speed()
            except Exception:
                print("Error in calculate_speed")
        try:
            display_image()
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
# try:
#     count_parameters()
# except Exception:
#     print("Error in count_parameters")
try:
    capture.release()
    if record:
        out.release()
    cv2.destroyAllWindows()
except Exception:
    print("Error in releases")

# print("Final values:")
# print('Average Flowrate: {} veh/15min | {} veh/hr'.format(np.mean(right_flow_rate_list_15min),
#                                                           np.mean(right_flow_rate_list)))
# print('Average Density: {} veh/mi/15min  | {} veh/mi'.format(round(np.mean(right_density_list_15min), 2),
#                                                              round(np.mean(right_density_list), 2)))
# print('Average of average travelling speed: {}mi/hr'.format(round(np.mean(right_average_travel_speed_list), 2)))
print("--------------------------------------------------------")
print('Total vehicle: {} \nLeft total: {}\n'
      'Right total: {}'
      .format(left_traffic_counter + right_traffic_counter, left_traffic_counter,
              right_traffic_counter))
print("--------------------------------------------------------")
print('{}: {} | {}:{}'.format(cluster[2], cluster[0], cluster[3], cluster[1]))
print("--------------------------------------------------------")
# try:
#     write_traffic_parameters_to_csv()
# except Exception:
#     print("Error in CSV_writer")
# write_vehicle_parameters_to_csv()
# plot_data()
