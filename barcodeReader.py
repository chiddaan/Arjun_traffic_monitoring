from __future__ import print_function
import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar
from os import listdir
from os.path import isfile, join
import csv
import os
from wand.image import Image
from wand.color import Color


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)

    # # Print results
    # for obj in decodedObjects:
    #     print("Type : " + str(obj.type))
    #     print("Data : " + str(obj.data) + "\n")

    return decodedObjects

# Display barcode and QR code location
def display(im, decodedObjects):
    # Loop over all decoded objects
    out_jpg_filename = "noEAN13Data"
    for decodedObject in decodedObjects:
        # skips all other format thatn ean 13
        if(decodedObject.type is not "EAN13"):
            continue
        out_jpg_filename = decodedObject.data
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        # Number of points in the convex hull
        n = len(hull)

        # Draw the convex hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)
        csv_data.append(
            (jpg_file, decodedObject.type, decodedObject.data))

    # current_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    cv2.imwrite(dir_path + "/out/" + str(out_jpg_filename) + "_1.jpg", im)


    # Display results
    # cv2.imshow("Results", im)
    # cv2.waitKey(0)

def write_to_csv():
    if os.path.exists(csv_filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_filename, append_write) as csvfile:
        writer = csv.writer(csvfile)
        if append_write == 'w':
            writer.writerow(csv_header)
        writer.writerows(csv_data)

def create_directories(dir_path):
    # Creates a CSV directory if it doesn't already exist
    if not os.path.exists(dir_path +"csv/"):
        os.makedirs(dir_path +"csv/")
    # Creates a directory for output files if it doesn't already exist
    if not os.path.exists(dir_path + "out/"):
        os.makedirs(dir_path + "out/")
    # Creates a directory for jpg files converted from pdf if it doesn't already exist
    if not os.path.exists(dir_path + "jpg/"):
        os.makedirs(dir_path + "jpg/")

def convert_pdf(filename, output_path, resolution=500):
    all_pages = Image(filename=filename, resolution=resolution)
    for i, page in enumerate(all_pages.sequence):
        with Image(page) as img:
            img.format = 'jpg'
            img.background_color = Color('white')
            img.alpha_channel = 'remove'

            image_filename = os.path.splitext(os.path.basename(filename))[0]
            image_filename = '{}-{}.jpg'.format(image_filename, i)
            image_filename = os.path.join(output_path, image_filename)
            img.save(filename=image_filename)


# Main
if __name__ == "__main__":

    #  path for directory where all pdf files reside and only input for the file
    dir_path ="/home/arjun/Documents/virtualEnvs/imagepp3/src/darkflow/darkflow/pdf_in/"
    csv_data = list()
    create_directories(dir_path)

    # converts each pdf file to corresponding jpg file and stores in jpg directory
    #-------PDF TO JPG---------------------------------------------------------------
    jpg_dir_path = dir_path + "jpg/"
    only_pdf_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    for pdf_file in only_pdf_files:
        convert_pdf(dir_path + pdf_file, jpg_dir_path)
    # -------------------------------------------------------------------------------

    # CSV file to store file_name, barcode type, barcode data
    csv_filename = dir_path +"csv/" + dir_path.split("/")[-2] + '.csv'
    # header for CSV file
    csv_header = list()
    csv_header.extend(("File_name", "Type", "Data"))

    # list of files in the directory given in "dir_path"
    onlyfiles = [f for f in listdir(jpg_dir_path) if isfile(join(jpg_dir_path, f))]

    # processing each file one by one
    # storing output in OUT folder and appending data to CSV file
    for jpg_file in onlyfiles:
        im = cv2.imread(jpg_dir_path + jpg_file)
        decodedObjects = decode(im)
        if len(decodedObjects) > 0:
            display(im, decodedObjects)
    write_to_csv()
