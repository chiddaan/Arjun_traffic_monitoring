import csv
import numpy as np
import collections
from sklearn.cluster import KMeans

leftCoords = np.array([])
rightCoords = np.array([])
file_handler= open("sampla.txt","w+")
with open('jan26.csv', newline='') as myFile:
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
left_undersaturated_percentage = round(left_undersaturated /(left_undersaturated + left_oversaturated) * 100)
left_oversaturated_percentage = round(left_oversaturated /(left_undersaturated + left_oversaturated) * 100)
right_undersaturated_percentage = round(right_undersaturated /(right_undersaturated + right_oversaturated) * 100)
right_oversaturated_percentage = round(right_oversaturated /(right_undersaturated + right_oversaturated) * 100)
file_handler.write("From the traffic going left, \nTotal Time processed : {} minutes \nUndersaturated traffic : {} "
      "minutes ({}%)\nOversaturated traffic : {} minutes ({}%)".format(left_undersaturated + left_oversaturated,
                                                                         left_undersaturated,left_undersaturated_percentage, left_oversaturated, left_oversaturated_percentage))
file_handler.write("\n\nFrom the traffic going right, \nTotal Time processed : {} minutes \nUndersaturated traffic : {} "
      "minutes ({}%) \nOversaturated traffic : {} minutes ({}%)".format(right_undersaturated+right_oversaturated,
                                                                              right_undersaturated, right_undersaturated_percentage, right_oversaturated, right_oversaturated_percentage))