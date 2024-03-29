import cv2
import numpy as np
import pandas as pd

# filename = 'videos/D-1/DJI_0003.mp4'
filename = 'videos/D-1/example.avi'
smoothing_radius = 30

# Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
cap = cv2.VideoCapture(filename)
success, prev = cap.read()
if not success:
    raise Exception('No frames found!')
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
last_transform = None
dx, dy, da = [], [], []
frames = 0
height = prev.shape[0]
width = prev.shape[1]
print(height)
while (cap.isOpened()):
    success, cur = cap.read()
    print(success)
    if not success:
        break
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    prev_corner = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 30)
    cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_corner, np.array([]))
    prev_corner = [c for c, s in zip(prev_corner, status) if s[0]]
    cur_corner = [c for c, s in zip(cur_corner, status) if s[0]]
    transform = cv2.estimateRigidTransform(np.array(prev_corner), np.array(cur_corner), False)
    if transform is None:
        transform = last_transform
    dx.append(transform[0, 2])
    dy.append(transform[1, 2])
    da.append(np.arctan2(transform[1, 0], transform[0, 0]))
    prev = cur
    prev_gray = cur_gray
    last_transform = transform
    frames += 1
cap.release()

dx, dy, da = map(lambda transform: pd.Series(transform), [dx, dy, da])

# Step 2 - Accumulate the transformations to get the image trajectory
x, y, a = map(lambda transform: transform.cumsum(), [dx, dy, da])

# Step 3 - Smooth out the trajectory using an averaging window
x_smooth, y_smooth, a_smooth = map(lambda t: t.rolling(window=smoothing_radius,
                                                       min_periods=1,
                                                       center=False).mean(),
                                   [x, y, a])

# Step 4 - Generate new set of previous to current transform, such that the trajectory
#          ends up being the same as the smoothed trajectory vector
dx = dx + x_smooth - x
dy = dy + y_smooth - y
da = da + a_smooth - a

# Step 5 - Write results to video
transforms = np.array([np.cos(da), -np.sin(da), dx, np.sin(da), np.cos(da), dy]).T.reshape((-1, 2, 3))
cap = cv2.VideoCapture(filename)
frame_index = 0
codec = cv2.VideoWriter_fourcc(*'XVID')
in_size = (width, height)
out_size = (width * 2, height)
writer = cv2.VideoWriter('data/output.avi', codec, 20.0, out_size)
while (cap.isOpened()):
    success, cur = cap.read()
    if not success or frame_index >= frames:
        break
    stabilized = cv2.warpAffine(cur, transforms[frame_index], in_size)
    combined = np.concatenate([stabilized, cur], axis=1)
    cv2.imshow("out", combined)
    cv2.waitKey(0)
    writer.write(combined)
    frame_index += 1
cap.release()