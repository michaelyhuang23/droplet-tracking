import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

video = cv2.VideoCapture("videos/video1.mp4")

def hampel_filter_forloop(input_series, window_size, n_sigmas=3):
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    
    # possibly use np.nanmedian 
    for i in range((window_size),(n - window_size)):
        x0 = np.mean(input_series[(i - window_size):(i + window_size)], axis=0)
        S0 = k * np.mean(np.abs(input_series[(i - window_size):(i + window_size)] - x0[None,...]), axis=0)
        if (np.linalg.norm(input_series[i] - x0) > n_sigmas * np.linalg.norm(S0)):
            new_series[i] = x0
            indices.append(i)
    return new_series, indices

prev_frames = []
prev_frame_len = 20  # higher better duh
for i in range(prev_frame_len):
	ret, o_frame = video.read()
	prev_frames.append(cv2.cvtColor(o_frame, cv2.COLOR_BGR2GRAY) / 255.0)


locations = []

while True:
    ret, o_frame = video.read()
    if not ret:
    	print('you have f*cked up, not reading anything')
    	continue

    frame = cv2.cvtColor(o_frame, cv2.COLOR_BGR2GRAY) / 255.0

    contrast_frames = [frame-p_frame for p_frame in prev_frames]
    total_mask = frame.copy()
    for c_frame in contrast_frames:
    	total_mask *= c_frame
    total_mask /= np.max(total_mask)

    loc = np.unravel_index(total_mask.argmax(), total_mask.shape)

    o_frame = cv2.circle(o_frame, (loc[1],loc[0]), radius=5, color=(0, 0, 255), thickness=5)

    locations.append(loc)

    cv2.imshow("Origin", o_frame)
    cv2.imshow("Mask", total_mask)
    if cv2.waitKey(1) & 0xff == 27: break

    prev_frames = prev_frames[1:]
    prev_frames.append(frame)
        
video.release()
cv2.destroyAllWindows()

f_locations, e_indices = hampel_filter_forloop(locations, 5, 3)
f_locations = np.array(f_locations)

sns.scatterplot(x=f_locations[:,0], y=f_locations[:,1])
plt.show()




