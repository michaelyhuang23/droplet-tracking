import cv2
import numpy as np
import pandas as pd

locations_df = pd.read_hdf('locations.h5', key='location')
locations = np.concatenate([locations_df['x'][...,None], locations_df['y'][...,None]], axis=-1).astype(np.int32)
scale = 10
locations *= scale
print(len(locations))
locations = locations[:5000]
xmax = np.max(locations[:,0])
ymax = np.max(locations[:,1])
image = np.zeros((ymax+1, xmax+1, 3))
maxsep = 500*scale
for i in range(1, len(locations)):
	sep = np.linalg.norm(locations[i]-locations[i-1], axis=-1)*30
	c = np.clip(int(255-255*np.tanh(sep/maxsep)), 0, 255)
	print(c)
	image = cv2.line(image, locations[i-1], locations[i], (250,int(c),int(c)), 1) 

cv2.imshow('tracker', image)
cv2.imwrite('pathtrace5000.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()





































