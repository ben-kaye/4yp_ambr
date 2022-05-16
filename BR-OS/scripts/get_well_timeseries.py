import cv2
from cv2 import ROTATE_90_CLOCKWISE
import numpy as np

# loc:
path = './../Unit tests/Exp-D ALS/ims/wells_'
out_path = './../Unit tests/out.bmp'

final_index = 1239
num = 10
offset = 10 # indices

mult = round(final_index/num)

indices = tuple(offset + i*mult for i in range(num))

ims = []

for i in indices:
    link = path + str(i) + '.png'

    im = cv2.imread(link)
    ims.append(cv2.rotate(im,ROTATE_90_CLOCKWISE))

out_im = np.hstack(ims)
cv2.imshow(None, out_im)
cv2.waitKey(0)

cv2.imwrite(out_path,out_im)