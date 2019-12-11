import meshrenderer_phong
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from pysixd_stuff import transform
import random
import cv2 as cv
from pysixd_stuff import view_sampler
import random


def _aug():
    from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, \
        Noop, Lambda, AssertLambda, AssertShape, Scale, CropAndPad, \
        Pad, Crop, Fliplr, Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, \
        Grayscale, GaussianBlur, AverageBlur, MedianBlur, Convolve, \
        Sharpen, Emboss, EdgeDetect, DirectedEdgeDetect, Add, AddElementwise, \
        AdditiveGaussianNoise, Multiply, MultiplyElementwise, Dropout, \
        CoarseDropout, Invert, ContrastNormalization, Affine, PiecewiseAffine, \
        ElasticTransformation
    str = "Sequential([	Sometimes(0.5, Affine(scale=(1.0, 1.2))),\
	        Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),\
	        Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),\
            Sometimes(0.5, Add((-25, 25), per_channel=0.3)),\
            Sometimes(0.3, Invert(0.2, per_channel=True)),\
            Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),\
            Sometimes(0.5, Multiply((0.6, 1.4))),\
            Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))\
	        ], random_order=False)"

    str = "'Sometimes(0.5, GaussianBlur(sigma=(1.2*np.random.rand())))'"

    return eval(str)


def render(height, width, s):
    height = int(height*s)
    width = int(width*s)
    clip_near = 10
    clip_far = 5000
    K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
    renderer = meshrenderer_phong.Renderer('/home/sid/thesis/ply/models_cad/obj_05_red.ply',
                                           samples=1, vertex_tmp_store_folder='.', clamp=False, vertex_scale=1.0)
    # R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    R = transform.random_rotation_matrix()[:3, :3]
    t = np.array([0, 0, random.random()], dtype=np.float32) * 1000
    start = time.time()
    num_runs = 3000
    # for i in tqdm(range(num_runs)):
    color, depth_x = renderer.render(
        0, width, height, K
        , R, t, clip_near, clip_far)
    # cv2.imshow("win", color)
    # cv2.waitKey(1)
    mask_x = depth_x == 0
    ys, xs = np.nonzero(depth_x > 0)

    try:
        obj_bb = view_sampler.calc_2d_bbox(xs, ys, (width, height))
        print obj_bb
    except ValueError as e:
        print('Object in Rendering not visible. Have you scaled the vertices to mm?')

    x, y, w, h = obj_bb
    return color, mask_x, obj_bb

def channel_first(img):
    new_img = np.ones((3, img.shape[0], img.shape[1]))
    new_img[0] = img[:, :, 0]
    new_img[1] = img[:, :, 1]
    new_img[2] = img[:, :, 2]
    return new_img

def channel_last(img):
    new_img = np.ones((img.shape[1], img.shape[2],3))
    new_img[:, :, 0] = img[0, :]
    new_img[:, :, 1] = img[1, :]
    new_img[:, :, 2] = img[2, :]
    return new_img

M = np.array((
    (1, 0, 90),
    (0, 1, 90)
))
height = int(760)
width = int(1080)
obj, mask_x, obj_bb = render(height, width, 1)
x, y, w, h = obj_bb
cv.rectangle(obj, (x, y), (x + w, y + h), (0, 255, 0), 2)
obj = cv.warpAffine(obj, M, (obj.shape[1], obj.shape[0]))
cv.imshow('res', obj)
cv.waitKey(0)
cv.destroyAllWindows()
bkr = cv.imread("/home/sid/thesis/VOCdevkit/VOC2012/JPEGImages/2007_001834.jpg")
print bkr.shape
bkr = cv.resize(bkr, (width, height))
"""
#bit = cv2.bitwise_xor(obj,bkr)
#bit = cv2.addWeighted(obj,2,bkr,1,0)
bit = cv2.add(obj,bkr)
cv2.imshow("bit", bit)
"""
img1 = bkr
img2 = obj
# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
print rows, cols
a, b = rows, cols
roi = img1[0:a, 0:b]
# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2, img2, mask=mask)
# Put logo in ROI and modify the main image
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
#cv.imwrite("1.png",img1)
import imageio
import imgaug as ia
"""
image = imageio.imread("1.png")
#img1 = channel_first(img1)
img1 = np.rollaxis(img1, 2)
img1 = _aug().augment_images(img1)
#img1 = channel_last(img1)
img1 = np.rollaxis(img1, 0, 3)
"""


print "hi"
