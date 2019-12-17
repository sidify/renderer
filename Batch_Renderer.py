from __future__ import print_function
import meshrenderer_phong
import numpy as np
from pysixd_stuff import transform
import cv2 as cv
from pysixd_stuff import view_sampler
import random
import cv2
from tqdm import tqdm
import glob
import os

class Batch_Renderer:
    def __init__(self, kw):
        self._kw = kw
        self.shape = (int(kw['H']), int(kw['W']), 3)
        self.no_of_samples = kw['no_of_samples']
        self.batch_size = kw['batch_size']
        assert self.no_of_samples > 0
        assert self.batch_size > 0
        assert self.no_of_samples >= self.batch_size
        self.bg_imgs_path = []
        self.bg_imgs = np.empty((self.no_of_samples, self.shape[0], self.shape[1], self.shape[2]), dtype=np.uint8 )
        self.create_bkg_list()
        self.shape = (int(kw['H']), int(kw['W']), 3)
        self.gen_ae = kw['gen_data_for_ae']
        self.ae_x = kw['ae_train_x']
        self.ae_y = kw['ae_train_y']
        self.target_loc = self._kw['target_loc']
        self.renderer = meshrenderer_phong.Renderer(self._kw['cad_model'], samples=1, vertex_tmp_store_folder='.', clamp=False, vertex_scale=1.0)


    def create_bkg_list(self):
        self.bg_imgs_path = glob.glob(self._kw['bgr_dir'])
        random.shuffle(self.bg_imgs_path)
        self.bg_imgs_path = self.bg_imgs_path[:self.no_of_samples]


    def _load_bg_imgs(self, start):
        for j, fname in enumerate(self.bg_imgs_path[start:start + self.batch_size]):
            bgr = cv2.imread(fname)
            self.bg_imgs[j] = cv.resize(bgr, (self.shape[1], self.shape[0]))

    @property
    def _aug(self):
        from imgaug.augmenters import Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, \
            Noop, Lambda, AssertLambda, AssertShape, Scale, CropAndPad, \
            Pad, Crop, Fliplr, Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, \
            Grayscale, GaussianBlur, AverageBlur, MedianBlur, Convolve, \
            Sharpen, Emboss, EdgeDetect, DirectedEdgeDetect, Add, AddElementwise, \
            AdditiveGaussianNoise, Multiply, MultiplyElementwise, Dropout, \
            CoarseDropout, Invert, ContrastNormalization, Affine, PiecewiseAffine, \
            ElasticTransformation
        return eval(self._kw['aug'])


    def _render(self, renderer, no_of_samples):
        train_x = np.empty((no_of_samples,) + self.shape, dtype=np.uint8)
        mask_x = np.empty((no_of_samples,) + self.shape[:2], dtype=bool)
        obj_bb = np.empty((no_of_samples, 4), dtype=np.uint16)
        height = self._kw['H']
        width = self._kw['W']
        clip_near = self._kw['clip_near']
        clip_far = self._kw['clip_far']
        i = 0
        while i < no_of_samples :
            R = transform.random_rotation_matrix()[:3, :3]
            t = np.array([0, 0, random.randint(50, 400)], dtype=np.float32)
            color, depth_x = renderer.render(
                0, int(width), int(height), K
                , R, t, clip_near, clip_far)
            ys, xs = np.nonzero(depth_x > 0)
            ys = np.array(ys, dtype=np.int16)
            xs = np.array(xs, dtype=np.int16)
            tx, ty = random.randint(-height, height), random.randint(-width, width)
            M = np.array((
                (1, 0, tx),
                (0, 1, ty)
            ), dtype=np.float)
            color = cv.warpAffine(color, M, (color.shape[1], color.shape[0]))
            depth_x = cv.warpAffine(depth_x, M, (depth_x.shape[1], depth_x.shape[0]))
            try:
                x, y, w, h = view_sampler.calc_2d_bbox(xs + tx, ys + ty, (width, height))
                if x <= 0 or y <= 0 or x >= width or y >= height or x + tx >= width or y + ty >= height or w < 0 or h < 0:
                    #print('Object translated out of bounds. Regenerating ...')
                    continue
                train_x[i] = np.copy(color)
                obj_bb[i] = x, y, w, h
                #self.show_image("train_x[i]", train_x[i])
                mask_x[i] = np.copy(depth_x == 0)
                cv2.waitKey(0)
                i += 1
            except ValueError as e:
                #print('Object in Rendering not visible. Regenerating ...')
                continue
        return train_x, mask_x, obj_bb

    def show_image(self, name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()


    def clean_directories(self):
        # empty the directories
        dirs = [self.ae_x, self.ae_y]
        for dir in dirs:
            files = glob.glob(dir)
            for f in files:
                os.remove(f)

    def process(self):
        # empty the directories
        #self.clean_directories()
        import csv
        with open(self.target_loc + 'train.csv', 'w' ) as csv_file:
            count = 0
            csv_writer = csv.writer(csv_file, delimiter=',')
            pbar = tqdm(total = self.no_of_samples)
            while count < self.no_of_samples:
                #rand_idcs = np.random.choice(self.batch_size, self.batch_size, replace=False)
                #rand_idcs_bg = np.random.choice(self.batch_size, self.batch_size, replace=False)
                rand_idcs = np.array(range(self.batch_size))
                rand_idcs_bg = rand_idcs
                self._load_bg_imgs(count)
                rand_vocs = self.bg_imgs[rand_idcs_bg]
                train_x, mask_x, obj_bb = self._render(self.renderer, self.batch_size)
                batch_x, masks = train_x[rand_idcs], mask_x[rand_idcs]
                batch_x[masks] = rand_vocs[masks]
                obj_bb = obj_bb[rand_idcs]
                aug = self._aug.augment_images(batch_x)
                for i, img in enumerate(aug):
                    file_name = self._kw['target_loc']+str(count + i)+".jpg"
                    cv2.imwrite(file_name, img)
                    x, y, w, h = obj_bb[i]
                    csv_writer.writerow([file_name, x, y, x + w, y + h, 'object'])
                    #write for autoencoder
                    if self.gen_ae:
                        crop_img = img[y:y+h, x:x+w]
                        crop_img_ren = train_x[i][y:y+h, x:x+w]
                        #self.show_image("img", img)
                        #self.show_image("crop_img", crop_img)
                        #self.show_image("crop_img_ren", crop_img_ren)
                        img_num = (count + i)
                        crop_file_x = self.ae_x+str(img_num) + ".jpg"
                        crop_file_y = self.ae_y+str(img_num) + ".jpg"
                        cv2.imwrite(crop_file_x, crop_img)
                        cv2.imwrite(crop_file_y, crop_img_ren)

                    if self._kw['show_output']:
                        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.imwrite(self._kw['show_output_dir'] + str(count + i) + ".jpg", img)
                count += self.batch_size
                pbar.update(self.batch_size)



if __name__ == "__main__" :
    #create dictionary of properties
    kw = {'bgr_dir'         : '/home/sid/thesis/VOCdevkit/VOC2012/JPEGImages/*.jpg',
          'cad_model'       : '/home/sid/thesis/ply/models_cad/obj_05_red.ply',
          'aug'             : 'Sequential([Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),\
	                            Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),\
                                Sometimes(0.5, Add((-25, 25), per_channel=0.3)),\
                                Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),\
                                Sometimes(0.5, Multiply((0.6, 1.4))),\
                                Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))\
	                            ], random_order=False)',
          'H'               : 768,
          'W'               : 960,
          'clip_near'       : 10,
          'clip_far'        : 50,
          'K'               : '[[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]',
          'batch_size'      : 200,
          'show_output'     : False,
          'show_output_dir' : '/home/sid/thesis/dummy/output/',
          'target_loc'      : '/home/sid/thesis/dummy/',
          'no_of_samples'   : 4000,
          'gen_data_for_ae' : True,
          'ae_train_x'      : '/home/sid/thesis/dummy/ae_train/x/',
          'ae_train_y'      : '/home/sid/thesis/dummy/ae_train/y/',
          }

    batch_renderer = Batch_Renderer(kw)
    batch_renderer.process()