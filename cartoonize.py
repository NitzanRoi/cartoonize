import requests
from io import BytesIO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from skimage import color
from colorsys import rgb_to_hls, hls_to_rgb
import cv2

class BasicOperations:
    # class for basic image operations: 
    # read, normalize, show, convert colors, get random, save
    def __init__(self):
        pass

    def read_image(self, img_path):
        return np.array(Image.open(img_path)).astype('float64')

    def normalize_img(self, np_img): # range: [0 -> 1]
        # the formula is: (x - min(x)) / (max(x) - min(x))
        if (np.max(np_img) == np.min(np_img)):
            return False
        return (np_img - np.min(np_img)) / (np.max(np_img) - np.min(np_img))

    def show_img(self, np_img, is_grayscale=False):
        np_img = self.normalize_img(np_img)
        plt.figure(figsize = (10,10))
        if is_grayscale:
            plt.imshow(np_img, cmap=plt.get_cmap('gray'))
        else:
            plt.imshow(np_img)
        plt.show()

    def convert_img_colors(self, dest_color, np_img):
        if dest_color == 'RGB':
            return color.gray2rgb(np_img)
        elif dest_color == 'GS':
            return color.rgb2gray(np_img)
        else:
            return False

    def get_random_image(self):
        height=500
        width=300
        # url='https://picsum.photos/'+str(height)+'/'+str(width)+'.jpg'
        url = 'https://source.unsplash.com/'+str(width)+'x'+str(height)+'/?face'
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return np.array(img).astype('float64')

    def save_img(self, img, path):
        if (len(img.shape) == 2):
            img = convert_img_colors('RGB', img)
        if (np.max(img) > 1.0):
            img = normalize_img(img)
        image.imsave(path + ".jpg", img)
        print('image saved')

class ImageManipulations:
    # class for image manipulations
    def __init__(self, img):
        self.img = img

    def is_k_means_converged(self, last_centroids, new_centroids):
        diff_dactor = 2
        return np.linalg.norm(new_centroids - last_centroids) < diff_dactor

    def calc_dist(self, pixA, pixB):
        return np.sqrt(np.square(pixA[0]-pixB[0]) + np.square(pixA[1]-pixB[1]) + np.square(pixA[2]-pixB[2]))

    def get_min_dist_idx(self, pixel, centroids_arr):
        distances = np.zeros(centroids_arr.shape[0])
        for i in range(centroids_arr.shape[0]):
            distances[i] = self.calc_dist(pixel, centroids_arr[i])
        return np.argmin(distances)

    def get_assigned_indexes(self, idx, img):
        return img[idx[0], idx[1]]

    def map_pixels_to_centroid(self, idx, centroid):
        self.img[idx[0], idx[1]] = centroid

    def k_means(self):
        k = 5 # number of colors in quantized image
        img = self.img

        # step 1: pick k random centroids:
        centroids = np.random.randint(256, size=(k,3))
        last_centroids = np.random.randint(256, size=(k,3))

        # repeat until convergence:
        iterations_ctr = 0
        max_iterations = 10
        while (not self.is_k_means_converged(last_centroids, centroids) and iterations_ctr < max_iterations):
            print(iterations_ctr)
            iterations_ctr += 1

            last_centroids = np.copy(centroids)
            # step 2: assign each pixel to the closest centroid (Euclidean distance)
            min_dist_color_idx = np.apply_along_axis(self.get_min_dist_idx, 2, img, centroids_arr=centroids) # apply func on each pixel

            # get all pixels assigned to i_th centroid, and update centroid i_th to their mean
            for i in range(centroids.shape[0]):
                idxs_assigned_ith = np.tile(np.where(min_dist_color_idx==i), 1).T
                if (idxs_assigned_ith.shape[0] > 0):
                    pixels_assigned = np.apply_along_axis(self.get_assigned_indexes, 1, idxs_assigned_ith, img=img)               
                    centroids[i] = np.sum(pixels_assigned, axis=0) / pixels_assigned.shape[0]
        
        # step 3: map each pixel to its centroid
        min_dist_color_idx = np.apply_along_axis(self.get_min_dist_idx, 2, img, centroids_arr=centroids) # apply func on each pixel

        # get all pixels assigned to i_th centroid, and map them to the i_th centroid
        for i in range(centroids.shape[0]):
            idxs_assigned_ith = np.tile(np.where(min_dist_color_idx==i), 1).T
            if (idxs_assigned_ith.shape[0] > 0):
                np.apply_along_axis(self.map_pixels_to_centroid, 1, idxs_assigned_ith, centroid=centroids[i])
        return self.img

    def make_color_lighter(self, pixel, factor=1.3):
        h, l, s = rgb_to_hls(pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0)
        l = max(min(l * factor, 1.0), 0.0)
        r, g, b = hls_to_rgb(h, l, s)
        return np.array([int(r * 255), int(g * 255), int(b * 255)])

    def bilateral_filter(self, img):
        img = img.astype('uint8')
        return cv2.bilateralFilter(img, 5, 30, 30)

    def cartoonize(self):
        img = self.k_means() # quantize
        img = np.apply_along_axis(self.make_color_lighter, 2, img) # lightening the colors
        img = self.bilateral_filter(self.img)
        return img

class Test:
    # class for tests
    def __init__(self):
        self.bo_obj = BasicOperations()
        self.im_obj = None

    def test_get_img(self):
        return self.bo_obj.get_random_image()

    def test_show_img(self, img):
        self.bo_obj.show_img(img)

    def test_manipulations(self, img):
        self.im_obj = ImageManipulations(img)
        img = self.im_obj.cartoonize()
        return img

def main():
    test = Test()
    img = test.test_get_img()
    test.test_show_img(img)
    print('--')
    img = test.test_manipulations(img)
    test.test_show_img(img)

if __name__ == "__main__":
    main()