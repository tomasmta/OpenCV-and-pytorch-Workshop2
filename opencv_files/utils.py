import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt 



def load_gray_image(image_path):
    """
    Loads image, convert to gray, and normalizes pixels values (divide by 255)
    """ 

    image = cv.imread(image_path)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = np.asarray(gray_image)/255.0
    return gray_image


def get_magnitude(x, y):
    magnitude = np.sqrt(pow(x, 2) + pow(y, 2))
    return magnitude


class Filters():

    def __init__(self):

        self.gauss = np.array([
                            [0.045, 0.122, 0.045],
                            [0.122, 0.332, 0.122],
                            [0.045, 0.122, 0.045]])
        
        self.sobel_x = np.array([
                            [-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]])

        self.sobel_y = np.array([
                            [1.0, 2.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [-1.0, -2.0, -1.0]])

        self.kernel_len = len(self.gauss[0]) # Square Kernel of length 3

    def pad_image(self, image, height, width):
        """
        Returns image padded with zeros. For example: Input [2, 2] --> Output [0, 0, 0, 0]
                                                            [2, 2]            [0, 2, 2, 0]
                                                                              [0, 2, 2, 0]
                                                                              [0, 0, 0, 0]
        """
        pad = int((self.kernel_len - 1)/2) # padding to preserve image size 
        padded_img = np.zeros((height + 2*pad, width + 2*pad))
        padded_img[pad : height + pad, pad : width + pad] = image
        return padded_img

    def conv2d(self, image, kernel):
        """ 
        Computes 2d convolution multyplying kernel by image.
        """ 
        output_img = np.zeros_like(image)
        height, width = image.shape
        padded_img = self.pad_image(image, height, width)

        for x_out in range(height):
            for y_out in range(width):
                """ 
                 Mapping center coordinates from gray img to padded image 
                 E.g. Coord [0,0] in gray --> [1, 1] in padded image                
                """
                x_pad, y_pad = x_out+1, y_out+1
                x_indices = [x_pad-1,x_pad, x_pad+1]
                y_indices = [y_pad-1, y_pad, y_pad+1]
                img_window = padded_img[x_indices, :][:, y_indices]
                output_img[x_out][y_out] = np.sum(np.multiply(img_window, kernel)).astype('float32')

        return output_img

    def apply_gauss(self, image):
        kernel = self.gauss
        blurred_img = self.conv2d(image, kernel)
        return blurred_img
    
    def apply_x_sobel(self, image):
        kernel = self.gauss
        blur_image = self.conv2d(image, kernel)
        kernel = self.sobel_x
        sobel_X_image = self.conv2d(blur_image, kernel)
        return sobel_X_image

    def apply_y_sobel(self, image):
        kernel = self.gauss
        blur_image = self.conv2d(image, kernel)
        kernel = self.sobel_y
        sobel_Y_image = self.conv2d(blur_image, kernel)
        return sobel_Y_image

def apply_resize(image_path):
    image = cv.imread(image_path)
    h, w = len(image[0]), len(image[1])
    scale = 0.5
    transf_matrix = cv.getRotationMatrix2D(center=(0,0), angle=0, scale = scale)
    new_image = cv.warpAffine(image, transf_matrix, (h, w))
    return new_image

def apply_translation(image_path):
    image1 = cv.imread(image_path)
    image2= image1.copy()
    h, w = len(image1[0]), len(image1[1])
    scale = 0.5

    transformation1 = cv.getRotationMatrix2D(center=(0,0), angle = 0, scale = scale)
    transformation2 = cv.getRotationMatrix2D(center=(h,w), angle = 0, scale = scale)

    new_image_1 = cv.warpAffine(image1, transformation1, (h, w))
    new_image_2 = cv.warpAffine(image2, transformation2, (h, w))

    combined_img = cv.addWeighted(new_image_1, 1.0, new_image_2, 1.0, 0.0)

    return combined_img

def apply_rotation_and_scaling(image_path):
    translated_image = apply_translation(image_path)
    out_size = (430, 430)
    scale = 0.5
    transf = cv.getRotationMatrix2D((215, 215), 45, scale)
    rotated_image = cv.warpAffine(translated_image, transf, out_size)
    return rotated_image

def apply_shear(image_path):
    out_size = (430, 430)
    input_image = apply_rotation_and_scaling(image_path)
    old_loc = np.array([[50,50],[200,50],[50,200]]).astype("float32")
    new_loc = np.array([[10,100],[100,50],[100,250]]).astype("float32")
    transf = cv.getAffineTransform(old_loc, new_loc)
    out_image = cv.warpAffine(input_image, transf, out_size)
    return out_image

