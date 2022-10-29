import cv2 as cv
import math
import numpy as np



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


if __name__ == "__main__":
    main()