import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt

OVERLAP = 0.3
SPLIT_SIZE = 3

def scan_directory(selected_dir):
    input_image_list = []
    for root, dirs, files in os.walk(selected_dir):
        for name in files :
            input_image_list.append(os.path.join(root,name))
    return input_image_list

#  test only


def create_test_images(input_image):
    img = cv2.imread(input_image)
    imgs = []
    size = SPLIT_SIZE
    overlap = OVERLAP
    height,width = img.shape[:2]

    # split the image to 3x3 with overlap

    for h in range(size):
        for w in range(size):
            roi = img[max(0,round(((h-overlap)*height)/size)):round(((h+1)*height)/size),max(0,round(((w-overlap)*width)/size)):round(((w+1)*width)/size)]
            imgs.append(roi)
    return imgs


def load_images(input_list):
    imgs = []
    for i in len(input_list):
        imgs.append(cv2.imread(input_list[i]))
    return imgs


def image_stitch(imgs, output_dir):
    stitcher = cv2.createStitcher(False)
    result = stitcher.stitch(imgs)
    if result[0] != 0 :
        print("Image stitching is failed, please check the input images!")
    else:
        cv2.imwrite(output_dir+'/stitched.jpg',result[1])


def get_histogram_gray(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img_gray.jpg',img_gray)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()
    return hist


def get_histogram(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colors = ('b','g','r')
    # for i, col in enumerate(colors):
    #     if i == 0:
    #         range = 180
    #     else :
    #         range = 256
    #     hist = cv2.calcHist([img_hsv],[i], None, [range], [0, range])
    #     #plt.plot(hist, color = col)
    #     #plt.xlim([0,256])

    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [180,256,256],[0,180,0,256,0,256])
    plt.plot(hist[1])
    plt.show()
    return hist



def get_color_mask_image(img,hsv_low_range,hsv_high_range, mode):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_hsv = cv2.inRange(img_hsv,0,150)
    print(mask_hsv)
    if mode == 'REMOVED':
        mask_hsv = 255- mask_hsv
    img_mask = cv2.bitwise_and(img_hsv,img_hsv,mask=mask_hsv)
    #img_bgr = cv2.cvtColor(img_mask,cv2.COLOR_HSV2BGR)
    cv2.imwrite('out_mask.jpg',img_mask)
    #plt.imshow(img_bgr)
    plt.show()
    #return img_bgr



def get_contour(img_in):
    #convert to binary image for contour

    img = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    img_binary = cv2.Canny(img,0,100)
    image, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0 :
        c = max(contours, key=cv2.contourArea)
        cv2.polylines(img_in,c,True,(0,0,255),20)
        cx,cy = calculate_moment(c)
        cv2.circle(img_in,(cx,cy),30,(0,0,255),-1)
        cv2.imwrite('out.jpg',img_in)
        cv2.imwrite('out1.jpg',img_binary)

    return c


def calculate_moment(c):
    M = cv2.moments(c)
    cx,  cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    return cx, cy

# imgs = create_test_images('C:\DL_project\image_proc\images\B.jpg')
# image_stitch(imgs,'C:\DL_project\image_proc\output')
img = cv2.imread('C:\DL_project\image_proc\images\B.jpg')
#get_contour(img)
get_histogram_gray(img)

#remove some colors
red_low_range = np.array([0])
red_high_range = np.array([130])

get_color_mask_image(img,red_low_range,red_high_range, 'REMOVED')