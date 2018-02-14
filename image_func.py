import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OVERLAP = 0.3
SPLIT_SIZE = 3
GREEN = (0,255,0)
BLUE   = (255,0,0)
RED  = (0,0,255)

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
    for i in range(len(input_list)):
        imgs.append(cv2.imread(input_list[i]))
    return imgs


def image_stitch(imgs, output_dir):
    stitcher = cv2.createStitcher(False)
    result = stitcher.stitch(imgs)
    if result[0] != 0 :
        print("Image stitching is failed, please check the input images!")
        return None
    else:
        cv2.imwrite(output_dir+'/stitched.jpg',result[1])
    return result[1]


def get_histogram_gray(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()
    return hist


def get_histogram(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [180,256,256],[0,180,0,256,0,256])
    plt.plot(hist[2])
    plt.show()
    return hist

def get_histogram_3d(img):
    print(img.shape)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    for x, c, z in zip([h,s,v], ['r', 'g', 'b'], [30, 20, 10]):
        xs = np.arange(256)
        ys = cv2.calcHist([x], [0], None, [256], [0,256])
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys.ravel(), zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def get_color_mask_image(img,hsv_low_range,hsv_high_range):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(img_hsv,hsv_low_range,hsv_high_range)
    return mask_hsv

def generate_final_mask(mask_hsv,c,mode):
    if mode == 1:
        mask_hsv = 255 - mask_hsv

    for i in range(c.shape[0]):
        mask_hsv[c[i,0,1],c[i,0,0]] = 255

    return mask_hsv

def generate_image_from_mask(img,mask_hsv,cx,cy,max_x,max_y,min_x,min_y,grid_size):
    res = cv2.bitwise_and(img,img,mask = mask_hsv)
    cv2.line(res,(cx,min_y),(cx,max_y),(255,0,0),5)
    cv2.line(res,(min_x,cy),(max_x,cy),(0,255,0),5)
    cv2.circle(res, (cx, cy), 30, GREEN, -1)
   # draw_grid(res,cx,cy,grid_size,max_x,max_y,min_x,min_y)
    cv2.imwrite('masked_image.jpg',res)
    cv2.imwrite('mask.jpg',mask_hsv)

def get_contour(img_in):
    #convert to binary image for contour
    img = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    img_binary = cv2.Canny(img,0,100)
    image, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0 :
        c = max(contours, key=cv2.contourArea)
        c_reshape = np.reshape(c,(-1,2))
        max_x,max_y = np.max(c_reshape,axis=0)
        min_x,min_y = np.min(c_reshape,axis=0)
        print(min_x,min_y,max_x,max_y)
        cv2.polylines(img_in,c,True,RED,20)
        cx,cy = calculate_moment(c)
        cv2.circle(img_in,(cx,cy),30,RED,-1)
    else:
        cx = None

    return cx,cy,c,max_x,max_y,min_x,min_y,img_in


def calculate_moment(c):
    M = cv2.moments(c)
    cx,  cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    return cx, cy

def draw_grid(img,cx,cy,max_x,max_y,min_x,min_y,row,col,color=GREEN):
    dx = img.shape[1]//col
    dy = img.shape[0]//row
    print(dx,dy,img.shape[1],img.shape[0])
    lineWidth = 10
    for pt_y in np.arange(cy,min_y,-dy):
        cv2.line(img,(min_x,pt_y),(max_x,pt_y),color,lineWidth)

    for pt_y in np.arange(cy,max_y,dy):
        cv2.line(img, (min_x,pt_y), (max_x,pt_y),color,lineWidth)

    for pt_x in np.arange(cx,min_x,-dx):
        cv2.line(img,(pt_x,min_y),(pt_x,max_y),color,lineWidth)

    for pt_x in np.arange(cx,max_x,dx):
        cv2.line(img, (pt_x,min_y), (pt_x,max_y),color,lineWidth)

    #draw external rectangle
    cv2.rectangle(img,(min_x,min_y),(max_x,max_y),color,lineWidth)

    cv2.imwrite('grid.jpg',img)
    return img

def get_statistics_per_bin():
    pass



# #
# #imageName = r'/home/pyp/project_stitch/project_qt/images/B.jpg'
# imageName = r'C:\DL_project\project_qt\images\B.jpg'
# # # imgs = create_test_images(imageName)
# # # image_stitch(imgs,'C:\DL_project\image_proc\output')
# img = cv2.imread(imageName)
# # cx,cy,c,max_x,max_y,min_x,min_y = get_contour(img)
# # #draw_grid(img,cx,cy,8,max_x,max_y,min_x,min_y)
# get_histogram_3d(img)
# #
# # #remove some colors
# # red_low_range = np.array([125,43,33])
# # red_high_range = np.array([155,255,100])
# # # #
# # mask_hsv = get_color_mask_image(img,red_low_range,red_high_range)
# # mask_hsv = generate_final_mask(mask_hsv,c,'dfdf')
# # generate_image_from_mask(img,mask_hsv,cx,cy,max_x,max_y,min_x,min_y,6)