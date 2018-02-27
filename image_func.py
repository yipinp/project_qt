import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xlwt
from mpl_toolkits.mplot3d import Axes3D
#from numpy import arange,max,min,reshape

OVERLAP = 0.3
SPLIT_SIZE = 3
GREEN = (0,255,0)
BLUE   = (255,0,0)
RED  = (0,0,255)
BACKGROUND_COLOR = np.array([255,255,255])
LINEWIDTH = 10

'''
        Preprocessing  function
'''

#scan the directory to get the image list for stitching
def scan_directory(selected_dir):
    input_image_list = []
    for root, dirs, files in os.walk(selected_dir):
        for name in files :
            input_image_list.append(os.path.join(root,name))
    return input_image_list

#  test only, not normal function.
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

'''
        function 1 : stitch image
'''
#call opencv to stitch the image, if success, the stitched image will be written to output directory
def image_stitch(imgs, output_dir):
    stitcher = cv2.createStitcher(False)
    result = stitcher.stitch(imgs)
    if result[0] != 0 :
        print("Image stitching is failed, please check the input images!")
        return None
    else:
        cv2.imwrite(output_dir+'/stitched.jpg',result[1])
    return result[1]


'''
    function 2 : extract contour and calculate the moment
'''
def calculate_moment(c):
    M = cv2.moments(c)
    cx,  cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    return cx, cy


#convert to gray image and use the canny to remove some noises before contour extraction
def get_contour(stitched_image):
    img_in = stitched_image
    img = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    img_binary = cv2.Canny(img,0,100)
    image, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0 :
        c = max(contours, key=cv2.contourArea)
        c_reshape = np.reshape(c,(-1,2))
        max_x,max_y = np.max(c_reshape,axis=0)
        min_x,min_y = np.min(c_reshape,axis=0)
        cv2.polylines(img_in,c,True,RED,20)
        cx,cy = calculate_moment(c)
        cv2.circle(img_in,(cx,cy),30,RED,-1)
    else:
        cx = None
    print(cx,cy,min_x,min_y,max_x,max_y)
    return cx,cy,c,max_x,max_y,min_x,min_y,img_in


'''
      function 3:  Create histogram for gray or color image (HSV)
'''
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    fig = plt.figure('HSV 3D颜色统计直方图')

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

'''
     function 4: image mask generation and  masked image generation (HSV domain)
'''
def get_color_mask_image(img,hsv_low_range,hsv_high_range):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(img_hsv,hsv_low_range,hsv_high_range)
    return mask_hsv

def isBackGround(pixel):
    if pixel[0] == BACKGROUND_COLOR[0] and pixel[1] == BACKGROUND_COLOR[1] and pixel[2] == BACKGROUND_COLOR[2]:
        return True
    else:
        return False

def generate_final_mask(img,mask_hsv,c,mode):
    print(img.shape,mask_hsv.shape)
    if mode == 1:
        mask_hsv = cv2.bitwise_not(mask_hsv)
        #reset background
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if isBackGround(img[i,j,:]) == True:
                    mask_hsv[i,j] = 0

    for i in range(c.shape[0]):
        mask_hsv[c[i,0,1],c[i,0,0]] = 255

    cv2.imwrite('mask_hsv.jpg',mask_hsv)
    return mask_hsv

def generate_image_from_mask(img,mask_hsv,cx,cy,max_x,max_y,min_x,min_y,row,col):
    res = cv2.bitwise_and(img,img,mask = mask_hsv)
    cv2.line(res,(cx,min_y),(cx,max_y),(255,0,0),LINEWIDTH)
    cv2.line(res,(min_x,cy),(max_x,cy),(0,255,0),LINEWIDTH)
    cv2.circle(res, (cx, cy), 30, GREEN, -1)
    draw_grid(res,cx,cy,max_x,max_y,min_x,min_y,row,col)
    cv2.imwrite('masked_image.jpg',res)
    #cv2.imwrite('mask.jpg',mask_hsv)
    return res

'''
                Grid generation
'''
def draw_grid(img,cx,cy,max_x,max_y,min_x,min_y,row,col,color=GREEN):
    dx = img.shape[1]//col
    dy = img.shape[0]//row
    #print(dx,dy,img.shape[1],img.shape[0])
    for pt_y in np.arange(cy,min_y,-dy):
        cv2.line(img,(min_x,pt_y),(max_x,pt_y),color,LINEWIDTH)

    for pt_y in np.arange(cy,max_y,dy):
        cv2.line(img, (min_x,pt_y), (max_x,pt_y),color,LINEWIDTH)

    for pt_x in np.arange(cx,min_x,-dx):
        cv2.line(img,(pt_x,min_y),(pt_x,max_y),color,LINEWIDTH)

    for pt_x in np.arange(cx,max_x,dx):
        cv2.line(img, (pt_x,min_y), (pt_x,max_y),color,LINEWIDTH)

    #draw external rectangle
    cv2.rectangle(img,(min_x,min_y),(max_x,max_y),color,LINEWIDTH)
    return img


'''
     function 5 : create grid and calculate bin area statistics
'''
def get_grid_info(mask_hsv,cx,cy,max_x,max_y,min_x,min_y,row,col):
    dx = mask_hsv.shape[1] // col
    dy = mask_hsv.shape[0] // row
    #copy from draw grid to match it
    grid_row=[]
    grid_col=[]
    grid_temp=[]
    for pt_y in np.arange(cy,min_y,-dy):
        grid_temp.append(pt_y)

    if grid_temp[0] != min_y:
        grid_temp.append(min_y)
    #reorder it
    for i in np.arange(len(grid_temp)-1,-1,-1):
        grid_row.append(grid_temp[i])

    for pt_y in np.arange(cy+dy, max_y+1, dy):
        grid_row.append(pt_y)

    if pt_y != max_y:
        grid_row.append(max_y)

    #col
    grid_temp = []
    for pt_x in np.arange(cx, min_x, -dx):
        grid_temp.append(pt_x)

    if grid_temp[0] != min_x:
        grid_temp.append(min_x)
    # reorder it
    for i in np.arange(len(grid_temp)-1, -1, -1):
        grid_col.append(grid_temp[i])

    for pt_x in np.arange(cx+dx, max_x+1, dx):
        grid_col.append(pt_x)

    if pt_x != max_x:
        grid_col.append(max_x)

    return grid_row,grid_col



def get_grid_info2(mask_hsv,cx,cy,max_x,max_y,min_x,min_y,row,col):
    dx = mask_hsv.shape[1]//col
    dy = mask_hsv.shape[0]//row

    grid_row = []
    half_row_num = np.ceil((cy - min_y)/dy)
    #check the first row if not even grid split
    grid_row.append((cy-min_y)%dy)
    for i in np.arange(half_row_num - 1):
        grid_row.append(dy)
    half_row_num = np.ceil((max_y-cy)/dy)

    for i in np.arange(half_row_num - 1):
        grid_row.append(dy)
    #check the last row if not even grid split
    grid_row.append((max_y-cy)%dy)

    grid_col = []
    half_col_num = np.ceil((cx - min_x) / dx)
    # check the first col if not even grid split
    grid_col.append((cx - min_x) % dx)
    for i in np.arange(half_col_num - 1):
        grid_col.append(dx)
    half_col_num = np.ceil((max_x - cx) / dx)

    for i in np.arange(half_col_num - 1):
        grid_col.append(dx)
    # check the last col if not even grid split
    grid_col.append((max_x - cx) % dx)

    return grid_row,grid_col

def get_statistics_per_bin(mask_hsv,grid_row,grid_col):
    #open excel
    data = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = data.add_sheet('网格统计',cell_overwrite_ok=True)

    #write header
    sheet.col(0).width = 256*20
    sheet.write(0,0,'网格坐标(行,列)')
    sheet.col(1).width = 256 * 12
    sheet.write(0,1,'网格面积')
    sheet.col(2).width = 256 * 20
    sheet.write(0,2,'网格中组分图像面积')
    sheet.col(3).width = 256 * 20
    sheet.write(0,3,'组分图像面积百分比')

    #grid area calculation
    index = 0
    start_x = 0
    start_y = 0
    end_x = 0
    end_y = 0
    total = 0
    for i in range(len(grid_row)):
        end_y += grid_row[i]
        for j in range(len(grid_col)):
            index = index + 1
            sheet.write(index, 0, '%d,%d' % (i, j))
            area = grid_row[i]*grid_col[j]
            end_x += grid_col[j]
            mask_area = get_bin_area(mask_hsv,start_x,end_x,start_y,end_y)
            total += mask_area
            sheet.write(index,1,int(area))
            sheet.write(index,2,int(mask_area))
            style_percent = xlwt.easyxf(num_format_str='0.00%')
            sheet.write(index,3,int(mask_area)/int(area),style_percent)
            start_x += grid_col[j]

        start_y += grid_row[i]
        start_x = 0
        end_x  = 0
    data.save(r'./网格统计信息表.xls')

def get_bin_area(mask_hsv,start_x,end_x,start_y,end_y):
    num = 0
    for i in np.arange(start_y,end_y,1):
        for j in np.arange(start_x,end_x,1):
            if mask_hsv[i,j] > 0:
                num = num + 1
    return num

# #
imageName = r'/home/pyp/project_stitch/project_qt/images/B.jpg'
#imageName = r'C:\DL_project\project_qt\images\B.jpg'
# # # imgs = create_test_images(imageName)
# # # image_stitch(imgs,'C:\DL_project\image_proc\output')
img = cv2.imread(imageName)
cx,cy,c,max_x,max_y,min_x,min_y,image = get_contour(img)
# draw_grid(img,cx,cy,max_x,max_y,min_x,min_y,8,8)
# # get_histogram_3d(img)
#
# # # #remove some colors
red_low_range = np.array([125,43,33])
red_high_range = np.array([155,255,100])
# # # # #
mask_hsv = get_color_mask_image(img,red_low_range,red_high_range)
mask_hsv = generate_final_mask(img,mask_hsv,c,1)

generate_image_from_mask(img,mask_hsv,cx,cy,max_x,max_y,min_x,min_y,8,8)
grid_row,grid_col = get_grid_info(mask_hsv,cx,cy,max_x,max_y,min_x,min_y,8,8)
get_statistics_per_bin(mask_hsv,grid_row,grid_col)