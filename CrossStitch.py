import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import cv2



def save_image(a, f):
    imageio.imwrite(f, a)

def downres(image, num_gridlines, backstitch):

    num_boxes = [0, 0]

    img = np.copy(image)
    s = img.shape
    print('original dims', img.shape)
    if num_gridlines > s[0] or num_gridlines > s[1]:
        print("error: tried to have more grid lines than pixels")
        exit(-1)

    grid_size = int(min(s[0], s[1])/num_gridlines)
    print("grid_size: " + str(grid_size))

    img = fix_dims(img, grid_size)
    print('new dims: ', img.shape)

    s = img.shape

    #left, right, top, bottom coordinate of sub box
    l = 0
    t = 0
    r = grid_size
    b = grid_size

    for _ in range(0, s[0], grid_size):
        for _ in range(0, s[1], grid_size):
            sub_mat1 = img[t:b, l:r, 0]
            avg1 = sub_mat1.mean()
            sub_mat2 = img[t:b, l:r, 1]
            avg2 = sub_mat2.mean()
            sub_mat3 = img[t:b, l:r, 2]
            avg3 = sub_mat3.mean()

            cutoff = 1  # grayscale value below which we consider the pixel black

            backstitch_type = None
            if backstitch:
                backstitch_type = get_backstitch(img, l, t, r, b, cutoff)

            for k in range(min(grid_size, s[0]-t)):
                for m in range(min(grid_size, s[1]-l)):
                    img[t+k, l+m, 0] = avg1
                    img[t+k, l+m, 1] = avg2
                    img[t+k, l+m, 2] = avg3

            if backstitch_type:
                set_backstitch(img, l, t, r, b, backstitch_type)

            l = min(l + grid_size, s[1])
            r = min(r + grid_size, s[1])
            if l==r:
                l = 0
                r = grid_size
                break
        t = min(t + grid_size, s[0])
        b = min(b + grid_size, s[0])
        num_boxes[0] += 1
        if t==b:
            break

    return img


def fix_dims(img, grid_size):
    remove_rows = img.shape[0] % grid_size
    remove_cols = img.shape[1] % grid_size

    new_img = np.delete(img, np.arange(remove_rows) + (img.shape[0] - remove_rows), 0)
    #print('fixed rows shape:', new_img.shape)
    new_img = np.delete(new_img, np.arange(remove_cols) + (img.shape[1] - remove_cols), 1)
    #print('fixed cols shape:', new_img.shape)

    return new_img


def set_backstitch(img, l, t, r, b, backstitch_type):
    stitch_width = 3
    red = 255
    blue = 0
    green = 0

    if backstitch_type == 'slash':
        top = True
        for i in range(0, b-t):
            img[b-i-1, l+i, 0] = red
            img[b-i-1, l+i, 1] = green
            img[b-i-1, l+i, 2] = blue

        num = 1
        n = 1
        while num < stitch_width:
            if top:
                for i in range(0, (r-l)-n):
                    img[b-n-i-1, l+i, 0] = red
                    img[b-n-i-1, l+i, 1] = green
                    img[b-n-i-1, l+i, 2] = blue
                top = not top
                num += 1
            else:
                for i in range(0, (r-l)-n):
                    img[b-i-1, l+n+i, 0] = red
                    img[b-i-1, l+n+i, 1] = green
                    img[b-i-1, l+n+i, 2] = blue
                top = not top
                num += 1
                n += 1
    elif backstitch_type == 'backslash':
        top = True
        for i in range(0, r-l):
            img[t+i, l+i, 0] = red
            img[t+i, l+i, 1] = green
            img[t+i, l+i, 2] = blue

        num = 1
        n = 1
        while num < stitch_width:
            if top:
                for i in range(0, (r-l)-n):
                    img[t+i, l+n+i, 0] = red
                    img[t+i, l+n+i, 1] = green
                    img[t+i, l+n+i, 2] = blue
                top = not top
                num += 1
            else:
                for i in range(0, (r-l)-n):
                    img[t+n+i, l+i, 0] = red
                    img[t+n+i, l+i, 1] = green
                    img[t+n+i, l+i, 2] = blue
                top = not top
                num += 1
                n += 1
    elif backstitch_type == 'top':
        img[t: t+stitch_width, l:r, 0] = red
        img[t: t+stitch_width, l:r, 1] = green
        img[t: t+stitch_width, l:r, 2] = blue
    elif backstitch_type == 'left':
        img[t:b, l: l+stitch_width, 0] = red
        img[t:b, l: l+stitch_width, 1] = green
        img[t:b, l: l+stitch_width, 2] = blue
    elif backstitch_type == 'right':
        img[t:b, r-stitch_width:r, 0] = red
        img[t:b, r-stitch_width:r, 1] = green
        img[t:b, r-stitch_width:r, 2] = blue
    elif backstitch_type == 'bottom':
        img[t-stitch_width:t, l:r, 0] = red
        img[t-stitch_width:t, l:r, 1] = green
        img[t-stitch_width:t, l:r, 2] = blue


def get_backstitch(img, l, t, r, b, cutoff):
    if needs_stitch(img, l, t, r, b, cutoff):
        counts = {'slash': 0, 'backslash': 0, 'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        max_distance = max(b - t, r - l)

        # loop through every pixel
        for i in range(t, b):
            for j in range(l, r):
                # if the pixel is black
                p1 = img[i, j, 0]
                p2 = img[i, j, 1]
                p3 = img[i, j, 2]
                if p1 < cutoff and p2 < cutoff and p3 < cutoff and p1 == p2 == p3:
                    distances = {'slash': max_distance, 'backslash': max_distance, 'top': max_distance, 'bottom': max_distance, 'left': max_distance, 'right': max_distance}
                    distances['slash'] = get_slash_distance(i, j, l, r, t, b)
                    distances['backslash'] = get_backslash_distance(i, j)
                    distances['top'] = i-t
                    distances['bottom'] = (b-1)-i
                    distances['left'] = j-l
                    distances['right'] = (r-1)-j

                    for key in distances:
                        counts[key] += distances[key]*1.0

        return min(counts, key=counts.get)
    else:
        return None

def get_slash_distance(i, j, l, r, t, b):
    horizontal = abs(((r-1-l)-i)-j)
    vertical = abs(((b-1-t)-j)-i)
    return 0.5 * np.sqrt(vertical**2 + horizontal**2)

def get_backslash_distance(i, j):
    horizontal = abs(j-i)
    vertical = abs(j-i)
    return 0.5 * np.sqrt(vertical**2 + horizontal**2)

def needs_stitch(img, l, t, r, b, cutoff):
    # percentage of black pixels in grid square past which we need to do a backstitch
    threshold = 0.001


    # loop through grid square and count the number of black pixels
    count = 0
    for i in range(t, b):
        for j in range(l, r):
            #if img[i, j, :].tolist() == [0, 0, 0]:
            p1 = img[i, j, 0]
            p2 = img[i, j, 1]
            p3 = img[i, j, 2]
            if p1 < cutoff and p2 < cutoff and p3 < cutoff and p1 == p2 == p3:
                count += 1

    # percentage of the grid square that is black
    percent = (count/((b-t)*(r-l)))

    return 1.0 > percent > threshold


def main():
    filename = './images/originals/shapes.jpg'
    num_gridlines = 100
    backstitch = True

    img = imageio.imread(filename)
    new_img = downres(img, num_gridlines, backstitch)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Original")
    ax2.set_title("After Algorithm")
    ax1.imshow(img)
    ax2.imshow(new_img)
    plt.show()

if __name__=='__main__':
    main()

#edge_stuff()

#arr = imageio.imread("sunrise.jpg")
#arr = imageio.imread("steven_universe.jpg")
#arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
#print(arr.shape)
#new_arr = downres(arr, 100)
#black_removed = remove_black(arr)
#imageio.imwrite("downres_edges.jpeg", new_arr)





#filename = "abc.jpg"
#my_image = get_image(filename)


# def edge_stuff():
#     img = cv2.imread('steven_universe.jpg',0)
#     edges = cv2.Canny(img,100,200)
#
#     plt.subplot(121),plt.imshow(img,cmap = 'gray')
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#     plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#     cv2.imwrite('other_edge.jpg',edges)
#     plt.show()
#
#     img = cv2.imread('steven_universe.jpg')
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray,50,150,apertureSize = 3)
#     minLineLength = 100
#     maxLineGap = 10
#     lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#     for x1,y1,x2,y2 in lines[0]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
#     cv2.imwrite('houghlines5.jpg',img)
#     plt.show()

# def remove_black(a):
#     s = a.shape
#     print(s)
#     for i in range(s[0]):
#         for j in range(s[1]):
#             if a[i, j, 0] == 0 and a[i, j, 1] == 0 and a[i, j, 2] == 0:
#                 a[i, j, 0] = 255
#                 a[i, j, 1] = 0
#                 a[i, j, 2] = 0
#
#     return a

# def get_image(f):
#     return 0
#
# def makeimage(dimensions, color):
#     if color == "red":
#         pixel = [1.0, 0, 0]
#     if color == "green":
#         pixel = [0, 1.0, 0]
#     if color == "blue":
#         pixel = [0, 0, 1.0]
#     if color == "random":
#         pixel = [np.random.uniform(0,1.0), np.random.uniform(0,1.0), np.random.uniform(0,1.0)]
#
#     image = []
#     for i in range(dimensions[0]):
#         row = []
#         for j in range(dimensions[1]):
#             row.append(pixel)
#         image.append(row)
#
#     #print(np.array(image).shape)
#     return np.array(image), pixel