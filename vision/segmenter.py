import cv2

"""
# structure of an image array as returned by cv2.imread(img_filepath)
# the colour channels are not in the RGB pattern but BGR pattern (blue,green,red)

[
    [ [255,   0,   0], [  0, 255,   0], [  0,   0, 255] ],  # Row 0 (Top row)
    [ [  0,   0,   0], [255, 255, 255], [128, 128, 128] ]   # Row 1 (Bottom row)
]

"""



def segment_image(img_path : str, n : int = 10):

    # imread returns a 3D numpy array (height,width,colour_channels)
    img_arr = cv2.imread(img_path)
    h, w = img_arr.shape[:2]

    print('height of the image: ')
    print('width of the image: ')

    # this will eventually miss segments of the image
    cell_h = h // n
    cell_w = w // n

    cells = []

    for r in range(n):

        cells.append([])

        for c in range(n):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w

            cells[r].append(img_arr[y1:y2,x1:x2])

    print(cells)
    return cells