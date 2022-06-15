import os

import matplotlib.pyplot as plt
import pydicom as dicom

def show_slice(scan_num=17, slice_num=33):
    path = 'data/images'
    image = 'images_' + str(scan_num)
    print(image)
    image_dir = os.path.join(path, image)
    slice_num = str(slice_num)
    scan = sorted(os.listdir(image_dir))[slice_num-1]

    test_img = dicom.read_file(scan)
    test_img = test_img.pixel_array
    print(test_img.shape)

    plt.imshow(test_img, cmap=plt.cm.bone)  # set the color map to bone
    plt.show()
    return 0

show_slice()