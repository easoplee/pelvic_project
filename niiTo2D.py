import os
import numpy as np

def niiTo2D(mask_3d_dir):
    mask_dir = 'mask/'
    mask_3d_dir = mask_3d_dir

    mask_3d_list = sorted(os.listdir(os.path.join(mask_dir, mask_3d_dir)))
    mask_len = len(mask_3d_list)
    print(mask_3d_list)

    prefix = '2d_scan_'
    count = 1
    for i in range(mask_len):
        scan = np.load(os.path.join(mask_dir, mask_3d_dir, mask_3d_list[i]))
        width = scan.shape[0]
        height = scan.shape[1]
        depth = scan.shape[2]
        print("height: ", height, "width: ", width, "depth: ", depth)
        for j in range(depth):
            temp = np.zeros((width, height))
            temp = scan[:,:,j]
            print(temp.shape)
            if count < 10:
                num = '00' + str(count)
            elif count < 100:
                num = '0' + str(count)
            else:
                num = str(count)

            name = prefix + num + '.npy'
            dir = mask_dir + mask_3d_dir + '_2d/' + name
            print(dir)
            count +=1

            print(np.unique(temp))
            np.save(dir, temp)

niiTo2D('pelvic_binary_mask')
#niiTo2D('spine_binary_mask')
#niiTo2D('ilium_binary_mask')
#niiTo2D('femur_binary_mask')