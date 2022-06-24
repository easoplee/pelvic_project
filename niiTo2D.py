import os
import numpy as np
import nibabel as nib

def niiTo2D(mask_3d_dir, mask_dir):
    mask_3d_dir = mask_3d_dir

    mask_3d_list = sorted(os.listdir(os.path.join(mask_dir, mask_3d_dir)))
    mask_len = len(mask_3d_list)
    print(mask_3d_list)

    prefix = '2d_scan_'
    count = 1
    for i in range(mask_len):
        #scan = np.load(os.path.join(mask_dir, mask_3d_dir, mask_3d_list[i]))
        if mask_3d_list[i].startswith('.DS'):
            continue
        scan = nib.load(os.path.join(mask_dir, mask_3d_dir, mask_3d_list[i])).get_fdata()
        width = scan.shape[0]
        height = scan.shape[1]
        depth = scan.shape[2]
        print("height: ", height, "width: ", width, "depth: ", depth)
        for j in range(depth):
            temp = np.zeros((width, height))
            temp = scan[:,:,j]
            print(temp.shape)
            if count < 10:
                num = '000' + str(count)
            elif count < 100:
                num = '00' + str(count)
            elif count < 1000:
                num = '0' + str(count)
            else:
                num = str(count)

            name = prefix + num + '.npy'
            dir = mask_dir + mask_3d_dir + '_2d/' + name
            print(dir)
            count +=1

            print(np.unique(temp))
            np.save(dir, temp)

# niiTo2D('sacrum_binary_mask', 'mask/')
# niiTo2D('spine_binary_mask', 'mask/')
# niiTo2D('ilium_binary_mask', 'mask/')
# niiTo2D('femur_binary_mask', 'mask/')
niiTo2D('multiclass_mask_final', 'mask/')
# niiTo2D('images_3d', 'data/')