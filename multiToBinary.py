# given a multi-class mask list, convert each 3D scan to binary mask
# nii.gz -> npy

from multiprocessing import connection
import nibabel as nib
import os
import numpy as np

def multiToBinary(bone_name):
    choice = bone_name

    NameToClass = {
        'background': 0,
        'ilium': 1,
        'pelvic': 2,
        'spine': 3,
        'sacrum': 4
    }

    choice_num = NameToClass[choice]

    non_interest = [1, 2, 3, 4]
    non_interest.remove(choice_num)

    mask_dir = 'mask/'
    multimask_dir = 'mask/multiclass_mask_corrected'
    mask_content = sorted(os.listdir(multimask_dir))
    mask_content_len = len(mask_content)
    for j in range(mask_content_len):
        scan_dir = mask_content[j]
        if scan_dir.startswith('.DS'):
            continue
        scan_content = os.path.join(multimask_dir, scan_dir)
        scan = nib.load(scan_content).get_fdata()
        orig = scan.copy()

        height = orig.shape[0]
        width = orig.shape[1]
        depth = orig.shape[2]

        print("height: ", height, "width: ", width, "depth: ", depth)

        for i in range(3):
            where_element = np.where(orig==non_interest[i])
            orig[where_element] = 0

        print(np.unique(orig))

        new_dir = os.path.join(mask_dir, choice + '_binary_mask/' + scan_dir[:len(scan_dir)-4] + '.npy')
        print(new_dir)

        np.save(new_dir, orig)

multiToBinary('pelvic')