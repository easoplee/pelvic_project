import nibabel as nib
import os
import numpy as np

dir = 'mask/multiclass_mask_corrected'
new_dir = 'mask/multiclass_mask_final'
dir_list = os.listdir(dir)

for each in dir_list:
    if not each.startswith('.DS'):
        mask = nib.load(os.path.join(dir, each))
        new_mask = mask.get_fdata()
        #change 1's to 2's and 2's to 1's
        where_1 = np.where(new_mask == 1)
        where_2 = np.where(new_mask == 2)
        new_mask[where_1] = 2
        new_mask[where_2] = 1

        final_mask = nib.Nifti1Image(new_mask, mask.affine)
        new_name = os.path.join(new_dir, each)

        nib.save(final_mask, new_name)