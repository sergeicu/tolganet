##########################################
# CHECKING FILE STRUCTURE 
##########################################

###### validation ###### 
cd /fileserver/external/body/abd/anum/dualecho/FD-Net/FD-Net/data/val/val

# choose image 
image=199958_3T_DWI_dir96*slice94.nii

# image info 
fslinfo slices_like_topup/$image # 144 x 168 x 2 
fslinfo slices_field_topup/$image # 144 x 168 x 1  
fslinfo slices_topup/$image # 144 x 168 x 1  

# see images 
ls slices_like_topup/$image # 6 images - volume 0, 16, 32, 48, 64, 80 
ls slices_field_topup/$image # 1 image 
ls slices_topup/$image # 6 images - same as above 

###### train ###### 
cd /fileserver/external/body/abd/anum/dualecho/FD-Net/FD-Net/data/train # only one folder -> slices_like_topup
ls 104416_3T_DWI_dir97_*_slice96.nii # 6 images again 
