import glob 
import os 
import sys 

import nibabel as nb 
import numpy as np 
from tqdm import tqdm


if __name__ == '__main__':
    indir = sys.argv[1] + "/"
    outdir = sys.argv[2]+ "/"



    files = glob.glob(indir + "*.nii.gz")
    assert files 
    os.makedirs(outdir, exist_ok=True)
    L = len(files)
    for f in tqdm(files): 
        
        basename = os.path.basename(f)
        #print(f"{basename}")
        imo = nb.load(f)
        newimo = nb.Nifti1Image(imo.get_fdata(), affine=np.eye(4))
        nb.save(newimo, outdir + basename)