"""We are optimizing the network for inferrence here"""

import os
import argparse
import sys 
import time 
import glob as glob
from datetime import datetime

import nibabel as nb 
import tensorflow as tf


# measure init loading time 
start_time = time.time()

# check if gpu is available 
if not tf.config.experimental.list_physical_devices('GPU'):
    sys.exit("not gpu available / registered") 
    
# load custom 
from fdnet_utils import DataGenerator, model_compile

# set memory growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
     

    
def load_args(): 
    parser = argparse.ArgumentParser()
    
    # files & folders 
    parser.add_argument('--custompath', type=str, default=None, help="custom path to train data")
    parser.add_argument('--customshape', nargs="+", type=int, default=None, help="custom shape")
    parser.add_argument('--savedir', type=str, help="custom save directory for test inferrence (or train)")

    # network 
    parser.add_argument('--batch_size', type=int,  default=4)
    parser.add_argument('--weights', type=str,  default=None, help="load from saved weights")
    
    # misc 
    parser.add_argument('--legacy', action="store_true", help="uses legacy optimizer -  use for original weights from original paper")
    parser.add_argument('--dontskip', action="store_true", help="dont skip existing files during inferrence")
    parser.add_argument('--verbose', type=int, default=3, help="how much to print")
    parser.add_argument('--debug', action="store_true", help="sets epochs to 1 and limits data to 2 batches")
        
    # model compiler settings 
    parser.add_argument('--loss', type=str,  choices=["mse", "nce", "mine", "mi"],default="mse",help='which loss to use')
    parser.add_argument('--lr', type=float,  default=1e-4, help="learning rate")
    parser.add_argument('--lambda_reg', type=float,  default=1e-5, help="TBD")
    
    args = parser.parse_args()
    
    return args     
    
    
# helper function to yield file names 
def batch_generator(lst, batch_size=8):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
    
if __name__ == '__main__':
    args = load_args()

    # data (legacy)
    assert os.path.exists(args.custompath), f"Path does not exist"

    # compile model 
    if args.legacy:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model = model_compile(optimizer=optimizer, reg=args.lambda_reg, input_shape=args.customshape,loss_type=args.loss)
    
    # where to save 
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M") 
    prefix = current_time if not args.debug else "debug"
    savedir = args.custompath + "/weights/" + prefix + "/"
 
    # check weights 
    weights = args.weights
    assert weights is not None 
    if args.weights.endswith(".h5"):
        assert os.path.exists(weights)
    else:
    
        # check for .index files if not saved as .h5 
        weights = weights.replace(".data-00000-of-00001", "")
        weights = weights.replace(".index", "")
        assert os.path.exists(weights+ ".index"), f"weights .index file does not exist"
        assert os.path.exists(weights+ ".data-00000-of-00001"), f"weights .data-00000-of-00001 file does not exist"
    
    # LOAD WEIGHTS
    model.load_weights(weights)
    
    # savedir
    if args.savedir is None:
        savedir = os.path.dirname(weights) + "/"    
        os.makedirs(savedir +"predicted/", exist_ok=True)
    else: 
        savedir = args.savedir + "/"
        os.makedirs(savedir, exist_ok=True)
        print(f"Files are saved to: {savedir}")
                
    # create generator for list names 
    list_test = sorted(glob.glob(args.custompath + "/*.nii.gz"))
    assert list_test
    dg_test_names = batch_generator(list_test, batch_size=args.batch_size)    
    
    dg_test  = DataGenerator(
        list_test,
        None,
        None,
        batch_size=args.batch_size,
        shuffle=False,
        train=False,
        dim=args.customshape,
        )
        

    
    # get test results
    L = len(dg_test)
    
 
    # measure initialization time   
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"START time: {execution_time} seconds")      
        
        
    # MAIN LOOP - load files in batches
    for counter, (dat, names) in enumerate(zip(dg_test,dg_test_names)):
        print(f"{counter}/{L}")

    
        i=0 # WARNING!!!! we should refer to specific name... this is not correct 
        
        
        # dat = dg_test[j]
        X = dat[0]

        # filename 
        file_name = os.path.basename(names[i]) 

        # skip existing 
        if os.path.exists(savedir+file_name.replace(".nii.gz", "_XLR.nii.gz")) and not args.dontskip:
            continue                
        
        # measure PER batch time
        start_time = time.time()
        
        # MAIN NEURAL NETWORK CALL - predicts images
        Y, Y1, Y2, Y3, rigid = model.predict(X, verbose=0)
        
        
        # measure PER batch time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        
        # fetch predicted data
        XLR = X[0][i,:,:,0]
        XRL = X[1][i,:,:,0]
        YLR = Y[i,:,:,0,0]
        YRL = Y[i,:,:,0,1]
        network_image = Y[i,:,:,0,3]
        network_field = Y[i,:,:,0,2]  
        rigid_transform = rigid[i,:]
        
        #########################
        # SAVE IMAGES 
        #########################        
                            
        # get affine 
        imo=nb.load(names[i])

        # save 
        nb.save(nb.Nifti1Image(XLR.numpy(),affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_XLR.nii.gz"))
        nb.save(nb.Nifti1Image(XRL.numpy(),affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_XRL.nii.gz"))
        nb.save(nb.Nifti1Image(YLR,affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_YLR.nii.gz"))
        nb.save(nb.Nifti1Image(YRL,affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_YRL.nii.gz"))
        
        nb.save(nb.Nifti1Image(network_image,affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_network_image.nii.gz"))
        nb.save(nb.Nifti1Image(network_field,affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_network_field.nii.gz"))
        with open(savedir+file_name.replace(".nii.gz","_rigid_transform.txt"), "w") as f:
            lines = [str(i) for i in list(rigid_transform)]
            f.writelines(lines)
            
                
    print(f"Files are saved to: {savedir}")
    







        



