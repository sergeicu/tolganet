"""We are optimizing the network for inferrence here"""

import os
import argparse
import sys 
import glob as glob
from datetime import datetime


import numpy as np
import nibabel as nb 

import tensorflow as tf
import time 

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
    parser.add_argument('--dataset', type=str,  choices=["custom", "individual_RLRL"],default="custom",help='choose dataset type')
    
    # parameters 
    parser.add_argument('--rootdir', type=str,  default="/fileserver/external/body/abd/anum/",help='where all the data is contained')
    parser.add_argument('--batch_size', type=int,  default=4)
    parser.add_argument('--patience', type=int,  default=20,help='patience')
    parser.add_argument('--early_stopping_delta', type=float,  default=5e-6, help="early stopping criteria threshold after which training is cut off")
    parser.add_argument('--lr', type=float,  default=1e-4, help="learning rate")
    parser.add_argument('--lambda_reg', type=float,  default=1e-5, help="TBD")
    parser.add_argument('--weights', type=str,  default=None, help="load from saved weights")
    parser.add_argument('--debug', action="store_true", help="sets epochs to 1 and limits data to 2 batches")
    
    parser.add_argument('--resume_training', type=str, default=None, help="provide path to saved weights")
    
    
    
    
    # try different losses
    parser.add_argument('--loss', type=str,  choices=["mse", "nce", "mine", "mi"],default="mse",help='which loss to use')
    
    
    parser.add_argument('--savedir', type=str, help="custom save directory for test inferrence (or train)")
    parser.add_argument('--legacy', action="store_true", help="uses legacy optimizer -  use for original weights from original paper")
    parser.add_argument('--dontskip', action="store_true", help="dont skip existing files during inferrence")
    parser.add_argument('--fileprefix', type=str, default=None, help="define prefix for a file - only these files will be processed ")
    
    
    parser.add_argument('--save_batch_freq', type=str, default=10, help="how often to save - if epoch - saves every epoch, if number - saves every N batches")
    parser.add_argument('--period', type=int, default=10, help="how many epochs to save")
    parser.add_argument('--verbose', type=int, default=3, help="how much to print")

    
    parser.add_argument('--limit_test', type=int, default=None, help="how many slices to run test inferrence over")
    parser.add_argument('--testdir', type=str, help="path to testdir with slices")
    parser.add_argument('--testdir_field', type=str, default=None, help="path to testdir with slices - field")
    parser.add_argument('--testdir_topup', type=str, default=None, help="path to testdir with slices - topup")
    
    
    parser.add_argument('--custompath', type=str, default=None, help="custom path to train data")
    parser.add_argument('--customshape', nargs="+", type=int, default=None, help="custom shape")
    parser.add_argument('--notopup', action="store_true", help="skip topup validation")
    
    
    parser.add_argument('--nowandb', action="store_true", help="skip weights and biases logging")
    
    parser.add_argument('--name', type=str, default=None, help="weights and biases run name")
    parser.add_argument('--project', type=str, default='tolganet-hcp', help="weights and biases project name")
    parser.add_argument('--tags', type=str, nargs="+", default="", help="weights and biases tags")
    
    args = parser.parse_args()
    
    return args     
    
    
    
def load_data(load_path, batch_size,debug=False, fileprefix=None,imshape=None):
    
    topup=False
    

        
    load_path = load_path + "/"
     
    ################
    # Paths init
    ################

    slice_path_train = load_path + "*.nii*"
    slice_path_val = load_path + "*.nii*"
    slice_path_test = load_path + "*.nii*"
        
     

    ################
    # Get files and sort 
    ################
        
    # inputs - train 
    list_train = sorted(glob.glob(slice_path_train)) 
    list_val = sorted(glob.glob(slice_path_val))
    list_test = sorted(glob.glob(slice_path_test))

    
    # define topup inputs
    if topup:
        
        # train 
        list_topup_train = sorted(glob.glob(topup_path_train))
        list_field_train = sorted(glob.glob(field_path_train))

        # val 
        list_topup_val = sorted(glob.glob(topup_path_val))
        list_field_val = sorted(glob.glob(field_path_val))
    
        # test 
        list_topup_test = sorted(glob.glob(topup_path_test))
        list_field_test = sorted(glob.glob(field_path_test))
        
    else: 
        list_topup_train = list_field_train = list_topup_val = list_field_val = list_topup_test = list_field_test = None

    # slices to remove 
    assert imshape is not None 
    assert len(imshape)==2

    

    
    

    ############    
    # choose specific files
    ############    
    
    list_test_t = []
    list_topup_test_t = []
    list_field_test_t = []
    
    if fileprefix is not None:
        for i1,i2,i3 in zip(list_test,list_topup_test,list_field_test):
            if fileprefix in i1: 
                list_test_t.append(i1)
                list_topup_test_t.append(i2)
                list_field_test_t.append(i3)
                
        list_test = list_test_t
        list_topup_test = list_topup_test_t
        list_field_test = list_field_test_t
        assert list_test, f"No files matched to fileprefix"
        

    ############    
    # DEBUG - shorten lists 
    ############    

    if debug:
        list_train = list_train[:2*batch_size]
        list_topup_train = list_topup_train[:2*batch_size]
        list_field_train = list_field_train[:2*batch_size]
        
        list_val = list_val[:2*batch_size]
        list_topup_val = list_topup_val[:2*batch_size]
        list_field_val = list_field_val[:2*batch_size]    
        
        list_test = list_test[:2]
        list_topup_test = list_topup_test[:2]
        list_field_test = list_field_test[:2]        

    ############    
    # GENERATORS 
    ############    
    
    
    dg_train = DataGenerator(
        list_train,
        list_topup_train,
        list_field_train,
        batch_size=batch_size,
        shuffle=True,
        train=True, 
        dim=imshape,
        )
        

    dg_val   = DataGenerator(
        list_val,
        list_topup_val,
        list_field_val, 
        batch_size=batch_size,
        shuffle=True,
        train=True,
        dim=imshape,
        )



    dg_test  = DataGenerator(
        list_test,
        list_topup_test,
        list_field_test,
        batch_size=batch_size,
        shuffle=False,
        dim=imshape,
        )
    

    return dg_train,dg_val, dg_test, list_test, 

def batch_generator(lst, batch_size=8):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
    
if __name__ == '__main__':
    
    

    
    
    args = load_args()



    # data 
    data_path = args.custompath
    imshape = args.customshape    
    assert os.path.exists(data_path), f"Path does not exist"
    
    dg_train, dg_val, dg_test,list_test = load_data(data_path, args.batch_size, args.debug, args.fileprefix, args.customshape)
    
    

    # compile model 
    if args.legacy:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model = model_compile(optimizer=optimizer, reg=args.lambda_reg, input_shape=imshape,loss_type=args.loss,resume=args.resume_training)
    

    

    # where to save 
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M") 
    prefix = current_time if not args.debug else "debug"
    savedir = data_path + "/weights/" + prefix + "/"


            
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
    
    #%% LOAD WEIGHTS
    model.load_weights(weights)
    
    
    if args.savedir is None:
        savedir = os.path.dirname(weights) + "/"    
        os.makedirs(savedir +"predicted/", exist_ok=True)
    else: 
        savedir = args.savedir + "/"
        os.makedirs(savedir, exist_ok=True)
        print(f"Files are saved to: {savedir}")
            
            
    # pull results from customdir
    if args.testdir: 


    
        train = True 
        
        # build a generator 
        list_test = sorted(glob.glob(args.testdir + "/*.nii.gz"))
        assert list_test
        
        dg_test  = DataGenerator(
            list_test,
            None,
            None,
            batch_size=args.batch_size,
            shuffle=False,
            train=False,
            dim=imshape,
            )
                

        
        
    # create generator for list names 
    dg_test_names = batch_generator(list_test, batch_size=args.batch_size)
    
    
    # get test results
    
    L = len(dg_test)
    dfs = []
    
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"START time: {execution_time} seconds")      
        
    for counter, (dat, names) in enumerate(zip(dg_test,dg_test_names)):
        print(f"{counter}/{L}")
        
        if args.limit_test is not None: 
            if counter > args.limit_test/args.batch_size:
                print(f"\n\n\n\nExiting. Already processed {args.limit_test} slices\n\n\n")
                sys.exit("Finished.") 
            
        
        
    
        i=0 # WARNING!!!! we should refer to specific name... 
        
        
        # dat = dg_test[j]
        X = dat[0]

        # filename 
        file_name = os.path.basename(names[i]) 

        # skip existing 
        if os.path.exists(savedir+file_name.replace(".nii.gz", "_XLR.nii.gz")) and not args.dontskip:
            continue                
        
        start_time = time.time()
        
        # predict
        # from IPython import embed; embed()
        Y, Y1, Y2, Y3, rigid = model.predict(X, verbose=0)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        
                

        # fetch individual data
        XLR = X[0][i,:,:,0]
        XRL = X[1][i,:,:,0]
        YLR = Y[i,:,:,0,0]
        YRL = Y[i,:,:,0,1]
        network_image = Y[i,:,:,0,3]
        network_field = Y[i,:,:,0,2]  
        rigid_transform = rigid[i,:]
                            
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
    







        



