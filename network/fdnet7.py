"""This is heavily refactored version of the original code by Serge: 
    - add __name__ call
    - split code into functions 
    - added argparse 
    - split train and test routines 
    """

# -*- coding: utf-8 -*-

import os
import re 

import argparse
import sys 
import glob as glob
from datetime import datetime
from tqdm import tqdm


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nb 

import tensorflow as tf

import wandb

# check if gpu is available 
if not tf.config.experimental.list_physical_devices('GPU'):
    sys.exit("not gpu available / registered") 
    
# load custom 
from fdnet_utils import DataGenerator, print_metrics2, model_compile

# set memory growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
     

    
def load_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,  choices=["train", "test"],default="train",help='train or test')
    parser.add_argument('--dataset', type=str,  choices=["dualecho", "hcp","hcp_tolga12", "tsc"],default="hcp",help='choose which dataset to train / test on')
    
    # parameters 
    parser.add_argument('--rootdir', type=str,  default="/fileserver/external/body/abd/anum/",help='where all the data is contained')
    parser.add_argument('--batch_size', type=int,  default=4)
    parser.add_argument('--patience', type=int,  default=20,help='patience')
    parser.add_argument('--epochs', type=int,  default=200)
    parser.add_argument('--early_stopping_delta', type=float,  default=5e-6, help="early stopping criteria threshold after which training is cut off")
    parser.add_argument('--lr', type=float,  default=1e-4, help="learning rate")
    parser.add_argument('--lambda_reg', type=float,  default=1e-5, help="TBD")
    parser.add_argument('--weights', type=str,  default=None, help="load from saved weights")
    parser.add_argument('--debug', action="store_true", help="sets epochs to 1 and limits data to 2 batches")
    
    parser.add_argument('--savedir', type=str, help="custom save directory for test inferrence (or train)")
    parser.add_argument('--limit_test', type=int, default=None, help="how many slices to run test inferrence over")
    
    parser.add_argument('--legacy', action="store_true", help="uses legacy optimizer -  use for original weights from original paper")
    parser.add_argument('--dontskip', action="store_true", help="dont skip existing files during inferrence")
    parser.add_argument('--fileprefix', type=str, default=None, help="define prefix for a file - only these files will be processed ")
    
    
    parser.add_argument('--save_batch_freq', type=int, default=10, help="how often to save batches")
    
    
    parser.add_argument('--testdir', type=str, help="path to testdir with slices")
    parser.add_argument('--testdir_field', type=str, default=None, help="path to testdir with slices - field")
    parser.add_argument('--testdir_topup', type=str, default=None, help="path to testdir with slices - topup")
    
    args = parser.parse_args()
    
    return args     
    
    
def choose_dataset(rootdir, dataset):
    
    if dataset == 'hcp':
        load_path = rootdir + "data/HCP/"
        min_shape = (144,168)
    elif dataset == 'hcp_tolga12':
        load_path = rootdir + "data/HCP/tolga_12_replicated_v2/"
        min_shape = (144,168)
    elif dataset == 'tsc':
        load_path = rootdir + "data/TSC/"
        min_shape = None
    elif dataset == 'dualecho':
        load_path = rootdir + "data/dualecho/derivatives/topup_python/whole-volume/"
        min_shape = (316, 288)
    else:
        sys.exit('wrong dataset specified')
    
    assert os.path.exists(load_path), f"Path does not exist"
    
    return load_path,min_shape

def check_filenames(list1, list2, list3, load_path, prefix, debug=False):
    newlist1=[]
    newlist2=[]
    newlist3=[]
    for c, (im_f, topup_f, field_f) in enumerate(zip(list1,list2,list3)):
        skip=False 
        
        im = im_f.replace(load_path+prefix+"/slices_input/", "")
        topup = topup_f.replace(load_path+prefix+"/slices_topup_image/","")
        field = field_f.replace(load_path+prefix+"/slices_topup_field/", "")
        
        # find dir number
        im_dir=re.search(r'dir(\d+)', im).group(1)
        topup_dir=re.search(r'dir(\d+)', topup).group(1)
        field_dir=re.search(r'dir(\d+)', field).group(1)
        
        if not im_dir == topup_dir == field_dir:
            skip=True
            if debug:
                print("\n\n\nDIRS DO NOT MATCH. Check im, topup, field names\n\n\n")
                embed()
                break

        # find subject number
        im_id=re.search(r'(\d+)', im).group(1)
        topup_id=re.search(r'(\d+)', topup).group(1)
        field_id=re.search(r'(\d+)', field).group(1)
        
        if not im_id == topup_id == field_id:
            skip=True
            if debug:
                print("\n\n\nSUB IDS DO NOT MATCH. Check im, topup, field names\n\n\n")
                embed()
                break

        # find slice number
        im_sl=re.search(r'slice(\d+)', im).group(1)
        topup_sl=re.search(r'slice(\d+)', topup).group(1)
        field_sl=re.search(r'slice(\d+)', field).group(1)
        
        if not im_sl == topup_sl == field_sl:
            skip=True
            if debug:
                print("\n\n\nSLICES DO NOT MATCH. Check im, topup, field names\n\n\n")
                embed()
                break
            
        if not skip:
            newlist1.append(im_f)
            newlist2.append(topup_f) 
            newlist3.append(field_f) 
            
    print(f"ORIGINAL length: {len(list1)}")
    print(f"UPDATED length: {len(newlist1)}")
            
    return newlist1, newlist2, newlist3

def verify_slices_hcp():
    # custom function that makes sure that slices are matched 
    # also - we remove most 
    pass

def remove_slices(paths, start_end_range):
    
    start_from, end_with = start_end_range
    
    filtered_paths = [path for path in paths if not any(f"_slice{str(i).zfill(4)}.nii.gz" in path for i in range(start_from,end_with))]    
    
    return filtered_paths
    
        
    
def load_data(load_path, batch_size,debug=False, dataset='hcp', fileprefix=None):

    topup_image = "slices_topup_image"
    if dataset == 'dualecho':
        topup_image = topup_image + "_e1"
        
    
    ################
    # Paths init
    ################
    slice_path_train = load_path + "/train/slices_input/*.nii*"
    topup_path_train = load_path + "/train/"+topup_image+"/*.nii*"
    field_path_train = load_path + "/train/slices_topup_field/*.nii*"
    
    slice_path_val = load_path + "/val/slices_input/*.nii*"
    topup_path_val = load_path + "/val/"+topup_image+"/*.nii*"
    field_path_val = load_path + "/val/slices_topup_field/*.nii*"
    
    slice_path_test = load_path + "/test/slices_input/*.nii*"
    topup_path_test = load_path + "/test/"+topup_image+"/*.nii*"
    field_path_test = load_path + "/test/slices_topup_field/*.nii*"
    
    
    # slices to remove 
    if dataset=='hcp':
        # these ranges define top and bottom of head in HCP (we want to remove the edges)
        range1=[0,30]
        range2=[100,111]
        imshape=(144,168)
    elif dataset == 'hcp_tolga12':           
        range1=[0,0]
        range2=[0,0]    
        imshape=(144,168)
    elif dataset == 'dualecho':
        # not removing anything at the moment as number of slices varies significantly per subject - need to curate better data
        range1=[0,0]
        range2=[0,0]    
        imshape=(316,288)    

    ################
    # Get files and sort 
    ################
        
    # train 
    list_slices_train = glob.glob(slice_path_train)
    list_topups_train = glob.glob(topup_path_train)
    list_fields_train = glob.glob(field_path_train)

    list_slices_train.sort()
    list_topups_train.sort()
    list_fields_train.sort()

    list_train = list_slices_train
    list_topup_train = list_topups_train
    list_field_train = list_fields_train
    

    # val
    list_slices_val = glob.glob(slice_path_val)
    list_topups_val = glob.glob(topup_path_val)
    list_fields_val = glob.glob(field_path_val)

    list_slices_val.sort()
    list_topups_val.sort()
    list_fields_val.sort()

    list_val   = list_slices_val
    list_topup_val = list_topups_val
    list_field_val = list_fields_val
    
    # test
    list_slices_test = glob.glob(slice_path_test)
    list_topups_test = glob.glob(topup_path_test)
    list_fields_test = glob.glob(field_path_test)

    list_test  = list_slices_test
    list_topup_test = list_topups_test
    list_field_test = list_fields_test

    list_test.sort()
    list_topup_test.sort()
    list_field_test.sort()
    
    
    ############    
    # Check files
    ############    

        
    # remove slices 
    for rangei in [range1,range2]:
        list_train = remove_slices(list_train, rangei)
        list_topup_train = remove_slices(list_topup_train, rangei)
        list_field_train = remove_slices(list_field_train, rangei)
        
        list_val = remove_slices(list_val, rangei)
        list_topup_val = remove_slices(list_topup_val, rangei)
        list_field_val = remove_slices(list_field_val, rangei)
        
        list_test = remove_slices(list_test, rangei)
        list_topup_test = remove_slices(list_topup_test, rangei)
        list_field_test = remove_slices(list_field_test, rangei)    

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
    # [custom to each dataset] 
    # Check that files match 
    ############    
    
    
    if dataset == 'hcp' or 'hcp_tolga12':

        # check if files match
        list_train, list_topup_train, list_field_train = check_filenames(list_train, list_topup_train, list_field_train, load_path, '/val/', debug=debug)
    
        # check if files match
        list_val, list_topup_val, list_field_val = check_filenames(list_val,list_topup_val, list_field_val, load_path, '/val/', debug=debug)
        
        # let's check if the subjectID, and slice numbers match 
        list_test,list_topup_test, list_field_test= check_filenames(list_test,list_topup_test, list_field_test, load_path, '/test/',debug=debug)
        

        # let's limit the size of val to a few batches of data instead -> e.g. 80 slices only 
        list_val = list_val[:10*batch_size]
        list_topup_val = list_topup_val[:10*batch_size]
        list_field_val = list_field_val[:10*batch_size]    
        
    if dataset == 'dualecho':

        list_length = 5 if len(list_field_val) >=5 else len(list_field_val)
        # from IPython import embed; embed()
        random_numbers = np.random.choice(range(1, len(list_field_val)), list_length, replace=True)
        for i in random_numbers:
            
            i1=list_train[i]
            i2=list_topup_train[i]
            i3= list_field_train[i]
            assert os.path.basename(i1) == os.path.basename(i2) == os.path.basename(i3)

            i1=list_val[i]
            i2=list_topup_val[i]
            i3= list_field_val[i]
            assert os.path.basename(i1) == os.path.basename(i2) == os.path.basename(i3)        
            
            
            if list_test:
                i1=list_test[i]
                i2=list_topup_test[i]
                i3= list_field_test[i]
                assert os.path.basename(i1) == os.path.basename(i2) == os.path.basename(i3)      
            else:
                list_test = list_val
                list_topup_test = list_topup_val
                list_field_test = list_field_val
        
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
    
    # wandb setup 
    wandb.login()
    if wandb.run is not None:
        wandb.finish()    
    if args.mode != 'test':
        project="tolganet-hcp"    
        run = wandb.init(
            # Set the project where this run will be logged
            project=project, 
            tags=['debug'], 
            notes='basic test')

        config = dict(test_name='basic-test')
                        # initial_width=64,base_width=10, current_width=10,
                        # dropout=True,dropout_rate=0.2,
                        # epochs=600,learning_rate = 0.0001,
                        # patience=100, output_size=2,batch_size=8,

        w = wandb.config = config        


    # data 
    data_path, imshape = choose_dataset(args.rootdir, args.dataset)    
    dg_train, dg_val, dg_test,list_test = load_data(data_path, args.batch_size, args.debug,args.dataset,args.fileprefix)

    # compile model 
    #optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    if args.legacy:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model = model_compile(optimizer=optimizer, reg=args.lambda_reg, input_shape=imshape)

    # where to save 
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M") 
    prefix = current_time if not args.debug else "debug"
    savedir = data_path + "output/" + prefix + "/"
    
    # limit epochs if debug
    epochs = args.epochs if not args.debug else 1

    if args.mode == 'train':
        
        # early stopping 
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=args.early_stopping_delta,
            patience=args.patience,
            verbose=3,
            mode="min",
            baseline=None,
            restore_best_weights=True
            )       
        
        callback2 = tf.keras.callbacks.ModelCheckpoint(
            savedir + "/"+"ckpt-{batch:02d}-{loss:.2f}.h5", 
            monitor='val_loss',
            save_best_only=False,
            mode='min', 
            save_weights_only=True,
            save_freq=args.save_batch_freq) #            period=10,
        
        callback3 = wandb.keras.WandbMetricsLogger(log_freq="batch")
                
        class BatchValidationCallback(tf.keras.callbacks.Callback):
            def __init__(self, val_data, batch_freq):
                super(BatchValidationCallback, self).__init__()
                self.val_data = val_data
                self.batch_freq = batch_freq
                self.batch_counter = 0

            def on_batch_end(self, batch, logs=None):
                self.batch_counter += 1
                if self.batch_counter % self.batch_freq == 0:
                    val_logs = self.model.evaluate(self.val_data[0], self.val_data[1], verbose=0)
                    for key, value in val_logs.items():
                        print(f'Validation {key}: {value}')                        
                        
                        
        # train 
        #from IPython import embed; embed()
        hist = model.fit(
            dg_train,
            epochs=epochs,
            validation_data=dg_val,
            callbacks=[callback, callback2,callback3]
            )
        
        # save weights (at the end)
        model.save_weights(savedir + "/weights-name")                    

        # plot results after training 
        plt.figure()
        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.ylabel("Loss")
        plt.xlabel("Epoch Number")
        plt.legend(["Train", "Val"], loc="upper right")
        plt.show()
        # plt.savefig("plot-name", dpi=300, bbox_inches="tight")

        # save training history 
        hist_df = pd.DataFrame(hist.history)
        os.makedirs(savedir, exist_ok=True)
        print(f"Results saved to:\n{savedir}")
        hist_csv_file = savedir + "/file-name.csv"
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            

    elif args.mode == 'test':
                
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
            if args.testdir_field is None and args.testdir_topup is None: 
                no_topup = True
                train = True 
                
                # build a generator 
                list_test = sorted(glob.glob(args.testdir + "*.nii.gz"))
                assert list_test
                list_topup_test = None 
                list_field_test = None 
                
                dg_test  = DataGenerator(
                    list_test,
                    list_topup_test,
                    list_field_test,
                    batch_size=args.batch_size,
                    shuffle=False,
                    train=train,
                    dim=imshape,
                    )
                
                
            else:
                no_topup = False    
                train = False
                sys.exit("Not implemented")    
        
        
             
            
        # create generator for list names 
        dg_test_names = batch_generator(list_test, batch_size=args.batch_size)
        
        
        # get test results
        
        L = len(dg_test)
        dfs = []
        for counter, (dat, names) in enumerate(zip(dg_test,dg_test_names)):
            print(f"{counter}/{L}")
            
            if args.limit_test is not None: 
                if counter > args.limit_test/args.batch_size:
                    print(f"\n\n\n\nExiting. Already processed {args.limit_test} slices\n\n\n")
                    sys.exit("Finished.") 
                
            
            
            

            for i in range(0, args.batch_size):
                
                # dat = dg_test[j]
                X = dat[0]

                # filename 
                file_name = os.path.basename(names[i])

                # skip existing 
                if os.path.exists(savedir+file_name.replace(".nii.gz", "_XLR.nii.gz")) and not args.dontskip:
                    continue                
                
                
                # predict
                Y, Y1, Y2, Y3, rigid = model.predict(X, verbose=0)
                        

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

                if not no_topup: 
                    topup_image =  dat[1][0][i,:,:,0,3]
                    topup_field = dat[1][0][i,:,:,0,2]                    
                
                    psnr_i, psnr_f, ssim_i, ssim_f  = print_metrics2(topup_image,topup_field,network_image,network_field,file_name,mask_field=True)  
                    vol = re.sub(r'_slice.*$', '', file_name)
                    slicee=file_name[-11:-7]
                    # df = pd.DataFrame(psnr_image=psnr_i, psnr_field=psnr_f, ssim_image=ssim_i,ssim_field=ssim_f,file=vol, slicee=slicee)     
                    df = pd.DataFrame({'psnr_image': [psnr_i], 'psnr_field': [psnr_f], 'ssim_image': [ssim_i], 'ssim_field': [ssim_f], 'file': [vol], 'slicee': [slicee]})

                    dfs.append(df)
                    with open(savedir+file_name.replace(".nii.gz","_metrics.txt"), "w") as file:
                        df.to_csv(file)                    
                    
                    nb.save(nb.Nifti1Image(topup_image.numpy(),affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_topup_image.nii.gz"))
                    nb.save(nb.Nifti1Image(topup_field.numpy(),affine=imo.affine,header=imo.header), savedir+file_name.replace(".nii.gz", "_topup_field.nii.gz"))
                    

        # get all metrics 
        # with open(savedir+"/combined_metrics.txt", "w") as file:
        #     pd.concat(dfs).to_csv(file)                       
                    
        print(f"Files are saved to: {savedir}")

            



