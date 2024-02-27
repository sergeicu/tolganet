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


# check if gpu is available 
if not tf.config.experimental.list_physical_devices('GPU'):
    sys.exit("not gpu available / registered") 
    
# load custom 
from fdnet_utils import DataGenerator, print_metrics, model_compile

# set memory growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
     

    
def load_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,  choices=["train", "test"],default="train",help='train or test')
    parser.add_argument('--dataset', type=str,  choices=["dualecho", "hcp", "tsc"],default="hcp",help='choose which dataset to train / test on')
    
    # parameters 
    parser.add_argument('--rootdir', type=str,  default="/fileserver/external/body/abd/anum/",help='where all the data is contained')
    parser.add_argument('--batch_size', type=int,  default=8)
    parser.add_argument('--patience', type=int,  default=1,help='patience')
    parser.add_argument('--epochs', type=int,  default=1)
    parser.add_argument('--early_stopping_delta', type=float,  default=5e-6, help="early stopping criteria threshold after which training is cut off")
    parser.add_argument('--lr', type=float,  default=1e-4, help="learning rate")
    parser.add_argument('--lambda_reg', type=float,  default=1e-5, help="TBD")
    parser.add_argument('--weights', type=str,  default=None, help="load from saved weights")
    parser.add_argument('--debug', action="store_true", help="sets epochs to 1 and limits data to 2 batches")
    
    args = parser.parse_args()
    
    return args     
    
    
def choose_dataset(rootdir, dataset):
    
    if dataset == 'hcp':
        load_path = rootdir + "data/HCP/"
    elif dataset == 'tsc':
        load_path = rootdir + "data/TSC/"
    elif dataset == 'dualecho':
        load_path = rootdir + "data/dualecho/derivatives/topup_python/whole-volume/"
    else:
        sys.exit('wrong dataset specified')
    
    assert os.path.exists(load_path), f"Path does not exist"
    
    return load_path

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
    
        
    
def load_data(load_path, batch_size,debug=False, dataset='hcp'):

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
    
    if dataset=='hcp':
        range1=[0,30]
        range2=[100,111]
    elif dataset == 'dualecho':
        range1=[0,0]
        range2=[0,0]        
        
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
    
    
    if dataset == 'hcp':

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
    
    xx,yy,_ = nb.load(list_train[0]).shape
    
    dg_train = DataGenerator(
        list_train,
        list_topup_train,
        list_field_train,
        batch_size=batch_size,
        shuffle=True,
        train=True, 
        dim=(xx,yy),
        )
        

    dg_val   = DataGenerator(
        list_val,
        list_topup_val,
        list_field_val, 
        batch_size=batch_size,
        shuffle=True,
        train=True,
        dim=(xx,yy),
        )



    dg_test  = DataGenerator(
        list_test,
        list_topup_test,
        list_field_test,
        batch_size=1,
        shuffle=False,
        dim=(xx,yy),
        )
    

    return dg_train,dg_val, dg_test, list_test, (xx,yy)


    
if __name__ == '__main__':
    
    args = load_args()

    # data 
    data_path = choose_dataset(args.rootdir, args.dataset)
    dg_train, dg_val, dg_test,list_test, imshape = load_data(data_path, args.batch_size, args.debug,args.dataset)

    # compile model 
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model = model_compile(optimizer=optimizer, reg=args.lambda_reg, input_shape=(imshape[0],imshape[1],1))

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


        # from IPython import embed; embed() 
        # from IPython.core.interactiveshell import InteractiveShell
        # InteractiveShell.ast_node_interactivity = "all" 
        # x,y = next(iter(dg_train))
        # len(x)
        # for xi in x:
        #     xi.shape
        # len(y)
        # for yi in y:
        #     yi.shape
        # x2,y2 = next(iter(dg_val))

        # train 
        hist = model.fit(
            dg_train,
            epochs=epochs,
            validation_data=dg_val,
            callbacks=[callback]
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
        weights = weights.replace(".data-00000-of-00001", "")
        weights = weights.replace(".index", "")
        assert os.path.exists(weights+ ".index"), f"weights .index file does not exist"
        assert os.path.exists(weights+ ".data-00000-of-00001"), f"weights .data-00000-of-00001 file does not exist"
        
        #%% LOAD WEIGHTS
        model.load_weights(weights)
        
                
        # get test results
        for j in tqdm(range(len(dg_test))):
            dat = dg_test[j]
            X = dat[0]
            Y, Y1, Y2, Y3, rigid = model.predict(X, verbose=0)
                    
            file_name = list_test[j].split('/')[-1]
            
            i = 0 # batch index
                
            #images
            XLR = X[0][i,:,:,0]
            XRL = X[1][i,:,:,0]
            YLR = Y[i,:,:,0,0]
            YRL = Y[i,:,:,0,1]
            topup_image =  dat[1][0][i,:,:,0,3]
            network_image = Y[i,:,:,0,3]
            topup_field = dat[1][0][i,:,:,0,2]
            network_field = Y[i,:,:,0,2]  
            rigid_transform = rigid[i,:]




        ## NOT SURE WHAT THIS DOES BUT OK 
        
        
        #%% PRINT METRICS (DWI)
        os.makedirs(savedir, exist_ok=True)
        print(f"Results saved to:\n{savedir}")
        list_print  = glob.glob(savedir + "/test/predict/slices_like_topup/*.nii")
        list_topup_print = glob.glob(savedir + "/test/predict/slices_topup/*.nii")
        list_field_print = glob.glob(savedir + "/test/predict/slices_field_topup/*.nii")

        dg_test_print  = DataGenerator(
            list_print,
            list_topup_print,
            list_field_print,
            batch_size=1,
            shuffle=False
            )

        for j in tqdm(range(len(dg_test_print))):
            dat = dg_test_print[j]
            X = dat[0]
            Y, _, _, _, abc = model.predict(X, verbose=0)
                    
            file_name = list_print[j].split('/')[-1]
                
            i = 0 # batch index
            
            #images
            topup_image =  dat[1][0][i,:,:,0,3]
            network_image = Y[i,:,:,0,3]
            topup_field = dat[1][0][i,:,:,0,2]
            network_field = Y[i,:,:,0,2]  
            
            print_metrics(topup_image,
                        topup_field,
                        network_image,
                        network_field,
                        file_name,
                        mask_field=True,
                        ext="DWI")    


