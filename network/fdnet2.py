"""This is heavily refactored version of the original code by Serge: 
    - add __name__ call
    - split code into functions 
    - added argparse 
    - split train and test routines 
    """

# -*- coding: utf-8 -*-

import os


import argparse
import sys 
import glob as glob
from datetime import datetime
from tqdm import tqdm

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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
        load_path = rootdir + "data/dualecho/"
    else:
        sys.exit('wrong dataset specified')
    
    assert os.path.exists(load_path), f"Path does not exist"
    
    return load_path
    
    
def load_data(load_path, batch_size,debug=False):
    
    # from IPython import embed; embed()

    # train
    slice_path_train = load_path + "/train/slices_like_topup/*.nii"
    topup_path_train = None
    field_path_train = None

    list_slices_train = glob.glob(slice_path_train)
    list_topups_train = None
    list_fields_train = None

    list_slices_train.sort()

    list_train = list_slices_train
    list_topup_train = list_topups_train
    list_field_train = list_fields_train
    
    if debug:
        list_train = list_train[:2*batch_size]
        #list_topup_train = list_topup_train[:2*batch_size]
        #list_field_train = list_field_train[:2*batch_size]

    dg_train = DataGenerator(
        list_train,
        list_topup_train,
        list_field_train,
        batch_size=batch_size,
        shuffle=True,
        train=True
        )

    # val
    slice_path_val = load_path + "/val/val/slices_like_topup/*.nii"
    topup_path_val = load_path + "/val/val/slices_topup/*.nii"
    field_path_val = load_path + "/val/val/slices_field_topup/*.nii"

    list_slices_val = glob.glob(slice_path_val)
    list_topups_val = glob.glob(topup_path_val)
    list_fields_val = glob.glob(field_path_val)

    list_slices_val.sort()

    list_val   = list_slices_val
    list_topup_val = list_topups_val
    list_field_val = list_fields_val
    
    if debug:
        list_val = list_val[:2*batch_size]
        list_topup_val = list_topup_val[:2*batch_size]
        list_field_val = list_field_val[:2*batch_size]    

    dg_val   = DataGenerator(
        list_val,
        list_topup_val,
        list_field_val, 
        batch_size=batch_size,
        shuffle=True,
        train=True
        )

    # test
    slice_path_test = load_path + "/test/test/slices_like_topup/*.nii"
    topup_path_test = load_path + "/test/test/slices_topup/*.nii"
    field_path_test = load_path + "/test/test/slices_field_topup/*.nii"

    list_slices_test = glob.glob(slice_path_test)
    list_topups_test = glob.glob(topup_path_test)
    list_fields_test = glob.glob(field_path_test)

    list_test  = list_slices_test
    list_topup_test = list_topups_test
    list_field_test = list_fields_test

    list_test.sort()
    list_topup_test.sort()
    list_field_test.sort()
    
    if debug:
        list_test = list_val[:2]
        list_topup_test = list_topup_test[:2]
        list_field_test = list_field_test[:2]        

    dg_test  = DataGenerator(
        list_test,
        list_topup_test,
        list_field_test,
        batch_size=1,
        shuffle=False
        )

    return dg_train,dg_val, dg_test, list_test


    
if __name__ == '__main__':
    
    args = load_args()

    # data 
    data_path = choose_dataset(args.rootdir, args.dataset)
    dg_train, dg_val, dg_test,list_test = load_data(data_path, args.batch_size, args.debug)

    # compile model 
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model = model_compile(optimizer=optimizer, reg=args.lambda_reg)

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

        # train 
        hist = model.fit(
            dg_train,
            epochs=epochs,
            validation_data=dg_val,
            callbacks=[callback]
            )
        
        # save weights (at the end)
        model.save_weights(data_path + "/weights-name")                    

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
        
        assert args.weights is not None 
        assert os.path.exists(args.weights)
        
        #%% LOAD WEIGHTS
        model.load_weights(args.weights)
        
                
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


