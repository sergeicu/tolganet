"""This is heavily refactored version of the original code by Serge: 
    - add __name__ call
    - split code into functions 
    - added argparse 
    - split train and test routines 
    
    """

# -*- coding: utf-8 -*-

import os
# set cuda visible to 1 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
import sys 
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

# check if gpu is available 
if not tf.config.experimental.list_physical_devices('GPU'):
    sys.exit("not gpu available / registered") 
    
    # tf.config.experimental.get_memory_info(tf.config.experimental.list_physical_devices('GPU')[0])

#from IPython import embed; embed()

from fdnet_utils import DataGenerator, DataGeneratorFMRI,\
    print_metrics, model_compile

rootdir="/fileserver/external/body/abd/anum/"
os.chdir(rootdir+"FD-Net")

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
     
import glob as glob
    
def load_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(mode, type=str,  choices=["train", "test"],default="train",help='train or test')
    #parser.add_argument('--config', type=str, required=True)    
    #parser.add_argument('--cuda', type=str, default="0",help='choose gpu')
    #parser.add_argument('--folder_path', type=str, help='folder to analyze')
    #parser.add_argument('--slice_range', nargs='+', default=None, type=int, help='slices to analyze')
    #parser.add_argument('--repetitionsuffix', default=1, type=int, help='indicate which repetition we should pick - 1 to 6')
    
    
    #parser.add_argument('--view', action='store_true', default=False,help='plot output in itksnap and in rview')
    #parser.add_argument('--split_channels', action='store_true', default=False,help='view split files')
    #parser.add_argument('--datadir','-d',type=str,default = '', help='path to a previously generated directory that also contains an options file (which may be read once again to re-run the experiment once again)')
    #parser.add_argument('--debug',action='store_true')
    args = parser.parse_args()
    
    return args     
    
def load_data(load_path):

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

    dg_test  = DataGenerator(
        list_test,
        list_topup_test,
        list_field_test,
        batch_size=1,
        shuffle=False
        )



    return dg_train,dg_val, dg_test


    
if __name__ == '__main__':
    
    # init vars 
    load_path = rootdir+"/data_HCP/"
    batch_size = 8
    epochs = 1
    patience = epochs//1
    

    # data 
    dg_train, dg_val, dg_test = load_data(load_path)

    if args.mode == 'train':
        
        # compile model 
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        lambda_reg = 1e-5            
        model = model_compile(optimizer=optimizer, reg=lambda_reg)
        
        # early stopping 
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=5e-6,
            patience=patience,
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
        model.save_weights(load_path + "/weights-name")                    

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
        hist_csv_file = load_path + "/file-name.csv"
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            

    elif args.mode == 'test':
        
        #%% LOAD WEIGHTS
        model.load_weights(load_path + "/weights-name")
        
                
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
        list_print  = glob.glob(load_path + "/test/predict/slices_like_topup/*.nii")
        list_topup_print = glob.glob(load_path + "/test/predict/slices_topup/*.nii")
        list_field_print = glob.glob(load_path + "/test/predict/slices_field_topup/*.nii")

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



















        # UNUSED  - FMRI PART 



        """
        #%% LOAD DATA + TRAIN (FMRI)
        load_path = rootdir
        batch_size = 4

        # train
        slice_path_train = load_path + "/fmri/slices/distorted/*.nii"
        topup_path_train = None
        field_path_train = None

        list_slices_train = glob.glob(slice_path_train)
        list_topups_train = None
        list_fields_train = None

        list_slices_train.sort()

        list_train = list_slices_train
        list_topup_train = list_topups_train
        list_field_train = list_fields_train

        dg_train = DataGeneratorFMRI(
            list_train,
            list_topup_train,
            list_field_train,
            batch_size=batch_size,
            shuffle=True,
            train=True
            )

        # train
        epochs = 64
        patience = 4

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=5e-6,
            patience=patience,
            verbose=3,
            mode="min",
            baseline=None,
            restore_best_weights=True
            )       

        hist = model.fit(
            dg_train,
            epochs=epochs,
            callbacks=[callback]
            )

        plt.figure()
        plt.plot(hist.history["loss"])
        plt.ylabel("Loss")
        plt.xlabel("Epoch Number")
        plt.legend(["Fine-tune"], loc="upper right")
        plt.show()
        # plt.savefig("plot-name-fine-tune", dpi=300, bbox_inches="tight")

        hist_df = pd.DataFrame(hist.history)
        hist_csv_file = load_path + "/file-name-fine-tune.csv"
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        # weights
        model.save_weights(load_path + "/weights-name-fine-tune")
        # model.load_weights(load_path + "/weights-name-fine-tune")
        #%% PRINT METRICS (FMRI)
        load_path = "dir-to-data"

        list_print  = glob.glob(load_path + "/fmri/slices/distorted/*.nii")
        list_topup_print = glob.glob(load_path + "/fmri/slices/corrected_topup_resized/image/*.nii")
        list_field_print = glob.glob(load_path + "/fmri/slices/corrected_topup_resized/field/*.nii")

        dg_test_print  = DataGeneratorFMRI(
            list_print,
            list_topup_print,
            list_field_print,
            batch_size=1,
            shuffle=False
            )

        for j in tqdm(range(len(dg_test_print))):
            dat = dg_test_print[j]
            X = dat[0]
            Y, _, _, _, abc = model.predict(X, verbose=0) # !!! choose appropriate weights
                    
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
                        ext="FMRI") # !!! "FMRI_finetuned" for finetuning weights
        """