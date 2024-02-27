
# Inferrence - Liam - please test 
```

# run our own kidney abd data through the best tolga net - to test timing 


# ssh to gpu machine 
ssh ankara 

# Init conda  
source /fileserver/external/body/abd/anum/miniconda3/bin/activate
conda activate fdnet3
export CUDA_VISIBLE_DEVICES=0,1


# git clone 
git clone git@github.com:sergeicu/tolganet.git
cd tolganet/network
git checkout inferrence_for_liam

# set the network weights (NB it is NOT an existing file but a regexp expression)
weights=/fileserver/external/body/abd/anum/tolganet/network/weights/fdnet_weights_42_subjects

# run network with a batch of 160 images 

# basic test - slices_input_test - liver (AX)
imagepath=/fileserver/external/body/abd/anum/data/abd/v4_ax/b50/slices_input_test_rotated_MANY/
savedir=$imagepath/predicted_tolga42_TEST
rm -rf $savedir
python fdnet11.py --custompath $imagepath --customshape 144 168 --weights $weights --savedir $savedir --legacy --batch_size 80 --dontskip

```



# Dataset & Training 
...details of training 

# Preprocessing
...


# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.

```
@article{ZaidAlkilani2023,
	author = {Zaid Alkilani, Abdallah and Çukur, Tolga and Saritas, Emine Ulku},
	title = {FD-Net: An unsupervised deep forward-distortion model for susceptibility artifact correction in EPI},
	journal = {Magnetic Resonance in Medicine},
	volume = {n/a},
	number = {n/a},
	pages = {},
	doi = {https://doi.org/10.1002/mrm.29851},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29851},
	eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.29851}
}

@misc{ZaidAlkilani2023arxiv,
	title = "FD-Net: An Unsupervised Deep Forward-Distortion Model for Susceptibility Artifact Correction in EPI", 
	author = "Zaid Alkilani, Abdallah AND {\c C}ukur, Tolga AND Saritas, Emine Ulku",
	year = "2023",
  	eprint = "2303.10436",
  	archivePrefix = "arXiv",
  	primaryClass = "eess.IV"
}

@InProceedings{ZaidAlkilani2022,
  	author = "Zaid Alkilani, Abdallah AND {\c C}ukur, Tolga AND Saritas, Emine Ulku",
	title = "A Deep Forward-Distortion Model for Unsupervised Correction of Susceptibility Artifacts in EPI",
	booktitle = "Proceedings of the 30th Annual Meeting of ISMRM",
	year = "2022",
	pages = "0959",
 	month = "May",
	address = "London, United Kingdom"
}

```

# Acknowledgements
A preliminary version of this work was presented in the Annual Meeting of ISMRM in London, 2022. This work was supported by the Scientific and Technological Council of Turkey (TÜBİTAK) via Grant 117E116. Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

For questions/comments please send an email to: `<><><>`
