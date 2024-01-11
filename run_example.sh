# ssh 
ssh <ankara / gamakichi / izmir> 

# load conda 
source /fileserver/external/body/abd/anum/miniconda3/bin/activate
conda activate fdnet3

# cd & set cuda
cd /fileserver/external/body/abd/anum/FD-Net/network/
export CUDA_VISIBLE_DEVICES=0,1


# train on HCP with Anum's code 
python fdnet.py 