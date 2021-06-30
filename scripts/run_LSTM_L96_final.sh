#!/bin/bash

WS=`pwd`
LOCAL_RUN="xargs -L1 -P16 python"


# send_email start foo_experiment
# if there is only one para, no {}
# "{'imperfect_onelayer','imperfect_large'}"


printf "%s\n" "$WS/tasks/LSTM_train_L96_final.py --training_type=imperfect_onelayer --batch_size="64" --num_ensembles_train="30" --hidden_size="64" --num_layers="1"  --lr="0.001"  --num_epochs="300" --n_segments="40" --lead_time="{1,2,3,4,5,6,7,8,9}" --model=Two_layer_L96_final --foldername="/n_steps=500001,dim_I=40,sparse_obs=True,f=4.0,h=2.0,b=2.0,c=2.0,sigma_obs=1.0,sigma_u=1.0,sigma_v=1.0 | $LOCAL_RUN


printf "%s\n" "$WS/tasks/LSTM_train_L96_final.py --training_type="{'imperfect_onelayer','imperfect_large'}" --batch_size="64" --num_ensembles_train="50" --hidden_size="64" --num_layers="1"  --lr="0.001"  --num_epochs="100" --n_segments="40" --lead_time="{11,12,13,14,16,17,18,19}" --model=Two_layer_L96_final --foldername="/n_steps=500001,dim_I=40,sparse_obs=True,f=4.0,h=2.0,b=2.0,c=2.0,sigma_obs=1.0,sigma_u=1.0,sigma_v=1.0 | $LOCAL_RUN


printf "%s\n" "$WS/tasks/LSTM_train_L96_final.py --training_type="{'imperfect_onelayer','imperfect_large'}" --batch_size="64" --num_ensembles_train="50" --hidden_size="32" --num_layers="1"  --lr="0.001"  --num_epochs="100" --n_segments="40" --lead_time="{18,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,61,62,63,64,65,66,67,68,69}" --model=Two_layer_L96_final --foldername="/n_steps=500001,dim_I=40,sparse_obs=True,f=4.0,h=2.0,b=2.0,c=2.0,sigma_obs=1.0,sigma_u=1.0,sigma_v=1.0 | $LOCAL_RUN

printf "%s\n" "$WS/tasks/LSTM_train_L96_final.py --training_type="{'imperfect_onelayer','imperfect_large'}" --batch_size="64" --num_ensembles_train="50" --hidden_size="16" --num_layers="1"  --lr="0.001"  --num_epochs="100" --n_segments="40" --lead_time="{61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,89,81,82,83,84,85,86,87,88,89,91,92,93,94,95,96,97,98,99}" --model=Two_layer_L96_final --foldername="/n_steps=500001,dim_I=40,sparse_obs=True,f=4.0,h=2.0,b=2.0,c=2.0,sigma_obs=1.0,sigma_u=1.0,sigma_v=1.0 | $LOCAL_RUN

