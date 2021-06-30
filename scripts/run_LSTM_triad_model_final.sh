#!/bin/bash

WS=`pwd`
LOCAL_RUN="xargs -L1 -P16 python"


printf "%s\n" "$WS/tasks/LSTM_train_triad_final.py --model=triad_model_final --training_type=imperfect --num_ensembles_train="50" --batch_size="64" --lead_time="2" --hidden_size="64"  --num_epochs="200" --lr="0.001" --n_segments="120" --foldername="/n_steps=200001,regime=1,sigma_obs=0.2 | $LOCAL_RUN


printf "%s\n" "$WS/tasks/LSTM_train_triad_final.py --model=triad_model_final  --training_type=imperfect --num_ensembles_train="50" --batch_size="64" --lead_time="20" --hidden_size="32"  --num_epochs="50" --lr="0.001" --n_segments="120" --foldername="/n_steps=200001,regime=1,sigma_obs=0.2 | $LOCAL_RUN

printf "%s\n" "$WS/tasks/LSTM_train_triad_final.py --model=triad_model_final  --training_type=imperfect --num_ensembles_train="50" --batch_size="64" --lead_time="80" --hidden_size="16"  --num_epochs="30" --lr="0.001" --n_segments="100" --foldername="/n_steps=200001,regime=1,sigma_obs=0.2 | $LOCAL_RUN


printf "%s\n" "$WS/tasks/LSTM_train_triad_final.py --model=triad_model_final --training_type=imperfect_smooth --num_ensembles_train="1" --batch_size="64" --lead_time="2" --hidden_size="64"  --num_epochs="1000" --lr="0.001" --n_segments="120" --foldername="/n_steps=200001,regime=1,sigma_obs=0.2 | $LOCAL_RUN


printf "%s\n" "$WS/tasks/LSTM_train_triad_final.py --model=triad_model_final  --training_type=imperfect_smooth --num_ensembles_train="1" --batch_size="64" --lead_time="20" --hidden_size="32"  --num_epochs="500" --lr="0.001" --n_segments="120" --foldername="/n_steps=200001,regime=1,sigma_obs=0.2 | $LOCAL_RUN


printf "%s\n" "$WS/tasks/LSTM_train_triad_final.py --model=triad_model_final  --training_type=imperfect_smooth --num_ensembles_train="1" --batch_size="64" --lead_time="80" --hidden_size="16"  --num_epochs="500" --lr="0.001" --n_segments="100" --foldername="/n_steps=200001,regime=1,sigma_obs=0.2 | $LOCAL_RUN

