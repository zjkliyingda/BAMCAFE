#!/bin/bash

WS=`pwd`
LOCAL_RUN="xargs -L1 -P16 python"

printf "%s\n" "$WS/tasks/Two_layer_L96_model_prediction_final.py --model=Two_layer_L96_final --finalized_mode=True --generating_initial_data=True --initial_data_type=onelayer --sparse_obs=True --n_steps="500001" --dim_I="40" --pred_start_time="400" --pred_total_time="80" --last_lead_time="5 | $LOCAL_RUN
