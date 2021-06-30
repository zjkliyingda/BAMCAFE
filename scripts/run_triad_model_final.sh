#!/bin/bash

WS=`pwd`
LOCAL_RUN="xargs -L1 -P16 python"

printf "%s\n" "$WS/tasks/triad_model.py --finalized_mode=True --initial_data_type=onelayer --regime="1" --sigma_obs="0.2" --pred_start_time="800" --pred_total_time="80" --last_lead_time="10 | $LOCAL_RUN
