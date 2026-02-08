#!/bin/bash

newTmuxSession(){ #new tmux session
    session=$1
    tmux has-session -t $session 2>/dev/null
    if [ $? == 0 ]; then
        echo "Session $session already exists"
        tmux kill-session -t $session
        tmux new-session -d -s $session
        echo "Session $session created done"
    else
        tmux new-session -d -s $session
        echo "Session $session created done"
    fi
}

envs=("halfcheetah-kinematic")

srctypes=("expert")

seeds=("100")
device="cuda:0"
algo="DROCO"
save_model="True"
penalty_coefficient="0.3"
huber_delta="30"

dir="./logs/DROCO"

for env in "${envs[@]}"
do  
    for srctype in "${srctypes[@]}"
    do
        for seed in "${seeds[@]}"
        do  
            tmux_name="${algo}_${env}_${srctype}_${seed}"
            newTmuxSession ${tmux_name}
            tmux send -t ${tmux_name} "cd path/to/project" C-m
            tmux send -t ${tmux_name} "conda activate odrl" C-m
            tmux send -t ${tmux_name} "python train_droco.py --algo=${algo} --env=${env} --srctype=${srctype} --seed=${seed} --device=${device} --save-model=${save_model} --penalty_coefficient=${penalty_coefficient} --huber_delta=${huber_delta} --dir=${dir}" C-m
        done
    done
done

