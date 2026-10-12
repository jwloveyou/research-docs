#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
LOG=research_logs/multids_progress.txt
echo "driver start $(date)" >> $LOG
while [ ! -f research_logs/round3_done.flag ]; do sleep 30; done
echo "round3 done; multidataset start $(date)" >> $LOG
EP=12
python main.py --recommender=NoiseSGL --dataset=amazon-book --seed=42 --epochs=1 --verbose=1 --stop_cnt=20 --freq_scaler=0 > research_logs/smoke_amazon.log 2>&1
if ! grep -q "epoch 1:" research_logs/smoke_amazon.log; then
  echo "SMOKE FAIL amazon-book $(date)" >> $LOG; echo FAIL > research_logs/multids_fail.flag; exit 1
fi
echo "smoke ok $(date)" >> $LOG
for DS in amazon-book gowalla; do
  for CFG in ctrl low high; do
    case $CFG in
      ctrl) EXTRA="--freq_scaler=0";;
      low)  EXTRA="--freq_scaler=1 --freq_type=low";;
      high) EXTRA="--freq_scaler=1 --freq_type=high";;
    esac
    echo "run $DS $CFG $(date)" >> $LOG
    python main.py --recommender=NoiseSGL --dataset=$DS --seed=42 --epochs=$EP --verbose=1 --stop_cnt=20 $EXTRA > research_logs/${CFG}_${DS}_s42.log 2>&1
    if grep -q "epoch ${EP}:" research_logs/${CFG}_${DS}_s42.log; then
      echo "ok $DS $CFG $(date)" >> $LOG
    else
      echo "SHORT/FAIL $DS $CFG $(date)" >> $LOG
    fi
  done
done
echo DONE > research_logs/multids_done.flag
echo "all done $(date)" >> $LOG
