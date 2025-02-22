#!/bin/bash

# 누적 시간 초기화
elapsed_time=0
interval=1800   # 10분 = 600초
total_time=18000   # 1시간 = 3600초

# 3600초가 될 때까지 10분 간격으로 누적 시간을 출력
while [ $elapsed_time -lt $total_time ]; do
	echo "Elapsed time: $elapsed_time seconds"
	sleep $interval
	elapsed_time=$((elapsed_time + interval))
done

	    # 3600초가 지나면 experiment.sh 실행
echo "$total_time seconds reached. Running experiment.sh"
bash mae_pretrain_fb237.sh

