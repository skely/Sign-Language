#!/bin/bash
test=LSTM_test_AE.py
testver=test_glo_v8
mkdir -p /home/jedle/data/Sign-Language/_source_clean/testing/$testver
nohup python /home/jedle/Projects/Sign-Language/tests/LSTM/$test > /home/jedle/data/Sign-Language/_source_clean/testing/$testver/error.log 2>&1 &
