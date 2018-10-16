#!/usr/bin/env bash

START=$(date +%s.%N)

mfuMain.py MainInput.in &> log.enkf

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF  > log.time

