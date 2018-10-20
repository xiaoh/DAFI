#!/usr/bin/env bash

START=$(date +%s.%N)

vt_dainv.py main_input.in &> log.enkf

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF  > log.time

