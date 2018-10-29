#!/usr/bin/env bash

START=$(date +%s.%N)

vt_dainv.py dainv.in # > log.dainv

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF  > log.time

