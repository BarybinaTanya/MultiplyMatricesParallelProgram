#!/bin/bash
source /opt/intel/oneapi/setvars.sh

PROC_COUNT=$1
PROC_X=$2
PROC_Y=$3
N1=$4
N2=$5
N3=$6
MODE=$7

mpiicc -trace main.c -o main_traced -lm
mpirun -np $PROC_COUNT ./main_traced $PROC_X $PROC_Y $N1 $N2 $N3 $MODE
tracemgr -merge *.stf -o main.stf 2>/dev/null
[ -n "$DISPLAY" ] && traceanalyzer main.stf &
