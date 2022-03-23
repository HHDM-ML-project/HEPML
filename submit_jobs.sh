#!/bin/bash

flavor=\"$1\"
N_models=$2
signal=$3

# Check if HEP_OUTPATH variable exists then read
if [[ -z "${HEP_OUTPATH}" ]]; then
  echo "HEP_OUTPATH environment varibale is undefined. Aborting script execution..."
  exit 1
else
  outpath=${HEP_OUTPATH}
fi

# Check if MACHINES variable exists then read
if [[ -z "${MACHINES}" ]]; then
  echo "MACHINES environment varibale is undefined. Aborting script execution..."
  exit 1
else
  machines=${MACHINES}
fi

if [ "$1" == "help" ]
then
    echo "command: ./submit_grid.sh Flavour NumberOfJobs" 
    echo "Options for Flavour (maximum time to complete all jobs):"
    echo "espresso     = 20 minutes"
    echo "microcentury = 1 hour"
    echo "longlunch    = 2 hours"
    echo "workday      = 8 hours"
    echo "tomorrow     = 1 day"
    echo "testmatch    = 3 days"
    echo "nextweek     = 1 week"
else
    sed -i "s/.*queue.*/queue ${N_models}/" train.sub
    sed -i "s~.*arguments.*~arguments             = \$(ProcId) $(pwd) ${outpath} ${machines} ${signal}~" train.sub
    sed -i "s/.*+JobFlavour.*/+JobFlavour = ${flavor}/" train.sub

    condor_submit train.sub
fi
