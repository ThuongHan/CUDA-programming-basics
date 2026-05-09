#!/bin/bash
#
# create temp.sh file with the executable
# schedule it with sbatch
executable=$1
cp template.sh temp.sh
sed -i -e "s/EXECUTABLE/$executable/g" temp.sh
sbatch temp.sh