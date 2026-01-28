#!/bin/bash

for i in {1..9}
do
    qsub "jobscript${i}.sh"
done

for i in {1..3}
do
    qsub "jobscript-a${i}.sh"
done
