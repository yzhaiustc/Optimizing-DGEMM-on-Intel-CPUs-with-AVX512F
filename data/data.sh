#!/bin/bash
for ((i=0;i<20;i+=1))
do
        input_file_name="res_${i}.txt"
        output_file_name="perf_${i}.txt"
        grep "elasped" ${input_file_name} | cut -c 52-60  &> ${output_file_name}
done
