#!/usr/bin/bash


for method in {0..2}; do
    echo "Method: ${method}"
    echo "Method: ${method} " > method-${method}-output.txt
for dataset in {6..6}; do

    echo "Dataset: ${dataset}"  >> method-${method}-output.txt
    echo "Dataset: ${dataset}"

for test in {0..9}; do
    echo "Test: ${test}"

    (./Histogram_Solution -e Histogram/Dataset/${dataset}/output.raw -i Histogram/Dataset/${dataset}/input.raw ${method} | grep compute | awk '{print $5}' >> method-${method}-output.txt)
done
done
done
