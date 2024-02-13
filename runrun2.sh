#!/bin/bash

for i in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"; do
    . run_a100_test2.sh --target_sparsity $i
done
