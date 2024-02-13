#!/bin/bash

for reg in "none" "l2"; do
    . run_2a1002_.sh --param_reg_type $reg
done