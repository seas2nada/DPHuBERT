#!/usr/bin/env bash
# THIS FILE IS GENERATED BY tools/setup_anaconda.sh
if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /home/Workspace/DPHuBERT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate dphubert
