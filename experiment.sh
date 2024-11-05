#!/bin/bash

if [ -z $1 ]
then
    echo "Error: Missing log directory"
    echo "Usage: $0 <name of log directory>"
    exit 1
fi

declare -a src_langs=("so" "ba" "fo" "tt" "ga")
expdir="${1}"

for s in "${src_langs[@]}"
do
    time python3 train_eval.py --title "${s} baseline" --src "${s}" --trg "en" --eval --baseline --expdir "${expdir}"
    time python3 train_eval.py --title "${s} finetune" --src "${s}" --trg "en" --train --eval --baseline --expdir "${expdir}"
    time python3 train_eval.py --title "${s} experimental" --src "${s}" --trg "en" --train --eval --expdir "${expdir}"
done