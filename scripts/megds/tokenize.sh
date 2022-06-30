/usr/bin/time -v python tools/preprocess_data_many_cores.py \
       --input ${1} \
       --output-prefix ${2} \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path ${3} \
       --append-eod \
       --workers 40