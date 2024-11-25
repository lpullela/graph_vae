# if permission denied on exec, run chmod 777 run_tests.sh and then you can do ./baselines/graphvae/run_tests.sh
#!/bin/bash

# can change default values
EPOCHS=20
MAX_NUM_NODES=30
DATASET="enzymes"

# parsing args
while getopts e:n: flag; do
    case "${flag}" in
        e) EPOCHS=${OPTARG};;       
        n) MAX_NUM_NODES=${OPTARG};;  
    esac
done

SIMILARITY_FUNCTIONS=("dummy" "original" "binned" "page_rank")

for FUNC in "${SIMILARITY_FUNCTIONS[@]}"; do
    echo "Running with similarity_function=${FUNC}, epochs=${EPOCHS}, max_num_nodes=${MAX_NUM_NODES}"
    python3 baselines/graphvae/train.py \
        --dataset ${DATASET} \
        --max_num_nodes ${MAX_NUM_NODES} \
        --similarity_function ${FUNC} \
        --epochs ${EPOCHS}
done
