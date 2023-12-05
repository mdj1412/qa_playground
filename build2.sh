export CUDA_VISIBLE_DEVICES=2 # Don't change, file default
# export FEW=0
export ENCODER_NAME=princeton-nlp/sup-simcse-roberta-large
export VERSION=2

for n_retrieval_data in 1000, 10000, 100000
do
    for few in 1 2 4 8 16
    do
        make FEW=$few CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} N_RETRIEVAL_DATA=$n_retrieval_data ENCODER_NAME=${ENCODER_NAME} VERSION=${VERSION}
    done
done
make clean