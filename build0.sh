export CUDA_VISIBLE_DEVICES=0 # Don't change, file default
# export FEW=0
export VERSION=0

for few in 0 1 2 4 8 16
do
    make FEW=$few CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} VERSION=${VERSION}
done
make clean