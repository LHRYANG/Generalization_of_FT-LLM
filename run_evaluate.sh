
OUTPUT_DIR=./results
MODEL=saved_models/summary/xsum_


DATA_PREFIX=./data/summary

for model in 2000 4000 6000; do

python evaluate_cross.py \
  --infile_list ${DATA_PREFIX}/xsum_xlsum_pdfs_cnn \
  --model_name_or_path ${MODEL}${model} \
  --batch_size 8 \
  --world_size 1 \
  --outfile ${OUTPUT_DIR}/ \
  --gpus_per_model 1 \
  --max_new_tokens 60 

done


DATA_PREFIX=./data/question

for model in 2000 4000 6000; do

python evaluate_cross.py \
  --infile_list ${DATA_PREFIX}/tweetqa_socialqa_sciqa \
  --model_name_or_path ${MODEL}${model} \
  --batch_size 8 \
  --world_size 1 \
  --outfile ${OUTPUT_DIR}/ \
  --gpus_per_model 1 \
  --max_new_tokens 60 

done


DATA_PREFIX=./data/inference

for model in 2000 4000 6000; do

python evaluate_cross.py \
  --infile_list ${DATA_PREFIX}/multinli_multinli2_rte_gptnli \
  --model_name_or_path ${MODEL}${model} \
  --batch_size 8 \
  --world_size 1 \
  --outfile ${OUTPUT_DIR}/ \
  --gpus_per_model 1 \
  --max_new_tokens 5 

done

DATA_PREFIX=./data/sentiment

for model in 2000 4000 6000; do

python evaluate_cross.py \
  --infile_list ${DATA_PREFIX}/yelp_sst2_amazon_amazonfood \
  --model_name_or_path ${MODEL}${model} \
  --batch_size 8 \
  --world_size 1 \
  --outfile ${OUTPUT_DIR}/ \
  --gpus_per_model 1 \
  --max_new_tokens 5 

done


DATA_PREFIX=./data/detection

for model in 2000 4000 6000; do

python evaluate_cross.py \
  --infile_list ${DATA_PREFIX}/paws_stsb_qqp \
  --model_name_or_path ${MODEL}${model} \
  --batch_size 8 \
  --world_size 1 \
  --outfile ${OUTPUT_DIR}/ \
  --gpus_per_model 1 \
  --max_new_tokens 5 

done










