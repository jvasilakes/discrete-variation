#!/bin/bash

for datafile in data/SFU/processed_split/labeled*.data; do
    i=$(echo ${datafile} | sed 's/.*split\([0-9]\)\.data/\1/')
    python vsl_gg.py \
        --model hier \
        --random_seed 0 \
        --prefix "test_hier/split_${i}" \
        --data_file ${datafile} \
        --vocab_file data/SFU/processed_split/vocab \
        --tag_file data/SFU/processed_split/tagfile \
        --embed_file data/twitter/twitter_wordvects \
        --use_unlabel 0 \
        --prior_file "test_hier/split_${i}" \
        --embed_type twitter \
        --n_iter 15000 \
        --batch_size 10 \
        --vocab_size 100000 \
        --train_emb 0 \
        --save_prior 1 \
        --learning_rate 1e-3 \
        --embed_dim 100 \
        --tie_weights 1 \
        --char_embed_dim 50 \
        --char_hidden_size 100 \
        --latent_z_size 75 \
        --latent_y_size 25 \
        --rnn_size 100 \
        --mlp_hidden_size 100 \
        --mlp_layer 2 \
        --kl_anneal_rate 1e-4 \
        --update_freq_label 1 \
        --update_freq_unlabel 1 \
        --f1_score 1 \
        --print_every 100 \
        --eval_every 1000 \
        --summarize 1
    wait
done
