#!/bin/bash
python medicalbert/ --experiment_name alsentzer --tokenizer /home/apatterson/pretrained_bert_tf/biobert_pretrain_output_disch_100000 --pretrained_model /home/apatterson/pretrained_bert_tf/biobert_pretrain_output_disch_100000 --train --eval --classifier bert-general;
python medicalbert/ --experiment_name general  --train --eval --classifier bert-general;
python medicalbert/ --experiment_name random  --train --eval --classifier bert-general;
python medicalbert/ --experiment_name alsentzer-mean-pool-tokens --tokenizer /home/apatterson/pretrained_bert_tf/biobert_pretrain_output_disch_100000 --pretrained_model /home/apatterson/pretrained_bert_tf/biobert_pretrain_output_disch_100000 --train --eval --classifier bert-mean-pool;
