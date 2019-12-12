#!/bin/bash
python medicalbert/ --experiment_name bert-alsentzer --tokenizer /home/apatterson/pretrained_bert_tf/biobert_pretrain_output_disch_100000 --pretrained_model /home/apatterson/pretrained_bert_tf/biobert_pretrain_output_disch_100000 --train --eval --target readm_30d --classifier bert-general;
python medicalbert/ --experiment_name bert-uncase --train --eval --target readm_30d --classifier bert-general;
python medicalbert/ --experiment_name bert-random --train --eval --target readm_30d --classifier bert-random;