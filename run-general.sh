python medicalbert/ --experiment_name bert-uncase-lr5e5B64 --train --eval --target readm_30d --classifier bert-general --learning_rate 0.00005 --train_batch_size 64 --gradient_accumulation_steps 64;
python medicalbert/ --experiment_name bert-uncase-lr5e5B32 --train --eval --target readm_30d --classifier bert-general --learning_rate 0.00005 --train_batch_size 32 --gradient_accumulation_steps 32;
python medicalbert/ --experiment_name bert-uncase-lr2e5B32 --train --eval --target readm_30d --classifier bert-general --learning_rate 0.00002 --epochs 10;