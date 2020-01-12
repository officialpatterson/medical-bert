python medicalbert/ --experiment_name bert-uncase-lr5e5B16-12-layers --train --eval --target readm_30d --classifier bert-general --learning_rate 0.000005 --num_layers 12 --epochs 3;
python medicalbert/ --experiment_name bert-uncase-lr5e5B16-10-layers --train --eval --target readm_30d --classifier bert-general --learning_rate 0.000005 --num_layers 10 --epochs 3;
python medicalbert/ --experiment_name bert-uncase-lr5e5B16-8-layers --train --eval --target readm_30d --classifier bert-general --learning_rate 0.000005 --num_layers 8 --epochs 3;
python medicalbert/ --experiment_name bert-uncase-lr5e5B16-6-layers --train --eval --target readm_30d --classifier bert-general --learning_rate 0.000005 --num_layers 6 --epochs 3;
python medicalbert/ --experiment_name bert-uncase-lr5e5B16-4-layers --train --eval --target readm_30d --classifier bert-general --learning_rate 0.000005 --num_layers 4 --epochs 3;
python medicalbert/ --experiment_name bert-uncase-lr5e5B16-2-layers --train --eval --target readm_30d --classifier bert-general --learning_rate 0.000005 --num_layers 2 --epochs 3;
python medicalbert/ --experiment_name test --data_dir gs://storage.andrewrpatterson.com/aclImdb --output_dir /Users/apatterson/Desktop --train --eval --target sentiment --classifier bert-general --learning_rate 0.000005 --num_layers 1 --epochs 1;
