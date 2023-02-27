#1. Study of generation huge data vs real minimal data

# python data_formatting/delete_cache.py

# #Evaluate baseline-synth+real
# python test.py --data data/baseline_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/redo_baseline_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 0redo_baseline_synth-100e+real_ontestagency \
# --cubesat_output_folders 0redo_baseline_synth-100e+real_ontestdeployment \
# --cubesat_output_folders 0redo_baseline_synth-100e+real_ontestfar \
# --cubesat_output_folders 0redo_baseline_synth-100e+real_onfull \
# --cubesat_output_folders 0redo_baseline_synth-100e+real_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py

#Evaluate onlyReal
# python test.py --data data/only_real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/only-real-250e/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 1onlyreal-250e_ontestagency \
# --cubesat_output_folders 1onlyreal-250e_ontestdeployment \
# --cubesat_output_folders 1onlyreal-250e_ontestfar \
# --cubesat_output_folders 1onlyreal-250e_onfull \
# --cubesat_output_folders 1onlyreal-250e_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py

# #Evaluate onlySynth
# python test.py --data data/baseline_synth_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/baseline_synth-250e/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 2onlysynth-100e_ontestagency \
# --cubesat_output_folders 2onlysynth-100e_ontestdeployment \
# --cubesat_output_folders 2onlysynth-100e_ontestfar \
# --cubesat_output_folders 2onlysynth-100e_onfull \
# --cubesat_output_folders 2onlysynth-100e_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py

# #Evaluate onlyEmptySpace
# python test.py --data data/empty_space_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/empty_space-100e/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 3empty_space-100e_ontestagency \
# --cubesat_output_folders 3empty_space-100e_ontestdeployment \
# --cubesat_output_folders 3empty_space-100e_ontestfar \
# --cubesat_output_folders 3empty_space-100e_onfull \
# --cubesat_output_folders 3empty_space-100e_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images


# python data_formatting/delete_cache.py
# #Evaluate official5000
# python test.py --data data/official5000_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/official5000_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 4official5000_synth+real-100e_ontestagency \
# --cubesat_output_folders 4official5000_synth+real-100e_ontestdeployment \
# --cubesat_output_folders 4official5000_synth+real-100e_ontestfar \
# --cubesat_output_folders 4official5000_synth+real-100e_onfull \
# --cubesat_output_folders 4official5000_synth+real-100e_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# #Evaluate official2000Tuned
# python test.py --data data/baseline_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/tunedSquared2000_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 5tunedSquared2000synth+real-100e_ontestagency \
# --cubesat_output_folders 5tunedSquared2000synth+real-100e_ontestdeployment \
# --cubesat_output_folders 5tunedSquared2000synth+real-100e_ontestfar \
# --cubesat_output_folders 5tunedSquared2000synth+real-100e_onfull \
# --cubesat_output_folders 5tunedSquared2000synth+real-100e_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images


# python data_formatting/delete_cache.py
# #Evaluate official2000Tuned
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CORAL_100e_1l/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 6domain_adapt_CORAL_100e_1l_ontestagency \
# --cubesat_output_folders 6domain_adapt_CORAL_100e_1l_ontestdeployment \
# --cubesat_output_folders 6domain_adapt_CORAL_100e_1l_ontestfar \
# --cubesat_output_folders 6domain_adapt_CORAL_100e_1l_onfull \
# --cubesat_output_folders 6domain_adapt_CORAL_100e_1l_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# #Evaluate CORAL 1l
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CORAL_400e_1l/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 7domain_adapt_CORAL_400e_1l_ontestagency \
# --cubesat_output_folders 7domain_adapt_CORAL_400e_1l_ontestdeployment \
# --cubesat_output_folders 7domain_adapt_CORAL_400e_1l_ontestfar \
# --cubesat_output_folders 7domain_adapt_CORAL_400e_1l_onfull \
# --cubesat_output_folders 7domain_adapt_CORAL_400e_1l_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CMD_300e/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 8domain_adapt_CMD_300e_ontestagency \
# --cubesat_output_folders 8domain_adapt_CMD_300e_ontestdeployment \
# --cubesat_output_folders 8domain_adapt_CMD_300e_ontestfar \
# --cubesat_output_folders 8domain_adapt_CMD_300e_onfull \
# --cubesat_output_folders 8domain_adapt_CMD_300e_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CMD_400e_1l/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 9domain_adapt_CMD_400e_1l_ontestagency \
# --cubesat_output_folders 9domain_adapt_CMD_400e_1l_ontestdeployment \
# --cubesat_output_folders 9domain_adapt_CMD_400e_1l_ontestfar \
# --cubesat_output_folders 9domain_adapt_CMD_400e_1l_onfull \
# --cubesat_output_folders 9domain_adapt_CMD_400e_1l_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

#3L CMD retraining and inference
# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CMD_100e_3l/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 10domain_adapt_CMD_100e_3l_ontestagency \
# --cubesat_output_folders 10domain_adapt_CMD_100e_3l_ontestdeployment \
# --cubesat_output_folders 10domain_adapt_CMD_100e_3l_ontestfar \
# --cubesat_output_folders 10domain_adapt_CMD_100e_3l_onfull \
# --cubesat_output_folders 10domain_adapt_CMD_100e_3l_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CMD_500e_3l/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 11domain_adapt_CMD_500e_3l_ontestagency \
# --cubesat_output_folders 11domain_adapt_CMD_500e_3l_ontestdeployment \
# --cubesat_output_folders 11domain_adapt_CMD_500e_3l_ontestfar \
# --cubesat_output_folders 11domain_adapt_CMD_500e_3l_onfull \
# --cubesat_output_folders 11domain_adapt_CMD_500e_3l_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

#3L CORAL retraining and inference
# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CORAL_100e_3l/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 12domain_adapt_CORAL_100e_3l_ontestagency \
# --cubesat_output_folders 12domain_adapt_CORAL_100e_3l_ontestdeployment \
# --cubesat_output_folders 12domain_adapt_CORAL_100e_3l_ontestfar \
# --cubesat_output_folders 12domain_adapt_CORAL_100e_3l_onfull \
# --cubesat_output_folders 12domain_adapt_CORAL_100e_3l_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CORAL_400e_3l/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 13domain_adapt_CORAL_400e_3l_ontestagency \
# --cubesat_output_folders 13domain_adapt_CORAL_400e_3l_ontestdeployment \
# --cubesat_output_folders 13domain_adapt_CORAL_400e_3l_ontestfar \
# --cubesat_output_folders 13domain_adapt_CORAL_400e_3l_onfull \
# --cubesat_output_folders 13domain_adapt_CORAL_400e_3l_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

#3LD CMD retraining and inference
# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CMD_100e_3ld/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 14domain_adapt_CMD_100e_3ld_ontestagency \
# --cubesat_output_folders 14domain_adapt_CMD_100e_3ld_ontestdeployment \
# --cubesat_output_folders 14domain_adapt_CMD_100e_3ld_ontestfar \
# --cubesat_output_folders 14domain_adapt_CMD_100e_3ld_onfull \
# --cubesat_output_folders 14domain_adapt_CMD_100e_3ld_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CMD_400e_3ld/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 15domain_adapt_CMD_400e_3ld_ontestagency \
# --cubesat_output_folders 15domain_adapt_CMD_400e_3ld_ontestdeployment \
# --cubesat_output_folders 15domain_adapt_CMD_400e_3ld_ontestfar \
# --cubesat_output_folders 15domain_adapt_CMD_400e_3ld_onfull \
# --cubesat_output_folders 15domain_adapt_CMD_400e_3ld_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

#3L CORAL retraining and inference
# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CORAL_100e_3ld/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 16domain_adapt_CORAL_100e_3ld_ontestagency \
# --cubesat_output_folders 16domain_adapt_CORAL_100e_3ld_ontestdeployment \
# --cubesat_output_folders 16domain_adapt_CORAL_100e_3ld_ontestfar \
# --cubesat_output_folders 16domain_adapt_CORAL_100e_3ld_onfull \
# --cubesat_output_folders 16domain_adapt_CORAL_100e_3ld_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

# python data_formatting/delete_cache.py
# python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
# --device 0 --weights runs/train/domain_adapt_CORAL_400e_3ld/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders 17domain_adapt_CORAL_400e_3ld_ontestagency \
# --cubesat_output_folders 17domain_adapt_CORAL_400e_3ld_ontestdeployment \
# --cubesat_output_folders 17domain_adapt_CORAL_400e_3ld_ontestfar \
# --cubesat_output_folders 17domain_adapt_CORAL_400e_3ld_onfull \
# --cubesat_output_folders 17domain_adapt_CORAL_400e_3ld_onsynth \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
# --cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

#600 epochs inference
python data_formatting/delete_cache.py
python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
--device 0 --weights runs/train/domain_adapt_CORAL_600e_1l/weights/best.pt \
--no-trace --task cubesat_test --verbose --save-json \
--cubesat_output_folders 18domain_adapt_CORAL_600e_1l_ontestagency \
--cubesat_output_folders 18domain_adapt_CORAL_600e_1l_ontestdeployment \
--cubesat_output_folders 18domain_adapt_CORAL_600e_1l_ontestfar \
--cubesat_output_folders 18domain_adapt_CORAL_600e_1l_onfull \
--cubesat_output_folders 18domain_adapt_CORAL_600e_1l_onsynth \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
--cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images

python data_formatting/delete_cache.py
python test.py --data data/domain_adapt_DAloss.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
--device 0 --weights runs/train/domain_adapt_CORAL_600e_3ld/weights/best.pt \
--no-trace --task cubesat_test --verbose --save-json \
--cubesat_output_folders 19domain_adapt_CORAL_600e_3ld_ontestagency \
--cubesat_output_folders 19domain_adapt_CORAL_600e_3ld_ontestdeployment \
--cubesat_output_folders 19domain_adapt_CORAL_600e_3ld_ontestfar \
--cubesat_output_folders 19domain_adapt_CORAL_600e_3ld_onfull \
--cubesat_output_folders 19domain_adapt_CORAL_600e_3ld_onsynth \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images  \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/images \
--cubesat_testsets /home/iasrl/Documents/synthetic_dataset/synthetic_testset/images
