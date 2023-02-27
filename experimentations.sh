#1. Study of generation huge data vs real minimal data
#Train baseline-synth + real
# python data_formatting/delete_cache.py


#Evaluate baseline-synth
# python test.py --data data/baseline_synth_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/baseline_synth-250e/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders baseline_synth-250e_ontestagency \
# --cubesat_output_folders baseline_synth-250e_ontestdeployment \
# --cubesat_output_folders baseline_synth-250e_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# #Evaluate only-real
#Train baseline-synth + real
# python train.py --workers 4 --device 0 --batch-size 4 --data data/only_real_custom_st.yaml \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name baseline_synth-100e+real --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze


#Train baseline-synth + real
# python train.py --workers 4 --device 0 --batch-size 4 --data data/baseline_synth+real_custom_st.yaml \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name baseline_synth-250e+real --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze

# #Evaluate baseline-synth+real
# python test.py --data data/baseline_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/baseline_synth-250e+real/weights/last.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders baseline_synth-250e+real_ontestagency \
# --cubesat_output_folders baseline_synth-250e+real_ontestdeployment \
# --cubesat_output_folders baseline_synth-250e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images


#2. Study of generation mode
# python train.py --workers 4 --device 0 --batch-size 4 --data data/empty_space_custom_st.yaml    \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name empty_space-100e    --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# Evaluate empty space
# python test.py --data data/empty_space_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/empty_space-100e/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders empty_space-100e_ontestagency \
# --cubesat_output_folders empty_space-100e_ontestdeployment \
# --cubesat_output_folders empty_space-100e_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# Study of obj per image
#Few objects
# python train.py --workers 4 --device 0 --batch-size 4 --data data/fewobj_custom_st.yaml   \
#       --img-size 640 640 --cfg cfg/training/yolov7.yaml --name few_obj-100e        --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/fewobj_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/few_obj-100e/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders few_obj-100e_ontestagency \
# --cubesat_output_folders few_obj-100e_ontestdeployment \
# --cubesat_output_folders few_obj-100e_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

#Many objects
# python train.py --workers 4 --device 0 --batch-size 4 --data data/manyobj_custom_st.yaml \
#         --img-size 640 640 --cfg cfg/training/yolov7.yaml --name many_obj-100e       --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/manyobj_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/many_obj-100e/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders many_obj-100e_ontestagency \
# --cubesat_output_folders many_obj-100e_ontestdeployment \
# --cubesat_output_folders many_obj-100e_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# 4. Sub class study
#OnlyST
# python train.py --workers 4 --device 0 --batch-size 4 --data data/onlyST_custom_st.yaml \
#         --img-size 640 640 --cfg cfg/training/yolov7.yaml --name onlyST-100e       --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/onlyST_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/onlyST-100e/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders onlyST-100e_ontestagency \
# --cubesat_output_folders onlyST-100e_ontestdeployment \
# --cubesat_output_folders onlyST-100e_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/onlyST/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/onlyST/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/onlyST/far_testset/images

#1U3UST
# python train.py --workers 4 --device 0 --batch-size 4 --data data/1U3UST_custom_st.yaml \
#         --img-size 640 640 --cfg cfg/training/yolov7.yaml --name 1U3UST-100e       --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/1U3UST_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/1U3UST-100e/weights/best.pt \
# --no-trace --task cubesat_test --verbose --v5_metric \
# --cubesat_output_folders 1U3UST-100e_ontestagency \
# --cubesat_output_folders 1U3UST-100e_ontestdeployment \
# --cubesat_output_folders 1U3UST-100e_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images


#dataset size: num images
#Train baseline-synth + real in 100e
# python train.py --workers 4 --device 0 --batch-size 4 --data data/official2000_synth+real_custom_st.yaml \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name official2000_synth-100e+real --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# #Evaluate baseline-synth+real for 100e
# python test.py --data data/baseline_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/official2000_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders official2000_synth-100e+real_ontestagency \
# --cubesat_output_folders official2000_synth-100e+real_ontestdeployment \
# --cubesat_output_folders official2000_synth-100e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images


#dataset size: num images
#Train baseline-synth + real in 100e
# python train.py --workers 4 --device 0 --batch-size 4 --data data/official5000_synth+real_custom_st.yaml \
#     --image-weights  \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name official5000_synth-250e+real --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze

# #Evaluate baseline-synth+real for 100e
# python test.py --data data/baseline_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/official5000_synth-250e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders official5000_synth-250e+real_ontestagency \
# --cubesat_output_folders official5000_synth-250e+real_ontestdeployment \
# --cubesat_output_folders official5000_synth-250e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# #TUNING
#Train baseline-synth + real in 100e with rect

#Train hyperparameters
# python train.py --workers 2 --device 0 --batch-size 2 --data data/official2000_synth+real_custom_st.yaml \
#     --multi-scale --image-weights  --evolve \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --hyp data/hyp.scratch.tune.yaml --epochs 3 --weights 'yolov7_training.pt' --cache-images #--freeze

# Train tuned squared
# python train.py --workers 4 --device 0 --batch-size 4 --data data/official2000_synth+real_custom_st.yaml \
#     --multi-scale --image-weights \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name tunedSquared2000_synth-100e+real --hyp runs/train/evolve/hyp_evolved.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/official2000_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/tunedSquared2000_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders tunedSquared2000_synth-100e+real_ontestagency \
# --cubesat_output_folders tunedSquared2000_synth-100e+real_ontestdeployment \
# --cubesat_output_folders tunedSquared2000_synth-100e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# python train.py --workers 2 --device 0 --batch-size 2 --data data/official2000_synth+real_custom_st.yaml \
#     --multi-scale --image-weights --img-size 640 640 --rect --cfg cfg/training/yolov7.yaml --name tunedRect2000_synth-100e+real --hyp runs/train/evolve/hyp_evolved.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/official2000_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/tunedRect2000_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders tunedRect2000_synth-100e+real_ontestagency \
# --cubesat_output_folders tunedRect2000_synth-100e+real_ontestdeployment \
# --cubesat_output_folders tunedRect2000_synth-100e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# python data_formatting/delete_cache.py

# #Train untuned squared multi-scale image-weights
# python train.py --workers 2 --device 0 --batch-size 2 --data data/official2000_synth+real_custom_st.yaml \
#     --multi-scale --image-weights \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name multiscaled_2000_synth-100e+real --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/official2000_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/multiscaled_2000_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders multiscaled_2000_synth-100e+real_ontestagency \
# --cubesat_output_folders multiscaled_2000_synth-100e+real_ontestdeployment \
# --cubesat_output_folders multiscaled_2000_synth-100e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# python data_formatting/delete_cache.py

#Train single-cls squared multi-scale image-weights
# python train.py --workers 2 --device 0 --batch-size 2 --data data/official2000_synth+real_custom_st.yaml \
#     --multi-scale --image-weights --single-cls \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name singlecls_multiscaled_2000_synth-100e+real --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# python test.py --data data/official2000_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/singlecls_multiscaled_2000_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose --single-cls \
# --cubesat_output_folders singlecls_multiscaled_2000_synth-100e+real_ontestagency \
# --cubesat_output_folders singlecls_multiscaled_2000_synth-100e+real_ontestdeployment \
# --cubesat_output_folders singlecls_multiscaled_2000_synth-100e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images


# #Changed loss weights
# python train.py --workers 4 --device 0 --batch-size 4 --data data/baseline_synth+real_custom_st.yaml \
#     --img-size 640 640 --cfg cfg/training/yolov7.yaml --name wloss_baseline_synth-100e+real --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

# # Evaluate baseline-synth+real for 100e
# python test.py --data data/baseline_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/wloss_baseline_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders wloss_baseline_synth-100e+real_ontestagency \
# --cubesat_output_folders wloss_baseline_synth-100e+real_ontestdeployment \
# --cubesat_output_folders wloss_baseline_synth-100e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

# Few shot learning with 37 images
# python train.py --workers 4 --device 0 --batch-size 4 --data data/coco_fsl.yaml  \
# --img-size 640 640 --cfg cfg/training/yolov7.yaml --name coco_fsl-100e+real --hyp data/hyp.scratch.custom.yaml --epochs 10 --weights 'yolov7_training.pt' #--cache-images #--freeze

# python test.py --data data/coco_fsl.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/coco_fsl-10e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders coco_fsl-10e+real_ontestagency \
# --cubesat_output_folders coco_fsl-10e+real_ontestdeployment \
# --cubesat_output_folders coco_fsl-10e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset_cocolab/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset_cocolab/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset_cocolab/far_testset/images


# #Few shot learning with 37 images
# python train.py --workers 4 --device 0 --batch-size 4 --data data/coco_fsl_synth.yaml  \
# --img-size 640 640 --cfg cfg/training/yolov7.yaml --name coco_fsl_synth-15e+real --hyp data/hyp.scratch.custom.yaml --epochs 15 --weights 'yolov7_training.pt' #--cache-images #--freeze

# python test.py --data data/coco_fsl_synth.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/coco_fsl_synth-15e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose --save-json \
# --cubesat_output_folders coco_fsl_synth-15e+real_ontestagency \
# --cubesat_output_folders coco_fsl_synth-15e+real_ontestdeployment \
# --cubesat_output_folders coco_fsl_synth-15e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset_cocolab/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset_cocolab/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset_cocolab/far_testset/images
