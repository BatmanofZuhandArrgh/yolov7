python data_formatting/delete_cache.py

python train.py --workers 4 --device 0 --batch-size 4 \
 --data data/domain_adapt_CORAL.yaml --img 640 640 \
  --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_100e --hyp data/hyp.scratch.custom.yaml --epochs 100  \
  --weights 'yolov7_training.pt' --cache-images #--freeze

#Evaluate baseline-synth+real for 100e
python test.py --data data/domain_adapt_CORAL.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.3 \
--device 0 --weights runs/train/domain_adapt_CORAL_100e/weights/best.pt \
--no-trace --task cubesat_test --verbose \
--cubesat_output_folders domain_adapt_CORAL_100e_ontestagency \
--cubesat_output_folders domain_adapt_CORAL_100e_ontestdeployment \
--cubesat_output_folders domain_adapt_CORAL_100e_ontestfar \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images
