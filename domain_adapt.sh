python train.py --workers 4 --device 0 --batch-size 4 \
 --data data/domain_adapt_test.yaml --img 640 640 \
  --cfg cfg/training/yolov7.yaml --name domain_adapt_test-3e --hyp data/hyp.scratch.custom.yaml --epochs 3  \
  --weights 'yolov7_training.pt' --cache-images #--freeze
