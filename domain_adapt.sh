python data_formatting/delete_cache.py

python train.py --workers 4 --device 0 --batch-size 4 \
 --data data/domain_adapt_official.yaml --img 640 640 \
  --cfg cfg/training/yolov7.yaml --name domain_adapt_test --hyp data/hyp.scratch.custom.yaml --epochs 3  \
  --weights 'yolov7_training.pt' --cache-images   #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CMD_500e_3l --hyp data/hyp.scratch.custom.yaml --epochs 500  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_100e_3l --hyp data/hyp.scratch.custom.yaml --epochs 100  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_400e_3l --hyp data/hyp.scratch.custom.yaml --epochs 400  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CMD_100e_3ld --hyp data/hyp.scratch.custom.yaml --epochs 100  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CMD_400e_3ld --hyp data/hyp.scratch.custom.yaml --epochs 400  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

#   python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_100e_3ld --hyp data/hyp.scratch.custom.yaml --epochs 100  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

#   python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_600e_3ld --hyp data/hyp.scratch.custom.yaml --epochs 600  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_400e_3ld --hyp data/hyp.scratch.custom.yaml --epochs 400  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze


#   python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_100e_1l --hyp data/hyp.scratch.custom.yaml --epochs 100  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

#   python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_400e_1l --hyp data/hyp.scratch.custom.yaml --epochs 400  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

#   python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_DAloss.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CORAL_600e_1l --hyp data/hyp.scratch.custom.yaml --epochs 600  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze

#   python train.py --workers 4 --device 0 --batch-size 4 \
#  --data data/domain_adapt_CORAL.yaml --img 640 640 \
#   --cfg cfg/training/yolov7.yaml --name domain_adapt_CMD_400e_1l --hyp data/hyp.scratch.custom.yaml --epochs 400  \
#   --weights 'yolov7_training.pt' --cache-images   #--freeze