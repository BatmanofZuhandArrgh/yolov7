python train.py --workers 4 --device 0 --batch-size 4 --data data/baseline_synth_custom_st.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --name baseline_synth-100e+real --hyp data/hyp.scratch.custom.yaml --epochs 100 --weights 'yolov7_training.pt' --cache-images #--freeze

python train.py --workers 4 --device 0 --batch-size 4 --data data/only_real_custom_st.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --name only-real-250e --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 --data data/far_custom_st.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --name far-250e --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --data data/near_custom_st.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --name near-250e --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --data data/medium_custom_st.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --name medium-250e --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze

# python train.py --workers 4 --device 0 --batch-size 4 --data data/empty_space_custom_st.yaml    --img 640 640 --cfg cfg/training/yolov7.yaml --name empty_space-250e    --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --data data/only_cubes_custom_st.yaml     --img 640 640 --cfg cfg/training/yolov7.yaml --name only_cubes-250e     --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --data data/fewobj_custom_st.yaml         --img 640 640 --cfg cfg/training/yolov7.yaml --name few_obj-250e        --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --data data/manyobj_custom_st.yaml        --img 640 640 --cfg cfg/training/yolov7.yaml --name many_obj-250e       --hyp data/hyp.scratch.custom.yaml --epochs 250 --weights 'yolov7_training.pt' --cache-images #--freeze




