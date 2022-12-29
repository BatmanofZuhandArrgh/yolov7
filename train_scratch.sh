# python train.py --workers 4 --device 0 --batch-size 4 --weights '' --data data/baseline_synth_custom_st.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --name scratch_baseline_synth-200e --hyp data/hyp.scratch.p5.yaml --epochs 200 --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --weights '' --data data/empty_space_custom_st.yaml    --img 640 640 --cfg cfg/training/yolov7.yaml --name scratch_empty_space-200e    --hyp data/hyp.scratch.p5.yaml --epochs 200 --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --weights '' --data data/only_cubes_custom_st.yaml     --img 640 640 --cfg cfg/training/yolov7.yaml --name scratch_only_cubes-200e     --hyp data/hyp.scratch.p5.yaml --epochs 200 --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --weights '' --data data/fewobj_custom_st.yaml         --img 640 640 --cfg cfg/training/yolov7.yaml --name scratch_few_obj-200e        --hyp data/hyp.scratch.p5.yaml --epochs 200 --cache-images #--freeze
# python train.py --workers 4 --device 0 --batch-size 4 --weights '' --data data/manyobj_custom_st.yaml        --img 640 640 --cfg cfg/training/yolov7.yaml --name scratch_many_obj-200e       --hyp data/hyp.scratch.p5.yaml --epochs 200 --cache-images #--freeze
#  --weights 'yolov7_training.pt'
#  --weights 'yolov7_training.pt'
#  --weights 'yolov7_training.pt'
#  --weights 'yolov7_training.pt'
#  --weights 'yolov7_training.pt'