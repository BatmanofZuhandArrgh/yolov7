python test.py --data data/baseline_synth_custom_st.yaml --img 640 --batch 4 --conf 0.01 --iou 0.5 \
--device 0 --weights runs/train/baseline_synth-200e/weights/best.pt \
--name baseline_synth-200e_ontestagency-temp --no-trace --task test --verbose

# python test.py --data data/baseline_synth_custom_st.yaml --img 640 --batch 4 --conf 0.001 --iou 0.5 \
# --device 0 --weights runs/train/baseline_synth-cubesat-200e/weights/best.pt \
# --name baseline_synth-cubesat-200e_onval --no-trace --task val --verbose

# python test.py --data data/baseline_synth_custom_st.yaml --img 640 --batch 4 --conf 0.001 --iou 0.5 \
# --device 0 --weights runs/train/trivial_unfrozen-cubesat-200e/weights/best.pt \
# --name trivial_unfrozen-cubesat-200e_ontestdeployment --no-trace --task test --verbose

# python test.py --data data/baseline_synth_custom_st.yaml --img 640 --batch 4 --conf 0.001 --iou 0.5 \
# --device 0 --weights runs/train/trivial_unfrozen-cubesat-200e/weights/best.pt \
# --name trivial_unfrozen-cubesat-200e_ontestagency --no-trace --task test --verbose