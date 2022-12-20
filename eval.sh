# python test.py --data data/baseline_synth_custom_st.yaml --img 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/baseline_synth-100e+real/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders baseline_synth-100e+real_ontestagency \
# --cubesat_output_folders baseline_synth-100e+real_ontestdeployment \
# --cubesat_output_folders baseline_synth-100e+real_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images


# python test.py --data data/baseline_synth_custom_st.yaml --img 640 --batch 4 --conf 0.1 --iou 0.3 \
# --device 0 --weights runs/train/baseline_synth-250e/weights/best.pt \
# --no-trace --task cubesat_test --verbose \
# --cubesat_output_folders baseline_synth-250e_ontestagency \
# --cubesat_output_folders baseline_synth-250e_ontestdeployment \
# --cubesat_output_folders baseline_synth-250e_ontestfar \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
# --cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images


python test.py --data data/only_real_custom_st.yaml --img 640 --batch 4 --conf 0.1 --iou 0.3 \
--device 0 --weights runs/train/only-real-250e/weights/best.pt \
--no-trace --task cubesat_test --verbose \
--cubesat_output_folders only-real-250e_ontestagency \
--cubesat_output_folders only-real-250e_ontestdeployment \
--cubesat_output_folders only-real-250e_ontestfar \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images

