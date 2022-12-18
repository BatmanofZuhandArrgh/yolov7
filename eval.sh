python test.py --data data/baseline_synth_custom_st.yaml --img 640 --batch 4 --conf 0.1 --iou 0.5 \
--device 0 --weights runs/train/baseline_synth-250e/weights/best.pt \
--no-trace --task cubesat_test --verbose \
--cubesat_output_folders \ 
    baseline_synth-250e_ontestagency \
    baseline_synth-250e_ontestdeployment \
    baseline_synth-250e_ontestfar \
--cubesat_testsets \
    /home/iasrl/Documents/real_dataset/full_real_testset/agency_testset/images \
    /home/iasrl/Documents/real_dataset/full_real_testset/deployment_testset/images \
    /home/iasrl/Documents/real_dataset/full_real_testset/far_testset/images
