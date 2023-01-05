python data_formatting/delete_cache.py

# #Evaluate baseline-synth+real
python test.py --data data/baseline_synth+real_custom_st.yaml --img-size 640 --batch 4 --conf 0.1 --iou 0.5 \
--device 0 --weights runs/train/redo_baseline_synth-100e+real/weights/best.pt \
--no-trace --task cubesat_test --verbose --save-json \
--cubesat_output_folders 0redo_baseline_synth-100e+real_ontestagency \
--cubesat_testsets /home/iasrl/Documents/real_dataset/full_real_testset/final_agency_testset/agencies_testset/images
