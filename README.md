The repo is based on yolov7, added with training of the Space Inspection classes and domain adaptation training.
For the implementation of da loss, see utils/loss.py, CORAL Loss and CMD Loss

To install of this repo, follow original_README.md

1. Training progress (model weights, tensorboard files,...) and evaluation results(map evaluation, confusion matrix, ...) folders can be found in runs/train and runs/test, which can be download from [link](https://mailuc-my.sharepoint.com/:f:/g/personal/kim3dn_ucmail_uc_edu/EllIKUzCRCZCihs_OfgNpFwBLWEunm5c6aogllb30xA_Xg?e=aWtRYJ). The experiments mentioned in the paper are:

- onlyReal 37 training image: 
    + runs/train/only-real-250e
    + runs/test/1onlyreal-250e*
- RS2000: 
    + runs/train/redo_baseline_synth-100e+real
    + runs/test/0redo_baseline_synth-100e+real_*
- baseline only 2000 synthetic images:
    + runs/train/baseline_synth-250e
    + runs/test/2onlySynth-100e*
- trained on empty space synth images and real images: 
    + runs/train/empty_space-100e
    + runs/test/3empty_space-100e_*
- 5000 image training:
    + runs/train/official5000_synth-100e+real
    + runs/test/4official5000_synth+real-100e*
- Tuned
    + runs/train/tunedSquared2000_synth-100e+real
    + runs/test/5tunedSquared2000synth+real-100e_*
- Domain Adaptation:
    + runs/train/*domain_adapt_*
    + runs/test/*domain_adapt_*

For weight folders, see [link](https://mailuc-my.sharepoint.com/:u:/g/personal/kim3dn_ucmail_uc_edu/ESVQ4TXobTtNvcInirIkXF0Bino7Fi2uz4ua9JzIoGFl9A?e=AWOuMB)

2. For evaluation folders:
- onfull: evaluated on a full real dataset
- onsynth: evaluated on a full synthetic dataset
- ontestagency: evaluated on a full agencies dataset
- ontestdeployment: evaluated on a full deployment dataset
- ontestfar: evaluated on a full far test set

3. To train with domain adaptation:

DA training only supported for ComputeLoss and ComputeLossOTA, and not for ComputeLossBinOTA, ComputeLossAuxOTA

- Edit data/domain_adapt_official.yaml
+ domain_adapt: List of folder containing unsupervised target domain images
+ da_loss: string, either "CORAL" or "CMD"
+ feature_map_layer: list of ints, the numbering of the layers that are applied domain adaptation losses. The 1l, 3l and 3ld cases were commented
+ For the rest of the yaml, edit as a normal training config file

- run: (As in domain_adapt.sh)

python train.py --workers 4 --device 0 --batch-size 4 \
 --data data/domain_adapt_official.yaml --img 640 640 \
  --cfg cfg/training/yolov7.yaml --name domain_adapt_test --hyp data/hyp.scratch.custom.yaml --epochs 3  \
  --weights 'yolov7_training.pt' --cache-images   #--freeze

4. To test and eval: Same as normal, see original_README.md

* To train without domain adaptation, switch to git branch cubesats.

