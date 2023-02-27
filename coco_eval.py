from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

annFile = '/home/iasrl/Documents/real_dataset/full_real_testset/full_labelled_set/ful_labelled_set.json'
cocoGt=COCO(annFile)

resFile='/home/iasrl/Documents/yolov7/runs/test/0redo_baseline_synth-100e+real_onfull/best_predictions.json'
cocoDt=cocoGt.loadRes(resFile)

print('here')
print(cocoDt.getImgIds())

imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:100]
# imgIds = imgIds[np.random.randint(100)]


# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
# cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()