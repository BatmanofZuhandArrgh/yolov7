import os
import glob as glob
import shutil

from tqdm import tqdm

img_ext = ['png', 'jpg', 'jpeg']
def yolov1_merge_to_yolov7(list_of_source_paths, output_path = None):
    #Create output path:
    image_output_path = os.path.join(output_path, 'images')
    label_output_path = os.path.join(output_path, 'labels')
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    
    #Get source path
    for dir_path in tqdm(list_of_source_paths):
        cur_image_paths = [x for x in glob.glob(f'{dir_path}/obj_train_data/*') if x.split('.')[-1].lower() in img_ext]
        cur_label_paths = glob.glob(f'{dir_path}/obj_train_data/*.txt')
        folder_name = dir_path.split('/')[-1].split('-yolo 1.1')[0].replace(" ", "_")

        assert len(cur_image_paths) == len(cur_label_paths)

        for index, img_path in enumerate(cur_image_paths):
            img_dst_path = os.path.join(image_output_path, folder_name + '_' + os.path.basename(img_path))
            shutil.copyfile(img_path, img_dst_path)
            
            txt_path = cur_label_paths[index]
            txt_dst_path = os.path.join(label_output_path, folder_name + '_' + os.path.basename(txt_path))
            shutil.copyfile(txt_path, txt_dst_path)


if __name__ == '__main__':
    source_paths = glob.glob(f'/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/*')

    far_source_paths = [
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/job_6-2022_10_16_02_13_40-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/job_14-2022_10_17_00_22_06-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_gt1_001535_001639-2022_10_16_02_23_36-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_opusat_010650_010656-2022_10_16_03_52_59-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_opusat_010722_010734-2022_10_17_00_01_13-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_tausat_001619_001627-2022_10_17_00_38_16-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_tausat_001638_001647-2022_10_17_00_42_14-yolo 1.1'
    ]
    deployment_source_paths = [
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/job_8-2022_10_16_03_42_53-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/job_12-2022_10_16_04_39_36-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_17_cubesats_deployed_for_research_this_week_000001_000004-2022_07_10_15_12_16-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_first_elementary_school_built_cubesat_seq-2022_10_12_03_49_25-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_gt1_001525_001530-2022_10_12_03_56_15-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_light1_1080p_002021_002025-2022_10_16_01_53_05-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_tausat_001522_001527-2022_10_16_04_14_17-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_kibo_tausat_004522_004526-2022_10_16_04_28_44-yolo 1.1',
        '/home/anhnguyen/Documents/CubeSat/sequences/labelled_seq/task_space_to_ground_ satellites_away!_ 01_20_2017_000022_000026-2022_10_17_00_30_59-yolo 1.1'
    ]
    yolov1_merge_to_yolov7(list_of_source_paths=deployment_source_paths, output_path='/home/anhnguyen/Documents/real_testsets/full_real_testset/deployment_testset')
