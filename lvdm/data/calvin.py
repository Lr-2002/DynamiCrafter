import numpy as np
from PIL import Image,ImageSequence
import torch
from torch.utils.data import Dataset
import cv2
import random
def save_as_video(array,output):

    num_frames, height, width, channels = array.shape


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output, fourcc, 30, (width, height)) 

    for i in range(num_frames):
        frame = array[i]
        video_writer.write(frame)

    video_writer.release()

class Calvin(Dataset):
    def __init__(self,
                 path,
                 video_length=16,
                 resolution = [256,256],
                 ):
        self.path = path
        lang_path = path + '/lang_annotations/auto_lang_ann.npy'
        lang = np.load(lang_path,allow_pickle=True).reshape(-1)[0]

        self.task_description = lang['language']['task']
        self.language_annotation = lang['language']['ann']
        self.img_ids = lang['info']['indx']
        self.perspective = ['rgb_static','rgb_gripper']
        self.init_fps = 30
        self.video_length = video_length
        self.h = resolution[0]
        self.w = resolution[1]
        print(len(self.task_description))

    def __len__(self):
        return len(self.task_description)

    def __getitem__(self, index):
        "return : video,text,frame_stride"
        start_id, end_id = self.img_ids[index]
        stride = (end_id-start_id+1) // self.video_length
        file_paths = []
        task_description = self.task_description[index]
        language_annotation = self.language_annotation[index]

        for i in range(start_id, end_id + 1):
            id = str(i).zfill(7)
            file_path = self.path + f'/episode_{id}.npz' 
            file_paths.append(file_path)
            
        perspective = random.choice(self.perspective)
        #perspective = 'rgb_static'
        rgb_statics = []
        for file_path in file_paths:
            data = np.load(file_path)
            rgb_static = data[perspective] 
            image = Image.fromarray(rgb_static)
            resized_image = image.resize((self.w, self.h)) 
            #resized_image.save('0.png')
            rgb_statics.append(np.array(resized_image))

        # rgb_grippers = []
        # for file_path in file_paths:
        #     data = np.load(file_path)
        #     rgb_gripper = data['rgb_gripper'] 
        #     rgb_grippers.append(rgb_gripper)

        rgb_statics = np.stack(rgb_statics, axis=0)
        
        ## sample by stride
        frames = rgb_statics[::stride]
        frames = frames[0:self.video_length]
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2 
        fps = self.init_fps // stride 
        data = {'video': frames, 'caption': language_annotation, 'fps':fps,'frame_stride': stride}
        
        return data
    
# dataset = Calvin('/root/autodl-tmp/task_D_D/validation')

# # # 获取数据集大小
# # print(len(dataset))

# # # 获取单个样本
# data = dataset[0]

# print(data['video'].shape)

# print(1)

