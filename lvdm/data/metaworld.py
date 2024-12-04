import numpy as np
import os
from PIL import Image,ImageSequence
import torch
from torch.utils.data import Dataset
import cv2
import random

METAWORLD_DESCIPTION ={
    "reach": "Move the robot end-effector to a specific target position in space.",
    "push": "Push an object to a designated target location on a flat surface.",
    "pick-place-wall": "Pick up an object and place it at a specific target position. Remember the wall",
    "door-open": "Open a hinged door by gripping and pulling or pushing it.",
    "door-lock": "Lock a hinged door by gripping .",
    "door-unlock": "Unlock a hinged door by gripping .",
    "drawer-open": "Open a drawer by pulling it outward.",
    "drawer": "Push a drawer back into its closed position.",
    "drawer_close": "Push a drawer back into its closed position.",
    "button_press": "Press a button by moving the robot’s end-effector onto it.",
    "button_press_topdown": "Press a button positioned on a surface, with a top-down motion.",
    "peg_insertion_side": "Insert a peg into a hole located on the side of an object.",
    "peg_insertion_topdown": "Insert a peg into a vertical hole with a top-down motion.",
    "sweep": "Sweep an object off a surface into a goal region using the robot arm.",
    "sweep_into_goal": "Sweep an object into a specific target area or goal.",
    "window_open": "Open a sliding window by gripping and pulling it.",
    "window_close": "Close a sliding window by pushing it back into place.",
    "lever-pull": "Pull a lever to a specific position.",
    "handle_press_side": "Press a handle to the side using lateral force.",
    "handle_press_topdown": "Push a vertical handle downward.",
    "stick-push": "Push a stick or lever into its socket.",
    "basketball": "Pick up a ball and place it into a basketball hoop.",
    "coffee_button": "Press a button on a coffee machine.",
    "coffee_push": "Push a lever or button on a coffee machine.",
    "coffee_pull": "Pull a lever or handle on a coffee machine.",
    "coffee_serve": "Pick up and serve a coffee cup into a designated area.",
    "bin_picking": "Pick an object from a bin and place it in a designated area.",
    "box_close": "Close the lid of a box by pushing it.",
    "plate_slide": "Slide a plate along a table surface into a goal area.",
    "plate_slide_back": "Slide a plate backward into a designated area.",
    "plate_slide_side": "Slide a plate laterally into a designated area.",
    "handle_pull": "Pull a handle attached to a fixed surface.",
    "handle_pull_side": "Pull a handle laterally to open or move an object.",
    "dial_turn": "Rotate a dial to a target angle.",
    "shelf_place": "Place an object onto a shelf at a specific location.",
    "push_wall": "Push a sliding wall to expose a target area.",
    "reach_wall": "Reach around an obstacle to touch a target.",
    "press_wall": "Press a button located behind a wall.",
    "faucet_open": "Rotate a faucet handle to turn it on.",
    "faucet_close": "Rotate a faucet handle to turn it off.",
    "hammer": "Use a hammer to strike a nail into a designated area.",
    "assembly": "Assemble cicle with bar onto a stick using the robot gripper.",
    "push_back": "Push an object backward into a target area.",
    "pick_place_wall": "Pick up an object and place it at a target location behind a wall.",
    "shelf_remove": "Remove an object from a shelf and place it in a designated area.",
    "stick_rotate": "Rotate a stick to a target angle or orientation.",
    "plate_place": "Pick up a plate and place it at a designated location.",
    "ball_place": "Pick up a ball and place it into a goal area.",
    "drawer_lock": "Lock a drawer by pushing it and engaging the lock mechanism.",
    "cup_place": "Pick up a cup and place it into a target area.",
    "block_stack": "Stack one block onto another in a stable configuration.",
    "block_unstack": "Remove a block from a stack and place it at a new location."
}

class MetaWorld(Dataset):
    def __init__(self,
                 path,
                 video_length=16,
                 resolution = [256,256],
                 ):
        self.path = path

        self.videos = []
        self.init_fps = 30
        self.video_length = video_length
        self.h = resolution[0]
        self.w = resolution[1]
        self.get_data()

    def get_videos(self, path, task):
        """
        input_path should be at 0 assembley level
        redered_images  
            assembley 
                0 
                    00001 
                        00001.png
                    00002 
                1
        """

        for ep in os.listdir(path):
            ep_path = os.path.join(path, ep)
            for view in os.listdir(ep_path):
                view_path = os.path.join(ep_path, view) 
                images = os.listdir(view_path)
                stride = len(images) // self.video_length

                images = sorted(images)
                images = [os.path.join(view_path, image) for image in images]
                # print(images)
                
                self.videos.append({'paths': images[::stride][:self.video_length], 'caption': METAWORLD_DESCIPTION[task.split('-v')[0]], 'stride': stride})
                

    def get_data(self):
        """
            task_name 
            video_path(use path to figure out )
        """
        self.task_name = os.listdir(self.path)
        # self.task_name = [t.split('-')[0] for t in self.task_name]
        for task in self.task_name:
            task_path = os.path.join(self.path, task)
            self.get_videos(task_path, task)

    def __len__(self):
        return len(self.videos)


    def load_images(self, image_paths):
        assert len(image_paths) == self.video_length
        images= []
        for image_path in image_paths:
            image = Image.open(image_path)
            resized_image = image.resize((self.w, self.h)) 
            images.append(np.array(resized_image))
        return np.stack(images, axis=0)

    def __getitem__(self, index):
        "return : video,text,frame_stride"
        ep = self.videos[index]
        frames = self.load_images(ep['paths'])
        language_annotation = ep['caption']
        stride = ep['stride']
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2 
        fps = 30 
        data = {'video': frames, 'caption': language_annotation, 'fps':fps,'frame_stride': stride}
        return data

if __name__ == '__main__':
    dataset = MetaWorld('/root/DynamiCrafter/data/rendered_images')
    print(len(dataset))
