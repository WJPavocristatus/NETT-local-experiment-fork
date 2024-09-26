#!/usr/bin/env python3

import collections
import gym
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DVSWrapper(gym.ObservationWrapper):


    def __init__(self, env, change_threshold=60, kernel_size=(3, 3), sigma=1, is_color = True):
        super().__init__(env)
        
        self.change_threshold = change_threshold
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.num_stack = 2 ## default
        self.env = gym.wrappers.FrameStack(env,self.num_stack)
        self.stack = collections.deque(maxlen=self.num_stack)
        self.is_color = is_color
        
        try:
            _, _, width, height = self.env.observation_space.shape # stack, channels,
            self.shape=(3, width, height)
            self.observation_space = gym.spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)
            logger.info("In dvs wrapper")
        except Exception as e:
            raise e
        
        
    def create_grayscale(self, image):

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

    def gaussianDiff(self, previous, current):
       
        previous = cv2.GaussianBlur(previous, self.kernel_size, self.sigma)
        np_previous = np.asarray(previous, dtype=np.int64)
        
        current = cv2.GaussianBlur(current, self.kernel_size, self.sigma)
        np_current = np.asarray(current, dtype=np.int64)
        
        change = np_current - np_previous
        
        return change
    
    def observation(self, obs):
     
        
        if len(obs)>0:
            prev = np.transpose(obs[0], (1, 2, 0))
            current = np.transpose(obs[1], (1, 2, 0))
            
            if not self.is_color:
                prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
                current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
                
            change = self.gaussianDiff(prev, current)
            
            ## threshold
            dc = self.threshold(change)
            
        else:
            obs = np.transpose(obs, (1, 2, 0))
            
            if not self.is_color:
                obs = self.create_grayscale(obs)
            
            obs = np.array(obs, dtype=np.float32) / 255.0
            dc = self.threshold(obs)
        
        # change to channel first, w, h
        dc = np.transpose(dc, (2, 0, 1))
        
        return  dc.astype(np.uint8)

    def threshold(self, change):
     
        if not self.is_color:
            ret_frame = np.ones(shape=change.shape) * 128
            ret_frame[change >= self.change_threshold] = 255
            ret_frame[change <= -self.change_threshold] = 0
        else:
            ret_frame = abs(change)
            ret_frame[ret_frame < self.change_threshold] = 0
            
        return ret_frame
    
    def reset(self, **kwargs):
        initial_obs = self.env.reset(**kwargs)
        return self.observation(initial_obs)
