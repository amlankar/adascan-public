import tensorflow as tf
import random
import numpy as np

def resize_tf(model_options):
    with tf.device("/cpu:0"):
        full_images = tf.placeholder("float",[None,None,None,model_options['num_channels']])
        newSize = tf.constant([224,224])
	resize_op = tf.image.resize_images(full_images,newSize)
	return full_images,resize_op

def getCrops(sess,model_options,model_vars,
             frames,tmpLab):
    '''
    Gets exactly 10 crops as in github.com/yjxiong/caffe/ for val or test
    Gets 1 random center crop for train
    frames is of shape (num_examples,240,320,channel_size)
    '''

    num_crops = 1
    if model_options['flip']:
        num_crops = 2

    if model_options['mode'] == 'train':
        tmpLab = np.tile(tmpLab,num_crops)
        crops_choice = [240,224,192,168]
        height = crops_choice[np.random.randint(len(crops_choice))]
        res_height = (frames.shape[1] - height)//2
        width = crops_choice[np.random.randint(len(crops_choice))]
        res_width = (frames.shape[2] - width)//2

        frames = frames[:,res_height:res_height+height,res_width:res_width+width,:]

        if model_options['flip']:
            if model_options['input'] == 'flow':
                frames_flip = np.zeros(frames.shape,dtype=np.float32)
                frames_flip[:,:,:,::2] = 255 - frames[:,:,:,::2]
                frames_flip[:,:,:,1::2] = frames[:,:,:,1::2]
                # Flip only the x component of flow (even numbered channels)
                frames = np.concatenate([frames,frames_flip],axis=0)

            else:
                frames_flip = np.zeros(frames.shape,dtype=np.float32)
                frames_flip = frames[:,:,::-1,:]
                # Spatial Flip (x-y)
                frames = np.concatenate([frames,frames_flip],axis=0)

            resized = sess.run(model_vars['resize_op'],feed_dict={model_vars['full_images']:frames})

        else:
            # Only resize and return
            resized = sess.run(model_vars['resize_op'],feed_dict={model_vars['full_images']:frames})

    else:
        # Hardcoded for inputs of shape (240,320)
        crop_1 = frames[:,:224,:224,:]
        crop_2 = frames[:,:224,-224:,:]
        crop_3 = frames[:,8:232,48:272,:]
        crop_4 = frames[:,-224:,:224,:]
        crop_5 = frames[:,-224:,-224:,:]

        if model_options['flip']:
            num_crops = 10
            frames_flip = np.zeros(frames.shape,dtype=np.float32)

            if model_options['input'] == 'flow':
                frames_flip[:,:,:,1::2] = frames[:,:,:,1::2]
                frames_flip[:,:,:,::2] = 255 - frames[:,:,:,::2]

            else:
                frames_flip[:,:,:,:] = frames[:,:,::-1,:]

            crop_1_f = frames_flip[:,:224,:224,:]
            crop_2_f = frames_flip[:,:224,-224:,:]
            crop_3_f = frames_flip[:,8:232,48:272,:]
            crop_4_f = frames_flip[:,-224:,:224,:]
            crop_5_f = frames_flip[:,-224:,-224:,:]

            resized = np.stack([crop_1,crop_2,crop_3,crop_4,crop_5,crop_1_f,crop_2_f,crop_3_f,crop_4_f,crop_5_f],axis=0)

        else:
            num_crops = 5
            resized = np.stack([crop_1,crop_2,crop_3,crop_4,crop_5],axis=0)

        tmpLab = np.repeat(tmpLab,num_crops)
        tmpLab = np.expand_dims(tmpLab,1)

    return resized,tmpLab,num_crops
