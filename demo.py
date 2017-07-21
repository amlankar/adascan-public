import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math
import sys
import tensorflow as tf
import numpy as np
from skvideo.io import vread
from skimage.transform import resize

def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_name', type=str, default='pre-trained/rgb_Split1')
    parser.add_argument('-vid_file', type=str)
    model_options = vars(parser.parse_args())
    model_options['meta_name'] = model_options['ckpt_name']+'.meta'
    return model_options

def model(model_options,sess):
    print 'Restoring Graph..'
    saver = tf.train.import_meta_graph(model_options['meta_name'])
    sess.run(tf.initialize_all_variables())
    print 'Restoring Variables..'
    saver.restore(sess,model_options['ckpt_name'])

    masks = sess.graph.get_tensor_by_name('Imp_Val:0')
    images = sess.graph.get_tensor_by_name('images:0')
    return masks,images

def gen_vis(masks,frames,file_name):
    n_frames = masks.shape[0]
    width = int(math.sqrt(n_frames))
    height = int(n_frames/width)
    plt.clf()
    plt.figure()
    f,axarr = plt.subplots(height,width)
    for idx,axis in enumerate(axarr.flatten()):
        axis.imshow(frames[idx+1])
        axis.set_axis_off()
        axis.set_title('Importance: '+str(masks[idx]),fontsize=4)
    save_name = file_name[:-4]+'.png'
    plt.savefig(file_name[:-4]+'.png')
    print 'Saved to', save_name

def get_mask(model_options):
    def _center_crop(frames):
                    y,x = frames.shape[1:3]
                    assert y >= 224 and x >= 224, 'Video too small!'

                    if y <= 430 and x <= 430: 
                        # central crop     
                        y_d = (y-224)//2
                        x_d = (x-224)//2
                        frames = frames[:,y_d:y_d+224,x_d:x_d+224,:]

                    return frames

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            masks,images = model(model_options,sess)
            print 'Reading Video..'
            frames = vread(model_options['vid_file'])
            print 'Video File: ',model_options['vid_file'], 'has shape ',frames.shape
            length = frames.shape[0]

            frames = _center_crop(frames)
            frames = np.stack([frames[int(math.ceil(i*length/17)),:,:,:] for i in range(17)],0)

            if frames.shape[1] != 224:
                print 'Big sized video, resizing'
                # make the larger side close to 420
                f = max(frames.shape[1:3])/420
                sh = (np.array(frames.shape[1:3])/f).astype(np.int32)
                for i in range(frames.shape[0]):
                    frames[i] = (resize(frames[i],sh)*255).astype(np.uint8)

                frames = _center_crop(frames)
            
            print 'New shape: ',frames.shape
            assert frames.shape[1:3] == (224,224), 'Bad aspect ratio!'           
            feed_dict = {}
            feed_dict[images] = np.reshape(frames,(-1,224,224,3))
            print 'Getting Mask...'
            mask = sess.run(masks,feed_dict=feed_dict)
            mask = mask[0]
        
    gen_vis(mask,frames,model_options['vid_file'])

if __name__=='__main__':
    model_options = argparser()
    get_mask(model_options)
