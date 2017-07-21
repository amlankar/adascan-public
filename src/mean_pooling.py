import tensorflow as tf
import numpy as np
import time
import gc
import os
import glob
import threading

import img_proc
import convNet
from datasets import minibatch_iterator
from evaluation import test_model

def remove_none(l):
# To remove None from gradient computation
    ind = []
    for i in range(len(l)):
	if (l[i][0] == None):
	    ind.append(i)
    if len(ind) == 0:
	return l
    else:
	for i in range(len(ind)):
            l.pop(ind[i] - i)
	return l

def get_fc_layer(name,size):
    '''
    name - Name to be added after W_ and b_ (can be any datatype convertible to str)
    size - [inp_size,out_size]
    tf.get_variable looks for variable name in current scope and returns it.
    If not found, it uses the initializer
    '''
    with tf.device('/cpu:0'):
        W = tf.get_variable('W_'+str(name),
                            shape=[size[0],size[1]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                             seed=None,
                                                                             dtype=tf.float32))
        b = tf.get_variable('b_'+str(name),
                            shape=[size[1]],
                            initializer=tf.random_uniform_initializer(-model_options['init_scale'],\
                                                                      model_options['init_scale']))
    return W,b

def grad_clip(grads,clip=10):
    clipped = []
    for (grad,var) in grads:
        if var != None and grad != None:
            clipped.append((tf.clip_by_norm(grad,clip_norm=clip),var))

    return clipped

def make_input(model_options):
    '''
    Prepare the input placeholders and queues
    '''
    model_vars = {}
    if model_options['mode'] == 'train':
        images = tf.placeholder("float",[None,224,224,model_options['num_channels']])
        model_vars['images'] = images

        labels = tf.placeholder("uint8",[1])
        model_vars['labels'] = labels

        q = tf.RandomShuffleQueue(200, model_options['min_to_keep'], [tf.float32, tf.uint8],
                                  shapes=[[model_options['example_size'],224,224,\
                                  model_options['num_channels']],1])
        model_vars['queue'] = q
        enqueue_op = q.enqueue([images, labels])
        model_vars['enqueue_op'] = enqueue_op

    else:
        num_crops = 10 if model_options['flip'] else 5;
        images = tf.placeholder("float",[num_crops,model_options['example_size']\
                                         ,224,224,model_options['num_channels']])
        labels = tf.placeholder("uint8",[num_crops,1])
        names = tf.placeholder("string",[num_crops,1])
        model_vars['images'] = images
        model_vars['labels'] = labels
        model_vars['names'] = names

        q = tf.FIFOQueue(100, [tf.float32, tf.uint8, "string"],
                              shapes=[[model_options['example_size'],224,224,\
                              model_options['num_channels']],[1],[1]])

        model_vars['queue'] = q
        enqueue_op = q.enqueue_many([images, labels, names])
        model_vars['enqueue_op'] = enqueue_op

    return model_vars

def model(model_options,model_vars):
    gpu_vars = {}      
    with tf.variable_scope('model') as model_scope:
        if model_options['mode'] == 'train':
            minibatch_images,minibatch_labels =\
            model_vars['queue'].dequeue_many(model_options['batch_size'])

            minibatch_images =\
            tf.reshape(minibatch_images,shape=[-1,224,224,model_options['num_channels']])
            minibatch_labels = \
            tf.reshape(minibatch_labels,shape=[-1])

            keep_prob = model_options['keep_prob']
            vgg_keep_prob = model_options['vgg_keep_prob']

        else:
            minibatch_images,minibatch_labels,minibatch_names = \
            model_vars['queue'].dequeue_many(model_options['batch_size'])
            minibatch_images = tf.reshape(minibatch_images,shape=[-1,224,224,model_options['num_channels']])
            minibatch_labels = tf.reshape(minibatch_labels,shape=[-1])
           
            gpu_vars['minibatch_names'] = minibatch_names
            keep_prob = 1
            vgg_keep_prob = 1

        gpu_vars['minibatch_labels'] = minibatch_labels
        labels_ = tf.one_hot(minibatch_labels,depth=model_options['num_classes']+1)[:,1:]
        # Ignore 0th dimension as no label is 0 (labels start from 1)

        # Build VGG
        vgg = convNet.Vgg16(model_options['vgg_path'])
        with tf.variable_scope('vgg') as vgg_scope:
            vgg.build(model_options, minibatch_images, input_type = model_options['input'], keep_prob= vgg_keep_prob)


        with tf.variable_scope('mean') as mean_scope:

            vgg_output = tf.reshape(vgg.fc6,shape=[model_options['batch_size'],-1, 4096])

            pooled = tf.reduce_mean(vgg_output,1)
            norm_pooled = tf.nn.l2_normalize(pooled,1)
            norm_pooled_dropped = tf.nn.dropout(norm_pooled, keep_prob)

            num_classes = 51 if model_options['hmdb'] else 101

            W_final,b_final = \
            get_fc_layer('final',[model_options['cnn_feat_size'],num_classes])

            logits = tf.nn.bias_add(tf.matmul(norm_pooled_dropped,W_final),b_final)

            prediction = tf.nn.softmax(logits,name='prediction')
            gpu_vars['prediction'] = prediction

            if model_options['mode'] == 'train':
                ce = \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                       labels_,
                                                                       name='cross_entropy'))

                gpu_vars['ce'] = ce
                total_loss = ce

                # Optimizer
                opt_vgg = model_vars['opt_vgg']
                opt_mean = model_vars['opt_mean']

                train_vars_vgg = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   "model/vgg")

                train_vars_mean = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   "model/mean")

                gpu_vars['grads_vgg'] = remove_none(opt_vgg.compute_gradients(total_loss,train_vars_vgg))
                gpu_vars['grads_mean'] = remove_none(opt_mean.compute_gradients(total_loss,train_vars_mean))

            vgg_scope.reuse_variables()
        model_scope.reuse_variables()

        # Reuse variables once made
        return gpu_vars

def make_optimizers(model_options,model_vars):
    with tf.device('/cpu:0'):
        model_vars['opt_vgg'] = tf.train.AdamOptimizer(learning_rate=0.000001, beta1=0.9, beta2=0.999,\
                                                       epsilon=1e-08, use_locking=False, name='Adam')

        model_vars['opt_mean'] = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,\
                                                           epsilon=1e-08, use_locking=False, name='Adam')
        
    return model_vars

def average_gradients(tower_grads):
    '''
    Takes a list of (gradient,variable) tuples and returns
    the average gradient

    Taken from Tensorflow cifar 10 example
    '''
    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        # * is the splat operator
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

def make_model(model_options):
    gpus = model_options['gpus'].split(',')
    print 'Trying to use GPUs:',gpus
    n_gpus = len(gpus)
    model_options['n_gpus'] = n_gpus
    model_vars_list = []

    model_vars = make_input(model_options)
    model_vars = make_optimizers(model_options,model_vars)

    for i,gpu in enumerate(gpus):
        with tf.device(gpu):
            with tf.name_scope('GPU'+gpu[-1]):
                model_vars_list.append(model(model_options,model_vars))
                tf.get_variable_scope().reuse_variables()

    with tf.device('/cpu:0'):
        if model_options['mode'] != 'train':
            model_vars['prediction_adascan'] =\
            tf.concat(0,[tmp['prediction'] for tmp in model_vars_list])

            model_vars['minibatch_names'] =\
            tf.concat(0,[tmp['minibatch_names'] for tmp in model_vars_list])

            model_vars['minibatch_labels'] =\
            tf.concat(0,[tmp['minibatch_labels'] for tmp in model_vars_list])

        if model_options['mode'] == 'train':
            model_vars['ce'] = model_vars_list[-1]['ce']
            tf.scalar_summary('meanpool/train', model_vars['ce'])

            model_vars['grads_vgg'] =\
            average_gradients([tmp['grads_vgg'] for tmp in model_vars_list])
 
            model_vars['grads_mean'] =\
            average_gradients([tmp['grads_mean'] for tmp in model_vars_list])

            model_vars['grads_ce'] = model_vars['grads_vgg'] + model_vars['grads_mean']
            print "Trainable Variables : Adascan"

            for (grad,var) in model_vars['grads_ce']:
                 if grad != None:
                     print var.name

            train_step_vgg = \
            model_vars['opt_vgg'].apply_gradients(grad_clip(model_vars['grads_vgg'],
                                                            clip=model_options['grad_clip']))

            train_step_mean = \
            model_vars['opt_mean'].apply_gradients(grad_clip(model_vars['grads_mean'],
                                                            clip=model_options['grad_clip']))
            model_vars['train_step_vgg'] = train_step_vgg
            model_vars['train_step_mean'] = train_step_mean

            merged_summaries = tf.merge_all_summaries()
            model_vars['merged_summaries'] = merged_summaries

    return model_options,model_vars

def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='test')
    parser.add_argument('-mode', type=str, default='train')
    # 'train' or 'test'
    parser.add_argument('-gpus', type=str, default='/gpu:0')
    # Comma separate string with gpu names to be used
    parser.add_argument('-hmdb', type=bool, default=False)
    parser.add_argument('-hmdb_restore', type=str, default=None)
    parser.add_argument('-input', type=str, default='rgb')
    # input can be either rgb or flow
    parser.add_argument('-num_data_threads', type=int, default=2)
    parser.add_argument('-min_to_keep', type=int, default=10)
    parser.add_argument('-init_scale', type=float, default=0.1)
    parser.add_argument('-reg_strength', type=float, default=1000000)
    parser.add_argument('-batch_size', type=int, default=6)
    parser.add_argument('-num_epochs', type=int, default=15)
    parser.add_argument('-print_freq', type=int, default=10)
    parser.add_argument('-save_freq', type=int, default=500)
    parser.add_argument('-grad_clip',type=float,default=10.)
    parser.add_argument('-split_dir', type=str, default=None)
    parser.add_argument('-data_list', type=str, default=None)
    # data_list should be a file in split_dir with file names
    parser.add_argument('-data_dir', type=str, default=None)
    # This should contain the .npy files
    parser.add_argument('-grad_monitor', type=bool, default=False)
    # Slows down training! Use if required.
    parser.add_argument('-keep_prob', type=float, default=0.2)
    parser.add_argument('-vgg_keep_prob', type=float, default=0.8)
    parser.add_argument('-flip', type=bool, default=True)
    parser.add_argument('-num_channels', type=int, default=3)
    parser.add_argument('-vgg_path', type=str, default=None)
    parser.add_argument('-eps', type=float, default=0.)
    parser.add_argument('-save', type=bool, default=False)
    parser.add_argument('-num_classes',type=int, default=101)
    parser.add_argument('-logdir', type=str, default='./log/')
    parser.add_argument('-example_size', type=int, default=25)
    parser.add_argument('-cnn_feat_size', type=int, default=4096)

    model_options = vars(parser.parse_args())

    return model_options

if __name__=='__main__':
    model_options = argparser()
    if model_options['mode'] == 'train':
        model_options['name'] = model_options['input']+'_'+model_options['name']
    model_options,model_vars = make_model(model_options)
    # Build Model
    model_vars['full_images'],model_vars['resize_op'] = img_proc.resize_tf(model_options)

    np.set_printoptions(precision = 4)
    print model_options

    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        model_vars['train_writer'] = \
        tf.train.SummaryWriter(model_options['logdir']+'/'+model_options['name'],sess.graph)
        variables = tf.all_variables()
        if model_options['hmdb']:
            # Restore vgg from a UCF checkpoint
            restore_vars = []
            for var in tf.all_variables():
                if 'vgg' in var.name:
                    if 'Adam' in var.name:
                        continue
                    restore_vars.append(var)

            saver_restore = tf.train.Saver(var_list=restore_vars)
            print 'Restoring'
            saver_restore.restore(sess,model_options['hmdb_restore'])

        if model_options['save']:
            saver = tf.train.Saver(var_list=variables,max_to_keep=100)
        
        if not os.path.exists('checkpoints/'+model_options['name']):
            os.system('mkdir -p checkpoints/'+model_options['name'])

        save_name = model_options['name']
        # -name is also used as the path to checkpoint file for testing

        if model_options['mode'] != 'train':
            model_options['start_from']=0 
            print 'Restoring -', save_name
            saver.restore(sess,save_name)
            test_model(model_options,sess,model_vars)

        if model_options['mode'] == 'train':
            save_name = './checkpoints/'+model_options['name']+'/'+model_options['name']
            minibatch_idx=-1
            epoch_idx=0
            step=-1
            model_options['start_from']=0 
            ######## UNCOMMENT AND CHANGE WHILE RESUMING ##############
            '''
            minibatch_idx=500
            epoch_idx=1
            step=2880
            # add saver.restore here
            saver.restore(sess,'checkpoints/rgb_test_multiGPU_split2/rgb_test_multiGPU_split2epoch1_minibatch2880.model')
            model_options['start_from']=minibatch_idx*model_options['batch_size']*model_options['n_gpus']
            if model_options['flip']:
                model_options['start_from']/=2
            # The point in the data list to start from, it is shuffled with a seed
            '''
	    ###########################################################

            for epoch_idx in range(epoch_idx,model_options['num_epochs']):
                t = threading.Thread(target= minibatch_iterator,
                                     args=(sess,model_options,model_vars,model_options['data_list']))
                t.start()
                update_start = time.time()
                while t.is_alive():
                    minibatch_idx += 1
                    step += 1
                    if minibatch_idx%model_options['print_freq'] != 0:
                        summary,_,_ = sess.run([model_vars['merged_summaries'],
                                              model_vars['train_step_mean'],
                                              model_vars['train_step_vgg']])

                    else:
                        if not model_options['grad_monitor']:
                            summary,ce,_,_ = \
                            sess.run([model_vars['merged_summaries'],
                                      model_vars['ce'],
                                      model_vars['train_step_mean'],
                                      model_vars['train_step_vgg']])

                        else:
                            summary,ce,_,_,grads_ce = \
                            sess.run([model_vars['merged_summaries'],
                                      model_vars['ce'],
                                      model_vars['train_step_mean'],
                                      model_vars['train_step_vgg'],
                                      model_vars['grads_ce']])

                            print '--------Cross Entropy Gradients--------'
                            for i in xrange(len(grads_ce)):
                                print '%s: %f' \
                                %(model_vars['grads_ce'][i][1].name, np.linalg.norm(grads_ce[i][0]))

                        freed_space = gc.collect()
                        print "Epoch: %d" %(epoch_idx),\
                              "Minibatches Done: %d" %(minibatch_idx),\
                              "Loss: %f" %(ce),\
                              "Update Time, %f" %(time.time()-update_start)

                        update_start = time.time()

                    model_vars['train_writer'].add_summary(summary,step*model_options['batch_size'])
                    if minibatch_idx%model_options['save_freq']==0:
                        if model_options['save']:
                            print 'Saving: ',
                            saver.save(sess,save_name+\
                            'epoch'+str(epoch_idx)+\
                            '_minibatch'+str(step)+'.model')
                            os.remove(save_name+\
                            'epoch'+str(epoch_idx)+\
                            '_minibatch'+\
                            str(step)+\
                            '.model.meta')
                            # To save meta file only once
                # Reset minibatch_idx       
                minibatch_idx = -1
                model_options['start_from'] = 0
