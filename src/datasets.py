import os
import numpy as np
import time
from img_proc import *
import random
import gc
import Queue
from threading import Thread

def worker(sess,model_options,model_vars,Queue,CLASS_DICT):
    while True:
        # print 'Queue Size', Queue.qsize()
        try:
            fname = Queue.get()
        except:
            return
        start = time.time()
        file_name_orig = fname.split(' ')[0].split('/')[1].strip()
        file_name = file_name_orig.replace('.avi','.npz')
        class_name = fname.split(' ')[0].split('/')[0].strip().lower()
        class_idx = CLASS_DICT[class_name]
        try:
            frames = np.load(model_options['data_dir']+file_name)['arr_0']
        except:
            print "Couldn't Open: ",model_options['data_dir']+file_name
	    Queue.task_done()
            continue

        idx = 0
        if model_options['mode'] == 'train':
            idx = random.randint(0,frames.shape[0]-1)

        frames = frames[idx]
        tmpImg,tmpLab,num_crops = getCrops(sess,model_options,model_vars,frames,np.array((class_idx)))

        if model_options['mode'] == 'train':
            for j in range(num_crops):
                size = model_options['example_size']
                sess.run(model_vars['enqueue_op'],feed_dict={model_vars['images']:tmpImg[j*size:(j+1)*size],
                         model_vars['labels']:tmpLab[j:(j+1)]})
        else:
            sess.run(model_vars['enqueue_op'],feed_dict={model_vars['images']:tmpImg,
                     model_vars['labels']:tmpLab,
                     model_vars['names']:[[file_name_orig]]*num_crops})

        Queue.task_done()

def minibatch_iterator(sess,model_options,model_vars,split_name='trainlist01.txt',shuffle=True):
    with open(model_options['split_dir']+'classInd.txt','r') as f:
        CLASS_DICT = {}
        for line in f:
            parts = line.strip().split(' ')
            CLASS_DICT[parts[1].lower()] = parts[0]

    with open(model_options['split_dir']+split_name,'r') as f:
        fnames = f.read().split('\n')[:-1]
        if shuffle:
            random.seed(1337)
            random.shuffle(fnames)
        fname_queue = Queue.Queue()
        for fname in fnames[model_options['start_from']:]:
	    fname_queue.put(fname)
        print 'Read ',len(fnames[model_options['start_from']:]),'file names from',split_name

    for i in range(model_options['num_data_threads']):
        t = Thread(target=worker,args=(sess,model_options,model_vars,fname_queue,CLASS_DICT))
        t.daemon = True
        t.start()

    fname_queue.join() # Block till all files processed
    print 'Joined -- all in queue'
    queue_size = -1
    min_to_keep = model_options['min_to_keep']+4*model_options['batch_size']*model_options['n_gpus']
    # stop after 4 minibatches left in queue
    if model_options['mode'] != 'train':
        min_to_keep = 0

    while True:
        old_queue_size = queue_size
        queue_size = sess.run(model_vars['queue'].size())
        print queue_size
        if queue_size <= min_to_keep:
            print 'Returning'
	    return 0
	else:
            time.sleep(2)
