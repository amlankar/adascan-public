from datasets import minibatch_iterator
import numpy as np
import threading
import sys
import os
import time

def process_preds(model_options,preds,labels,names):
    print "Processing"
    num_crops = 10 if model_options['flip'] else 5

    names =  np.reshape(names,(names.shape[0]))
    
    new_preds = []
    new_labels = []

    file_list = []
    with open(model_options['split_dir']+model_options['data_list'],'r') as f:
        lines = f.read().split('\n')[:-1]
        for line in lines:
            file_list.append(line.strip().split('/')[1])
   
    name_scores = {}
    for i,fname in enumerate(file_list):
        if i%100 == 0:
            print 'Processed: ',i 
        idx = (names == fname)
        try:
           assert(sum(idx) == num_crops)
        except:
           print 'Not all crops present for', fname
           continue

        pred_fname = np.mean(preds[idx,:],axis=0)
        label_fname = labels[idx][0]
        name_scores[fname] = pred_fname
        new_preds.append(pred_fname)
        new_labels.append(label_fname)  

    save_scores(model_options,name_scores)
    preds = np.array(new_preds)
    labels = np.array(new_labels)

    print "Done"
    return preds, labels

def print_metrics(model_options,preds,labels):
    #import sklearn.metrics as metrics
    #one_hot_labels = np.eye(model_options['num_classes']+1)[labels][:,1:]
    #average_precision = metrics.average_precision_score(one_hot_labels,preds)

    preds = np.argmax(preds,axis=1)+1
    acc=sum(preds==labels)*100.0/labels.shape[0]

    # Mean Class Accuracy
    means = []
    for i in range(1,model_options['num_classes']+1):
        examples = (labels==i)
        predictions = (preds[examples] == i)
        if len(predictions) == 0:
            continue
        means.append(sum(predictions)*1.0/len(predictions))

    mean_class = np.mean(means)
    print "Accuracy: ",acc,\
          "Average Class Accuracy", mean_class
          #"Average Precision", average_precision,\

def save_scores(model_options,name_scores):
    if not os.path.exists('scores/'):
        os.system('mkdir scores/')
    save_name = 'scores/scores_'+model_options['name'].split('/')[-1]
    print 'Dumping scores to: '+save_name
    if not os.path.isdir('scores/'):
        os.mkdir('scores')
    np.savez_compressed(save_name,name_scores)

def test_model(model_options,sess,model_vars):
    preds = []
    labels = []
    names = []

    i = 0
    t = threading.Thread(target= minibatch_iterator,
                         args=(sess,model_options,model_vars,model_options['data_list']))
    t.start()
    time.sleep(20)
    print "Getting Predictions"
    count = 0
    #ipdb.set_trace()
    while True:
        queue_size = sess.run(model_vars['queue'].size())
        if queue_size < model_options['batch_size']*model_options['n_gpus']:
            print 'Breaking'
            break

        out = sess.run([model_vars['prediction_adascan'],
                        model_vars['minibatch_labels'],
                        model_vars['minibatch_names']])

        preds.extend(out[0])
        labels.extend(out[1])
        names.extend(out[2])

        count += 1
        pred_class = np.argmax(preds,axis=1)+1
        acc = (np.sum(np.equal(pred_class,labels))*100.0)/len(labels)
        sys.stdout.write('Minibatches Done: '+str(count)+'\t One crop accuracy: '+str(acc)+'\r')

        sys.stdout.flush()

    preds,labels= process_preds(model_options,np.array(preds),np.array(labels),np.array(names))
    print preds.shape
    print labels.shape
    print_metrics(model_options,preds,labels)
