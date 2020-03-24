# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:41:35 2019

"""
import os
import tensorflow as tf
import numpy as np
import scipy.io
from CNN_models import SE_ResNeXt
from Prepare_data import preprocess
import pandas as pd
from lifelines.utils import concordance_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


weight_decay = 5e-5
momentum = 0.95
base_learning_rate = 1e-3
batch_size = 64
r0 = 40
r1 = batch_size - r0
y_label = np.zeros(batch_size)
y_label[r0:] = 1

#weights of different losses
alpha = 1.5
belta = 0.5
gamma = 0.5
sita = 0.5

num_epochs = 20
img_width = 128
img_heigh = 128


output_path = "../ICT-NPC-result_T1"
checkpoint_path = os.path.join(output_path, "checkpoint_path")
if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

main_path = "/ICT-NPC-summary_T1.mat"
clinic_path = "../ICT-PNC.csv"
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    # data1 is a dict, 'key' is ID of a patients and 'value' is a list that
    #records paths of all instance of the patient
    data1 = scipy.io.loadmat(main_path)
    clinic_msag = pd.read_csv(clinic_path, header=0, index_col = 0)
    #obtain training cohort
    tra_msag = clinic_msag[clinic_msag.data_cohort != 3]
    #obtain validation cohort
    nn = len(tra_msag)
    np.random.seed(1)
    out = np.random.choice(nn,int(0.3*nn))
    val_msag = tra_msag[out]

    tra_Pat_ID = np.array(tra_msag.index)
    tra_DFS_time = np.array(tra_msag.loc[:, 'DFS.time'], np.float32)
    tra_DFS_event = np.array(tra_msag.loc[:, 'DFS.event'], np.int32)

    val_Pat_ID = np.array(val_msag.index)
    val_DFS_time = np.array(val_msag.loc[:, 'DFS.time'], np.float32)
    val_DFS_event = np.array(val_msag.loc[:, 'DFS.event'], np.int32)

    tra_msag_0 = tra_msag[tra_msag.loc[:,'DFS.event'] == 0]
    tra_msag_1 = tra_msag[tra_msag.loc[:,'DFS.event'] == 1]
    tra_Pat_ID_0 = np.array(tra_msag_0.index)
    tra_Pat_ID_1 = np.array(tra_msag_1.index)
    
    #censored data
    tra_Pat_ind_0 = np.array(range(len(tra_msag_0)))
    #complete data, i.e. experiencing disease progression
    tra_Pat_ind_1 = np.array(range(len(tra_msag_1)))
    nn = len(tra_Pat_ind_0)
    #to make the ratio of the two types of data 1:2
    np.random.seed(1)
    out = np.random.choice(len(tra_Pat_ind_1),int(0.6*nn))
    tra_Pat_ind_1 = tra_Pat_ind_1[out]
    
    print(nn)
    print(len(tra_Pat_ind_1))
    # number of batch per epoch
    num_batchs = int(len(tra_Pat_ind_1) / r1)
    print('num_epochs: %d' % num_batchs)

def _prepare_surv_data(surv_time, surv_event):
    surv_data_y = surv_time * ([item == 1 and 1.0 or -1.0 for item in surv_event])
    surv_data_y = np.array(surv_data_y, np.float32)
    T = - np.abs(np.squeeze(surv_data_y))
    sorted_idx = np.argsort(T)
    _Y = surv_data_y[sorted_idx]
    
    return sorted_idx, _Y

def DeepSurv_loss(Y, Y_hat,pat_ind):
    # Obtain T and E from self.Y
    # NOTE: negtive value means E = 0
    Y_c = tf.squeeze(Y)
    Y_hat_c = tf.squeeze(Y_hat)
    Y_hat_c = tf.gather(Y_hat_c,pat_ind)
    Y_label_T = tf.abs(Y_c)
    Y_label_E = tf.cast(tf.greater(Y_c, 0), dtype=tf.float32)
    Obs = tf.reduce_sum(Y_label_E)
    
    Y_hat_hr = tf.exp(Y_hat_c)
    Y_hat_cumsum = tf.log(tf.cumsum(Y_hat_hr))
    
    # Start Computation of Loss function
    
    # Get Segment from T
    _, segment_ids = tf.unique(Y_label_T)
    # Get Segment_max
    loss_s2_v = tf.segment_max(Y_hat_cumsum, segment_ids)
    # Get Segment_count
    loss_s2_count = tf.segment_sum(Y_label_E, segment_ids)
    # Compute S2
    loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
    # Compute S1
    loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
    # Compute Breslow Loss
    loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)
    
    return loss_breslow

def _create_fc_layer(x, output_dim, activation, scope, keep_prob,is_training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weights', [x.shape[1], output_dim], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    
        b = tf.get_variable('biases', [output_dim], 
            initializer=tf.constant_initializer(0.0))
    
        # add weights and bias to collections
        tf.add_to_collection("var_fc", w)
        tf.add_to_collection("var_fc", b)
        if keep_prob > 0.0:
            layer_out = tf.layers.dropout(tf.matmul(x, w) + b, rate = keep_prob, training = is_training)
        else:
            layer_out = tf.matmul(x, w) + b
        if activation == 'relu':
            layer_out = tf.nn.relu(layer_out)
        elif activation == 'sigmoid':
            layer_out = tf.nn.sigmoid(layer_out)
        elif activation == 'tanh':
            layer_out = tf.nn.tanh(layer_out)
        else:
            raise NotImplementedError('activation not recognized')
    
        return layer_out
def GetWeight(ind0, ind1):
    #get the corresponding target about DFS
    DFS_time_0 = np.array(tra_msag_0.iloc[ind0, 4], np.float32)
    DFS_time_1 = np.array(tra_msag_1.iloc[ind1, 4], np.float32)
    DFS_time = np.concatenate([DFS_time_0,DFS_time_1],axis=0)
    sorted_idx, label_batch = _prepare_surv_data(DFS_time, y_label)
    
    #get the corresponding target about OS
    OS_time_0 = np.array(tra_msag_0.iloc[ind0, 2], np.float32)
    OS_time_1 = np.array(tra_msag_1.iloc[ind1, 2], np.float32)
    y_label_0 = np.array(tra_msag_0.iloc[ind0, 3], np.int32)
    y_label_1 = np.array(tra_msag_1.iloc[ind1, 3], np.int32)
    y_label_OS = np.concatenate([y_label_0,y_label_1], axis = 0)
    OS_time = np.concatenate([OS_time_0,OS_time_1],axis=0)
    sorted_idx1, label_batch1 = _prepare_surv_data(OS_time, y_label_OS)

    #get the corresponding target about DMFS
    DMFS_time_0 = np.array(tra_msag_0.iloc[ind0, 6], np.float32)
    DMFS_time_1 = np.array(tra_msag_1.iloc[ind1, 6], np.float32)
    y_label_0 = np.array(tra_msag_0.iloc[ind0, 7], np.int32)
    y_label_1 = np.array(tra_msag_1.iloc[ind1, 7], np.int32)
    y_label_DM = np.concatenate([y_label_0,y_label_1], axis = 0)
    DMFS_time = np.concatenate([DMFS_time_0,DMFS_time_1],axis=0)
    sorted_idx2, label_batch2 = _prepare_surv_data(DMFS_time, y_label_DM)

    #get the corresponding target about LRFS
    LRFS_time_0 = np.array(tra_msag_0.iloc[ind0, 8], np.float32)
    LRFS_time_1 = np.array(tra_msag_1.iloc[ind1, 8], np.float32)
    y_label_0 = np.array(tra_msag_0.iloc[ind0, 9], np.int32)
    y_label_1 = np.array(tra_msag_1.iloc[ind1, 9], np.int32)
    y_label_LR = np.concatenate([y_label_0,y_label_1], axis = 0)
    LRFS_time = np.concatenate([LRFS_time_0,LRFS_time_1],axis=0)
    sorted_idx3, label_batch3 = _prepare_surv_data(LRFS_time, y_label_LR)


    Pat_0 = tra_Pat_ID_0[ind0]
    Pat_1 = tra_Pat_ID_1[ind1]
    tmp = np.concatenate([Pat_0,Pat_1],axis=0)
    Pat_path = [str(i) for i in tmp]

    return Pat_path,label_batch,label_batch1,label_batch2,label_batch3,sorted_idx,sorted_idx1,sorted_idx2,sorted_idx3

def main():
        
    with tf.device('/gpu:0'):
        
        x = tf.placeholder(tf.float32, [None, img_heigh, img_width, 2], name = 'input')
        y = tf.placeholder(tf.float32, [None,], name = 'label')
        y1 = tf.placeholder(tf.float32, [None,], name = 'label-1')
        y2 = tf.placeholder(tf.float32, [None,], name = 'label-2')
        y3 = tf.placeholder(tf.float32, [None,], name = 'label-3')

        Pat_ind = tf.placeholder(tf.int32, [None,], name = 'Pat_ind')
        Pat_ind1 = tf.placeholder(tf.int32, [None,], name = 'Pat_ind-1')
        Pat_ind2 = tf.placeholder(tf.int32, [None,], name = 'Pat_ind-2')
        Pat_ind3 = tf.placeholder(tf.int32, [None,], name = 'Pat_ind-3')

        training_flag = tf.placeholder(tf.bool)
        pool_avg = SE_ResNeXt(x, training = training_flag).model
        pool_avg = tf.contrib.layers.flatten(pool_avg)
        # saver1 = tf.train.Saver()

        fc0 =  _create_fc_layer(pool_avg, 128, 'tanh', 'FC_layer1', 0.0, training_flag)
        #predict OS
        output1 = _create_fc_layer(fc0, 1, 'tanh', 'FC_layer2-1', 0.0, training_flag)
        #predict DMFS
        output2 = _create_fc_layer(fc0, 1, 'tanh', 'FC_layer2-2', 0.0, training_flag)
        #predict LRFS
        output3 = _create_fc_layer(fc0, 1, 'tanh', 'FC_layer2-3', 0.0, training_flag)
        output = tf.concat([output1,output2,output3], axis=1)
        # predict DFS
        output = tf.reduce_max(output,1)

        loss = DeepSurv_loss(y, output,Pat_ind)
        loss1 = DeepSurv_loss(y1, output1, Pat_ind1)
        loss2 = DeepSurv_loss(y2, output2, Pat_ind2)
        loss3 = DeepSurv_loss(y3, output3, Pat_ind3)
        
        loss_all = alpha*loss + belta*loss1 + gamma*loss2 + sita*loss3
        
        with tf.name_scope('MIL'):
            instance_index = tf.argmax(output, 0)
            pred_value = tf.reduce_max(output, 0)
            pred_value = tf.squeeze(pred_value)

        global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,num_batchs * 5, 0.9, staircase = True)
        optimizer = tf.train.MomentumOptimizer(learning_rate = base_learning_rate, momentum = momentum, use_nesterov = True)
        train_step = optimizer.minimize(loss_all, global_step = global_step) #,var_list = tf.get_collection("var_fc")

        saver = tf.train.Saver(max_to_keep = 20)
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)#设置每个GPU使用率0.7代表70%
        #
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
#            saver1.restore(sess, "../ICT-NPC-new/checkpoint_path/pretrained-model_epoch-5")
            best_val_C_index = 0.0
            num_up = 0
            # Loop over number of epochs
            for epoch in range(num_epochs):
            
                # print("{} Start epoch number: {}".format(datetime.now(), epoch))
                np.random.shuffle(tra_Pat_ind_0)
                np.random.shuffle(tra_Pat_ind_1)
                # Initialize iterator with the training dataset
                train_risk = 0.0
                epoch_loss = 0.0
                for i in range(num_batchs):
                    ind0 = tra_Pat_ind_0[i*r0:(i+1)*r0]
                    ind1 = tra_Pat_ind_1[i*r1:(i+1)*r1]
                    img_path,label_batch,label_batch1,label_batch2,label_batch3,sorted_idx,sorted_idx1,sorted_idx2,sorted_idx3 = GetWeight(ind0,ind1)
                    vd_img = []
                    #implementation of multiple instance learning
                    #instance_batch: a bag containing axial MR slices of a NPC patient
                    #each axial MR slice is a instance
                    for i,Pat_path in enumerate(img_path):
                        instance_paths = list(data1[Pat_path])
                        #obtain and preprocess data
                        instance_batch = preprocess(instance_paths, True)
                        IID = sess.run(instance_index, feed_dict = {x: instance_batch, training_flag: False})
                        #obtain the instance with the highest output value
                        vd_img.append(instance_batch[IID,:,:,:])
                    img_batch = np.stack(vd_img)
                    # training
                    _, all_loss, main_loss = sess.run([train_step, loss_all,  loss], feed_dict = { x: img_batch, y: label_batch, Pat_ind: sorted_idx, y1: label_batch1, Pat_ind1: sorted_idx1, 
                                            y2: label_batch2, Pat_ind2: sorted_idx2, y3: label_batch3, Pat_ind3: sorted_idx3, training_flag: True})

                    train_risk += main_loss
                    epoch_loss += all_loss

                train_risk /= num_batchs
                epoch_loss /= num_batchs
                line = 'epoch_loss: %.4f, train_risk: %.4f, epoch: %d' % (epoch_loss, train_risk, epoch + 1)
                print(line)
                with open(os.path.join(output_path,'logs.txt'), 'a') as f:
                    f.write(line + '\n')
                
                #evaluate model in the validation cohort
                val_pred = []
                for Pat_path in val_Pat_ID:
                    Pat_path = str(Pat_path)
                    instance_paths = list(data1[Pat_path])
                    instance_batch = preprocess(instance_paths, False)
                    Pat_pred = sess.run(pred_value, feed_dict = {x: instance_batch, training_flag: False}) 
                    val_pred.append(-np.exp(Pat_pred))
                val_pred = np.array(val_pred, np.float32)
                val_ci_value = concordance_index(val_DFS_time, val_pred, val_DFS_event)
                if (val_ci_value-best_val_C_index)/(best_val_C_index + 1e-8) > 0.005:
                    best_val_C_index = val_ci_value
                    num_up = 0
                elif num_up > 4:
                    #save model
                    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch')
                    saver.save(sess, checkpoint_name, global_step = epoch)
                    break
                else:
                    num_up += 1
                
                    
                    

if __name__ == '__main__':
    main()
