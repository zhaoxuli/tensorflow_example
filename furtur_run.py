import os
import os.path
import sys
sys.path.append('./symbol/')
import numpy as np
import tensorflow as tf
import  load_cmp_data as load_data
import VGG
import tools

################################################################
'''  Base Setting
train_samples_num = 13297  test_samples-num = 3326
'''
################################################################
Train_img_num = 13297
IMG_W = 128
IMG_H = 128
N_CLASSES = 20
#Epoch and step setting
BATCH_SIZE = 32
MAX_Epoch = 500
check_times_pre_epoch = 5
val_times_pre_epoch = 2
Epoch_to_save = 2
#dont need to be set
Epoch_steps = int(Train_img_num/BATCH_SIZE)  #415
MAX_STEP = Epoch_steps*MAX_Epoch
check_steps = int(Epoch_steps/check_times_pre_epoch)
val_steps = int(Epoch_steps/val_times_pre_epoch)
save_steps =  Epoch_steps * Epoch_to_save
#setting lr_rate
init_lr = 0.0002
stair_steps = Epoch_steps*5
decay_ratio = 0.5
#log_file
DATA_DIR='D:\Competation_data\process_img'
TRAIN_ANNO='./Anno/train_anno.txt'
VAL_ANNO='./Anno/test_anno.txt'
log_file = sys.argv[1]
Recoder_file = 'logs_com'

if os.path.exists(Recoder_file)==False:
    os.makedirs(Recoder_file)
train_log_dir = Recoder_file+os.sep+'train'
val_log_dir = Recoder_file+os.sep+'validation'

def add_log(out_log,log_file):
    out_log = out_log+'\n'
    ctx = open(log_file,'a')
    ctx.write(out_log)
    ctx.close()

def train():
    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = load_data.get_batch(True,DATA_DIR,TRAIN_ANNO,IMG_W,IMG_H,BATCH_SIZE,N_CLASSES)
        val_image_batch, val_label_batch = load_data.get_batch(False,DATA_DIR,VAL_ANNO,IMG_W,IMG_H,BATCH_SIZE,N_CLASSES)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = VGG.VGG16_comp(x, N_CLASSES)
    loss = tools.ce_loss(logits, y)
    accuracy = tools.accuracy(logits, y)


    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(init_lr,my_global_step,decay_steps=stair_steps,decay_rate=decay_ratio,staircase=True)
    train_op = tools.Momentum_optimize(my_global_step, loss, learning_rate, Momentum=0.9)

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    loss_lst = [0,0,0,0,0,0,0,0]
    acc_lst = [0,0,0,0,0,0,0,0]
    count = 0
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break

            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc ,lr_rate= sess.run([train_op, loss, accuracy,learning_rate],
                                            feed_dict={x:tra_images, y:tra_labels})
            if step % check_steps == 0 or (step + 1) == MAX_STEP:
                out_log = 'Step: %d, loss: %.4f, accuracy: %.4f ,lr_rate:%.7f' % (step, tra_loss, tra_acc,lr_rate)
                add_log(out_log,log_file)
                loss_lst[count]=tra_loss
                acc_lst[count]=tra_acc
                count = count +1
                print (out_log)

                if count == check_times_pre_epoch:
                    mean_loss = float(sum(loss_lst)/check_times_pre_epoch)
                    mean_acc = float(sum(acc_lst)/check_times_pre_epoch)
                    out_log ='Finished %d Epoch, mean_loss = %.4f, mean_accuracy = %.2f **' %(step//Epoch_steps, mean_loss, mean_acc)
                    add_log(out_log,log_file)
                    count = 0
                    print(out_log)

            if step % val_steps == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y:val_labels})
                out_log ='**  Step %d, val loss = %4f, val accuracy = %.2f **' %(step, val_loss, val_acc)
                add_log(out_log,log_file)
                print(out_log)


            if step % save_steps == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
if  __name__ =='__main__':
    train()
