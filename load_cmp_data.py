import os
import random
import  tensorflow as tf
def  get_lst(pre_dir,anno_txt,istrain):
    ctx = open(anno_txt,'r').readlines()
    img_lst = []
    label_lst = []
    for ele in ctx:
        ele = ele.strip()
        img_key = ele.split(' ')[0]
        label = ele.split(' ')[1]
        img_lst.append(pre_dir+os.sep+img_key)
        label_lst.append(int(label))

    if istrain ==True:
        random.shuffle(img_lst)
        random.shuffle(label_lst)
    return  img_lst,label_lst

def split_dataset(anno_txt):
    all_ctx = open(anno_txt,'r').readlines()
    test_ctx = open('test_anno.txt','w')
    train_ctx = open('train_anno.txt','w')
    count = 0
    for ele in  all_ctx:
        if count == 10:
            count  = 0
        if  count <2:
            test_ctx.write(ele)
        if 2<=count <10:
            train_ctx.write(ele)
        count = count +1


def get_batch(is_train,data_pre_dir,anno_txt, image_W, image_H, batch_size, num_class,capacity=2000):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    if is_train == True:
        image,label = get_lst(data_pre_dir,anno_txt,is_train)
    else:
        image,label = get_lst(data_pre_dir,anno_txt,is_train)

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    #image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.resize_images(image, (image_W, image_H))

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)

    #you can also use shuffle_batch
    image_batch, label_batch = tf.train.shuffle_batch([image,label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=capacity,
                                                      min_after_dequeue=capacity-1)
    label_batch = tf.one_hot(label_batch, depth= num_class)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, num_class])
    return image_batch, label_batch
#if __name__ =='__main__':
#    train_dir = 'D:\Competation_data\out_data'
#    image_H=128
#    image_W=128
#    batch_size=32
#    #anno_txt = 'D:\Competation_data\out_data\\anno_info.txt'
#    #split_dataset(anno_txt)
#    img_lst=[]
#    lable_lst = []
#    img_lst,label_lst=get_lst(train_dir,'./Anno/train_anno.txt')
#    #get_batch(img_lst, label_lst, image_W, image_H, batch_size)













