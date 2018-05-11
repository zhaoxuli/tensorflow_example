
import tensorflow as tf
import tools

#%%
def VGG16(x, num_class):

    x = tools.conv_bn('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv_bn('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv_bn('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv_bn('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    #x = tools.conv_bn('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    #x = tools.conv_bn('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    #x = tools.conv_bn('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    #x = tools.conv_bn('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    #x = tools.conv_bn('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    #x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    x = tools.conv_1x1('act_1x1' ,x,256,isActive=True)
    x = tools.conv_1x1('down_1x1',x,128,isActive=False)
    x = tools.GAP('GAP',x,is_max_pool=False)
    x = tools.FC_layer('fc6', x, out_nodes=64, isActive=True)
    #x = tools.batch_norm(x,'fc6')
    #x = tools.batch_norm(x,'fc7')
    x = tf.nn.dropout(x,0.5)
    x = tools.FC_layer('fc7', x, out_nodes=num_class)

    return x

def VGG16_comp(x, num_class):

    x = tools.conv_bn('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv_bn('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv_bn('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv_bn('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.conv_bn('conv5_1', x, 1024 , kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv5_2', x, 1024 , kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.conv_bn('conv5_3', x, 1024 , kernel_size=[3,3], stride=[1,1,1,1])
    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

    x = tools.FC_layer('fc6', x, out_nodes=4096, isActive=True)
    x = tools.FC_layer('fc7', x, out_nodes=2048, isActive=True)
    x = tf.nn.dropout(x,0.7)
    x = tools.FC_layer('fc8', x, out_nodes=num_class)

    return x



#%% TO get better tensorboard figures!

def VGG16N(x, num_class):

    with tf.name_scope('VGG16'):

        x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool1'):
            x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)

        x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool2'):
            x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)



        x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool3'):
            x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


        x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool4'):
            x = tools.pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


        x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool5'):
            x = tools.pool('pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)


        x = tools.FC_layer('fc6', x, out_nodes=4096)
        x = tools.batch_norm(x)
        x = tools.FC_layer('fc7', x, out_nodes=4096)
        x = tools.batch_norm(x)
        x = tools.FC_layer('fc8', x, out_nodes=num_class)

        return x



#%%








