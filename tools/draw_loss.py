import os
import numpy as np
import matplotlib.pyplot as plt
import random

def  draw(out_dir,cruve_type,x,y,x1=None,y1=None):
    num = random.randint(0,5)
    color_lst = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(15,10))
    if x1 is None:
        plt.plot(x,y,label=cruve_type,color=color_lst[num],marker='o',linewidth=2)
    else:
        plt.plot(x,y,label=cruve_type.split('&')[0],color=color_lst[num],marker='o',linewidth=2)
        plt.plot(x1,y1,label=cruve_type.split('&')[1],color=color_lst[num+1],marker='o',linewidth=2)
    x_step = x[1]-x[0]
    if x_step >50:
        x_step = 10*x_step
    plt.xticks(np.arange(min(x), max(x)+1,x_step ))
    plt.legend()
    plt.show()
    plt.savefig(out_dir+os.sep+cruve_type+'.png')


def get_info(anno_file):
    step_lst = []
    s_loss= []
    s_acc= []
    s_lr = []

    val_step = []
    val_loss =[]
    val_acc =[]

    epoch_lst = []
    e_loss=[]
    e_acc=[]

    ctx = open(anno_file,'r').readlines()
    for ele in  ctx:
        line = ele.split(',')
        if line[0][0:4]=='Step':
            step_lst.append(int(line[0].split(':')[-1].strip()))
            s_loss.append(float(line[1].split(':')[-1].strip()))
            s_acc.append( float(line[2].split(':')[-1].strip()))
            s_lr.append(  float(line[3].split(':')[-1].strip()))
        if line[0][0:2]=='**':
            val_step.append(int(line[0].split(' ')[-1].strip()))
            val_loss.append(float(line[1].split('=')[-1].strip()))
            val_acc.append(float(line[2].split(' ')[-2].strip()))
        if line[0][0:8]=='Finished':
            epoch_lst.append(int(line[0].split(' ')[1]))
            e_loss.append(float(line[1].split(' ')[-1].strip()))
            e_acc.append(float(line[2].split(' ')[-2].strip()))

    return step_lst,s_loss,s_acc,s_lr,val_step,val_loss,val_acc,epoch_lst,e_loss,e_acc


if __name__ =='__main__':
    anno_file = '../05-11-mrn.log'
    dir_key = anno_file.split('/')[-1][0:-4]
    print (dir_key)
    out_dir = '../tools'+os.sep+dir_key
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    step,s_loss,s_acc,s_lr,val_step,val_loss,val_acc,epoch_lst,e_loss,e_acc =get_info(anno_file)
    draw(out_dir,'train_loss&vall_loss',step,s_loss,val_step,val_loss)
    draw(out_dir,'train_acc&vall_acc',step,s_acc,val_step,val_acc)
    draw(out_dir,'lr_rate',step,s_lr)
    if len(epoch_lst)>0:
        draw(out_dir,'epoch_loss',epoch_lst,e_loss)
        draw(out_dir,'epoch_acc',epoch_lst,e_acc)

