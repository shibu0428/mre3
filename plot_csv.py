import numpy as np
import matplotlib.pyplot as plt


start_frame=500
frames=100

foldername="1128_plot_right_q/"

choice_parts=[1]        #部位選択,全指定は012345
choice_dof=[3,4,5,6]      #自由度選択,全指定は0123456
ylim_min=-1                 #グラフの表示範囲   加速度なら30程度,quatなら1,-1
ylim_max=1


motions={
    "freeze",
    "hslashleft",
    "clap",
    "lassoleft",
    "lasso2hand",
    "lassoright",
    "vslashleft",
    "vslash2hand",
    "walkfast",
    "walkslow",
}



labels=["ax","ay","az","qw","qx","qy","qz"] #ここは変更しない。勝手にchoiceされる

data_cols=(7+2)*6
#csv読み取り->配列を返す
def readcsv2np(filepath,frames,choice_parts,start_frame):
    full_parts=[0,1,2,3,4,5]
    for num in choice_parts:
        if num in full_parts:
            full_parts.remove(num)
    delete_parts=full_parts
    data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[start_frame:start_frame+frames, :data_cols]
    cap_data = np.delete(data, [7,8,16,17,25,26,34,35,43,44,52,53], 1)
    # デバイス選択
    cap_cols=len(choice_parts)*7
    delete_list=[]
    for i in delete_parts:
        delete_list.extend(range(i*7, i*7+7))
    #print("delete cols = ", delete_list)
    return np.delete(cap_data, delete_list, 1)

def cutDoF(data,choice_dof):
    full_dof=[0,1,2,3,4,5,6]
    for num in choice_dof:
        if num in full_dof:
            full_dof.remove(num)
    delete_list=full_dof
    return np.delete(data,delete_list,1)

def cutlabels(labels,choice_dof):
    full_dof=[0,1,2,3,4,5,6]
    for num in choice_dof:
        if num in full_dof:
            full_dof.remove(num)
    delete_list=full_dof
    for num in reversed(delete_list):
        labels.pop(num)
    return labels


def plot(data,labels,filename,foldername,ylim_min,ylim_max):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(filepath)
    ax.set_ylim(ylim_min,ylim_max)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xlabel("frames")
    ax.set_ylabel("Qurt or accel")
    ax.plot(data,label=labels)
    ax.legend()
    #plt.show()
    fig.savefig("./image/"+foldername+filename+"_right.png")

cutlabel=cutlabels(labels,choice_dof)

for filename in motions:
    filepath="./dataset/1128_sit_"+filename+".csv"
    data=readcsv2np(filepath,frames,choice_parts,start_frame)
    cut_data=cutDoF(data,choice_dof)
    print(filepath)
    plot(cut_data,cutlabel,filename,foldername,ylim_min,ylim_max)



'''
ax,ay,az,qw,qx,qy,qzの順番

0D7A2	=	Head	=	0
1437E	=	WristR	=	1
12AA1	=	WristL	=	2
13D54	=	AnkleR	=	3
0FC42	=	AnkleL	=	4
121DE	=	Hip		=	5
'''