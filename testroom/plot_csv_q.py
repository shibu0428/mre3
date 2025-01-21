import numpy as np
import matplotlib.pyplot as plt

filename=input("motion名を入力(datasetはデフォルトで入る)\n")
filepath="./dataset/"+filename+".csv"
all_data_frames=100
data_cols=(7+2)*6

choice_parts=[2]
delete_parts=[0,1,3,4,5]
labels=["ax","ay","az","qw","qx","qy","qz"]
data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]
cap_data = np.delete(data, [7,8,16,17,25,26,34,35,43,44,52,53], 1)
# デバイス選択
cap_cols=len(choice_parts)*7
delete_list=[]
for i in delete_parts:
    delete_list.extend(range(i*7, i*7+7))
#print("delete cols = ", delete_list)
cap_choice_data = np.delete(cap_data, delete_list, 1)


plt.title(filepath)
plt.xlabel("frames")
plt.ylabel("Y axis")
plt.plot(cap_choice_data,label=labels)
plt.legend()
#plt.show()
plt.savefig("./image/"+filename+".png")

