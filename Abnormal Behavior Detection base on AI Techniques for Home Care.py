# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:54:45 2018

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:28:35 2018

@author: user
"""
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import collections
from collections import Counter
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
import math
from sklearn import cluster, datasets
from sklearn import cluster, datasets, metrics
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
#from sklearn.datasets.samples_generator import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette

#########讀取資料###########
print("Code by David_Chu")
root = tk.Tk()
root.withdraw()

print("請選擇老年人行為記錄檔!")
fileread = filedialog.askopenfilename()
dataset = pd.read_csv(fileread , encoding = "big5")
dataset_export = dataset

number = int(input("請輸入想分群的數目 : "))

#########行為轉字元#########
dataset = dataset.replace("起床",'a')
dataset = dataset.replace("刷牙",'b')
dataset = dataset.replace("上廁所",'c')
dataset = dataset.replace("吃飯",'d')
dataset = dataset.replace("休閒活動",'e')
dataset = dataset.replace("睡覺",'f')
#########行為轉字元#########

y = dataset.iloc[:,0]
Z = dataset.iloc[:,1:].values #選擇第一行 ~ 最後一行特徵資料
r = Z.size/len(Z) #宣告 r 儲存 X 的 Column長度
r = int(r) # r 轉為 int

###########分割24小時段###########
X = dataset.iloc[:,1:9].values #選擇第一行 ~ 第三行特徵資料
X2 = dataset.iloc[:,9:17].values #選擇行
X3 = dataset.iloc[:,17:25].values
X4 = dataset.iloc[:,25:33].values
X5 = dataset.iloc[:,33:41].values
X6 = dataset.iloc[:,41:49].values
###########分割24小時段###########
################Counter###############
charlist = ['a','b','c','d','e','f']
Xdata = [X,X2,X3,X4,X5,X6]
Importlist = [[]* r for l in range(6)]#建構 List，維度 = X label長度
for j in range(6):
    for h in range(6): 
        m = 0
        X_nparray = np.array(Xdata[j]) #將原始資料 dataset 轉為 np.array
        X_list_count = X_nparray.tolist() #將 nparray 轉為 list
        for i in range(len(Z)):
            k = ((X_list_count[i].count(charlist[h])))
            m = m+k #計數疊代
            if m > 0:
                Importlist[j].append(charlist[h])
                Importlist[j] = list(set(Importlist[j])) #去除重複
################Counter###############
#X段處理
X_nparray = np.array(X) #將原始資料 dataset 轉為 np.array
X_list = X_nparray.tolist() #將 nparray 轉為 list
X_list
df = pd.DataFrame(index=y,columns = Importlist[0])
for j in range(len(Z)):
    dict_X = {}
    dict_X.update({X_list[j][0]:X_list[j][1]})
    df.iloc[j] = dict_X#將dict_X6值用data填入
    dict_X.update({X_list[j][2]:X_list[j][3]})
    df.iloc[j] = dict_X#將dict_X6值用data填入
    dict_X.update({X_list[j][4]:X_list[j][5]})
    df.iloc[j] = dict_X#將dict_X6值用data填入
    dict_X.update({X_list[j][6]:X_list[j][7]})
    df.iloc[j] = dict_X#將dict_X6值用data填入

df = df.fillna(86400)#將na值用data填入
df = df.rename(columns={'a':'a1'})#修改列名稱
df = df.rename(columns={'b':'b1'})
df = df.rename(columns={'c':'c1'})
df = df.rename(columns={'d':'d1'})
df = df.rename(columns={'e':'e1'})
df = df.rename(columns={'f':'f1'})   
    
#X段處理   

#X2段處理
X_nparray = np.array(X2) #將原始資料 dataset 轉為 np.array
X2_list = X_nparray.tolist() #將 nparray 轉為 list
X2_list
df2 = pd.DataFrame(index=y,columns = Importlist[1])
for j in range(len(Z)):
    dict_X2 = {}
    dict_X2.update({X2_list[j][0]:X2_list[j][1]})
    df2.iloc[j] = dict_X2#將dict_X6值用data填入
    dict_X2.update({X2_list[j][2]:X2_list[j][3]})
    df2.iloc[j] = dict_X2#將dict_X6值用data填入
    dict_X2.update({X2_list[j][4]:X2_list[j][5]})
    df2.iloc[j] = dict_X2#將dict_X6值用data填入
    dict_X2.update({X2_list[j][6]:X2_list[j][7]})
    df2.iloc[j] = dict_X2#將dict_X6值用data填入

df2 = df2.fillna(86400)#將na值用data填入
df2 = df2.rename(columns={'a':'a2'})#修改列名稱
df2 = df2.rename(columns={'b':'b2'})
df2 = df2.rename(columns={'c':'c2'})
df2 = df2.rename(columns={'d':'d2'})
df2 = df2.rename(columns={'e':'e2'})
df2 = df2.rename(columns={'f':'f2'})       
#X2段處理   

#X3段處理
X_nparray = np.array(X3) #將原始資料 dataset 轉為 np.array
X3_list = X_nparray.tolist() #將 nparray 轉為 list
X3_list
df3 = pd.DataFrame(index=y,columns = Importlist[2])
for j in range(len(Z)):
    dict_X3 = {}
    dict_X3.update({X3_list[j][0]:X3_list[j][1]})
    df3.iloc[j] = dict_X3#將dict_X6值用data填入
    dict_X3.update({X3_list[j][2]:X3_list[j][3]})
    df3.iloc[j] = dict_X3#將dict_X6值用data填入
    dict_X3.update({X3_list[j][4]:X3_list[j][5]})
    df3.iloc[j] = dict_X3#將dict_X6值用data填入
    dict_X3.update({X3_list[j][6]:X3_list[j][7]})
    df3.iloc[j] = dict_X3#將dict_X6值用data填入

df3 = df3.fillna(86400)#將na值用data填入
df3 = df3.rename(columns={'a':'a3'})#修改列名稱
df3 = df3.rename(columns={'b':'b3'})
df3 = df3.rename(columns={'c':'c3'})
df3 = df3.rename(columns={'d':'d3'})
df3 = df3.rename(columns={'e':'e3'})
df3 = df3.rename(columns={'f':'f3'})    
#X3段處理  

#X4段處理
X_nparray = np.array(X4) #將原始資料 dataset 轉為 np.array
X4_list = X_nparray.tolist() #將 nparray 轉為 list
X4_list
df4 = pd.DataFrame(index=y,columns = Importlist[3])
for j in range(len(Z)):
    dict_X4 = {}
    dict_X4.update({X4_list[j][0]:X4_list[j][1]})
    df4.iloc[j] = dict_X4#將dict_X6值用data填入
    dict_X4.update({X4_list[j][2]:X4_list[j][3]})
    df4.iloc[j] = dict_X4#將dict_X6值用data填入
    dict_X4.update({X4_list[j][4]:X4_list[j][5]})
    df4.iloc[j] = dict_X4#將dict_X6值用data填入
    dict_X4.update({X4_list[j][6]:X4_list[j][7]})
    df4.iloc[j] = dict_X4#將dict_X6值用data填入

df4 = df4.fillna(86400)#將na值用data填入
df4 = df4.rename(columns={'a':'a4'})#修改列名稱
df4 = df4.rename(columns={'b':'b4'})
df4 = df4.rename(columns={'c':'c4'})
df4 = df4.rename(columns={'d':'d4'})
df4 = df4.rename(columns={'e':'e4'})
df4 = df4.rename(columns={'f':'f4'})      
#X4段處理 
    
#X5段處理
X_nparray = np.array(X5) #將原始資料 dataset 轉為 np.array
X5_list = X_nparray.tolist() #將 nparray 轉為 list
X5_list
df5 = pd.DataFrame(index=y,columns = Importlist[4])
for j in range(len(Z)):
    dict_X5 = {}
    dict_X5.update({X5_list[j][0]:X5_list[j][1]})
    df5.iloc[j] = dict_X5#將dict_X6值用data填入
    dict_X5.update({X5_list[j][2]:X5_list[j][3]})
    df5.iloc[j] = dict_X5#將dict_X6值用data填入
    dict_X5.update({X5_list[j][4]:X5_list[j][5]})
    df5.iloc[j] = dict_X5#將dict_X6值用data填入
    dict_X5.update({X5_list[j][6]:X5_list[j][7]})
    df5.iloc[j] = dict_X5#將dict_X6值用data填入
    
df5 = df5.fillna(86400)#將na值用data填入
df5 = df5.rename(columns={'a':'a5'})#修改列名稱
df5 = df5.rename(columns={'b':'b5'})
df5 = df5.rename(columns={'c':'c5'})
df5 = df5.rename(columns={'d':'d5'})
df5 = df5.rename(columns={'e':'e5'})
df5 = df5.rename(columns={'f':'f5'})  
#X5段處理 
    
#X6段處理
X_nparray = np.array(X6) #將原始資料 dataset 轉為 np.array
X6_list = X_nparray.tolist() #將 nparray 轉為 list
X6_list
df6 = pd.DataFrame(index=y,columns =  Importlist[5])

for j in range(len(Z)): 
    dict_X6 = {}
    dict_X6.update({X6_list[j][0]:X6_list[j][1]})
    df6.iloc[j] = dict_X6#將dict_X6值用data填入
    dict_X6.update({X6_list[j][2]:X6_list[j][3]})
    df6.iloc[j] = dict_X6#將dict_X6值用data填入
    dict_X6.update({X6_list[j][4]:X6_list[j][5]})
    df6.iloc[j] = dict_X6#將dict_X6值用data填入
    dict_X6.update({X6_list[j][6]:X6_list[j][7]})
    df6.iloc[j] = dict_X6#將dict_X6值用data填入

df6 = df6.fillna(86400)#將na值用data填入
df6 = df6.rename(columns={'a':'a6'})#修改列名稱
df6 = df6.rename(columns={'b':'b6'})
df6 = df6.rename(columns={'c':'c6'})
df6 = df6.rename(columns={'d':'d6'})
df6 = df6.rename(columns={'e':'e6'})
df6 = df6.rename(columns={'f':'f6'})  
#X6段處理
 
frames = [df,df2,df3,df4,df5,df6]
df_all = pd.concat(frames,axis=1)#合併

#權重處理

for i in range(len(df_all.columns)):
    if df_all.columns[i] == "d1" or df_all.columns[i] == "d2" or df_all.columns[i] == "d3" or df_all.columns[i] == "d4" or df_all.columns[i] == "d5" or df_all.columns[i] == "d6":
        df_all[df_all.columns[i]] = df_all[df_all.columns[i]] * 2


for i in range(len(df_all.columns)):
    if df_all.columns[i] == "c1" or df_all.columns[i] == "c2" or df_all.columns[i] == "c3" or df_all.columns[i] == "c4" or df_all.columns[i] == "c5" or df_all.columns[i] == "c6":
        df_all[df_all.columns[i]] = df_all[df_all.columns[i]] * 2

#權重處理

print(df_all)


# Hierarchical Clustering 演算法
hclust = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n_clusters = number )

# 印出分群結果
hclust.fit(df_all)
cluster_labels = hclust.labels_
print(cluster_labels)
print("---")

hclust.fit(df_all)
cluster_labels = hclust.labels_
silhouette_avg = metrics.silhouette_score(df_all, cluster_labels)
print(silhouette_avg)
# 印出分群結果  
    
# Hierarchical Clustering 演算法   """    

###################樹狀圖形輸出##################
Z = linkage(df_all, 'single')  # X: 資料在下面的示範中會有
plt.figure()
dn = dendrogram(Z)
plt.show() 
###################樹狀圖形輸出##################
###################群集計算輸出##################
# 計算群集數量
unique = []
for i in cluster_labels:
    if i not in unique:
        unique.append(i)

counters = []
counter = collections.Counter(cluster_labels)
for i in unique:
    temp = []
    temp.append(i)
    temp.append(counter[i])
    counters.append(temp)   

minima = ""
c = 0

for ln in range(len(counters)):
    try:
        if c == 0:
            minima = counters[ln]
            c +=1
            
        if c != 0:
            if counters[ln][1] > minima[1] and c != 0:
                minima = minima
        
            elif counters[ln][1] < minima[1] and c != 0:
                minima = counters[ln]
        
    except IndexError:
        continue
    

print(counters)
print("最小群集", minima)


cluster_labels = list(cluster_labels)

for i in range(len(cluster_labels)): 
    if cluster_labels[i] == minima[0]:
        cluster_labels[i] = "異常"
    else: 
        cluster_labels[i] = "正常"
    
#print(cluster_labels)

dataset_export = dataset_export.set_index("Day")
dataset_export["判斷結果"] = cluster_labels
print(dataset_export)

dataset_export.to_excel("ABDT異常日判斷結果.xls")
###################群集計算輸出##################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
