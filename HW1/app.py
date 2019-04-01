#!/usr/bin/env python
# coding: utf-8

# # 使用lightGBM套件

# In[1]:


from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb


# In[2]:


import csv


#開啟 CSV 檔案
with open('台灣電力公司_過去電力供需資訊2018-201902.csv', newline='') as csvfile:

    #讀取 CSV 檔內容，將每一列轉成一個 dictionary
    rows = csv.DictReader(csvfile)
    datas_2018_201902 = []
    #以迴圈輸出指定欄位
    for row in rows:
        data = []
        data.append(float(row['\ufeff日期']))
        data.append(float(row['尖峰負載(MW)']))
        data.append(float(row['備轉容量(MW)']))
        data.append(float(row['備轉容量率(%)']))
        datas_2018_201902.append(data)


# 讀取資料集，取出所需的attribute，存成list

# In[3]:


new_data_list_x = []

week_x = 0
for i in range(len(datas_2018_201902)-(len(datas_2018_201902)%7)-7):
    week_x += 1
    new_data = []
    if datas_2018_201902[i+6]:
        for j in range(7):
            if week_x < 7:
                new_data.append(float(datas_2018_201902[i+j][1]))
                new_data.append(float(datas_2018_201902[i+j][2]))
                new_data.append(week_x)
            else:
                new_data.append(float(datas_2018_201902[i+j][1]))
                new_data.append(float(datas_2018_201902[i+j][2]))
                new_data.append(week_x)
                week_x = 0
    new_data_list_x.append(new_data)


# 以7天為單位，每次取出負載容量及備載容量，並新增星期幾為新的attribute

# In[4]:


new_data_list_y = []
for i in range(len(datas_2018_201902)):
    new_data = []
    if i >= 7 and len(new_data_list_y)<len(new_data_list_x):
        new_data.append(float(datas_2018_201902[i][1]))
        
        new_data_list_y.append(new_data)


# 取出7天後，第8天的負載容量作為target

# In[5]:


datas_x_array = np.array(new_data_list_x)
datas_y_array = np.array(new_data_list_y)

y_train = datas_y_array.ravel()
X_train = datas_x_array

X_test = [[28535, 1853, 1, 28756,1887,2,29140,1933,3,30093,1892,4,29673,2054,5,25810,2155,6,24466,2298,7]]

lgb_train = lgb.Dataset(X_train, y_train)


# 初始X_test為前一週03/25~03/31 的資料

# In[6]:


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

#print('Starting training...')
#train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=None)

# save model to file
#gbm.save_model('model.txt')


# lightgbm參數設定

# In[7]:


# predict
#X_test = [[28535, 1853, 1, 28756,1887,2,29140,1933,3,30093,1892,4,29673,2054,5,25810,2155,6,24466,2298,7]]
operating_reserve_0401 = float(1870)
operating_reserve_0402_0408 = [1860,1960,2440,2460,2670,2430,2100]
y_pred_0401 = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#0401
for j in X_test:
    j.pop(0)
    j.pop(0)
    j.pop(0)
    j.append(y_pred_0401[0])
    j.append(float(operating_reserve_0401))
    j.append(1)


# 預測出04/01的負載容量，使用"台灣電力公司_未來一週電力供需預測"中的備載容量作為attribute
# 拿到X_test中第一天的資料，新增04/01的資料上去
# 此時X_test資料為03/26~04/01

# In[8]:


y_pred_list = []
for i in range(7):
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    for j in X_test:
        j.pop(0)
        j.pop(0)
        j.pop(0)
        j.append(y_pred)
        j.append(float(operating_reserve_0402_0408[i]))
        if i > 7:
            j.append(i-7)
        else:
            j.append(i)
    y_pred_list.append(y_pred[0])


# 用迴圈依序得到預測的資料，並持續對X_test做修正，直到得出04/02~04/08的負載容量為止

# In[9]:


print_days = [20190402,20190403,20190404,20190405,20190406,20190407,20190408]

# 開啟輸出的 CSV 檔案
with open('submission.csv', 'w', newline='') as csvfile:
  # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

  # 寫入一列資料
    writer.writerow(['date', 'peak_load(MW)'])

  # 寫入另外幾列資料
    for i in range(7):
        writer.writerow([print_days[i], y_pred_list[i]])


# 輸出至submission.csv
