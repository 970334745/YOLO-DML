import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter1d  # 高斯滤波
from matplotlib import font_manager

pwd = os.getcwd()

names = ['ours', 'yolov8n', 'yolov10n', 'yolov11n', 'yolov12n']

plt.figure(figsize=(8, 8))

def process_column(data, column_name):
    # 去除列名中的空格
    data.columns = data.columns.str.strip()
    # 如果列名存在则处理
    if column_name in data.columns:
        data[column_name] = data[column_name].astype(np.float32).replace(np.inf, np.nan)
        data[column_name] = data[column_name].fillna(data[column_name].interpolate())
    else:
        print(f"Column '{column_name}' not found!")
    return data

# mAP50 plot
plt.subplot(1, 1, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data = process_column(data, 'metrics/mAP50(B)')
    # 应用高斯滤波对数据进行平滑
    smoothed_data = gaussian_filter1d(data['metrics/mAP50(B)'], sigma=2)
    plt.plot(smoothed_data, label=i, linewidth=2)  # 增加线条粗细

plt.xlabel('epoch', fontsize=20, family='Times New Roman')
plt.title('mAP_0.5', fontsize=20, family='Times New Roman')
plt.legend(fontsize=20, prop={'family': 'Times New Roman'})

plt.xticks(fontsize=20, family='Times New Roman')
plt.yticks(fontsize=20, family='Times New Roman')

plt.tight_layout()
plt.savefig('metrice_curve_smoothed.png')
print(f'metrice_curve_smoothed.png saved in {pwd}/metrice_curve_smoothed.png')
