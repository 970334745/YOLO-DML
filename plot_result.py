import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import font_manager  # 导入字体管理模块

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

# Precision plot
# plt.subplot(2, 2, 1)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'metrics/precision(B)')
#     plt.plot(data['metrics/precision(B)'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('precision', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# Recall plot
# plt.subplot(2, 2, 2)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'metrics/recall(B)')
#     plt.plot(data['metrics/recall(B)'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('recall', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# mAP50 plot
# plt.subplot(1, 1, 1)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'metrics/mAP50(B)')
#     plt.plot(data['metrics/mAP50(B)'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=20, family='Times New Roman')
# plt.title('mAP_0.5', fontsize=20, family='Times New Roman')
# plt.legend(fontsize=20, prop={'family': 'Times New Roman'})

# plt.xticks(fontsize=20, family='Times New Roman')
# plt.yticks(fontsize=20, family='Times New Roman')

# mAP50-95 plot
plt.subplot(1, 1, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data = process_column(data, 'metrics/mAP50-95(B)')
    plt.plot(data['metrics/mAP50-95(B)'], label=i, linewidth=2)  # 增加线条粗细
plt.xlabel('epoch', fontsize=20, family='Times New Roman')
plt.title('mAP_0.5:0.95', fontsize=20, family='Times New Roman')
plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

plt.xticks(fontsize=20, family='Times New Roman')
plt.yticks(fontsize=20, family='Times New Roman')

plt.tight_layout()
plt.savefig('metrice_curve.png')
print(f'metrice_curve.png saved in {pwd}/metrice_curve.png')

# Loss plots
# plt.figure(figsize=(15, 10))

# Box loss plot
# plt.subplot(2, 3, 1)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'train/box_loss')
#     plt.plot(data['train/box_loss'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('train/box_loss', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# DFL loss plot
# plt.subplot(2, 3, 2)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'train/dfl_loss')
#     plt.plot(data['train/dfl_loss'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('train/dfl_loss', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# # Class loss plot
# plt.subplot(2, 3, 3)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'train/cls_loss')
#     plt.plot(data['train/cls_loss'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('train/cls_loss', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# Validation box loss plot
# plt.subplot(2, 3, 4)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'val/box_loss')
#     plt.plot(data['val/box_loss'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('val/box_loss', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# Validation DFL loss plot
# plt.subplot(2, 3, 5)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'val/dfl_loss')
#     plt.plot(data['val/dfl_loss'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('val/dfl_loss', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# Validation class loss plot
# plt.subplot(2, 3, 6)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     data = process_column(data, 'val/cls_loss')
#     plt.plot(data['val/cls_loss'], label=i, linewidth=2)  # 增加线条粗细
# plt.xlabel('epoch', fontsize=14, family='Times New Roman')
# plt.title('val/cls_loss', fontsize=14, family='Times New Roman')
# plt.legend(fontsize=14, prop={'family': 'Times New Roman'})

# plt.tight_layout()
# plt.savefig('loss_curve.png')
# print(f'loss_curve.png saved in {pwd}/loss_curve.png')
