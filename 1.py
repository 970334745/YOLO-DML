# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # 生成随机特征数据（集中化处理）
# def generate_concentrated_features(num_samples=100, num_features=64, center_scale=0.5):
#     """生成集中于中心的特征数据"""
#     return center_scale * np.random.randn(num_samples, num_features)

# # 将特征投影到二维并绘制
# def plot_feature_gate_with_short_outward_arrows(features_before, features_after, title, save_path=None):
#     """绘制门控前后特征的二维分布，仅为红点添加短箭头"""
#     # PCA 降维到 2 维
#     pca = PCA(n_components=2)
#     features_before_pca = pca.fit_transform(features_before)
#     features_after_pca = pca.transform(features_after)

#     # 绘图
#     plt.figure(figsize=(10, 8))
#     plt.scatter(features_before_pca[:, 0], features_before_pca[:, 1], c='blue', alpha=0.6, label='Before Gate')
#     plt.scatter(features_after_pca[:, 0], features_after_pca[:, 1], c='red', alpha=0.6, label='After Gate')

#     # 添加红点的短箭头
#     for i in range(len(features_after_pca)):
#         dx = 0.1 * (features_after_pca[i, 0] - features_before_pca[i, 0])  # 缩短箭头的长度
#         dy = 0.1 * (features_after_pca[i, 1] - features_before_pca[i, 1])
#         plt.arrow(features_after_pca[i, 0], features_after_pca[i, 1], 
#                   dx, dy, color='black', alpha=0.8, head_width=0.03, head_length=0.05, linewidth=0.5)

#     plt.title(title, fontsize=8, family='Times New Roman')
#     plt.xlabel('PCA Component 1', fontsize=25, family='Times New Roman')
#     plt.ylabel('PCA Component 2', fontsize=25, family='Times New Roman')
    
#     # 设置全局字体样式
#     plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14})
    
#     # 绘制图例，使用默认字体设置
#     plt.legend(fontsize=14)

#     # 设置坐标轴的字体
#     plt.tick_params(axis='both', which='major', labelsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)

#     if save_path:
#         plt.savefig(save_path)
#     plt.show()

# # 主程序
# if __name__ == "__main__":
#     # 蓝点（门控前特征）集中生成
#     num_samples = 100
#     num_features = 64
#     features_before_gate = generate_concentrated_features(num_samples=num_samples, num_features=num_features)

#     # 红点（门控后特征）分两部分：集中部分+发散部分
#     concentrated_part = features_before_gate[:num_samples // 2]  # 保持集中
#     dispersed_part = features_before_gate[num_samples // 2:] + 2 * np.random.randn(num_samples // 2, num_features)  # 发散
#     features_after_gate = np.vstack((concentrated_part, dispersed_part))

#     # 绘制并保存图片
#     plot_feature_gate_with_short_outward_arrows(features_before_gate, features_after_gate, 
#                                                 "", 
#                                                 save_path="feature_gate_short_outward_arrows.png")
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# import scipy.ndimage
# import cv2

# # 读取输入图片
# input_image_path = "/home/wangyi/dataset/VisDrone2019/VisDrone2019-DET-val/images/0000001_05499_d_0000010.jpg"  # 修改为你的图像路径
# input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# # Sobel 特征提取，作为输入特征图
# def extract_sobel_features(image):
#     sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     sobel_features = np.sqrt(sobel_x**2 + sobel_y**2)
#     return sobel_features

# features = extract_sobel_features(input_image)

# # PCA 降维特征图
# def pca_reduced_feature_map(features, save_path="pca_reduced_surface.png"):
#     # 将特征进行扁平化，以便进行 PCA
#     flattened_features = features.reshape(-1, 1)
#     pca = PCA(n_components=1)
#     pca_features = pca.fit_transform(flattened_features)

#     # 将 PCA 结果还原为与原始图像相同的形状
#     pca_features_reshaped = pca_features.reshape(features.shape)

#     # 创建 3D 绘图的数据
#     X, Y = np.meshgrid(range(features.shape[1]), range(features.shape[0]))
#     Z = pca_features_reshaped

#     # 绘制 3D 表面图
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
#     ax.set_title("PCA Reduced Feature Map (3D Surface)")
#     ax.set_xlabel("Width")
#     ax.set_ylabel("Height")
#     ax.set_zlabel("Activation")
#     plt.savefig(save_path)
#     plt.close()

# # 插值特征图
# def visualize_interpolated_surface(features, save_path="interpolated_surface.png"):
#     smoothed_features = scipy.ndimage.zoom(features, (1.5, 1.5), order=1)
#     X, Y = np.meshgrid(range(smoothed_features.shape[1]), range(smoothed_features.shape[0]))
#     Z = smoothed_features

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
#     ax.set_title("Interpolated Feature Map (3D Surface)")
#     ax.set_xlabel("Width")
#     ax.set_ylabel("Height")
#     ax.set_zlabel("Activation")
#     plt.savefig(save_path)
#     plt.close()

# # 特征激活密度图
# def feature_activation_density(features, save_path="feature_activation_density.png"):
#     flattened_features = features.flatten()

#     plt.figure()
#     sns.kdeplot(flattened_features, fill=True, color="b", alpha=0.5)
#     plt.title("Feature Activation Density")
#     plt.xlabel("Feature Activation")
#     plt.ylabel("Density")
#     plt.savefig(save_path)
#     plt.close()

# # 特征热图
# def feature_map_heatmap(features, save_path="feature_map_heatmap.png"):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(features, cmap="viridis")
#     plt.title("Feature Map Heatmap")
#     plt.xlabel("Width")
#     plt.ylabel("Height")
#     plt.savefig(save_path)
#     plt.close()

# # 调用各函数绘制并保存图片
# pca_reduced_feature_map(features, save_path="pca_reduced_surface.png")
# visualize_interpolated_surface(features, save_path="interpolated_surface.png")
# feature_activation_density(features, save_path="feature_activation_density.png")
# feature_map_heatmap(features, save_path="feature_map_heatmap.png")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.ndimage
import cv2
from mpl_toolkits.mplot3d import Axes3D

# 读取输入图片
input_image_path = "E:/yoloxilie/v8/new_yolov8/0000294_00000_d_0000053.jpg"  # 修改为你的图像路径
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Sobel 特征提取，作为输入特征图
def extract_sobel_features(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_features = np.sqrt(sobel_x**2 + sobel_y**2)
    return sobel_features

features = extract_sobel_features(input_image)

# PCA 降维特征图
def pca_reduced_feature_map(features, save_path="pca_reduced_surface.png"):
    # 将特征进行扁平化，以便进行 PCA
    flattened_features = features.reshape(-1, 1)
    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(flattened_features)

    # 将 PCA 结果还原为与原始图像相同的形状
    pca_features_reshaped = pca_features.reshape(features.shape)

    # 创建 3D 绘图的数据
    X, Y = np.meshgrid(range(features.shape[1]), range(features.shape[0]))
    Z = pca_features_reshaped

    # 坐标值除以100
    X = X / 100
    Y = Y / 100
    Z = Z / 100

    # 绘制 3D 表面图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')

    # 关闭网格线
    ax.grid(False)

    # 设置坐标轴标签
    ax.set_title("", fontsize=14, fontname="Times New Roman")
    ax.set_xlabel("Height", fontsize=14, fontname="Times New Roman")  # Y 轴放左边
    ax.set_ylabel("Width", fontsize=14, fontname="Times New Roman")   # X 轴放底部
    ax.set_zlabel("Activation", fontsize=14, fontname="Times New Roman")

    # 调整视角，交换 X 和 Y
    ax.view_init(elev=30, azim=-60)

    # 添加箭头
    # 为 X, Y, Z 坐标轴添加箭头
    ax.quiver(0, 0, 0, 0, Y.max(), 0, color='black', arrow_length_ratio=0.1)  # Y 轴（竖直）
    ax.quiver(0, 0, 0, X.max(), 0, 0, color='black', arrow_length_ratio=0.1)  # X 轴（水平）
    ax.quiver(0, 0, 0, 0, 0, Z.max(), color='black', arrow_length_ratio=0.1)  # Z 轴（垂直）

    # 保存图像
    plt.savefig(save_path)
    plt.close()

# 插值特征图
def visualize_interpolated_surface(features, save_path="interpolated_surface.png"):
    smoothed_features = scipy.ndimage.zoom(features, (1.5, 1.5), order=1)
    X, Y = np.meshgrid(range(smoothed_features.shape[1]), range(smoothed_features.shape[0]))
    Z = smoothed_features

    # 坐标值除以100
    X = X / 100
    Y = Y / 100
    Z = Z / 100

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')

    # 关闭网格线
    ax.grid(False)

    # 设置坐标轴标签
    ax.set_title("", fontsize=14, fontname="Times New Roman")
    ax.set_xlabel("Height", fontsize=14, fontname="Times New Roman")  # Y 轴放左边
    ax.set_ylabel("Width", fontsize=14, fontname="Times New Roman")   # X 轴放底部
    ax.set_zlabel("Activation", fontsize=14, fontname="Times New Roman")

    # 调整视角，交换 X 和 Y
    ax.view_init(elev=30, azim=-60)

    # 添加箭头
    # 为 X, Y, Z 坐标轴添加箭头
    ax.quiver(0, 0, 0, 0, Y.max(), 0, color='black', arrow_length_ratio=0.1)  # Y 轴（竖直）
    ax.quiver(0, 0, 0, X.max(), 0, 0, color='black', arrow_length_ratio=0.1)  # X 轴（水平）
    ax.quiver(0, 0, 0, 0, 0, Z.max(), color='black', arrow_length_ratio=0.1)  # Z 轴（垂直）

    # 保存图像
    plt.savefig(save_path)
    plt.close()

# 调用各函数绘制并保存图片
pca_reduced_feature_map(features, save_path="pca_reduced_surface.png")
visualize_interpolated_surface(features, save_path="interpolated_surface.png")


