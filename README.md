本项目使用的ultralytics的版本为8.3.9

使用环境为：    python: 3.10.14
               torch: 2.2.2+cu121
               torchvision: 0.17.2+cu121
               timm: 1.0.7
               mmcv: 2.2.0
               mmengine: 0.10.4

需编译部分 sh YOLO-DML\ultralytics\nn\extra_modules\cutlass\examples\19_large_depthwise_conv2d_torch_extension\make.sh

额外需要安装包的命令
  pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.5.4 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python prettytable -i https://pypi.tuna.tsinghua.edu.cn/simple
      pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
      mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
      mim install "mmcv>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple