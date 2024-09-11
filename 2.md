# YOLOV8V10改进方案(持续更新)

<a id="b"></a>

#### 目前支持的一些block (部分block可能会与主结构有冲突,具体以是否能运行为主)

##### C2f系列
C2f, C2f_Faster, C2f_ODConv, C2f_Faster_EMA, C2f_DBB, C2f_CloAtt, C2f_SCConv, C2f_ScConv, C2f_EMSC, C2f_EMSCP, C2f_KW, C2f_DCNv2, C2f_DCNv3, C2f_OREPA, C2f_REPVGGOREPA, C2f_DCNv2_Dynamic, C2f_MSBlock, C2f_ContextGuided, C2f_DLKA, C2f_EMBC, C2f_Parc, C2f_DWR, C2f_RFAConv, C2f_RFCBAMConv, C2f_RFCAConv, C2f_MLCA, C2f_AKConv, C2f_UniRepLKNetBlock, C2f_DRB, C2f_DWR_DRB, C2f_AggregatedAtt....(C2f系列的改进都合适、太多了就不一一标注了)
##### C3系列  
C3, C3Ghost, C3_CloAtt, C3_SCConv, C3_ScConv, C3_EMSC, C3_EMSCP, C3_KW, C3_ODConv, C3_Faster, C3_Faster_EMA, C3_DCNv2, C3_DCNv3, C3_DBB, C3_OREPA, C3_REPVGGOREPA, C3_DCNv2_Dynamic, C3_MSBlock, C3_ContextGuided, C3_DLKA, C3_EMBC, C3_Parc, C3_DWR, C3_RFAConv, C3_RFCBAMConv, C3_RFCAConv, C3_MLCA, C3_AKConv, C3_UniRepLKNetBlock, C3_DRB, C3_DWR_DRB, C3_AggregatedAtt....(C3系列的改进都合适、太多了就不一一标注了)
##### 其他系列
VoVGSCSP, VoVGSCSPC, RCSOSA, CSP_EDLAN

<a id="c"></a>

#### 目前整合的一些注意力机制 还需要别的注意力机制可从[github](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)拉取对应的代码到ultralytics/nn/extra_modules/attention.py即可. 视频教程可看项目视频中的(如何在yaml配置文件中添加注意力层)
EMA, SimAM, SpatialGroupEnhance, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw, TripletAttention, CoordAtt, CBAM, BAMBlock, EfficientAttention(CloFormer中的注意力), LSKBlock, SEAttention, CPCA, deformable_LKA, EffectiveSEModule, LSKA, SegNext_Attention, DAttention(Vision Transformer with Deformable Attention CVPR2022), FocusedLinearAttention(ICCV2023), MLCA, TransNeXt_AggregatedAttention, LocalWindowAttention, ELA, CAA, CAFM, AFGCAttention(Neural Networks ECCV2024)

### YOLOV8
1. ultralytics/cfg/models/v8/yolov8-efficientViT.yaml

    (CVPR2023)efficientViT替换yolov8主干.

2. ultralytics/cfg/models/v8/yolov8-fasternet.yaml

    (CVPR2023)fasternet替换yolov8主干.

3. ultralytics/cfg/models/v8/yolov8-timm.yaml

    使用timm支持的主干网络替换yolov8主干.timm的内容可看[这期视频](https://www.bilibili.com/video/BV1Mx4y1A7jy/)

4. ultralytics/cfg/models/v8/yolov8-convnextv2.yaml

    使用convnextv2网络替换yolov8主干.

5. ultralytics/cfg/models/v8/yolov8-dyhead.yaml

    添加基于注意力机制的目标检测头到yolov8中.

6. ultralytics/cfg/models/v8/yolov8-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        其中支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.
    项目视频百度云链接:关于BIFPN的说明

7. ultralytics/cfg/models/v8/yolov8-C2f-Faster.yaml

    使用C2f-Faster替换C2f.(使用FasterNet中的FasterBlock替换C2f中的Bottleneck)
    项目视频百度云链接:20240729版本更新说明

8. ultralytics/cfg/models/v8/yolov8-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)
    项目视频百度云链接:20240729版本更新说明

9. ultralytics/cfg/models/v8/yolov8-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolov8主干.(需要看[常见错误和解决方案的第五点](#a))  
10. ultralytics/cfg/models/v8/yolov8-C2f-Faster-EMA.yaml

    使用C2f-Faster-EMA替换C2f.(C2f-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C2f-Faster)
    项目视频百度云链接:20240729版本更新说明
    
11. ultralytics/cfg/models/v8/yolov8-C2f-DBB.yaml

    使用C2f-DBB替换C2f.(使用DiverseBranchBlock替换C2f中的Bottleneck中的Conv)
    项目视频百度云链接:20240729版本更新说明

12. 增加Adaptive Training Sample Selection匹配策略.

    在ultralytics/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.  
    此ATSS匹配策略目前占用显存比较大,因此使用的时候需要设置更小的batch,后续会进行优化这一功能.

13. ultralytics/cfg/models/v8/yolov8-slimneck.yaml

    使用[VoVGSCSP\VoVGSCSPC和GSConv](https://github.com/AlanLi1997/slim-neck-by-gsconv)替换yolov8 neck中的C2f和Conv.

14. ultralytics/cfg/models/v8/yolov8-attention.yaml

    可以看项目视频-如何在yaml配置文件中添加注意力层  
    多种注意力机制在yolov8中的使用. [多种注意力机制github地址](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)  
    目前内部整合的注意力可看[链接](#c)

15. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/cfg/models/v8/yolov8-AFPN-P345.yaml  
    b. ultralytics/cfg/models/v8/yolov8-AFPN-P345-Custom.yaml  
    c. ultralytics/cfg/models/v8/yolov8-AFPN-P2345.yaml  
    d. ultralytics/cfg/models/v8/yolov8-AFPN-P2345-Custom.yaml  
    其中Custom中的block支持这些[结构](#b) [B站介绍说明](https://www.bilibili.com/video/BV1bh411A7yj/)

16. ultralytics/cfg/models/v8/yolov8-vanillanet.yaml

    vanillanet替换yolov8主干.

17. ultralytics/cfg/models/v8/yolov8-C2f-CloAtt.yaml

    使用C2f-CloAtt替换C2f.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C2f中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))  
    项目视频百度云链接:20240729版本更新说明

18. ultralytics/cfg/models/v8/yolov8-RevCol.yaml

    使用[(ICLR2023)Reversible Column Networks](https://github.com/megvii-research/RevCol)对yolov8主干进行重设计.
    视频:https://www.bilibili.com/video/BV1Mh4y1y76u/

19. ultralytics/cfg/models/v8/yolov8-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov8主干.

20. ultralytics/cfg/models/v8/yolov8-C2f-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C2f融合.
    项目视频百度云链接:SCConv和ScConv的使用教程

21. ultralytics/cfg/models/v8/yolov8-C2f-SCcConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C2f融合.  
    (取名为SCcConv的原因是在windows下命名是不区分大小写的)
    项目视频百度云链接:SCConv和ScConv的使用教程

22. MPDiou.[论文链接](https://arxiv.org/pdf/2307.07662v1.pdf)

    请看LOSS改进系列.md

23. ultralytics/cfg/models/v8/yolov8-LAWDS.yaml

    Light Adaptive-weight downsampling.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:LAWDS讲解

24. ultralytics/cfg/models/v8/yolov8-C2f-EMSC.yaml

    Efficient Multi-Scale Conv.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:EMSC,EMSCP讲解

25. ultralytics/cfg/models/v8/yolov8-C2f-EMSCP.yaml

    Efficient Multi-Scale Conv Plus.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:EMSC,EMSCP讲解

26. ultralytics/cfg/models/v8/yolov8-RCSOSA.yaml

    使用[RCS-YOLO](https://github.com/mkang315/RCS-YOLO/tree/main)中的RCSOSA替换C2f.

27. ultralytics/cfg/models/v8/yolov8-KernelWarehouse.yaml
    
    使用[Towards Parameter-Efficient Dynamic Convolution](https://github.com/OSVAI/KernelWarehouse)添加到yolov8中.  
    使用此模块需要注意,在epoch0-20的时候精度会非常低,过了20epoch会正常.
    项目视频百度云链接:20240803版本更新说明

28. Normalized Gaussian Wasserstein Distance.[论文链接](https://arxiv.org/abs/2110.13389)

    在Loss中使用:
        在ultralytics/utils/loss.py中的BboxLoss class中的__init__函数里面设置self.nwd_loss为True.  
        比例系数调整self.iou_ratio, self.iou_ratio代表iou的占比,(1-self.iou_ratio)为代表nwd的占比.  
    在TAL标签分配中使用:
        在ultralytics/utils/tal.py中的def iou_calculation函数中进行更换即可.
    以上这两可以配合使用,也可以单独使用.

29. SlideLoss and EMASlideLoss.[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)

    在ultralytics/utils/loss.py中的class v8DetectionLoss进行设定.

30. ultralytics/cfg/models/v8/yolov8-C2f-DySnakeConv.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C2f融合.
    项目视频百度云链接:20240803版本更新说明

31. ultralytics/cfg/models/v8/yolov8-EfficientHead.yaml

    对检测头进行重设计,支持10种轻量化检测头.详细请看ultralytics/nn/extra_modules/head.py中的Detect_Efficient class.[YOLOV8改进-带你分析V8的检测头并重设计10种结构轻量化检测头](https://www.bilibili.com/video/BV1cu411K7FE/)

32. ultralytics/cfg/models/v8/yolov8-aux.yaml

    参考YOLOV7-Aux对YOLOV8添加额外辅助训练头,在训练阶段参与训练,在最终推理阶段去掉.  
    其中辅助训练头的损失权重系数可在ultralytics/utils/loss.py中的class v8DetectionLoss中的__init__函数中的self.aux_loss_ratio设定,默认值参考yolov7为0.25.

33. ultralytics/cfg/models/v8/yolov8-C2f-DCNV2.yaml

    使用C2f-DCNV2替换C2f.(DCNV2为可变形卷积V2)
    项目视频百度云链接:DCNV2,DCNV3,DyHeadWithDCNV3相关讲解

34. ultralytics/cfg/models/v8/yolov8-C2f-DCNV3.yaml

    使用C2f-DCNV3替换C2f.([DCNV3](https://github.com/OpenGVLab/InternImage)为可变形卷积V3(CVPR2023,众多排行榜的SOTA))  
    官方中包含了一些指定版本的DCNV3 whl包,下载后直接pip install xxx即可.具体和安装DCNV3可看百度云链接中的视频.
    项目视频百度云链接:DCNV2,DCNV3,DyHeadWithDCNV3相关讲解

35. ultralytics/cfg/models/v8/yolov8-dyhead-DCNV3.yaml

    使用[DCNV3](https://github.com/OpenGVLab/InternImage)替换DyHead中的DCNV2.
    项目视频百度云链接:DCNV2,DCNV3,DyHeadWithDCNV3相关讲解

36. ultralytics/cfg/models/v8/yolov8-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.

37. ultralytics/cfg/models/v8/yolov8-C2f-OREPA.yaml

    使用C2f-OREPA替换C2f.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)
    项目视频百度云链接:关于OREPA,REPVGGOREPA,EMSC-OREPA,EMSCP-OREPA的说明

38. ultralytics/cfg/models/v8/yolov8-C2f-REPVGGOREPA.yaml

    使用C2f-REPVGGOREPA替换C2f.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)
    项目视频百度云链接:关于OREPA,REPVGGOREPA,EMSC-OREPA,EMSCP-OREPA的说明

39. ultralytics/cfg/models/v8/yolov8-swintransformer.yaml

    SwinTransformer-Tiny替换yolov8主干.

40. ultralytics/cfg/models/v8/yolov8-repvit.yaml

    [CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)替换yolov8主干.

41. ultralytics/cfg/models/v8/yolov8-fasternet-bifpn.yaml

    fasternet与bifpn的结合.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

42. ultralytics/cfg/models/v8/yolov8-C2f-DCNV2-Dynamic.yaml

    利用自研注意力机制MPCA强化DCNV2中的offset和mask.
    项目视频百度云链接:MPCA与DCNV2_Dynamic的说明

43. ultralytics/cfg/models/v8/yolov8-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块
    介绍可看这期视频：https://www.bilibili.com/video/BV1Bp4y1w7MB/

44. ultralytics/cfg/models/v8/yolov8-C2f-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided改进C2f.

45. ultralytics/cfg/models/v8/yolov8-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample进行下采样.

46. ultralytics/cfg/models/v8/yolov8-C2f-MSBlock.yaml

    使用[YOLO-MS](https://github.com/FishAndWasabi/YOLO-MS/tree/main)中的MSBlock改进C2f.
    项目视频百度云链接:20231010版本更新说明

47. ultralytics/cfg/models/v8/yolov8-C2f-DLKA.yaml

    使用[deformableLKA](https://github.com/xmindflow/deformableLKA)改进C2f.
    项目视频百度云链接:20231010版本更新说明

48. ultralytics/cfg/models/v8/yolov8-GFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN改进Neck.
    项目视频百度云链接:20231010版本更新说明

49. ultralytics/cfg/models/v8/yolov8-SPDConv.yaml

    使用[SPDConv](https://github.com/LabSAINT/SPD-Conv/tree/main)进行下采样.
    项目视频百度云链接:20231010版本更新说明

50. ultralytics/cfg/models/v8/yolov8-EfficientRepBiPAN.yaml

    使用[YOLOV6](https://github.com/meituan/YOLOv6/tree/main)中的EfficientRepBiPAN改进Neck.
    项目视频百度云链接:20231010版本更新说明

51. ultralytics/cfg/models/v8/yolov8-C2f-EMBC.yaml

    使用[Efficientnet](https://blog.csdn.net/weixin_43334693/article/details/131114618?spm=1001.2014.3001.5501)中的MBConv与EffectiveSE改进C2f.

52. ultralytics/cfg/models/v8/yolov8-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.

53. ultralytics/cfg/models/v8/yolov8-C2f-DAttention.yaml

    使用[Vision Transformer with Deformable Attention(CVPR2022)](https://github.com/LeapLabTHU/DAT)改进C2f.(需要看[常见错误和解决方案的第五点](#a))  
    使用注意点请看百度云视频.(DAttention(Vision Transformer with Deformable Attention CVPR2022)使用注意说明.)

54. ultralytics/cfg/models/v8/yolov8-CSwinTransformer.yaml

    使用[CSWin-Transformer(CVPR2022)](https://github.com/microsoft/CSWin-Transformer/tree/main)替换yolov8主干.(需要看[常见错误和解决方案的第五点](#a))

55. ultralytics/cfg/models/v8/yolov8-AIFI.yaml

    使用[RT-DETR](https://arxiv.org/pdf/2304.08069.pdf)中的Attention-based Intrascale Feature Interaction(AIFI)改进yolov8.
    项目视频百度云链接:20231107版本更新说明

56. ultralytics/cfg/models/v8/yolov8-C2f-Parc.yaml

    使用[ParC-Net](https://github.com/hkzhang-git/ParC-Net/tree/main)中的ParC_Operator改进C2f.(需要看[常见错误和解决方案的第五点](#a))  
    使用注意点请看百度云视频.(20231031更新说明)    

57. ultralytics/cfg/models/v8/yolov8-C2f-DWR.yaml

    使用[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块,加强从网络高层的可扩展感受野中提取特征.
    项目视频百度云链接:20231107版本更新说明

58. ultralytics/cfg/models/v8/yolov8-C2f-RFAConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFAConv改进yolov8.

59. ultralytics/cfg/models/v8/yolov8-C2f-RFCBAMConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFCBAMConv改进yolov8.

60. ultralytics/cfg/models/v8/yolov8-C2f-RFCAConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFCAConv改进yolov8.

61. ultralytics/cfg/models/v8/yolov8-HGNetV2.yaml

    使用HGNetV2作为YOLOV8的backbone.

62. ultralytics/cfg/models/v8/yolov8-GhostHGNetV2.yaml

    使用Ghost_HGNetV2作为YOLOV8的backbone.

63. ultralytics/cfg/models/v8/yolov8-RepHGNetV2.yaml

    使用Rep_HGNetV2作为YOLOV8的backbone.

64. ultralytics/cfg/models/v8/yolov8-seg-EfficientHead.yaml(实例分割)

    对检测头进行重设计,支持10种轻量化检测头.详细请看ultralytics/nn/extra_modules/head.py中的Detect_Efficient class.[YOLOV8改进-带你分析V8的检测头并重设计10种结构轻量化检测头](https://www.bilibili.com/video/BV1cu411K7FE/)  

65. ultralytics/cfg/models/v8/yolov8-C2f-FocusedLinearAttention.yaml

    使用[FLatten Transformer(ICCV2023)](https://github.com/LeapLabTHU/FLatten-Transformer)中的FocusedLinearAttention改进C2f.(需要看[常见错误和解决方案的第五点](#a))    
    使用注意点请看百度云视频.(20231114版本更新说明.)

66. IoU,GIoU,DIoU,CIoU,EIoU,SIoU更换方法.

    请看LOSS改进系列.md

67. Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU更换方法.

    请看LOSS改进系列.md

68. Inner-MPDIoU更换方法.

    请看LOSS改进系列.md

69. ultralytics/cfg/models/v8/yolov8-C2f-MLCA.yaml

    使用[Mixed Local Channel Attention 2023](https://github.com/wandahangFY/MLCA/tree/master)改进C2f.(用法请看百度云视频-20231129版本更新说明)

70. ultralytics/cfg/models/v8/yolov8-C2f-AKConv.yaml

    使用[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进C2f.(用法请看百度云视频-20231129版本更新说明)

71. ultralytics/cfg/models/v8/yolov8-unireplknet.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)替换yolov8主干.
    项目视频百度云链接:20231207版本更新说明

72. ultralytics/cfg/models/v8/yolov8-C2f-UniRepLKNetBlock.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的UniRepLKNetBlock改进C2f.
    项目视频百度云链接:20231207版本更新说明

73. ultralytics/cfg/models/v8/yolov8-C2f-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进C2f.
    项目视频百度云链接:20231207版本更新说明

74. ultralytics/cfg/models/v8/yolov8-C2f-DWR-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)的模块进行二次创新后改进C2f.
    项目视频百度云链接:20231207版本更新说明

75. ultralytics/cfg/models/v8/yolov8-ASF.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov8.
    项目视频百度云链接:20231217版本更新说明

76. ultralytics/cfg/models/v8/yolov8-ASF-P2.yaml

    在ultralytics/cfg/models/v8/yolov8-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.
    项目视频百度云链接:20231217版本更新说明

77. ultralytics/cfg/models/v8/yolov8-CSP-EDLAN.yaml

    使用[DualConv](https://github.com/ChipsGuardian/DualConv)打造CSP Efficient Dual Layer Aggregation Networks改进yolov8.
    项目视频百度云链接:20231217版本更新说明

78. ultralytics/cfg/models/v8/yolov8-TransNeXt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)改进yolov8的backbone.(需要看[常见错误和解决方案的第五点](#a))   

79. ultralytics/cfg/models/v8/yolov8-AggregatedAttention.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进yolov8的backbone.(需要看[常见错误和解决方案的第五点](#a))   
    项目视频百度云链接:20231227版本更新说明

80. ultralytics/cfg/models/v8/yolov8-C2f-AggregatedAtt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进C2f.(需要看[常见错误和解决方案的第五点](#a))   
    项目视频百度云链接:20231227版本更新说明

81. ultralytics/cfg/models/v8/yolov8-bifpn-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对BIFPN进行二次创新.
    项目视频百度云链接:20231227版本更新说明

82. ultralytics/cfg/models/v8/yolov8-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对yolov8中的feature fusion部分进行重设计.
    项目视频百度云链接:20231227版本更新说明

83. Shape-IoU,Inner-Shape-IoU更换方法.

    请看LOSS改进系列.md

84. FocalLoss,VarifocalLoss,QualityfocalLoss更换方法.

    请看LOSS改进系列.md

85. Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)更换方法.

    请看LOSS改进系列.md

86. Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)更换方法.

    请看LOSS改进系列.md

87. ultralytics/cfg/models/v8/yolov8-goldyolo-asf.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute与[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新改进yolov8的neck.
    项目视频百度云链接:20240116版本更新说明

88. ultralytics/cfg/models/v8/yolov8-C2f-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进C2f.(请关闭AMP进行训练,使用教程请看20240116版本更新说明)

89. ultralytics/cfg/models/v8/yolov8-dyhead-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)对DyHead进行二次创新.(请关闭AMP进行训练,使用教程请看20240116版本更新说明)

90. ultralytics/cfg/models/v8/yolov8-HSFPN.yaml

    使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进yolov8的neck.
    项目视频百度云链接:20240122版本更新说明

91. ultralytics/cfg/models/v8/yolov8-HSPAN.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进yolov8的neck.
    项目视频百度云链接:20240122版本更新说明

92. soft-nms(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,ShapeIoU)

    soft-nms替换nms.(建议:仅在val.py时候使用,具体替换请看20240122版本更新说明)

93. ultralytics/cfg/models/v8/yolov8-dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov8-neck中的上采样.
    项目视频百度云链接:20240122版本更新说明

94. ultralytics/cfg/models/v8/yolov8-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov8-neck中的上采样.
    项目视频百度云链接:20240122版本更新说明

95. ultralytics/cfg/models/v8/yolov8-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8的下采样.(请关闭AMP情况下使用)
    项目视频百度云链接:20240122版本更新说明

96. Focaler-IoU系列(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,WIoU,MPDIoU,ShapeIoU)

    请看LOSS改进系列.md

97. ultralytics/cfg/models/v8/yolov8-GDFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)进行二次创新改进Neck.
    项目视频百度云链接:20240203版本更新说明

98. ultralytics/cfg/models/v8/yolov8-HSPAN-DySample.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN再进行创新,使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进其上采样模块.
    项目视频百度云链接:20240203版本更新说明

99. ultralytics/cfg/models/v8/yolov8-ASF-DySample.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)组合得到Dynamic Sample Attentional Scale Sequence Fusion.
    项目视频百度云链接:20240203版本更新说明

100. ultralytics/cfg/models/v8/yolov8-SEAMHead.yaml

    使用[YOLO-Face V2](https://arxiv.org/pdf/2208.02019v2.pdf)中的遮挡感知注意力改进Head,使其有效地处理遮挡场景.
    项目视频百度云链接:20240203版本更新说明

101. ultralytics/cfg/models/v8/yolov8-MultiSEAMHead.yaml

    使用[YOLO-Face V2](https://arxiv.org/pdf/2208.02019v2.pdf)中的遮挡感知注意力改进Head,使其有效地处理遮挡场景.
    项目视频百度云链接:20240203版本更新说明

102. ultralytics/cfg/models/v8/yolov8-C2f-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进yolov8中的C2f.(详细请看20240216更新说明)
    项目视频百度云链接:20240216版本更新说明

103. ultralytics/cfg/models/v8/yolov8-C2f-iRMB.yaml

    使用[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB改进C2f.(详细请看20240216更新说明)

104. ultralytics/cfg/models/v8/yolov8-C2f-iRMB-Cascaded.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.(详细请看20240216更新说明)

105. ultralytics/cfg/models/v8/yolov8-C2f-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.(详细请看20240216更新说明)

106. ultralytics/cfg/models/v8/yolov8-C2f-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C2f.(详细请看20240216更新说明)

107. ultralytics/cfg/models/v8/yolov8-C2f-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)对C2f中的BottleNeck进行改进,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.
    项目视频百度云链接:20240219版本更新说明

108. ultralytics/cfg/models/v8/yolov8-C2f-LVMB.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)与Cross Stage Partial进行结合,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.
    项目视频百度云链接:20240219版本更新说明

109. Powerful-IoU系列.

    请看LOSS改进系列.md

110. ultralytics/cfg/models/v8/yolov8-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行改进yolov8.
    项目视频百度云链接:20240222版本更新说明

111. ultralytics/cfg/models/v8/yolov8-DBBNCSPELAN.yaml

    使用[Diverse Branch Block CVPR2021](https://arxiv.org/abs/2103.13425)对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolov8.
    项目视频百度云链接:20240222版本更新说明

112. ultralytics/cfg/models/v8/yolov8-OREPANCSPELAN.yaml

    使用[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolov8.
    项目视频百度云链接:20240222版本更新说明

113. ultralytics/cfg/models/v8/yolov8-DRBNCSPELAN.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolov8.
    项目视频百度云链接:20240222版本更新说明

114. ultralytics/cfg/models/v8/yolov8-v7DS.yaml

    使用[YOLOV7 CVPR2023](https://arxiv.org/abs/2207.02696)的下采样结构改进YOLOV8中的下采样.
    项目视频百度云链接:20240229版本更新说明

115. ultralytics/cfg/models/v8/yolov8-ADown.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的下采样结构改进YOLOV8中的下采样.
    项目视频百度云链接:20240229版本更新说明

116. ultralytics/cfg/models/v8/yolov8-PGI.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的programmable gradient information改进YOLOV8.(PGI模块可在训练结束后去掉)
    项目视频百度云链接:20240229版本更新说明

117. ultralytics/cfg/models/v8/yolov8-C2f-DynamicConv.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的DynamicConv改进C2f.
    项目视频百度云链接:20240303版本更新说明

118. ultralytics/cfg/models/v8/yolov8-C2f-GhostDynamicConv.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的GhostModule改进C2f.
    项目视频百度云链接:20240303版本更新说明

119. ultralytics/cfg/models/v8/yolov8-DynamicHGNetV2.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的DynamicConv对[CVPR2024 RTDETR](https://arxiv.org/abs/2304.08069)中的HGBlokc进行二次创新.
    项目视频百度云链接:20240303版本更新说明

120. ultralytics/cfg/models/v8/yolov8-C2f-RVB.yaml

    使用[CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)中的RepViTBlock改进C2f.
    项目视频百度云链接:20240309版本更新说明

121. ultralytics/cfg/models/v8/yolov8-C2f-RVB-EMA.yaml

    使用[CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)中的RepViTBlock和EMA注意力机制改进C2f.
    项目视频百度云链接:20240309版本更新说明

122. Lightweight Shared Convolutional Detection Head

    自研轻量化检测头.
    detect:ultralytics/cfg/models/v8/yolov8-LSCD.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-LSCD.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LSCD.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LSCD.yaml
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.
    3. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    综合以上，我们可以让检测头做到参数量更少、计算量更少的情况下，尽可能减少精度的损失.
    项目视频百度云链接:20240309版本更新说明

123. ultralytics/cfg/models/v8/yolov8-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进yolov8.
    项目视频百度云链接:20240309版本更新说明

124. Task Align Dynamic Detection Head

    自研任务对齐动态检测头
    detect:ultralytics/cfg/models/v8/yolov8-TADDH.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-TADDH.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-TADDH.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-TADDH.yaml
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.并且在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    3. 参照TOOD的思想,除了标签分配策略上的任务对齐,我们也在检测头上进行定制任务对齐的结构,现有的目标检测器头部通常使用独立的分类和定位分支,这会导致两个任务之间缺乏交互,TADDH通过特征提取器从多个卷积层中学习任务交互特征,得到联合特征,定位分支使用DCNV2和交互特征生成DCNV2的offset和mask,分类分支使用交互特征进行动态特征选择.
    项目视频百度云链接:20240314版本更新说明

125. ultralytics/cfg/models/v8/yolov8-rmt.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)改进yolov8的主干.
    项目视频百度云链接:20240323版本更新说明

126. ultralytics/cfg/models/v8/yolov8-C2f-RetBlock.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进C2f.
    项目视频百度云链接:20240323版本更新说明

127. ultralytics/cfg/models/v8/yolov8-ELA-HSFPN.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN.
    项目视频百度云链接:20240323版本更新说明

128. ultralytics/cfg/models/v8/yolov8-CA-HSFPN.yaml

    使用[Coordinate Attention CVPR2021](https://github.com/houqb/CoordAttention)改进HSFPN.
    项目视频百度云链接:20240323版本更新说明

129. ultralytics/cfg/models/v8/yolov8-ELA-HSFPN-TADDH.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN,使用自研动态动态对齐检测头改进Head.

130. ultralytics/cfg/models/v8/yolov8-pkinet.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)改进backbone.(需要安装mmcv和mmengine)
    项目视频百度云链接:20240330版本更新说明

131. ultralytics/cfg/models/v8/yolov8-C2f-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块改进C2f.
    项目视频百度云链接:20240330版本更新说明

132. ultralytics/cfg/models/v8/yolov8-RepNCSPELAN_CAA.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块改进RepNCSPELAN.
    项目视频百度云链接:20240330版本更新说明

133. ultralytics/cfg/models/v8/yolov8-CAA-HSFPN.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块HSFPN.
    项目视频百度云链接:20240330版本更新说明

134. ultralytics/cfg/models/v8/yolov8-C2f-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进C2f.
    项目视频百度云链接:20240406版本更新说明

135. ultralytics/cfg/models/v8/yolov8-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.
    项目视频百度云链接:20240406版本更新说明

136. ultralytics/cfg/models/v8/yolov8-FDPN-TADDH.yaml

    自研结构的融合.
    1. 自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    2. 自研任务对齐动态检测头(Task Align Dynamic Detection Head)
    项目视频百度云链接:20240408版本更新说明

137. ultralytics/cfg/models/v8/yolov8-C2f-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C2f.
    项目视频百度云链接:20240408版本更新说明

138. ultralytics/cfg/models/v8/yolov8-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.
    项目视频百度云链接:20240408版本更新说明

139. ultralytics/cfg/models/v8/yolov8-CSMHSA.yaml

    对Mutil-Head Self-Attention进行创新得到Cross-Scale Mutil-Head Self-Attention.
    1. 由于高维通常包含更高级别的语义信息，而低维包含更多细节信息，因此高维信息作为query，而低维信息作为key和Value，将两者结合起来可以利用高维的特征帮助低维的特征进行精细过滤，可以实现更全面和丰富的特征表达。
    2. 通过使用高维的上采样信息进行Query操作，可以更好地捕捉到目标的全局信息，从而有助于增强模型对目标的识别和定位能力。
    项目视频百度云链接:20240414版本更新说明

140. ultralytics/cfg/models/v8/yolov8-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进yolov8的下采样.
    项目视频百度云链接:20240420版本更新说明

141. ultralytics/cfg/models/v8/yolov8-CSFCN.yaml

    使用[Context and Spatial Feature Calibration for Real-Time Semantic Segmentation](https://github.com/kaigelee/CSFCN/tree/main)中的Context and Spatial Feature Calibration模块改进yolov8.
    项目视频百度云链接:20240420版本更新说明

142. ultralytics/cfg/models/v8/yolov8-mobilenetv4.yaml

    使用[MobileNetV4](https://github.com/jaiwei98/MobileNetV4-pytorch/tree/main)改进yolov8-backbone.
    项目视频百度云链接:20240428版本更新说明

143. ultralytics/cfg/models/v8/yolov8-CGAFusion.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的content-guided attention fusion改进yolov8-neck.
    项目视频百度云链接:20240428版本更新说明

144. ultralytics/cfg/models/v8/yolov8-CAFMFusion.yaml

    利用具有[HCANet](https://github.com/summitgao/HCANet)中的CAFM，其具有获取全局和局部信息的注意力机制进行二次改进content-guided attention fusion.
    项目视频百度云链接:20240428版本更新说明

145. ultralytics/cfg/models/v8/yolov8-RGCSPELAN.yaml

    自研RepGhostCSPELAN.
    1. 参考GhostNet中的思想(主流CNN计算的中间特征映射存在广泛的冗余)，采用廉价的操作生成一部分冗余特征图，以此来降低计算量和参数量。
    2. 舍弃yolov5与yolov8中常用的BottleNeck，为了弥补舍弃残差块所带来的性能损失，在梯度流通分支上使用RepConv，以此来增强特征提取和梯度流通的能力，并且RepConv可以在推理的时候进行融合，一举两得。
    3. 可以通过缩放因子控制RGCSPELAN的大小，使其可以兼顾小模型和大模型。
    项目视频百度云链接:20240501版本更新说明

146. Lightweight Asymmetric Detection Head

    detect:ultralytics/cfg/models/v8/yolov8-LADH.yaml
    segment:ultralytics/cfg/models/v8/yolov8-seg-LADH.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LADH.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LADH.yaml
    使用[Faster and Lightweight: An Improved YOLOv5 Object Detector for Remote Sensing Images](https://www.mdpi.com/2072-4292/15/20/4974)中的Lightweight Asymmetric Detection Head改进yolov8-head.
    项目视频百度云链接:20240505版本更新说明

147. ultralytics/cfg/models/v8/yolov8-C2f-Faster-CGLU.yaml

    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU对CVPR2023中的FasterNet进行二次创新.
    项目视频百度云链接:20240505版本更新说明

148. ultralytics/cfg/models/v8/yolov8-SDFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的superficial detail fusion module改进yolov8-neck.
    项目视频百度云链接:20240512版本更新说明

149. ultralytics/cfg/models/v8/yolov8-PSFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的profound semantic fusion module改进yolov8-neck.
    项目视频百度云链接:20240512版本更新说明

150. Lightweight Shared Convolutional Separamter BN Detection Head

    基于自研轻量化检测头(LSCD)上，参考NASFPN的设计思路把GN换成BN，并且BN层参数不共享.
    detect:ultralytics/cfg/models/v8/yolov8-LSCSBD.yaml
    seg:ultralytics/cfg/models/v8/yolov8-seg-LSCSBD.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LSCSBD.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LSCSBD.yaml
    1. 由于不同层级之间特征的统计量仍存在差异，Normalization layer依然是必须的，由于直接在共享参数的检测头中引入BN会导致其滑动平均值产生误差，而引入 GN 又会增加推理时的开销，因此我们参考NASFPN的做法，让检测头共享卷积层，而BN则分别独立计算。
    项目视频百度云链接:20240512版本更新说明

151. ultralytics/cfg/models/v8/yolov8-starnet.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)改进yolov8-backbone.
    项目视频百度云链接:20240513版本更新说明

152. ultralytics/cfg/models/v8/yolov8-C2f-Star.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock改进C2f.
    项目视频百度云链接:20240513版本更新说明

153. ultralytics/cfg/models/v8/yolov8-C2f-Star-CAA.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock和[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA改进C2f.
    项目视频百度云链接:20240513版本更新说明

154. ultralytics/cfg/models/v8/yolov8-starnet-C2f-Star-LSCD.yaml

    轻量化模型组合.
    1. CVPR2024-StarNet Backbone.
    2. C2f-Star.
    3. Lightweight Shared Convolutional Detection Head.
    项目视频百度云链接:20240513版本更新说明

155. ultralytics/cfg/models/v8/yolov8-C2f-KAN.yaml

    KAN In! Mamba Out! Kolmogorov-Arnold Networks.[Pytorch-Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan)
    目前支持:
    1. FastKANConv2DLayer
    2. KANConv2DLayer
    3. KALNConv2DLayer
    4. KACNConv2DLayer
    5. KAGNConv2DLayer
    项目视频百度云链接:20240523版本更新说明

156. ultralytics/cfg/models/v8/yolov8-nmsfree.yaml

    仿照yolov10的思想采用双重标签分配和一致匹配度量进行训练,后处理不需要NMS!
    项目视频百度云链接:20240526版本更新说明

157. ultralytics/cfg/models/v8/yolov8-EIEStem.yaml

    提出了一种新的EIEStem模块，旨在作为图像识别任务中的高效前端模块。该模块结合了提取边缘信息的SobelConv分支和提取空间信息的卷积分支，能够学习到更加丰富的图像特征表示。
    1. 通过SobelConv分支，可以提取图像的边缘信息。由于Sobel滤波器可以检测图像中强度的突然变化，因此可以很好地捕捉图像的边缘特征。这些边缘特征在许多计算机视觉任务中都非常重要，例如图像分割和物体检测。
    2. EIEStem模块还结合空间信息，除了边缘信息，EIEStem还通过池化分支提取空间信息，保留重要的空间信息。结合边缘信息和空间信息，可以帮助模型更好地理解图像内容。
    3. 通过3D组卷积高效实现Sobel算子。
    项目视频百度云链接:20240526版本更新说明

158. ultralytics/cfg/models/v8/yolov8-C2f-EIEM.yaml

    提出了一种新的C2f-EIEM模块，旨在作为图像识别任务中的高效前端模块。该模块结合了提取边缘信息的SobelConv分支和提取空间信息的卷积分支，能够学习到更加丰富的图像特征表示。
    1. 边缘信息学习: 卷积神经网络 (CNN)通常擅长学习空间信息，但是对于提取图像中的边缘信息可能稍显不足。EIEStem 模块通过SobelConv分支，显式地提取图像的边缘特征。Sobel滤波器是一种经典的边缘检测滤波器，可以有效地捕捉图像中强度的突然变化，从而获得重要的边缘信息。
    2. 空间信息保留: 除了边缘信息，图像中的空间信息也同样重要。EIEStem模块通过一个额外的卷积分支 (conv_branch) 来提取空间信息。与SobelCon 分支不同，conv_branch提取的是原始图像的特征，可以保留丰富的空间细节。
    3. 特征融合: EIEStem模块将来自SobelConv分支和conv_branch提取的特征进行融合 (concatenate)。 这种融合操作使得学习到的特征表示既包含了丰富的边缘信息，又包含了空间信息，能够更加全面地刻画图像内容。
    项目视频百度云链接:20240526版本更新说明

159. ultralytics/cfg/models/v8/yolov8-ContextGuideFPN.yaml

    Context Guide Fusion Module（CGFM）是一个创新的特征融合模块，旨在改进YOLOv8中的特征金字塔网络（FPN）。该模块的设计考虑了多尺度特征融合过程中上下文信息的引导和自适应调整。
    1. 上下文信息的有效融合：通过SE注意力机制，模块能够在特征融合过程中捕捉并利用重要的上下文信息，从而增强特征表示的有效性，并有效引导模型学习检测目标的信息，从而提高模型的检测精度。
    2. 特征增强：通过权重化的特征重组操作，模块能够增强重要特征，同时抑制不重要特征，提升特征图的判别能力。
    3. 简单高效：模块结构相对简单，不会引入过多的计算开销，适合在实时目标检测任务中应用。
    这期视频讲解在B站:https://www.bilibili.com/video/BV1Vx4y1n7hZ/
    项目视频百度云链接:20240601版本更新说明

160. ultralytics/cfg/models/v8/yolov8-C2f-DEConv.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的detail-enhanced convolution改进C2f.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.
    项目视频百度云链接:20240601版本更新说明

161. ultralytics/cfg/models/v8/yolov8-LSDECD.yaml

    基于自研轻量化检测头上(LSCD)，使用detail-enhanced convolution进一步改进，提高检测头的细节捕获能力，进一步改善检测精度.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.
    detect:ultralytics/cfg/models/v8/yolov8-LSDECD.yaml
    segment:ultralytics/cfg/models/v8/yolov8-seg-LSDECD.yaml
    pose:ultralytics/cfg/models/v8/yolov8-pose-LSDECD.yaml
    obb:ultralytics/cfg/models/v8/yolov8-obb-LSDECD.yaml
    1. DEA-Net中设计了一个细节增强卷积（DEConv），具体来说DEConv将先验信息整合到普通卷积层，以增强表征和泛化能力。然后，通过使用重参数化技术，DEConv等效地转换为普通卷积，不需要额外的参数和计算成本。
    项目视频百度云链接:20240601版本更新说明

162. ultralytics/cfg/models/v8/yolov8-C2f-SMPCGLU.yaml

    Self-moving Point Convolutional GLU模型改进C2f.
    SMP来源于[CVPR2023-SMPConv](https://github.com/sangnekim/SMPConv),Convolutional GLU来源于[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt).
    1. 普通的卷积在面对数据中的多样性和复杂性时，可能无法捕捉到有效的特征，因此我们采用了SMPConv，其具备最新的自适应点移动机制，从而更好地捕捉局部特征，提高特征提取的灵活性和准确性。
    2. 在SMPConv后添加CGLU，Convolutional GLU 结合了卷积和门控机制，能够选择性地通过信息通道，提高了特征提取的有效性和灵活性。
    3. 这种类似Transformer的encoder结构，利用DEConv进行特征提取，CGLU可以进一步的非线性变换和加强特征提取，提升模型的表达能力。
    项目视频百度云链接:20240601版本更新说明

163. ultralytics/cfg/models/v8/yolov8-C2f-Heat.yaml

    使用[vHeat](https://github.com/MzeroMiko/vHeat/tree/main)中的HeatBlock改进C2f.(需要看[常见错误和解决方案的第五点](#a)) 
    项目视频百度云链接:20240609版本更新说明

164. Re-CalibrationFPN

    为了加强浅层和深层特征的相互交互能力，推出重校准特征金字塔网络(Re-CalibrationFPN).
    P2345：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P2345.yaml(带有小目标检测头的ReCalibrationFPN)
    P345：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P345.yaml
    P3456：ultralytics/cfg/models/v8/yolov8-ReCalibrationFPN-P3456.yaml(带有大目标检测头的ReCalibrationFPN)
    1. 浅层语义较少，但细节丰富，有更明显的边界和减少失真。此外，深层蕴藏着丰富的物质语义信息。因此，直接融合低级具有高级特性的特性可能导致冗余和不一致。为了解决这个问题，我们提出了[SBA](https://github.com/Barrett-python/DuAT)模块，它有选择地聚合边界信息和语义信息来描绘更细粒度的物体轮廓和重新校准物体的位置。
    2. 相比传统的FPN结构，[SBA](https://github.com/Barrett-python/DuAT)模块引入了高分辨率和低分辨率特征之间的双向融合机制，使得特征之间的信息传递更加充分，进一步提升了多尺度特征融合的效果。
    3. [SBA](https://github.com/Barrett-python/DuAT)模块通过自适应的注意力机制，根据特征图的不同分辨率和内容，自适应地调整特征的权重，从而更好地捕捉目标的多尺度特征。
    项目视频百度云链接:20240609版本更新说明

165. ultralytics/cfg/models/v8/yolov8-WaveletPool.yaml

    使用[Wavelet Pooling](https://openreview.net/forum?id=rkhlb8lCZ)改进YOLOV8的上采样和下采样。
    项目视频百度云链接:20240613版本更新说明

166. ultralytics/cfg/models/v8/yolov8-CSP-PTB.yaml

    Cross Stage Partial - Partially Transformer Block
    在计算机视觉任务中，Transformer结构因其强大的全局特征提取能力而受到广泛关注。然而，由于Transformer结构的计算复杂度较高，直接将其应用于所有通道会导致显著的计算开销。为了在保证高效特征提取的同时降低计算成本，我们设计了一种混合结构，将输入特征图分为两部分，分别由CNN和Transformer处理，结合了卷积神经网络(CNN)和Transformer机制的模块，旨在增强特征提取的能力。
    我们提出了一种名为CSP_PTB(Cross Stage Partial - Partially Transformer Block)的模块，旨在结合CNN和Transformer的优势，通过对输入通道进行部分分配来优化计算效率和特征提取能力。
    1. 融合局部和全局特征：多项研究表明，CNN的感受野大小较少，导致其只能提取局部特征，但Transformer的MHSA能够提取全局特征，能够同时利用两者的优势。
    2. 保证高效特征提取的同时降低计算成本：为了能引入Transformer结构来提取全局特征又不想大幅度增加计算复杂度，因此提出Partially Transformer Block，只对部分通道使用TransformerBlock。
    3. MHSA_CGLU包含Mutil-Head-Self-Attention和[ConvolutionalGLU(TransNext CVPR2024)](https://github.com/DaiShiResearch/TransNeXt)，其中Mutil-Head-Self-Attention负责提取全局特征，ConvolutionalGLU用于增强非线性特征表达能力，ConvolutionalGLU相比于传统的FFN，具有更强的性能。
    4. 可以根据不同的模型大小和具体的运行情况调节用于Transformer的通道数。
    项目视频百度云链接:20240613版本更新说明

167. [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)

    集成Mamba-YOLO.(需要编译请看百度云视频-20240619版本更新说明)
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L.yaml
    ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml
    项目视频百度云链接:20240619版本更新说明

168. ultralytics/cfg/models/v8/yolov8-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块改进yolov8的neck.
    项目视频百度云链接:20240619版本更新说明

169. ultralytics/cfg/models/v8/yolov8-bifpn-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块对bifpn进行二次创新.
    项目视频百度云链接:20240619版本更新说明

170. ultralytics/cfg/models/v8/yolov8-SOEP.yaml  
    
    小目标在正常的P3、P4、P5检测层上略显吃力，比较传统的做法是加上P2检测层来提升小目标的检测能力，但是同时也会带来一系列的问题，例如加上P2检测层后计算量过大、后处理更加耗时等问题，日益激发需要开发新的针对小目标有效的特征金字塔，我们基于原本的PAFPN上进行改进，提出SmallObjectEnhancePyramid，相对于传统的添加P2检测层，我们使用P2特征层经过SPDConv得到富含小目标信息的特征给到P3进行融合，然后使用CSP思想和基于[AAAI2024的OmniKernel](https://ojs.aaai.org/index.php/AAAI/article/view/27907)进行改进得到CSP-OmniKernel进行特征整合，OmniKernel模块由三个分支组成，即三个分支，即全局分支、大分支和局部分支、以有效地学习从全局到局部的特征表征，最终从而提高小目标的检测性能。
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    出现这个报错的:RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR,如果你是40系显卡,需要更新torch大于2.0，并且cuda大于12.0.
    项目视频百度云链接:20240627版本更新说明

171. ultralytics/cfg/models/v8/yolov8-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进yolov8-neck.(需要看[常见错误和解决方案的第五点](#a))  
    项目视频百度云链接:20240627版本更新说明

172. ultralytics/cfg/models/v8/yolov8-p6-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进yolov8-neck.(带有p6版本)(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20240627版本更新说明  

173. ultralytics/cfg/models/v8/yolov8-C2f-WTConv.yaml

    使用[ECCV2024 Wavelet Convolutions for Large Receptive Fields](https://github.com/BGU-CS-VIL/WTConv)中的WTConv改进C2f-BottleNeck.
    项目视频百度云链接:20240720版本更新说明

174. ultralytics/cfg/models/v8/yolov8-CGRFPN.yaml

    Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    1. 借鉴[ECCV2024-CGRSeg](https://github.com/nizhenliang/CGRSeg)中的Rectangular Self-Calibration Module经过精心设计,用于空间特征重建和金字塔上下文提取,它在水平和垂直方向上捕获全局上下文，并获得轴向全局上下文来显式地建模矩形关键区域.
    2. PyramidContextExtraction Module使用金字塔上下文提取模块（PyramidContextExtraction），有效整合不同层级的特征信息，提升模型的上下文感知能力。
    3. FuseBlockMulti 和 DynamicInterpolationFusion 这些模块用于多尺度特征的融合，通过动态插值和多特征融合，进一步提高了模型的多尺度特征表示能力和提升模型对复杂背景下目标的识别能力。
    项目视频百度云链接:20240720版本更新说明

175. ultralytics/cfg/models/v8/yolov8-FeaturePyramidSharedConv.yaml

    1. 多尺度特征提取
        通过使用不同膨胀率的卷积层，模块能够提取不同尺度的特征。这对捕捉图像中不同大小和不同上下文的信息非常有利。
        低膨胀率捕捉局部细节，高膨胀率捕捉全局上下文。
    2. 参数共享
        使用共享的卷积层 self.share_conv，大大减少了需要训练的参数数量。相比于每个膨胀率使用独立的卷积层，共享卷积层能够减少冗余，提升模型效率。
        减少了模型的存储和计算开销，提升了计算效率。
    3. 高效的通道变换
        通过1x1卷积层 self.cv1 和 self.cv2，模块能够高效地调整通道数，并进行特征融合。1x1卷积层在减少参数量的同时还能保留重要的特征信息。
    4. 更细粒度的特征提取
        FeaturePyramidSharedConv 使用卷积操作进行特征提取，能够捕捉更加细粒度的特征。相比之下，SPPF 的池化操作可能会丢失一些细节信息。
        卷积操作在特征提取时具有更高的灵活性和表达能力，可以更好地捕捉图像中的细节和复杂模式。
    项目视频百度云链接:20240729版本更新说明

176. ultralytics/cfg/models/v8/yolov8-C2f-FMB.yaml

    使用[ECCV2024 SMFANet](https://github.com/Zheng-MJ/SMFANet/tree/main)的Feature Modulation block改进C2f.
    项目视频百度云链接:20240729版本更新说明

177. ultralytics/cfg/models/v8/yolov8-LDConv.yaml

    使用[LDConv](https://github.com/CV-ZhangXin/LDConv/tree/main)改进下采样.
    项目视频百度云链接:20240803版本更新说明

178. ultralytics/cfg/models/v8/yolov8-C2f-gConv.yaml

    使用[Rethinking Performance Gains in Image Dehazing Networks](https://arxiv.org/abs/2209.11448)的gConvblock改进C2f.
    项目视频百度云链接:20240803版本更新说明

179. ultralytics/cfg/models/v8/yolov8-MAFPN.yaml

    使用[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN改进Neck.
    项目视频百度云链接:20240803版本更新说明

180. ultralytics/cfg/models/v8/yolov8-BIMAFPN.yaml

    利用BIFPN的思想对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次改进得到BIMAFPN.
    项目视频百度云链接:20240803版本更新说明

181. APT(Adaptive Power Transformation)-TAL.

    为了使不同gt预测对的匹配质量和损失权重更具鉴别性，我们通过自定义的PowerTransformer显著增强高质量预测框的权重，抑制低质量预测框的影响，并使模型在学习的过程可以更关注质量高的预测框。
    修改方式：ultralytics/utils/tal.py中的get_box_metrics函数中的power参数设置为True.
    项目视频百度云链接:20240813版本更新说明

182. ultralytics/cfg/models/v8/yolov8-C2f-WDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的WDBB改进c2f.
    项目视频百度云链接:20240813版本更新说明

183. ultralytics/cfg/models/v8/yolov8-C2f-DeepDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的DeepDBB改进c2f.
    项目视频百度云链接:20240813版本更新说明

184. ultralytics/cfg/models/v8/yolov8-AIFIRepBN.yaml

    使用[ICML-2024 SLAB](https://github.com/xinghaochen/SLAB)中的RepBN改进AIFI.
    项目视频百度云链接:20240813版本更新说明

185. ultralytics/cfg/models/v8/yolov8-C2f-AdditiveBlock.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock改进c2f.
    项目视频百度云链接:20240822版本更新说明

186. ultralytics/cfg/models/v8/yolov8-C2f-AdditiveBlock-CGLU.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.
    项目视频百度云链接:20240822版本更新说明

187. ultralytics/cfg/models/v8/yolov8-EMBSFPN.yaml

    基于BIFPN、[MAF-YOLO](https://arxiv.org/pdf/2407.04381)、[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)提出全新的Efficient Multi-Branch&Scale FPN.
    Efficient Multi-Branch&Scale FPN拥有<轻量化>、<多尺度特征加权融合>、<多尺度高效卷积模块>、<高效上采样模块>、<全局异构核选择机制>。
    1. 具有多尺度高效卷积模块和全局异构核选择机制，Trident网络的研究表明，具有较大感受野的网络更适合检测较大的物体，反之，较小尺度的目标则从较小的感受野中受益，因此我们在FPN阶段，对于不同尺度的特征层选择不同的多尺度卷积核以适应并逐步获得多尺度感知场信息。
    2. 借鉴BIFPN中的多尺度特征加权融合，能把Concat换成Add来减少参数量和计算量的情况下，还能通过不同尺度特征的重要性进行自适用选择加权融合。
    3. 高效上采样模块来源于CVPR2024-EMCAD中的EUCB，能够在保证一定效果的同时保持高效性。
    项目视频百度云链接:20240822版本更新说明

188. ultralytics/cfg/models/v8/yolov8-C2f-MSMHSA-CGLU.yaml

    使用[CMTFNet](https://github.com/DrWuHonglin/CMTFNet/tree/main)中的M2SA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.
    项目视频百度云链接:20240831版本更新说明

189. ultralytics/cfg/models/v8/yolov8-CSP-PMSFA.yaml

    自研模块:CSP-Partial Multi-Scale Feature Aggregation.
    1. 部分多尺度特征提取：参考CVPR2020-GhostNet、CVPR2024-FasterNet的思想，采用高效的PartialConv，该模块能够从输入中提取多种尺度的特征信息，但它并不是在所有通道上进行这种操作，而是部分（Partial）地进行，从而提高了计算效率。
    2. 增强的特征融合: 最后的 1x1 卷积层通过将不同尺度的特征融合在一起，同时使用残差连接将输入特征与处理后的特征相加，有效保留了原始信息并引入了新的多尺度信息，从而提高模型的表达能力。
    项目视频百度云链接:20240831版本更新说明

### YOLOV10
#### 以下配置文件都基于v10n，如果需要使用其他大小的模型(s,m,b,l,x)可以看项目视频百度云链接-YOLOV10模型大小切换教程.

1. ultralytics/cfg/models/v10/yolov10n-bifpn.yaml

    添加BIFPN到yolov10中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        其中支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.
    项目视频百度云链接:关于BIFPN的说明

2. ultralytics/cfg/models/v10/yolov10n-C2f-EMSC.yaml

    Efficient Multi-Scale Conv.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:EMSC,EMSCP讲解

3. ultralytics/cfg/models/v10/yolov10n-C2f-EMSCP.yaml

    Efficient Multi-Scale Conv Plus.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:EMSC,EMSCP讲解

4. ultralytics/cfg/models/v10/yolov10n-LAWDS.yaml

    Light Adaptive-weight downsampling.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:LAWDS讲解

5. ultralytics/cfg/models/v10/yolov10n-LSCD.yaml

    自研轻量化检测头.(Lightweight Shared Convolutional Detection Head)
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.
    3. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    综合以上，我们可以让检测头做到参数量更少、计算量更少的情况下，尽可能减少精度的损失.
    项目视频百度云链接:20240713版本更新说明

6. ultralytics/cfg/models/v10/yolov10n-efficientViT.yaml

    (CVPR2023)efficientViT替换yolov10主干.

7. ultralytics/cfg/models/v10/yolov10n-fasternet.yaml

    (CVPR2023)fasternet替换yolov10主干.

8. ultralytics/cfg/models/v10/yolov10n-timm.yaml

    使用timm支持的主干网络替换yolov10主干.

9. ultralytics/cfg/models/v10/yolov10n-convnextv2.yaml

    使用convnextv2网络替换yolov10主干.

10. ultralytics/cfg/models/v10/yolov10n-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolov10主干.(需要看[常见错误和解决方案的第五点](#a))  

11. ultralytics/cfg/models/v10/yolov10n-vanillanet.yaml

    vanillanet替换yolov10主干.

12. ultralytics/cfg/models/v10/yolov10n-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolov10主干.

13. ultralytics/cfg/models/v10/yolov10n-swintransformer.yaml

    SwinTransformer-Tiny替换yolov10主干.

14. ultralytics/cfg/models/v10/yolov10n-repvit.yaml

    [CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)替换yolov10主干.

15. ultralytics/cfg/models/v10/yolov10n-CSwinTransformer.yaml

    使用[CSWin-Transformer(CVPR2022)](https://github.com/microsoft/CSWin-Transformer/tree/main)替换yolov10主干.(需要看[常见错误和解决方案的第五点](#a))

16. ultralytics/cfg/models/v10/yolov10n-HGNetV2.yaml

    使用HGNetV2作为YOLOV10的backbone.

17. ultralytics/cfg/models/v10/yolov10n-unireplknet.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)替换yolov10主干.

18. ultralytics/cfg/models/v10/yolov10n-TransNeXt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)改进yolov10的backbone.(需要看[常见错误和解决方案的第五点](#a))   

19. ultralytics/cfg/models/v10/yolov10n-rmt.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)改进yolov10的主干.

20. ultralytics/cfg/models/v10/yolov10n-pkinet.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)改进backbone.(需要安装mmcv和mmengine)

21. ultralytics/cfg/models/v10/yolov10n-mobilenetv4.yaml

    使用[MobileNetV4](https://github.com/jaiwei98/MobileNetV4-pytorch/tree/main)改进yolov10的backbone.

22. ultralytics/cfg/models/v10/yolov10n-starnet.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)改进yolov10-backbone.

23. ultralytics/cfg/models/v10/yolov10n-starnet-bifpn.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)和bifpn改进yolov10.

24. ultralytics/cfg/models/v10/yolov10n-C2f-WTConv.yaml

    使用[ECCV2024 Wavelet Convolutions for Large Receptive Fields](https://github.com/BGU-CS-VIL/WTConv)中的WTConv改进C2f-BottleNeck.
    项目视频百度云链接:20240720版本更新说明

25. ultralytics/cfg/models/v10/yolov10n-CGRFPN.yaml

    Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    1. 借鉴[ECCV2024-CGRSeg](https://github.com/nizhenliang/CGRSeg)中的Rectangular Self-Calibration Module经过精心设计,用于空间特征重建和金字塔上下文提取,它在水平和垂直方向上捕获全局上下文，并获得轴向全局上下文来显式地建模矩形关键区域.
    2. PyramidContextExtraction Module使用金字塔上下文提取模块（PyramidContextExtraction），有效整合不同层级的特征信息，提升模型的上下文感知能力。
    3. FuseBlockMulti 和 DynamicInterpolationFusion 这些模块用于多尺度特征的融合，通过动态插值和多特征融合，进一步提高了模型的多尺度特征表示能力和提升模型对复杂背景下目标的识别能力。
    项目视频百度云链接:20240720版本更新说明

26. SlideLoss and EMASlideLoss.[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)

    在ultralytics/utils/loss.py中的class v8DetectionLoss进行设定.

27. ultralytics/cfg/models/v10/yolov10n-RevCol.yaml

    使用[(ICLR2023)Reversible Column Networks](https://github.com/megvii-research/RevCol)对yolov10主干进行重设计,里面的支持更换不同的C2f-Block.
    视频:https://www.bilibili.com/video/BV1Mh4y1y76u/

28. ultralytics/cfg/models/v10/yolov10n-attention.yaml

    可以看项目视频-如何在yaml配置文件中添加注意力层  
    多种注意力机制在yolov10中的使用. [多种注意力机制github地址](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)  
    目前内部整合的注意力可看[链接](#c)

29. ultralytics/cfg/models/v10/yolov10n-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample进行下采样.

30. ultralytics/cfg/models/v10/yolov10n-SPDConv.yaml

    使用[SPDConv](https://github.com/LabSAINT/SPD-Conv/tree/main)进行下采样.
    项目视频百度云链接:20231010版本更新说明

31. ultralytics/cfg/models/v10/yolov10n-dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov10-neck中的上采样.
    项目视频百度云链接:20240122版本更新说明

32. ultralytics/cfg/models/v10/yolov10n-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov10-neck中的上采样.
    项目视频百度云链接:20240122版本更新说明

33. ultralytics/cfg/models/v10/yolov10n-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8的下采样.(请关闭AMP情况下使用)
    项目视频百度云链接:20240122版本更新说明

34. ultralytics/cfg/models/v8=10/yolov10n-v7DS.yaml

    使用[YOLOV7 CVPR2023](https://arxiv.org/abs/2207.02696)的下采样结构改进YOLOV10中的下采样.
    项目视频百度云链接:20240229版本更新说明

35. ultralytics/cfg/models/v10/yolov10n-ADown.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的下采样结构改进YOLOV10中的下采样.
    项目视频百度云链接:20240229版本更新说明

36. ultralytics/cfg/models/v10/yolov10n-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进yolov10的下采样.
    项目视频百度云链接:20240420版本更新说明

37. ultralytics/cfg/models/v10/yolov10n-WaveletPool.yaml

    使用[Wavelet Pooling](https://openreview.net/forum?id=rkhlb8lCZ)改进YOLOV10的上采样和下采样。
    项目视频百度云链接:20240613版本更新说明

38. ultralytics/cfg/models/v10/yolov10n-slimneck.yaml

    使用[VoVGSCSP\VoVGSCSPC和GSConv](https://github.com/AlanLi1997/slim-neck-by-gsconv)替换yolov10 neck中的C2f和Conv.

39. ultralytics/cfg/models/v10/yolov10n-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.
    项目视频百度云链接:20240720版本更新说明

40. ultralytics/cfg/models/v10/yolov10n-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.
    项目视频百度云链接:20240720版本更新说明

41. ultralytics/cfg/models/v10/yolov10n-FeaturePyramidSharedConv.yaml

    1. 多尺度特征提取
        通过使用不同膨胀率的卷积层，模块能够提取不同尺度的特征。这对捕捉图像中不同大小和不同上下文的信息非常有利。
        低膨胀率捕捉局部细节，高膨胀率捕捉全局上下文。
    2. 参数共享
        使用共享的卷积层 self.share_conv，大大减少了需要训练的参数数量。相比于每个膨胀率使用独立的卷积层，共享卷积层能够减少冗余，提升模型效率。
        减少了模型的存储和计算开销，提升了计算效率。
    3. 高效的通道变换
        通过1x1卷积层 self.cv1 和 self.cv2，模块能够高效地调整通道数，并进行特征融合。1x1卷积层在减少参数量的同时还能保留重要的特征信息。
    4. 更细粒度的特征提取
        FeaturePyramidSharedConv 使用卷积操作进行特征提取，能够捕捉更加细粒度的特征。相比之下，SPPF 的池化操作可能会丢失一些细节信息。
        卷积操作在特征提取时具有更高的灵活性和表达能力，可以更好地捕捉图像中的细节和复杂模式。
    项目视频百度云链接:20240729版本更新说明

42. ultralytics/cfg/models/v10/yolov10n-C2f-FMB.yaml

    使用[ECCV2024 SMFANet](https://github.com/Zheng-MJ/SMFANet/tree/main)的Feature Modulation block改进C2f.
    项目视频百度云链接:20240729版本更新说明

43. ultralytics/cfg/models/v10/yolov10n-C2f-Faster.yaml

    使用C2f-Faster替换C2f.(使用FasterNet中的FasterBlock替换C2f中的Bottleneck)
    项目视频百度云链接:20240729版本更新说明

44. ultralytics/cfg/models/v10/yolov10n-C2f-ODConv.yaml

    使用C2f-ODConv替换C2f.(使用ODConv替换C2f中的Bottleneck中的Conv)
    项目视频百度云链接:20240729版本更新说明

45. ultralytics/cfg/models/v10/yolov10n-C2f-Faster-EMA.yaml

    使用C2f-Faster-EMA替换C2f.(C2f-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C2f-Faster)
    项目视频百度云链接:20240729版本更新说明

46. ultralytics/cfg/models/v10/yolov10n-C2f-DBB.yaml

    使用C2f-DBB替换C2f.(使用DiverseBranchBlock替换C2f中的Bottleneck中的Conv)
    项目视频百度云链接:20240729版本更新说明

47. ultralytics/cfg/models/v10/yolov10n-C2f-CloAtt.yaml

    使用C2f-CloAtt替换C2f.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C2f中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20240729版本更新说明  

48. ultralytics/cfg/models/v10/yolov10n-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.
    介绍可看这期视频：https://www.bilibili.com/video/BV1Bp4y1w7MB/

49. ultralytics/cfg/models/v10/yolov10n-LDConv.yaml

    使用[LDConv](https://github.com/CV-ZhangXin/LDConv/tree/main)改进下采样.
    项目视频百度云链接:20240803版本更新说明

50. ultralytics/cfg/models/v10/yolov10n-C2f-gConv.yaml

    使用[Rethinking Performance Gains in Image Dehazing Networks](https://arxiv.org/abs/2209.11448)的gConvblock改进C2f.
    项目视频百度云链接:20240803版本更新说明

51. ultralytics/cfg/models/v10/yolov10n-MAFPN.yaml

    使用[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN改进Neck.
    项目视频百度云链接:20240803版本更新说明

52. ultralytics/cfg/models/v10/yolov10n-BIMAFPN.yaml

    利用BIFPN的思想对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次改进得到BIMAFPN.
    项目视频百度云链接:20240803版本更新说明

53. ultralytics/cfg/models/v10/yolov10n-C2f-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C2f融合.
    项目视频百度云链接:SCConv和ScConv的使用教程

54. ultralytics/cfg/models/v10/yolov10n-C2f-SCcConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C2f融合.  
    (取名为SCcConv的原因是在windows下命名是不区分大小写的)
    项目视频百度云链接:SCConv和ScConv的使用教程

55. ultralytics/cfg/models/v10/yolov10n-KernelWarehouse.yaml

    使用[Towards Parameter-Efficient Dynamic Convolution](https://github.com/OSVAI/KernelWarehouse)添加到yolov10中.  
    使用此模块需要注意,在epoch0-20的时候精度会非常低,过了20epoch会正常.
    项目视频百度云链接:20240803版本更新说明

56. ultralytics/cfg/models/v10/yolov10n-C2f-DySnakeConv.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C2f融合.
    项目视频百度云链接:20240803版本更新说明

57. APT(Adaptive Power Transformation)-TAL.

    为了使不同gt预测对的匹配质量和损失权重更具鉴别性，我们通过自定义的PowerTransformer显著增强高质量预测框的权重，抑制低质量预测框的影响，并使模型在学习的过程可以更关注质量高的预测框。
    修改方式：ultralytics/utils/tal.py中的get_box_metrics函数中的power参数设置为True.
    项目视频百度云链接:20240813版本更新说明

58. ultralytics/cfg/models/v10/yolov10n-C2f-WDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的WDBB改进c2f.
    项目视频百度云链接:20240813版本更新说明

59. ultralytics/cfg/models/v10/yolov10n-C2f-DeepDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的DeepDBB改进c2f.
    项目视频百度云链接:20240813版本更新说明

60. ultralytics/cfg/models/v10/yolov10n-AIFIRep.yaml

    使用[ICML-2024 SLAB](https://github.com/xinghaochen/SLAB)与AIFI改进yolov10.
    项目视频百度云链接:20240813版本更新说明

61. ultralytics/cfg/models/v10/yolov10n-C2f-AdditiveBlock.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock改进c2f.
    项目视频百度云链接:20240822版本更新说明

62. ultralytics/cfg/models/v10/yolov10n-C2f-AdditiveBlock-CGLU.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.
    项目视频百度云链接:20240822版本更新说明

63. ultralytics/cfg/models/v10/yolov10n-EMBSFPN.yaml

    基于BIFPN、[MAF-YOLO](https://arxiv.org/pdf/2407.04381)、[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)提出全新的Efficient Multi-Branch&Scale FPN.
    Efficient Multi-Branch&Scale FPN拥有<轻量化>、<多尺度特征加权融合>、<多尺度高效卷积模块>、<高效上采样模块>、<全局异构核选择机制>。
    1. 具有多尺度高效卷积模块和全局异构核选择机制，Trident网络的研究表明，具有较大感受野的网络更适合检测较大的物体，反之，较小尺度的目标则从较小的感受野中受益，因此我们在FPN阶段，对于不同尺度的特征层选择不同的多尺度卷积核以适应并逐步获得多尺度感知场信息。
    2. 借鉴BIFPN中的多尺度特征加权融合，能把Concat换成Add来减少参数量和计算量的情况下，还能通过不同尺度特征的重要性进行自适用选择加权融合。
    3. 高效上采样模块来源于CVPR2024-EMCAD中的EUCB，能够在保证一定效果的同时保持高效性。
    项目视频百度云链接:20240822版本更新说明

64. ultralytics/cfg/models/v10/yolov10n-ASF.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov10.
    项目视频百度云链接:20231217版本更新说明

65. ultralytics/cfg/models/v10/yolov10n-ASF-P2.yaml

    在ultralytics/cfg/models/v8/yolov8-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.
    项目视频百度云链接:20231217版本更新说明

66. ultralytics/cfg/models/v10/yolov10n-ASF-DySample.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)组合得到Dynamic Sample Attentional Scale Sequence Fusion.
    项目视频百度云链接:20240203版本更新说明

67. ultralytics/cfg/models/v10/yolov10n-goldyolo-asf.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute与[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新改进yolov10的neck.
    项目视频百度云链接:20240116版本更新说明

68. ultralytics/cfg/models/v10/yolov10n-SOEP.yaml 

    小目标在正常的P3、P4、P5检测层上略显吃力，比较传统的做法是加上P2检测层来提升小目标的检测能力，但是同时也会带来一系列的问题，例如加上P2检测层后计算量过大、后处理更加耗时等问题，日益激发需要开发新的针对小目标有效的特征金字塔，我们基于原本的PAFPN上进行改进，提出SmallObjectEnhancePyramid，相对于传统的添加P2检测层，我们使用P2特征层经过SPDConv得到富含小目标信息的特征给到P3进行融合，然后使用CSP思想和基于[AAAI2024的OmniKernel](https://ojs.aaai.org/index.php/AAAI/article/view/27907)进行改进得到CSP-OmniKernel进行特征整合，OmniKernel模块由三个分支组成，即三个分支，即全局分支、大分支和局部分支、以有效地学习从全局到局部的特征表征，最终从而提高小目标的检测性能。
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    出现这个报错的:RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR,如果你是40系显卡,需要更新torch大于2.0，并且cuda大于12.0.
    项目视频百度云链接:20240627版本更新说明

69. ultralytics/cfg/models/v10/yolov10n-C2f-MSMHSA-CGLU.yaml

    使用[CMTFNet](https://github.com/DrWuHonglin/CMTFNet/tree/main)中的M2SA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进c2f.
    项目视频百度云链接:20240831版本更新说明

70. ultralytics/cfg/models/v10/yolov10n-CSP-PMSFA.yaml

    自研模块:CSP-Partial Multi-Scale Feature Aggregation.
    1. 部分多尺度特征提取：参考CVPR2020-GhostNet、CVPR2024-FasterNet的思想，采用高效的PartialConv，该模块能够从输入中提取多种尺度的特征信息，但它并不是在所有通道上进行这种操作，而是部分（Partial）地进行，从而提高了计算效率。
    2. 增强的特征融合: 最后的 1x1 卷积层通过将不同尺度的特征融合在一起，同时使用残差连接将输入特征与处理后的特征相加，有效保留了原始信息并引入了新的多尺度信息，从而提高模型的表达能力。
    项目视频百度云链接:20240831版本更新说明

# 常见错误和解决方案(如果是跑自带的一些配置文件报错可以先看看第十大点对应的配置文件是否有提示需要修改内容)
1. RuntimeError: xxxxxxxxxxx does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.....

    解决方案：在ultralytics/utils/torch_utils.py中init_seeds函数中把torch.use_deterministic_algorithms里面的True改为False

2. ModuleNotFoundError：No module named xxx

    解决方案：缺少对应的包，先把YOLOV8环境配置的安装命令进行安装一下，如果还是缺少显示缺少包，安装对应的包即可(xxx就是对应的包).

3. OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.  

    解决方案：https://zhuanlan.zhihu.com/p/599835290

4. 训练过程中loss出现nan.

    可以尝试关闭AMP混合精度训练.(train.py中加amp=False)

<a id="a"></a>

5. 固定640x640尺寸的解决方案.

    运行train.py中的时候需要在ultralytics/models/yolo/detect/train.py的DetectionTrainer class中的build_dataset函数中的rect=mode == 'val'改为rect=False.其他模型可以修改回去.  
    运行val.py的时候,把val.py的rect=False注释取消即可.其他模型可以修改回去.  
    运行detect.py中的时候需要在ultralytics/engine/predictor.py找到函数def pre_transform(self, im),在LetterBox中的auto改为False,其他模型可以修改回去.  

6. 多卡训练问题.[参考链接](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training/#multi-gpu-dataparallel-mode-not-recommended:~:text=just%201%20GPU.-,Multi%2DGPU%20DistributedDataParallel%20Mode%20(%E2%9C%85%20recommended),-You%20will%20have)

    python -m torch.distributed.run --nproc_per_node 2 train.py

7. 指定显卡训练.

    1. 使用device参数进行指定.  
    2. 参考链接:https://blog.csdn.net/m0_55097528/article/details/130323125, 简单来说就是用这个来代替device参数.  

8. ValueError: Expected more than 1 value per channel when training, got input size torch.Size...

    如果是在训练情况下的验证阶段出现的话,大概率就是最后一个验证的batch为1,这种情况只需要把验证集多一张或者少一张即可,或者变更batch参数.

9. AttributeError: Can't pickle local object 'EMASlideLoss.__init__.<locals>.<lambda>'

    可以在ultralytics/utils/loss.py中添加import dill as pickle,然后装一下dill这个包.  
    pip install dill -i https://pypi.tuna.tsinghua.edu.cn/simple

10. RuntimeError: Dataset 'xxxxx' error ❌

    将data.yaml中的路径都改为绝对路径.

11. WARNING  NMS time limit 2.100s exceeded

    在ultralytics/utils/ops.py中non_max_suppression函数里面找到这个语句：
        time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    前面的2.0自己改大点即可，大到不会出现这个NMS time limit即可.

12. OSError: [WinError 1455] 页面文件太小，无法完成操作。

    此问题常见于windows训练.一般情况下有两种解决方案:
    1. 把workers设置小点直接不会报错.最小为0
    2. 扩大虚拟内存(可百度).