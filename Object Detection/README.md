# YOLO11 目标检测 TensorRT c++

本项目参考https://github.com/emptysoal/TensorRT-YOLO11
在windows环境下对其进行封装生成dll与lib，并用c++调用

本项目采用cuda12.5+cudnn9.2+tensorrt8.6.1.6+opencv4.8.1+cmake4+vs2017
需要保证cuda、cudnn、tensorrt版本兼容且正确安装配置

## ONNX 转 TensorRT（trt/engine）
例：TensorRT-8.6.1.6/bin/trtexec –onnx=yolo11n.onnx –saveEngine=yolo11n.trt
tensorrt版本需保持一贯

## camke编译
1. 修改CMakeLists，将tensorrt、opencv等路径修改为自己的路径
2. 使用cmake编译

## vs生成dll与lib
1. 将unistd文件夹下unistd.h文件复制到vs安装目录下.\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include
2. vs打开项目，切换Release x64，对ALL_BUILD进行生成，运行调试，生成yolo.dll与yolo.lib

## vs或qt c++调用
1. vs或qt正确配置包含目录与库目录、链接库
2. 在config.h修改大小、类别等
3. 调用方法见./vstest目录下testyolo项目testyolo.cpp
