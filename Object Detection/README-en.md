# YOLO11 detect TensorRT c++

This project refers to https://github.com/emptysoal/TensorRT-YOLO11
It is encapsulated in the Windows environment to generate DLL and LIB files, and is called using C++.

The environment configuration for this project is: cuda12.5+cudnn9.2+tensorrt8.6.1.6+opencv4.8.1+cmake4+vs2017
It is required to ensure that the versions of CUDA, cuDNN, and TensorRT are compatible and correctly installed and configured.

## ONNX -> TensorRT（trt/engine）
e.g. TensorRT-8.6.1.6/bin/trtexec –onnx=yolo11n.onnx –saveEngine=yolo11n.trt
The TensorRT version needs to remain consistent.

## camke
1. Modify the CMakeLists file to update the paths for TensorRT, OpenCV, etc., to your own paths.
2. Use CMake to compile.

## vs generate dll and lib
1. Copy the unistd.h file from the unistd folder to the VS installation directory.\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include
2. Open the project with VS, switch to Release x64, build ALL_BUILD, run and generate yolo.dll and yolo.lib.

## vs/qt c++
1. Correctly configure the include directories, library directories, and linked libraries in VS or Qt.
2. Modify kInputH, kInputW, vClassNames, etc., in config.h.
3. For the usage method, refer to the testyolo.cpp in the ./vstest directory under the testyolo project.
