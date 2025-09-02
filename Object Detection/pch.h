// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。
 
#pragma once
#ifndef PCH_H
#define PCH_H
 
// 添加要在此处预编译的标头
#include "framework.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "types.h"
 
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
 
#define CLASS_DECLSPEC __declspec(dllexport)//表示这里要把类导出//
 
struct Net_config
{
        float gd; // engine threshold
        float gw;  // engine threshold
        const char* netname;
};
 
 
class CLASS_DECLSPEC YOLO11
{
public:
        YOLO11() {};
        virtual ~YOLO11() {};
public:
        virtual void Initialize(const std::string trtFile,
        int gpuId,
        float nmsThresh,
        float confThresh,
        int numClass) = 0;
        virtual std::vector<Detection> inference(cv::Mat& img) = 0;
};
 
#endif //PCH_H