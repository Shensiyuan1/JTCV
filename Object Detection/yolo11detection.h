//Detection.h
#pragma once
#include <opencv2/opencv.hpp>
#include "public.h"
#include "config.h"
#include "types.h"
#include "pch.h"

using namespace nvinfer1;
 
 
 
class CLASS_DECLSPEC Connect
{
public:
	Connect();
	~Connect();
 
public:
	YOLO11* Create_YOLO11_Object();
	void Delete_YOLO11_Object(YOLO11* _bp);
};
 
 
 
class Yolo11Detection :public YOLO11
{
public:
    Yolo11Detection(
    );
    ~Yolo11Detection();
    void Initialize(const std::string trtFile,
        int gpuId = kGpuId,
        float nmsThresh = kNmsThresh,
        float confThresh = kConfThresh,
        int numClass = kNumClass);
    std::vector<Detection> inference(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult);
 
private:
    void get_engine();

private:
    Logger              gLogger;
    std::string         trtFile_;

    int                 numClass_;
    float               nmsThresh_;
    float               confThresh_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    float *             outputData;
    std::vector<void *> vBufferD;
    float *             transposeDevice;
    float *             decodeDevice;

    int                 OUTPUT_CANDIDATES;  // 8400: 80 * 80 + 40 * 40 + 20 * 20
};