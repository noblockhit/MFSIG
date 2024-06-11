#pragma once

#include <vector>
#include <CL/cl.h>
#include <mutex>


class ClManager {
public:
    static bool initialize();
    static void cleanup();
    static bool initializeOpenCL();
    static void cleanupOpenCL();
    static bool executeKernel(const uint8_t* inputData, uint8_t* outputData, int width, int height, uint8_t own_thresh, uint8_t nbg_thresh);

private:
    static bool isInitialized;
    static cl_platform_id platformId;
    static cl_device_id deviceId;
    static cl_context context;
    static cl_command_queue commandQueue;
    static cl_program program;
    static std::vector<cl_kernel> kernels;
    static std::mutex initMutex;

    static bool loadKernel(const char* source);
};