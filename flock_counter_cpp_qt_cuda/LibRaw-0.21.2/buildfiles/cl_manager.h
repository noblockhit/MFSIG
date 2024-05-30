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
    static bool executeKernel(const std::vector<float>& inputData, std::vector<float>& outputData);

private:
    static bool isInitialized;
    static cl_platform_id platformId;
    static cl_device_id deviceId;
    static cl_context context;
    static cl_command_queue commandQueue;
    static cl_program program;
    static cl_kernel kernel;
    static std::mutex initMutex;

    static bool loadKernel(const char* source);
};