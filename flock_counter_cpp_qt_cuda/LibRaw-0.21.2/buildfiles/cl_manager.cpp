// libs
#include <iostream>

// local
#include "cl_manager.h"


bool ClManager::isInitialized = false;
cl_platform_id ClManager::platformId = nullptr;
cl_device_id ClManager::deviceId = nullptr;
cl_context ClManager::context = nullptr;
cl_command_queue ClManager::commandQueue = nullptr;
cl_program ClManager::program = nullptr;
cl_kernel ClManager::kernel = nullptr;
std::mutex ClManager::initMutex;


bool ClManager::initialize() {
    // Example kernel source
    const char* kernelSource = "__kernel void hello(global float* input, global float* output) { \
                                    int gid = get_global_id(0); \
                                    output[gid] = input[gid] * 2.0f; \
                                }";
    return ClManager::loadKernel(kernelSource);
}

bool ClManager::initializeOpenCL() {
    std::lock_guard<std::mutex> lock(initMutex);
    if (isInitialized) {
        return true;
    }

    cl_int err;

    // Get platform
    err = clGetPlatformIDs(1, &platformId, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform ID." << std::endl;
        return false;
    }

    // Get device
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device ID." << std::endl;
        return false;
    }

    // Create context
    context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context." << std::endl;
        return false;
    }

    // Create command queue
    commandQueue = clCreateCommandQueue(context, deviceId, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        return false;
    }

    isInitialized = true;
    return true;
}

bool ClManager::loadKernel(const char* source) {
    cl_int err;

    // Create program from source
    program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program." << std::endl;
        return false;
    }

    // Build program
    err = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program." << std::endl;

        // Print build log
        size_t logSize;
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << log.data() << std::endl;

        return false;
    }

    // Create kernel
    kernel = clCreateKernel(program, "hello", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return false;
    }

    return true;
}

bool ClManager::executeKernel(const std::vector<float>& inputData, std::vector<float>& outputData) {
    cl_int err;

    // Create memory objects
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputData.size(), (void*)inputData.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create input buffer." << std::endl;
        return false;
    }

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * inputData.size(), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create output buffer." << std::endl;
        clReleaseMemObject(inputBuffer);
        return false;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel argument 0." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        return false;
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel argument 1." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        return false;
    }

    // Enqueue kernel for execution
    size_t globalSize = inputData.size();
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        return false;
    }

    // Read the results back
    outputData.resize(inputData.size());
    err = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, sizeof(float) * inputData.size(), outputData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read output buffer." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        return false;
    }

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);

    return true;
}


void ClManager::cleanup() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (context) clReleaseContext(context);
}