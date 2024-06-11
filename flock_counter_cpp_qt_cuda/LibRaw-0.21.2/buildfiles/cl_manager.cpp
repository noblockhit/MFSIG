// libs
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <CL/cl.h>

// local
#include "cl_manager.h"


bool ClManager::isInitialized = false;
cl_platform_id ClManager::platformId = nullptr;
cl_device_id ClManager::deviceId = nullptr;
cl_context ClManager::context = nullptr;
cl_command_queue ClManager::commandQueue = nullptr;
cl_program ClManager::program = nullptr;
std::vector<cl_kernel> ClManager::kernels;
std::mutex ClManager::initMutex;


bool ClManager::initialize() {
    FILE* fp;
    char* source_str;
    size_t source_size, program_size;

    fp = fopen("source.cl", "rb");
    if (!fp) {
        printf("Failed to load kernel\n");
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    source_str = (char*)malloc(program_size + 1);
    source_str[program_size] = '\0';
    fread(source_str, sizeof(char), program_size, fp);
    fclose(fp);


    if (atexit(cleanup) != 0) {
        fprintf(stderr, "Unable to register opencl cleanup.\n");
        return EXIT_FAILURE;
    }

    return ClManager::loadKernel(source_str);
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
        std::cout << "Failed to get platform ID. Error: " << err << std::endl;
        return false;
    }

    // Get devices count for the platform
    cl_uint numDevices;
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cout << "No devices found or failed to get device count. Error: " << err << std::endl;
        return false;
    }

    // Get device IDs for the platform
    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to get device IDs. Error: " << err << std::endl;
        return false;
    }

    // Select the desired device (for this example, let's assume the first device)
    deviceId = devices[0];

    // Print device name
    size_t deviceNameSize;
    clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize);
    std::vector<char> deviceName(deviceNameSize);
    clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr);
    std::cout << "Device: " << deviceName.data() << std::endl;

    // Create context
    context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create context. Error: " << err << std::endl;
        return false;
    }

    // Create command queue
    commandQueue = clCreateCommandQueue(context, deviceId, 0, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to create command queue. Error: " << err << std::endl;
        clReleaseContext(context); // Clean up context if command queue creation fails
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
    kernels.push_back(clCreateKernel(program, "threshold_approach", &err));
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        return false;
    }

    return true;
}

bool ClManager::executeKernel(const uint8_t* inputData, uint8_t* outputData, int width, int height, uint8_t own_thresh, uint8_t ngb_thresh) {
    cl_int err;
    auto inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * width * height * 3, (void*)inputData, &err);
    auto outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * width * height * 3, nullptr, &err);

    // Set kernel arguments
    clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernels[0], 2, sizeof(int), &width);
    clSetKernelArg(kernels[0], 3, sizeof(int), &height);
    clSetKernelArg(kernels[0], 4, sizeof(uint8_t), &own_thresh);
    clSetKernelArg(kernels[0], 5, sizeof(uint8_t), &ngb_thresh);

    
    size_t localWorkSize[2] = { 16, 16 }; // Adjust as per your hardware capabilities
    int gwsWidth = 0;
    while (gwsWidth < width) {
        gwsWidth += localWorkSize[0];
    }

    int gwsHeight = 0;
    while (gwsHeight < height) {
        gwsHeight += localWorkSize[1];
    }

    size_t globalWorkSize[2] = { gwsWidth, gwsHeight };

    err = clEnqueueNDRangeKernel(commandQueue, kernels[0], 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
    // Handle error
        std::cerr << "Error enqueuing the kernel: " << err << std::endl;
    }
    // Wait for the kernel to flush

    err = clFlush(commandQueue);
    if (err != CL_SUCCESS) {
        // Handle error
        std::cerr << "Error flushing the command queue: " << err << std::endl;
    }

    // Wait for the kernel to finish execution
    err = clFinish(commandQueue);
    if (err != CL_SUCCESS) {
        // Handle error
        std::cerr << "Error finishing the command queue: " << err << std::endl;
    }

    // Read the results back from the output buffer
    err = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, sizeof(uint8_t) * width * height * 3, outputData, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Handle error
        std::cerr << "Error reading the output buffer: " << err << std::endl;
    }

    // Release OpenCL resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    // Release other resources as necessary

    return true;
}


void ClManager::cleanup() {
    for (auto kernel : kernels) {
        clReleaseKernel(kernel);
    }
    if (program) clReleaseProgram(program);
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (context) clReleaseContext(context);
}
