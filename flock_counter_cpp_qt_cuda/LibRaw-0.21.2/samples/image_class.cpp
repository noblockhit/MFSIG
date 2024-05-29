// libs
#include <QDebug>
#include "libraw/libraw.h"
#include <iostream>
#include <thread>
#include <vector>
#include <CL/cl.h>

// local
#include "image_class.h"


bool IImage::isInitialized = false;
cl_platform_id IImage::platformId = nullptr;
cl_device_id IImage::deviceId = nullptr;
cl_context IImage::context = nullptr;
cl_command_queue IImage::commandQueue = nullptr;
cl_program IImage::program = nullptr;
cl_kernel IImage::kernel = nullptr; // Added kernel declaration
std::mutex IImage::initMutex;


IImage::IImage(std::string path) : path(path) {}

bool IImage::initialize() {
    // Example kernel source
    const char* kernelSource = "__kernel void hello(global float* input, global float* output) { \
                                    int gid = get_global_id(0); \
                                    output[gid] = input[gid] * 2.0f; \
                                }";
    return IImage::loadKernel(kernelSource);
}

bool IImage::initializeOpenCL() {
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

bool IImage::loadKernel(const char* source) {
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

bool IImage::executeKernel(const std::vector<float>& inputData, std::vector<float>& outputData) {
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


void IImage::cleanup() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (context) clReleaseContext(context);
}


std::string IImage::getPath() const { return path; }

void IImage::loadImage()
{
    const char* pcstr = path.c_str();
    std::cout << pcstr << std::endl;
    QPixmap pixmap;
    qInfo() << QString::fromUtf8(pcstr);

    if (!pixmap.load(QString::fromUtf8(pcstr)))
    {
        qInfo() << "QPixmap cannot load the image file, trying libraw";

        // Initialize LibRaw processor
        LibRaw rawProcessor;
        rawProcessor.imgdata.params.no_auto_bright = 1; // Disable auto-brightening
        rawProcessor.imgdata.params.use_camera_wb = 1;  // Use camera white balance
        rawProcessor.imgdata.params.output_bps = 8;     // Output 8 bits per sample to speed up processing
        rawProcessor.imgdata.params.output_color = 1;   // Use the simplest color space
        if (rawProcessor.open_file(pcstr) != LIBRAW_SUCCESS)
        {
            qInfo() << "Failed to open the NEF file";
            return;
        }

        // Convert the LibRaw image to QImage
        // Unpack the RAW image
        if (rawProcessor.unpack() != LIBRAW_SUCCESS)
        {
            throw std::runtime_error("Failed to unpack the RAW image");
        }

        // Process the RAW image to get RGB data
        if (rawProcessor.dcraw_process() != LIBRAW_SUCCESS)
        {
            throw std::runtime_error("Failed to process the RAW image");
        }

        // Get the processed image
        libraw_processed_image_t* image = rawProcessor.dcraw_make_mem_image();
        if (!image)
        {
            throw std::runtime_error("Failed to create memory image");
        }

        width = image->width;
        height = image->height;
        imageData = new uint8_t[width * height * 3];
        std::memcpy(imageData, image->data, width * height * 3);
        rawProcessor.recycle();
    }
    else
    {
        QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB888);
        const unsigned char* tempImageData = image.bits();

        // Get the raw pixel data
        width = image.width();
        height = image.height();
        imageData = new uint8_t[width * height * 3];
        std::memcpy(imageData, tempImageData, width * height * 3);
    }
}

QPixmap IImage::getPixmap() const
{
    if (imageData == nullptr)
    {
        qInfo() << "Image data is not loaded";
        return QPixmap();
    }
    if (width == -1 || height == -1)
    {
        qInfo() << "Image dimensions are not loaded";
        return QPixmap();
    }
    QImage image(imageData, width, height, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(image);
    return pixmap;
}
