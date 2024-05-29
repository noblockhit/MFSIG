#pragma once

#include <QPixmap>
#include <QLibrary>
#include <QDebug>
#include <libraw/libraw.h>
#include <CL/cl.h>
#include <mutex>

class IImage {
public:
    IImage(std::string path);


    QPixmap getPixmap() const;

    void loadImage();
    unsigned char* getBinaryImage();
    std::string getPath() const;

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

    std::string path;
    float referenceBrightness = -1;
    unsigned char* imageData = nullptr;
    int width = -1;
    int height = -1;
};

