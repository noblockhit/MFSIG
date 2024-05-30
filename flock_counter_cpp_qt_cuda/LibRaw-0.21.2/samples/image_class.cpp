// libs
#include <QDebug>
#include "libraw/libraw.h"
#include <iostream>

// local
#include "image_class.h"

IImage::IImage(std::string path) : path(path) {}

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
