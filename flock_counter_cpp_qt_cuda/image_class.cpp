// libs
#include <QDebug>
#include <QLibrary>
//#include <libraw.h>

#include <windows.h>


// local
#include "image_class.h"

typedef int (__stdcall *LibRawUnpack) (LibRaw);
typedef int (__stdcall *LibRawDcrawProcess) (LibRaw);



HMODULE libRawDll = LoadLibrary(L"libraw.dll");
LibRawUnpack libRawUnpack = reinterpret_cast<LibRawUnpack>(GetProcAddress(libRawDll, "LibRaw::unpack"));
LibRawDcrawProcess libRawDcrawProcess = reinterpret_cast<LibRawDcrawProcess>(GetProcAddress(libRawDll, "LibRaw::dcraw_process"));

QImage LibRawToQImage(LibRaw &rawProcessor) {
    // Unpack the RAW image
    if (libRawUnpack(rawProcessor) != LIBRAW_SUCCESS) {
        throw std::runtime_error("Failed to unpack the RAW image");
    }

    // Process the RAW image to get RGB data
    if (libRawDcrawProcess(rawProcessor) != LIBRAW_SUCCESS) {
        throw std::runtime_error("Failed to process the RAW image");
    }

    // // Get the processed image
    // libraw_processed_image_t *image = rawProcessor.dcraw_make_mem_image();
    // if (!image) {
    //     throw std::runtime_error("Failed to create memory image");
    // }

    // // Create a QImage from the raw data
    // QImage qimage(image->data, image->width, image->height, QImage::Format_RGB888);

    // // Clean up the LibRaw image data
    // LibRaw::dcraw_clear_mem(image);

    // return qimage.copy(); // Make a deep copy of the image data
    return QImage();
}

IImage::IImage(std::string path) : path(path) {}

std::string IImage::getPath() const {
    return path;
}

void IImage::loadImage() {
    QPixmap pixmap(path.c_str());
    if (pixmap.isNull()) {
        qInfo() << "QPixmap cannot load the image file, trying libraw";

        // Initialize LibRaw processor
        // LibRaw rawProcessor;
        // if (rawProcessor.open_file(path.c_str()) != LIBRAW_SUCCESS) {
        //     qInfo() << "Failed to open the NEF file";
        //     return;
        // }

        // // Convert the LibRaw image to QImage
        // QImage qimage;
        // try {
        //     qimage = LibRawToQImage(rawProcessor);
        // } catch (const std::exception &e) {
        //     qCritical() << "Error:" << e.what();
        //     return;
        // }
        // pixmap = QPixmap::fromImage(qimage);
    } else {
        QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB888);
        const unsigned char *tempImageData = image.bits();

        // Get the raw pixel data
        width = image.width();
        height = image.height();
        imageData = new uint8_t[width * height * 3];
        std::memcpy(imageData, tempImageData, width * height * 3);
    }
}

QPixmap IImage::getPixmap() const {
    if (imageData == nullptr) {
        qInfo() << "Image data is not loaded";
        return QPixmap();
    }
    if (width == -1 || height == -1) {
        qInfo() << "Image dimensions are not loaded";
        return QPixmap();
    }
    QImage image(imageData, width, height, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(image);
    return pixmap;
}

