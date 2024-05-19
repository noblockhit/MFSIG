#include "image_class.h"

IImage::IImage(std::string path) : path(path) {}

QPixmap IImage::getPixmap() const {
    // Create a QImage from the raw data
    QImage image(imageData, width, height, QImage::Format_RGB888);

    // Convert the QImage to a QPixmap
    return QPixmap::fromImage(image);
}