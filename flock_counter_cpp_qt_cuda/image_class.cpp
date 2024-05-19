// libs
#include <QDebug>

// local
#include "image_class.h"

IImage::IImage(std::string path) : path(path) {}

std::string IImage::getPath() const {
    return path;
}

void IImage::loadImage() {
    QPixmap pixmap(path.c_str());
    if (pixmap.isNull()) {
        qInfo() << "QPixmap cannot load the image file";
        return;
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
