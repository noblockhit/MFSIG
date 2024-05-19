#ifndef IIMAGE_H
#define IIMAGE_H

#include <QPixmap>

class IImage {
public:
    IImage(std::string path);

    QPixmap getPixmap() const;

    void loadImage();
    void bakeImage();

private:
    std::string path;
    float referenceBrightness = -1;
    uint8_t *imageData = nullptr;
    int width = -1;
    int height = -1;
};

#endif IIMAGE_H
