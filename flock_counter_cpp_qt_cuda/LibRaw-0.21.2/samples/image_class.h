#pragma once

#include <QPixmap>
#include <QLibrary>
#include <QDebug>
#include <libraw/libraw.h>


class IImage {
public:
    IImage(std::string path);


    QPixmap getPixmap() const;

    void loadImage();
    unsigned char* getBinaryImage();
    std::string getPath() const;

private:
    std::string path;
    float referenceBrightness = -1;
    unsigned char* imageData = nullptr;
    int width = -1;
    int height = -1;
};

