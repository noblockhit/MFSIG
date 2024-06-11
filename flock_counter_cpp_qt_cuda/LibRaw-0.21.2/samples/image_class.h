#pragma once

#include <QPixmap>
#include <QLibrary>
#include <QDebug>
#include <libraw/libraw.h>

#include "cl_manager.h"

class IImage {
public:
    IImage(std::string path);


    QPixmap getPixmap() const;

    void loadImage();
    QPixmap getBakedPixmap(int own_thresh, int ngb_thresh);
    int getPos(int x, int y, int channel);
    void setPixel(uint8_t* image, int x, int y, uint8_t r, uint8_t g, uint8_t b);
    void drawCirclePerimeter(uint8_t* image, int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b);
    void drawCircle(uint8_t* image, int cx, int cy, int radius, int thickness, uint8_t r, uint8_t g, uint8_t b);
    std::string getPath() const;
    int getWidth() const;
    int getHeight() const;
    unsigned char* imageData = nullptr;

private:
    std::string path;
    float referenceBrightness = -1;
    int width = -1;
    int height = -1;
};

