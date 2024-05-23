#ifndef IIMAGE_H
#define IIMAGE_H

#include <QPixmap>
#include <QLibrary>
#include <QDebug>
#include <libraw.h>

bool initializeLibRaw();

class IImage {
public:
    IImage(std::string path);

    QPixmap getPixmap() const;

    void loadImage();
    void bakeImage();
    std::string getPath() const;

private:
    std::string path;
    float referenceBrightness = -1;
    unsigned char *imageData = nullptr;
    int width = -1;
    int height = -1;
};

#endif //IIMAGE_H
