// libs
#include <QDebug>
#include "libraw/libraw.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// local
#include "image_class.h"
#include "cl_manager.h"

// temp
#include <QLabel>
#include <QVBoxLayout>
#include <QDialog>
#include <QPushButton>


IImage::IImage(std::string path) : path(path) {}

std::string IImage::getPath() const { return path; }

int IImage::getPos(int x, int y, int channel) {
    return (y * width + x) * 3 + channel;
}

void IImage::setPixel(uint8_t* image, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        image[getPos(x, y, 0)] = r;
        image[getPos(x, y, 1)] = g;
        image[getPos(x, y, 2)] = b;
    }
}

void IImage::drawCirclePerimeter(uint8_t* image, int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
    int x = radius;
    int y = 0;
    int decisionOver2 = 1 - x;   // Decision criterion divided by 2 evaluated at x=r, y=0

    while (y <= x) {
        setPixel(image, cx + x, cy + y, r, g, b); // Octant 1
        setPixel(image, cx + y, cy + x, r, g, b); // Octant 2
        setPixel(image, cx - y, cy + x, r, g, b); // Octant 3
        setPixel(image, cx - x, cy + y, r, g, b); // Octant 4
        setPixel(image, cx - x, cy - y, r, g, b); // Octant 5
        setPixel(image, cx - y, cy - x, r, g, b); // Octant 6
        setPixel(image, cx + y, cy - x, r, g, b); // Octant 7
        setPixel(image, cx + x, cy - y, r, g, b); // Octant 8
        y++;
        if (decisionOver2 <= 0) {
            decisionOver2 += 2 * y + 1; // Change in decision criterion for y -> y+1
        }
        else {
            x--;
            decisionOver2 += 2 * (y - x) + 1; // Change for y -> y+1, x -> x-1
        }
    }
}

// Function to draw a circle using Midpoint Circle Algorithm
void IImage::drawCircle(uint8_t* image, int cx, int cy, int radius, int thickness, uint8_t r, uint8_t g, uint8_t b) {
    for (int i = 0; i < thickness; ++i) {
        drawCirclePerimeter(image, cx, cy, radius - i, r, g, b);
    }
}

void rotateImage90Clockwise(unsigned char* src, unsigned char*& dst, int width, int height) {
    // Each pixel has 3 bytes (RGB)
    int numPixels = width * height;
    int srcRowSize = width * 3;
    int dstRowSize = height * 3;

    // Allocate memory for the destination image
    dst = new unsigned char[height * width * 3];

    // Rotate the image 90 degrees clockwise
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            // Source pixel position in the original image
            int srcIndex = (row * width + col) * 3;

            // Destination pixel position in the rotated image
            // The row in the original image becomes the column in the rotated image
            // The column in the original image becomes the row in the rotated image
            int dstRow = col;
            int dstCol = height - 1 - row;
            int dstIndex = (dstRow * height + dstCol) * 3;

            // Copy the RGB values
            std::memcpy(&dst[dstIndex], &src[srcIndex], 3);
        }
    }
}

void showPixmapInPopup(const QPixmap& pixmap) {
    QDialog* dialog = new QDialog;
    QVBoxLayout* layout = new QVBoxLayout(dialog);

    QLabel* label = new QLabel;

    // Scale the pixmap to fit within the label while maintaining aspect ratio
    QPixmap scaledPixmap = pixmap.scaled(pixmap.width() / 10, pixmap.height() / 10, Qt::KeepAspectRatio, Qt::SmoothTransformation); // Adjust the size as needed
    label->setPixmap(scaledPixmap);

    layout->addWidget(label);

    QPushButton* button = new QPushButton("Close");
    layout->addWidget(button);
    QObject::connect(button, &QPushButton::clicked, dialog, &QDialog::accept);

    dialog->exec();
}

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
        //rawProcessor.imgdata.params.no_auto_bright = 1; // Disable auto-brightening
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
        // always make the image be in landscape mode using libraw
        //rawProcessor.imgdata.sizes.flip = 0;
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

        if (width < height) {
            unsigned char* pixels = new uint8_t[width * height * 3];
            std::memcpy(pixels, image->data, width * height * 3);
            rotateImage90Clockwise(pixels, imageData, width, height);
            delete[] pixels;
            std::swap(width, height);

        }
        else {

            std::memcpy(imageData, image->data, width * height * 3);
        }
    }
    else
    {
        // rotate the image if its in portrait mode
        QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB888);
        if (image.width() < image.height())
        {
            QTransform transform;
            transform.rotate(90);
            image = image.transformed(transform);
        }

        const unsigned char* tempImageData = image.bits();

        // Get the raw pixel data
        width = image.width();
        height = image.height();
        imageData = new uint8_t[width * height * 3];
        std::memcpy(imageData, tempImageData, width * height * 3);
    }
}

int IImage::getWidth() const { return width; }
int IImage::getHeight() const { return height; }

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

QPixmap IImage::getBakedPixmap(int own_thresh, int ngb_thresh) {
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

    uint8_t* locImageData = new uint8_t[width * height * 3];

    ClManager::executeKernel(imageData, locImageData, width, height, own_thresh, ngb_thresh);
    QImage image(locImageData, width, height, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(image);
    delete[] locImageData;
    return pixmap;
}