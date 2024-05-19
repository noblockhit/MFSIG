
// local
#include "image_class.h"
#include "image_loader.h"

void loadImages(std::vector<IImage*>& images, const QList<QString>& fileNames) {
    for (const QString& fileName : fileNames) {
        // Create an IImage object using the fileName
        IImage* image = new IImage(fileName.toStdString());
        image->loadImage();
        // Add the created image object to the vector
        images.push_back(image);
    }
}
