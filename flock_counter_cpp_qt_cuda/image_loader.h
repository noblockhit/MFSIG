#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <vector>
#include <QString>

class IImage;

void loadImages(std::vector<IImage*>& images, const QList<QString>& fileNames);

#endif // IMAGE_LOADER_H