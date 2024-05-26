#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <vector>
#include <QString>

class IImage;

void loadImages(std::vector<IImage*>& images, const std::list<std::string> fileNames);

#endif // IMAGE_LOADER_H