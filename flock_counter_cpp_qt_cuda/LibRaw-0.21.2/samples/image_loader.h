#pragma once

#include <vector>
#include <QString>

class IImage;

void loadImages(std::vector<IImage*>& images, const std::list<std::string> fileNames);
