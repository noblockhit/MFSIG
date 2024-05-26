// libraries
#include <QList>
#include <QString>
#include <thread>
#include <vector>
#include <iostream>

// temp
#include <chrono>

// local
#include "image_class.h"
#include "image_loader.h"

void loadImageThread(IImage* image)
{
  // Load the image using the loadImage method of the IImage class
    image->loadImage();
}

void loadImages(std::vector<IImage*>& images, const std::list<std::string> fileNames)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (const std::string fileName : fileNames)
    {
      // Create an IImage object using the fileName
        IImage* image = new IImage(fileName);
        threads.emplace_back(loadImageThread, image);
        // Add the created image object to the vector
        images.push_back(image);
    }

    for (auto& thread : threads)
    {
        thread.join();
    }


    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output the duration
    std::cout << "Took " << duration.count() << " microseconds to load " << fileNames.size() << " images..." << std::endl;
}
