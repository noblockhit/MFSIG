// libs
#include <QApplication>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QLabel>
#include <QPixmap>
#include <QFileDialog>
#include <QDebug>

// temporary
#include <chrono>

// local
#include "image_class.h"
#include "image_loader.h"

class MainWindow : public QMainWindow {
    
public:
    MainWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
        setMinimumSize(640, 480);

        // Create the menu bar
        QMenuBar *menuBar = new QMenuBar(this);

        // Add menus to the menu bar
        QMenu *fileMenu = menuBar->addMenu(tr("&File"));
        QMenu *editMenu = menuBar->addMenu(tr("&Edit"));
        QMenu *helpMenu = menuBar->addMenu(tr("&Help"));
        
        // Add actions to the menus
        QAction *newAction = fileMenu->addAction(tr("&New"));
        QAction *openAction = fileMenu->addAction(tr("&Open"));
        QAction *saveAction = fileMenu->addAction(tr("&Save"));
        
        // Connect actions to slots (if needed)
        connect(newAction, &QAction::triggered, this, &MainWindow::newFile);
        connect(openAction, &QAction::triggered, this, &MainWindow::openFile);
        connect(saveAction, &QAction::triggered, this, &MainWindow::saveFile);

        // Set the menu bar to the main window
        setMenuBar(menuBar);

        // Create a central widget
        imageLabel = new QLabel(this);
        setCentralWidget(imageLabel);
    }

private slots:
    void newFile() {
        // Implement new file action here
    }

    void openFile() {
        qInfo() << "Open file action triggered";
        QStringList fileNames = QFileDialog::getOpenFileNames(this, "Open Image File", "", "Images (*.png *.jpeg *.jpg *.arw *.nef *.tif *.tiff *.cr2 *.crw *.dng *.orf *.pef *.rw2 *.srw *.raf *.x3f *.3fr)");
        

        auto start = std::chrono::high_resolution_clock::now();

        // Call the function whose execution time we want to measure
        loadImages(images, fileNames);

        // Get the current time after calling the function
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration of the function call
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        qDebug() << "Execution time: " << duration.count() << " microseconds for " << images.size() << " images";
        
        if (images.empty()) {
            qInfo() << "No images loaded";
            return;
        }
        QPixmap pixmap = images[0]->getPixmap();
        imageLabel->setPixmap(pixmap);
        imageLabel->adjustSize();
    }

    void saveFile() {
        // Implement save file action here
    }

private:
    QLabel *imageLabel;
    std::vector<IImage*> images;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}