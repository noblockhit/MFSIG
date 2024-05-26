// libs
#include <QApplication>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QLabel>
#include <QVBoxLayout>
#include <QResizeEvent>
#include <QPixmap>
#include <QFileDialog>
#include <QDebug>
#include <vector>
#include <iostream>
#include <thread>

// temporary
#include <chrono>

// local
#include "image_class.h"
#include "image_loader.h"

class ImageLabel : public QLabel {
public:
    ImageLabel(QWidget* parent = nullptr) : QLabel(parent) {
        setAlignment(Qt::AlignCenter);
        setScaledContents(false); // disable automatic scaling
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); // Allow the label to expand
    }

    void putPixmap(QPixmap pixmap) {
        _pxmp = pixmap;
        setPixmap(pixmap);
        redrawPixmap();
    }

    void redrawPixmap() {
        if (!_pxmp.isNull()) {
            setPixmap(_pxmp.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }

protected:
    void resizeEvent(QResizeEvent* event) override {
        redrawPixmap();
        QLabel::resizeEvent(event);
    }

private:
    QPixmap _pxmp;
};


class MainWindow : public QMainWindow {

public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setMinimumSize(640, 480);

        // Create the menu bar
        QMenuBar* menuBar = new QMenuBar(this);

        // Add menus to the menu bar
        QMenu* fileMenu = menuBar->addMenu(tr("&File"));
        QMenu* editMenu = menuBar->addMenu(tr("&Edit"));
        QMenu* helpMenu = menuBar->addMenu(tr("&Help"));

        // Add actions to the menus
        QAction* openAction = fileMenu->addAction(tr("&Open"));

        // Connect actions to slots (if needed)
        connect(openAction, &QAction::triggered, this, &MainWindow::openFile);

        // Set the menu bar to the main window
        setMenuBar(menuBar);

        // Create a central widget
        QWidget* centralWidget = new QWidget(this);
        mainLayout = new QVBoxLayout(centralWidget);
        settingsFrame = new QFrame(this);
        settingsFrame->setFrameShape(QFrame::Box);
        settingsFrame->setFixedHeight(50);

        imageLabel = new ImageLabel(this);
        imageLabel->setFrameShape(QFrame::Box);

        mainLayout->addWidget(settingsFrame);
        mainLayout->addWidget(imageLabel);

        this->setCentralWidget(centralWidget);
    }

private slots:
    void openFile() {
        qInfo() << "Open file action triggered";
        QStringList fileNames = QFileDialog::getOpenFileNames(this, "Open Image File", "", "Images (*.png *.jpeg *.jpg *.arw *.nef *.tif *.tiff *.cr2 *.crw *.dng *.orf *.pef *.rw2 *.srw *.raf *.x3f *.3fr)");

        std::list<std::string> fileNameList;
        for (QString qString : fileNames)
        {
            fileNameList.push_back(qString.toUtf8().constData());
        }
        //auto fileNameList = new std::list<std::string>{"D:/Images/Abi2023/AA_personen_zeugnisuebergabe/DSC_3045.NEF"};
        loadImages(images, fileNameList);

        if (images.empty()) {
            qInfo() << "No images loaded";
            return;
        }
        QPixmap pixmap = images[0]->getPixmap();
        imageLabel->putPixmap(pixmap);

    }

private:
    QVBoxLayout* mainLayout;
    QFrame* settingsFrame;
    ImageLabel* imageLabel;
    std::vector<IImage*> images;
};

int main(int argc, char* argv[]) {
    //initializeLibRaw();
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
