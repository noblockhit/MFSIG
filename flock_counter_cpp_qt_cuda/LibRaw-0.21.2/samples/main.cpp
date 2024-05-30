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
#include <QTextEdit>
#include <QSlider>
#include <QLineEdit>
// temporary
#include <chrono>

// local
#include "image_class.h"
#include "image_loader.h"
#include "indicator_action.h"
#include "image_label.h"
#include "num_setting.h"
#include "cl_manager.h"


class MainWindow : public QMainWindow {

public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setMinimumSize(640, 480);

        // MENU BAR
        QMenuBar* menuBar = new QMenuBar(this);

        QMenu* fileMenu = menuBar->addMenu(tr("&File"));
        QMenu* editMenu = menuBar->addMenu(tr("&Edit"));
        QMenu* helpMenu = menuBar->addMenu(tr("&Help"));

        QAction* openAction = fileMenu->addAction(tr("&Open"));

        connect(openAction, &QAction::triggered, this, &MainWindow::openFile);

        altIndicator = new IndicatorAction(QString("alton.png"), QString("altoff.png"), this);
        auto altOnOffCallback = std::bind(&MainWindow::setMoveAway, this, std::placeholders::_1);
        altIndicator->setMethod(altOnOffCallback);
        menuBar->addAction(altIndicator);
        connect(altIndicator, &QAction::triggered, this, &MainWindow::toggleAltIndicator);

        setMenuBar(menuBar);

        this->installEventFilter(this);

        QWidget* centralWidget = new QWidget(this);
        mainLayout = new QVBoxLayout(centralWidget);

        // SETTINGS
        settingsFrame = new QFrame(this);
        settingsFrame->setFrameShape(QFrame::NoFrame);
        settingsFrame->setFixedHeight(100);

        QHBoxLayout* settingsLayout = new QHBoxLayout;

        blurSetting = new NumSetting(1, 9, "Blur: ", "px");
        ownThreshholdSetting = new NumSetting(0, 255, "Own brightness: ", "");
        neighbourThreshholdSetting = new NumSetting(0, 255, "Neighbour brightness: ", "");
        minContourLengthSetting = new NumSetting(0, 2000, "Min contour length: ", "px");
        maxContourLengthSetting = new NumSetting(1, 2001, "Max contour length: ", "px");


        auto minContourLengthChangeCallback = std::bind(&MainWindow::onMinContourChange, this, std::placeholders::_1);
        minContourLengthSetting->setOnValueChangeCallback(minContourLengthChangeCallback);


        auto maxContourLengthChangeCallback = std::bind(&MainWindow::onMaxContourChange, this, std::placeholders::_1);
        maxContourLengthSetting->setOnValueChangeCallback(maxContourLengthChangeCallback);

        settingsLayout->addWidget(blurSetting);
        settingsLayout->addWidget(ownThreshholdSetting);
        settingsLayout->addWidget(neighbourThreshholdSetting);
        settingsLayout->addWidget(minContourLengthSetting);
        settingsLayout->addWidget(maxContourLengthSetting);


        settingsFrame->setLayout(settingsLayout);

            // MAIN FRAME
        imageLabel = new ImageLabel(this);
        imageLabel->setFrameShape(QFrame::NoFrame);


        // ASSIGN
        mainLayout->addWidget(settingsFrame);
        mainLayout->addWidget(imageLabel);

        this->setCentralWidget(centralWidget);
    }


private slots:
    void onMinContourChange(double newVal) {
        if (newVal >= maxContourLengthSetting->getValue()) {
            maxContourLengthSetting->setValue(newVal + 1);
        }
    }


    void onMaxContourChange(double newVal) {
        if (newVal <= minContourLengthSetting->getValue()) {
            minContourLengthSetting->setValue(newVal - 1);
        }
    }

    void setMoveAway(bool val) {
        imageLabel->moveAway = val;
    }
    void toggleAltIndicator() {
        altIndicator->toggleState();
    }
    void openFile() {
        qInfo() << "Open file action triggered";
        QStringList fileNames = QFileDialog::getOpenFileNames(this, "Open Image File", "", "Images (*.png *.jpeg *.jpg *.arw *.nef *.tif *.tiff *.cr2 *.crw *.dng *.orf *.pef *.rw2 *.srw *.raf *.x3f *.3fr)");

        std::list<std::string> fileNameList;
        for (QString qString : fileNames)
        {
            fileNameList.push_back(qString.toUtf8().constData());
        }

        loadImages(images, fileNameList);

        if (images.empty()) {
            qInfo() << "No images loaded";
            return;
        }

        updatePixmap();
    }

    void updatePixmap() {
        imageLabel->putPixmap(images[shownImageIndex]->getPixmap());
        imageLabel->setText(images[shownImageIndex]->getPath() + " (" + std::to_string(shownImageIndex + 1) + " / " + std::to_string(images.size()) + ")");
    }

private:
    int shownImageIndex = 0;
    QVBoxLayout* mainLayout;
    IndicatorAction* altIndicator;
    QFrame* settingsFrame;
    ImageLabel* imageLabel;
    std::vector<IImage*> images;

    NumSetting* blurSetting;
    NumSetting* ownThreshholdSetting;
    NumSetting* neighbourThreshholdSetting;
    NumSetting* minContourLengthSetting;
    NumSetting* maxContourLengthSetting;

protected:
    bool eventFilter(QObject* obj, QEvent* event) override {
        if (event->type() == QEvent::KeyPress) {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Alt) {
                std::cout << "altkey";
                altIndicator->toggleState(); // Toggle color when Alt key is pressed
                return true;
            }
            else if (keyEvent->key() == Qt::Key_Left && images.size() > 0) {
                shownImageIndex -= 1;
                if (shownImageIndex < 0) {
                    shownImageIndex = images.size() - 1;
                }
                updatePixmap();
            }
            else if (keyEvent->key() == Qt::Key_Right && images.size() > 0) {
                shownImageIndex = (shownImageIndex + 1) % images.size();
                updatePixmap();
            }
        }
        return QMainWindow::eventFilter(obj, event);
    }
};

int main(int argc, char* argv[]) {
    ClManager::initializeOpenCL();
    if (!ClManager::initialize()) {
        qInfo() << "failed to initialize";
    }
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();

    /*

    std::vector<float> inputData = {};
    for (int i = 1; i < 100000; i++) {
        inputData.push_back(i);
    }
    for (int i = 0; i < 5; i++) {
        std::vector<float> outputData;


        auto start = std::chrono::high_resolution_clock::now();

        if (!ClManager::executeKernel(inputData, outputData)) {
            std::cerr << "Failed to execute kernel." << std::endl;
            return -1;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Kernel execution time: " << duration.count() << " milliseconds" << std::endl;


        inputData = outputData;
    }
    */
    return 0;
}
