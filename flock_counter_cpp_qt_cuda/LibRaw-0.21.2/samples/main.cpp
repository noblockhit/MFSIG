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
        ownThreshholdSetting = new NumSetting(0, 254, "Own brightness: ", "");
        neighbourThreshholdSetting = new NumSetting(1, 255, "Neighbour brightness: ", "");
        minContourLengthSetting = new NumSetting(0, 2000, "Min contour length: ", "px");
        maxContourLengthSetting = new NumSetting(1, 2001, "Max contour length: ", "px");

        auto redrawCallback = std::bind(&MainWindow::callRedraw, this, std::placeholders::_1);
        auto ownThreshChangeCallback = std::bind(&MainWindow::onOwnThreshChange, this, std::placeholders::_1);
        auto ngbThreshChangeCallback = std::bind(&MainWindow::onNgbThreshChange, this, std::placeholders::_1);
        auto minContourLengthChangeCallback = std::bind(&MainWindow::onMinContourChange, this, std::placeholders::_1);
        auto maxContourLengthChangeCallback = std::bind(&MainWindow::onMaxContourChange, this, std::placeholders::_1);
        blurSetting->setOnValueChangeCallback(redrawCallback);
        ownThreshholdSetting->setOnValueChangeCallback(ownThreshChangeCallback);
        neighbourThreshholdSetting->setOnValueChangeCallback(ngbThreshChangeCallback);
        minContourLengthSetting->setOnValueChangeCallback(minContourLengthChangeCallback);
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
    void onOwnThreshChange(double newVal) {
        if (newVal >= neighbourThreshholdSetting->getValue()) {
            neighbourThreshholdSetting->setValue(newVal + 1);
        }
        updatePixmap();
    }

    void onNgbThreshChange(double newVal) {
        if (newVal <= ownThreshholdSetting->getValue()) {
            ownThreshholdSetting->setValue(newVal - 1);
        }
        updatePixmap();
    }

    void onMinContourChange(double newVal) {
        if (newVal >= maxContourLengthSetting->getValue()) {
            maxContourLengthSetting->setValue(newVal + 1);
        }
        updatePixmap();
    }


    void onMaxContourChange(double newVal) {
        if (newVal <= minContourLengthSetting->getValue()) {
            minContourLengthSetting->setValue(newVal - 1);
        }
        updatePixmap();
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
        // measure the time
        auto start = std::chrono::high_resolution_clock::now();

        imageLabel->putPixmap(images[shownImageIndex]->getBakedPixmap(ownThreshholdSetting->getValueAsInt(), neighbourThreshholdSetting->getValueAsInt()));
        imageLabel->setText(images[shownImageIndex]->getPath() + " (" + std::to_string(shownImageIndex + 1) + " / " + std::to_string(images.size()) + ")");
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Took " << duration.count() << " microseconds to update pixmap..." << std::endl;
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
    void callRedraw(double _) {
        updatePixmap();
    }

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
}
