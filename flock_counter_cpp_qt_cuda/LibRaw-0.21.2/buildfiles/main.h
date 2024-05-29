#pragma once


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
        auto callback = std::bind(&MainWindow::setMoveAway, this, std::placeholders::_1);
        altIndicator->setMethod(callback);
        menuBar->addAction(altIndicator);
        connect(altIndicator, &QAction::triggered, this, &MainWindow::toggleAltIndicator);

        setMenuBar(menuBar);

        this->installEventFilter(this);

        QWidget* centralWidget = new QWidget(this);
        mainLayout = new QVBoxLayout(centralWidget);

        // SETTINGS
        settingsFrame = new QFrame(this);
        settingsFrame->setFrameShape(QFrame::Box);
        settingsFrame->setFixedHeight(150);

        QVBoxLayout* settingsLayout = new QVBoxLayout;


        settingsFrame->setLayout(settingsLayout);

        // MAIN FRAME
        imageLabel = new ImageLabel(this);
        imageLabel->setFrameShape(QFrame::Box);




        // ASSIGN
        mainLayout->addWidget(settingsFrame);
        mainLayout->addWidget(imageLabel);

        this->setCentralWidget(centralWidget);
    }


private slots:
    void updateBlurLineEdit(int val) {
        blurLineEdit->setText("Blur: " + QString::number(val) + "px");
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
        QPixmap pixmap = images[0]->getPixmap();
        imageLabel->putPixmap(pixmap);
        imageLabel->setText(images[0]->getPath().c_str());
    }

private:
    QVBoxLayout* mainLayout;
    IndicatorAction* altIndicator;
    QFrame* settingsFrame;
    ImageLabel* imageLabel;
    std::vector<IImage*> images;


protected:
    bool eventFilter(QObject* obj, QEvent* event) override {
        if (event->type() == QEvent::KeyPress) {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Alt) {
                std::cout << "altkey";
                altIndicator->toggleState(); // Toggle color when Alt key is pressed
                return true;
            }
        }
        return QMainWindow::eventFilter(obj, event);
    }
};