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

// local
#include "indicator_action.h"

IndicatorAction::IndicatorAction(QString onImagePath, QString offImagePath, QWidget* parent)
    : QAction("", parent) {
    QString appDir = QCoreApplication::applicationDirPath();
    QString offPath = QDir::cleanPath(appDir + QDir::separator() + QString("images") + QDir::separator() + offImagePath);
    QString onPath = QDir::cleanPath(appDir + QDir::separator() + QString("images") + QDir::separator() + onImagePath);
    onIcon = QIcon(onPath);
    offIcon = QIcon(offPath);
    toggleState();
}

void IndicatorAction::setMethod(std::function<void(bool)> func) {
    _func = func;
}

void IndicatorAction::toggleState() {
    // Toggle the color between red and black
    if (enabled) {
        // Construct the full path to the icon file
        this->setIcon(offIcon);
        enabled = false;
    }
    else {
        this->setIcon(onIcon); // Set black indicator icon
        enabled = true;
    }
    if (_func) {
        _func(enabled);
    }
}
