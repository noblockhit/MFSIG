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


class IndicatorAction : public QAction {
public:
    IndicatorAction(QString onImagePath, QString offImagePath, QWidget* parent = nullptr);

    void setMethod(std::function<void(bool)> func);


    void toggleState();
private:
    QIcon onIcon;
    QIcon offIcon;
    bool enabled = true;
    std::function<void(bool)> _func;
};
