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


class ImageLabel : public QLabel {
public:
    ImageLabel(QWidget* parent = nullptr, std::string infoText = "");

    void putPixmap(QPixmap pixmap);

    void redrawPixmap();

    void setText(std::string infoText);
    bool moveAway = false;

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;
private:
    QLabel* imageLabel;
    QLabel* textLabel;
    QPixmap _pxmp;

    void handleMouseMove(QMouseEvent* event);
};

