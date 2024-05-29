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
#include "image_label.h"


ImageLabel::ImageLabel(QWidget* parent, std::string infoText) : QLabel(parent) {
    imageLabel = new QLabel(this);
    imageLabel->setMouseTracking(true); // Enable mouse tracking
    imageLabel->installEventFilter(this);

    textLabel = new QLabel(this);
    setText(infoText);
    textLabel->setFrameStyle(QFrame::NoFrame);
    textLabel->setStyleSheet("background: transparent; color: white;");
    textLabel->setAlignment(Qt::AlignCenter); // Center the text
    textLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(imageLabel);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    setMouseTracking(true);
    setAlignment(Qt::AlignCenter);
    setScaledContents(false); // disable automatic scaling
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); // Allow the label to expand
}


void ImageLabel::putPixmap(QPixmap pixmap) {
    _pxmp = pixmap;
    setPixmap(pixmap);
    redrawPixmap();
}


void ImageLabel::redrawPixmap() {
    textLabel->setFixedWidth(width());
    if (!_pxmp.isNull()) {
        setPixmap(_pxmp.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}


void ImageLabel::setText(std::string infoText) {
    textLabel->setText(QString::fromStdString(infoText));
}


void ImageLabel::resizeEvent(QResizeEvent* event) {
    redrawPixmap();
    QLabel::resizeEvent(event);
}

bool ImageLabel::eventFilter(QObject* watched, QEvent* event) {
    if (watched == imageLabel && event->type() == QEvent::MouseMove) {
        auto* mouseEvent = static_cast<QMouseEvent*>(event);
        handleMouseMove(mouseEvent);
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void ImageLabel::handleMouseMove(QMouseEvent* event) {
    int margin = 10;
    int textHeight = textLabel->sizeHint().height();

    if ((event->y() > height() / 2) ^ (!moveAway)) {
        textLabel->move(margin, height() - textHeight - margin); // Bottom
    }
    else {
        textLabel->move(margin, margin); // Top
    }
    textLabel->raise();
    textLabel->show();
}