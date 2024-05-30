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
#include <string>
#include <optional>


// local
#include "num_setting.h"

class FocusLoseCallbackLineEdit : public QLineEdit {
public:
    explicit FocusLoseCallbackLineEdit(QWidget* parent = nullptr) : QLineEdit(parent) {}

protected:
    void focusOutEvent(QFocusEvent* e) override {
        QLineEdit::focusOutEvent(e);
        emit focusOut();
    }

signals:
    void focusOut() {

    }
};


class ArrowKeyIgnoringSlider : public QSlider {
public:
    using QSlider::QSlider; // Inherit constructors

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Left || event->key() == Qt::Key_Right) {
            event->ignore(); // Ignore the key event
        }
        else {
            QSlider::keyPressEvent(event); // Handle other keys normally
        }
    }
};

NumSetting::NumSetting(double from, double to, std::string prefix, std::string suffix, std::optional<double> def, double step) : QWidget() {
    _from = from;
    _to = to;
    _step = step;
    _prefix = prefix;
    _suffix = suffix;

    if (def.has_value()) {
        std::cout << "def has value" << std::endl;
        _val = *def;
    }
    else {
        _val = _from;
    }

    layout = new QVBoxLayout(this);

    slider = new ArrowKeyIgnoringSlider(Qt::Horizontal, this);
    slider->setRange(0, (_to - _from) / _step);
    slider->setValue((_val - _from) / _step);

    lineEdit = new FocusLoseCallbackLineEdit(this);

    layout->addWidget(lineEdit);
    layout->addWidget(slider);

    connect(slider, &QSlider::valueChanged, this, &NumSetting::onSlider);
    connect(lineEdit, &QLineEdit::textChanged, this, &NumSetting::onLineEdit);
    connect(lineEdit, &QLineEdit::editingFinished, this, &NumSetting::onLineEditFocusOut);
    update();
}

void NumSetting::update() {
    lineEdit->setText(QString::fromStdString(_prefix) + QString::number(_val) + QString::fromStdString(_suffix));
}

void NumSetting::setValue(double val) {
    _val = val;
    update();
}

double NumSetting::getValue() {
    return _val;
}

void NumSetting::onSlider(int val) {
    _val = _from + ((double)val) * _step;
    if (!lastUpdateFromOnLineEdit) {
        update();
        if (_onValueChangeCallback) {
            _onValueChangeCallback(_val);
        }
    }
    lastUpdateFromOnLineEdit = false;
}

void NumSetting::onLineEdit(const QString& text) {
    QString parsed = text.mid(_prefix.length()).left(text.length() - _prefix.length() - _suffix.length());
    double value = parsed.toDouble();

    // Check if the conversion was successful
    if (value != 0.0 || (value == 0.0 && (parsed == "0" || parsed.startsWith("0.")))) {
        int tempPrevVal = slider->value();
        lastUpdateFromOnLineEdit = true;
        slider->setValue((value - _from) / _step);
        if (tempPrevVal == slider->value()) {
            lastUpdateFromOnLineEdit = false;
        }
    }
}

void NumSetting::onLineEditFocusOut() {
    if (_onValueChangeCallback) {
        _onValueChangeCallback(_val);
    }
}

void NumSetting::setOnValueChangeCallback(std::function<void(double)> onValueChangeCallback) {
    _onValueChangeCallback = onValueChangeCallback;
}