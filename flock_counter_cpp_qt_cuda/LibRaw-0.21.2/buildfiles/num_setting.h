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
#include <string>
#include <optional>
#include <QObject>


class NumSetting: public QWidget {
public:
    NumSetting(double from, double to, std::string prefix, std::string suffix, std::optional<double> def = std::nullopt, double step = 1);

    void setValue(double val);
    double getValue();
    int getValueAsInt();
    void setOnValueChangeCallback(std::function<void(double)> onValueChangeCallback);

    QVBoxLayout* layout;
    QSlider* slider;
    QLineEdit* lineEdit;

private:
    void update();
    void onSlider(int val);
    void onLineEdit(const QString& text);
    void onLineEditFocusOut();

    bool lastUpdateFromOnLineEdit = false;
    double _from;
    double _to;
    double _step;
    double _val = -1;
    std::string _prefix;
    std::string _suffix;
    std::function<void(double)> _onValueChangeCallback;
};
