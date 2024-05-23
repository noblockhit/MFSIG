TEMPLATE = app
TARGET = FlockCounter

# Include paths
INCLUDEPATH += .

# Specify the Qt version and modules
QT += core gui widgets
CONFIG += c++17

# Add additional include paths if necessary
INCLUDEPATH += D:/opt/QT/6.7.0/mingw_64/include

# Specify the source files
SOURCES += main.cpp
SOURCES += image_class.cpp
SOURCES += image_class.h
SOURCES += image_loader.cpp
SOURCES += image_loader.h

LIBS += -L $$PWD/LibRaw-0.21.2/bin -lraw

INCLUDEPATH += $$PWD/LibRaw-0.21.2/libraw
DEPENDPATH += $$PWD/LibRaw-0.21.2/libraw
