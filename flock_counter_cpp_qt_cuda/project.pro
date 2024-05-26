TEMPLATE = app
TARGET = FlockCounter

# Include paths
INCLUDEPATH += .

# Specify the Qt version and modules
QT += core gui widgets
CONFIG += c++17

# Add additional include paths if necessary
# INCLUDEPATH += D:/opt/QT/6.7.0/mingw_64/include

# Specify the source files
SOURCES += main.cpp
SOURCES += image_class.cpp
SOURCES += image_class.h
SOURCES += image_loader.cpp
SOURCES += image_loader.h

LIBS += $$PWD/LibRaw-0.21.2/buildfiles/debug-x86_64/libraw.lib

INCLUDEPATH += $$PWD/LibRaw-0.21.2/libraw
DEPENDPATH += $$PWD/LibRaw-0.21.2/libraw

message($$INCLUDEPATH)
# message($$LIBS)

# win32: LIBS += -L$$PWD/LibRaw-0.21.2/lib/ -llibraw

# INCLUDEPATH += $$PWD/LibRaw-0.21.2/libraw
# DEPENDPATH += $$PWD/LibRaw-0.21.2/libraw






# win32:CONFIG(release, debug|release): LIBS += -L$$PWD/LibRaw-0.21.2/lib/ -llibraw_static
# else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/LibRaw-0.21.2/lib/ -llibraw_staticd

# INCLUDEPATH += $$PWD/LibRaw-0.21.2
# DEPENDPATH += $$PWD/LibRaw-0.21.2

# win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/LibRaw-0.21.2/lib/liblibraw_static.a
# else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/LibRaw-0.21.2/lib/liblibraw_staticd.a
# else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/LibRaw-0.21.2/lib/libraw_static.lib
# else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/LibRaw-0.21.2/lib/libraw_staticd.lib
