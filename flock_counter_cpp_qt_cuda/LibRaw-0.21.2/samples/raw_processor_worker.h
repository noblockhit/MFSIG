#include <QObject>
#include <QImage>
#include <libraw/libraw.h>
#include <stdexcept>

class RawProcessorWorker : public QObject {
    Q_OBJECT

public:
    explicit RawProcessorWorker(const QString &filePath, QObject *parent = nullptr)
        : QObject(parent), filePath(filePath) {}
    
signals:
    void processedImage(const QImage &image);
    void errorOccurred(const QString &error);

public slots:
    void process() {}
}