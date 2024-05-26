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
    void process() {
        try {
            LibRaw rawProcessor;
            if (rawProcessor.open_file(filePath.toStdString().c_str()) != LIBRAW_SUCCESS) {
                throw std::runtime_error("Failed to open the NEF file");
            }
            if (rawProcessor.unpack() != LIBRAW_SUCCESS) {
                throw std::runtime_error("Failed to unpack the RAW image");
            }
            if (rawProcessor.dcraw_process() != LIBRAW_SUCCESS) {
                throw std::runtime_error("Failed to process the RAW image");
            }
            libraw_processed_image_t *image = rawProcessor.dcraw_make_mem_image();
            if (!image) {
                throw std::runtime_error("Failed to create memory image");
            }
            QImage qimage(image->data, image->width, image->height, QImage::Format_RGB888);
            LibRaw::dcraw_clear_mem(image);
            emit processedImage(qimage.copy());
        } catch (const std::exception &e) {
            emit errorOccurred(e.what());
        }
    }

private:
    QString filePath;
};
