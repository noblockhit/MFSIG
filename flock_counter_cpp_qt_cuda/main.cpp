#include <QApplication>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QLabel>
#include <QPixmap>
#include <QFileDialog>
#include <QDebug>


class MainWindow : public QMainWindow {
    
public:
    MainWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
        setMinimumSize(640, 480);

        // Create the menu bar
        QMenuBar *menuBar = new QMenuBar(this);

        // Add menus to the menu bar
        QMenu *fileMenu = menuBar->addMenu(tr("&File"));
        QMenu *editMenu = menuBar->addMenu(tr("&Edit"));
        QMenu *helpMenu = menuBar->addMenu(tr("&Help"));
        
        // Add actions to the menus
        QAction *newAction = fileMenu->addAction(tr("&New"));
        QAction *openAction = fileMenu->addAction(tr("&Open"));
        QAction *saveAction = fileMenu->addAction(tr("&Save"));
        
        // Connect actions to slots (if needed)
        connect(newAction, &QAction::triggered, this, &MainWindow::newFile);
        connect(openAction, &QAction::triggered, this, &MainWindow::openFile);
        connect(saveAction, &QAction::triggered, this, &MainWindow::saveFile);

        // Set the menu bar to the main window
        setMenuBar(menuBar);

        // Create a central widget
        imageLabel = new QLabel(this);
        setCentralWidget(imageLabel);
    }

private slots:
    void newFile() {
        // Implement new file action here
    }

    void openFile() {
        qInfo() << "Open file action triggered";
        QStringList fileNames = QFileDialog::getOpenFileNames(this, "Open Image File", "", "Images (*.png *.jpeg *.jpg *.arw *.nef *.tif *.tiff *.cr2 *.crw *.dng *.orf *.pef *.rw2 *.srw *.raf *.x3f *.3fr)");
        // if (fileNames.isEmpty()) {
        //     qInfo() << "No files selected";
        //     return;
        // }
        // QPixmap pixmap(fileNames[0]);
        // if (pixmap.isNull()) {
        //     qInfo() << "QPixmap cannot load the image file";
        //     return;
        // }
        // imageLabel->setPixmap(pixmap);
        // imageLabel->adjustSize();

        
    }

    void saveFile() {
        // Implement save file action here
    }

private:
    QLabel *imageLabel;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}