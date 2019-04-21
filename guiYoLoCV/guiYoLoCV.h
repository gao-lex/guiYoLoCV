#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_guiYoLoCV.h"


#include <QDebug>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QCloseEvent>
#include <QMessageBox>
#include <QTimer>

#include <fstream>
#include <sstream>
#include <iostream>
//#include <QElapsedTimer>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
//using namespace dnn;
using namespace cv;
using namespace cv::dnn;

class guiYoLoCV : public QMainWindow
{
	Q_OBJECT

public:
	guiYoLoCV(QWidget *parent = Q_NULLPTR);

public slots:
	void updateImage();

protected:
	void showEvent(QShowEvent *event);
	//void paintEvent(QPaintEvent *e);
private:
	Ui::guiYoLoCVClass ui;
	QTimer theTimer;
	QPixmap pixmap;
	cv::Mat frame, blob;
	Net net;
	cv::VideoCapture vcap;
	const std::string videoStreamAddress = "rtmp://localhost:1935/liveRaw";
};