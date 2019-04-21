#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_guiYoLoCV.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>

#include "yolo_v2_class.hpp"

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
	//void showEvent(QShowEvent *event);
	//void paintEvent(QPaintEvent *e);
private:
	Ui::guiYoLoCVClass ui;
	QTimer theTimer;
	QPixmap pixmap;
	cv::Mat frame, blob;
	Net net;
	cv::VideoCapture vcap;
	std::vector<Point> vec ;

	const std::string videoStreamAddress = "rtmp://localhost:1935/liveRaw";
};