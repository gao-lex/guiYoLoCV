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
	void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
		int current_det_fps = -1, int current_cap_fps = -1);
	//void showEvent(QShowEvent *event);
	//void paintEvent(QPaintEvent *e);
private:
	Ui::guiYoLoCVClass ui;
	QTimer theTimer;
	QPixmap pixmap;
	cv::Mat frame, blob;
	//Net net;
	cv::VideoCapture vcap;
	std::vector<Point> vec ;

	//const std::string videoStreamAddress = "rtmp://localhost:1935/liveRaw";

	//gpu

	cv::VideoCapture cap;
	cv::Mat cur_frame;
	cv::Size  frame_size;

	std::string  names_file = "obj.names";
	std::string  cfg_file = "yolov3-tiny-me.cfg";
	std::string  weights_file = "yolov3-tiny-me_2000.weights";
	std::string filename = "rtmp://127.0.0.1:1935/liveRaw";
	float const thresh = 0.2;

	bool const save_output_videofile = false;
	bool const send_network = false;
	bool const use_kalman_filter = false;

	bool detection_sync = true;
};