#include "guiYoLoCV.h"

std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba) {
	return bbox_vect;
}

#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>


#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)

#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")


//参数一：图片
//参数二：vector<bbox_t> result_vec
void guiYoLoCV::draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
	int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	// 总共人的个数提醒 / 为人概率提醒


	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);

		//越界逻辑


		//越界警示





		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);
			std::string coords_3d;
			if (!std::isnan(i.z_3d)) {
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
				coords_3d = ss.str();
				cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
				int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
				if (max_width_3d > max_width) max_width = max_width_3d;
			}

			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
		}
	}
}


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
	if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
	for (auto &i : result_vec) {
		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
		std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
			<< ", w = " << i.w << ", h = " << i.h
			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
	}
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
	const bool sync;
	std::atomic<T *> a_ptr;
public:

	void send(T const& _obj) {
		T *new_ptr = new T;
		*new_ptr = _obj;
		if (sync) {
			while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
		std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
	}

	T receive() {
		std::unique_ptr<T> ptr;
		do {
			while (!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
			ptr.reset(a_ptr.exchange(NULL));
		} while (!ptr);
		T obj = *ptr;
		return obj;
	}

	bool is_object_present() {
		return (a_ptr.load() != NULL);
	}

	send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
	{}
};

// 初始化参数
//float confThreshold = 0.3; // Confidence threshold
//float nmsThreshold = 0.4;  // Non-maximum suppression threshold
//int inpWidth = 416;  // Width of network's input image
//int inpHeight = 416; // Height of network's input image
//vector<std::string> classes;
//// Remove the bounding boxes with low confidence using non-maxima suppression
//void postprocess(Mat& frame, const vector<Mat>& out);
//// Draw the predicted bounding box
//void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
//// Get the names of the output layers
//vector<String> getOutputsNames(const Net& net);



guiYoLoCV::guiYoLoCV(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);


	// 定时刷新视频
	connect(&theTimer, &QTimer::timeout, this, &guiYoLoCV::updateImage);

	cap.open(filename);
	cap >> cur_frame;
	frame_size = cur_frame.size();


	//vcap.open("VIRAT_S_010201_00_000000_000053.mp4");
	//vcap.open("rtmp://127.0.0.1:1935/liveRaw");
	//vcap.set(CAP_PROP_BUFFERSIZE, 1);

	// 自适应视频窗口大小
	ui.videoLabel->resize(cur_frame.cols, cur_frame.rows);

	// 定时开始
	theTimer.start(33);







	// cpu
	//string classesFile = "obj.names";
	//ifstream ifs(classesFile.c_str());
	//string line;
	//while (getline(ifs, line))
	//	classes.push_back(line);
	//cv::String modelConfiguration = "yolov3-tiny-me.cfg";
	//cv::String modelWeights = "yolov3-tiny-me_2000.weights";
	//// Load the network
	//net = readNetFromDarknet(modelConfiguration, modelWeights);
	////net.setPreferableBackend(DNN_BACKEND_OPENCV);
	//net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	////net.setPreferableTarget(DNN_TARGET_CPU);
	//net.setPreferableTarget(DNN_TARGET_OPENCL);



	//加载限制区域信息
	FILE * fp = fopen("region.cfg", "r");
	int x, y;
	while (fscanf(fp, "%d,%d", &x, &y) == 2)
		vec.push_back(Point(x, y));

}




//void guiYoLoCV::paintEvent(QPaintEvent * e)
//{
//
//
//}

struct detection_data_t {
	cv::Mat cap_frame;
	std::shared_ptr<image_t> det_image;
	std::vector<bbox_t> result_vec;
	cv::Mat draw_frame;
	bool new_detection;
	uint64_t frame_id;
	bool exit_flag;
	cv::Mat zed_cloud;
	std::queue<cv::Mat> track_optflow_queue;
	detection_data_t() : exit_flag(false), new_detection(false) {}
};


void guiYoLoCV::updateImage()
{
	static Detector detector(cfg_file, weights_file);
	static auto obj_names = objects_names_from_file(names_file);


	//std::atomic<int> fps_cap_counter(0), fps_det_counter(0);
	//std::atomic<int> current_fps_cap(0), current_fps_det(0);
	std::atomic<bool> exit_flag(false);
	std::chrono::steady_clock::time_point steady_start, steady_end;
	//int video_fps = 25;
	track_kalman_t track_kalman;
	//cap >> cur_frame;
	//video_fps = cap.get(cv::CAP_PROP_FPS);
	/*cv::Size  frame_size = cur_frame.size();*/



	const bool sync = detection_sync; // sync data exchange

	send_one_replaceable_object_t<detection_data_t> cap2prepare(sync), cap2draw(sync),
		prepare2detect(sync), detect2draw(sync), draw2show(sync);

	std::thread t_cap, t_prepare, t_detect, t_post, t_draw;

	// capture new video-frame
	if (t_cap.joinable()) t_cap.join();
	t_cap = std::thread([&]()
	{
		uint64_t frame_id = 0;
		detection_data_t detection_data;
		do {
			detection_data = detection_data_t();

			{
				cap >> detection_data.cap_frame;
			}
			//fps_cap_counter++;
			detection_data.frame_id = frame_id++;
			if (detection_data.cap_frame.empty() || exit_flag) {
				std::cout << " exit_flag: detection_data.cap_frame.size = " << detection_data.cap_frame.size() << std::endl;
				detection_data.exit_flag = true;
				detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
			}

			if (!detection_sync) {
				cap2draw.send(detection_data);       // skip detection
			}
			cap2prepare.send(detection_data);
		} while (!detection_data.exit_flag);
		std::cout << " t_cap exit \n";
	});


	// pre-processing video frame (resize, convertion)
	t_prepare = std::thread([&]()
	{
		std::shared_ptr<image_t> det_image;
		detection_data_t detection_data;
		do {
			detection_data = cap2prepare.receive();

			det_image = detector.mat_to_image_resize(detection_data.cap_frame);
			detection_data.det_image = det_image;
			prepare2detect.send(detection_data);    // detection

		} while (!detection_data.exit_flag);
		std::cout << " t_prepare exit \n";
	});

	// detection by Yolo
	if (t_detect.joinable()) t_detect.join();
	t_detect = std::thread([&]()
	{
		std::shared_ptr<image_t> det_image;
		detection_data_t detection_data;
		do {
			detection_data = prepare2detect.receive();
			det_image = detection_data.det_image;
			std::vector<bbox_t> result_vec;

			if (det_image)
				result_vec = detector.detect_resized(*det_image, frame_size.width, frame_size.height, thresh, true);  // true
			//fps_det_counter++;
			//std::this_thread::sleep_for(std::chrono::milliseconds(150));

			detection_data.new_detection = true;
			detection_data.result_vec = result_vec;
			detect2draw.send(detection_data);
		} while (!detection_data.exit_flag);
		std::cout << " t_detect exit \n";
	});

	// draw rectangles (and track objects)
	t_draw = std::thread([&]()
	{
		std::queue<cv::Mat> track_optflow_queue;
		detection_data_t detection_data;
		do {

			//// for Video-file
			//if (detection_sync) {
				detection_data = detect2draw.receive();
			//}
			//// for Video-camera
			//else
			//{
			//	// get new Detection result if present
			//	if (detect2draw.is_object_present()) {
			//		cv::Mat old_cap_frame = detection_data.cap_frame;   // use old captured frame
			//		detection_data = detect2draw.receive();
			//		if (!old_cap_frame.empty()) detection_data.cap_frame = old_cap_frame;
			//	}
			//	// get new Captured frame
			//	else {
			//		std::vector<bbox_t> old_result_vec = detection_data.result_vec; // use old detections
			//		detection_data = cap2draw.receive();
			//		detection_data.result_vec = old_result_vec;
			//	}
			//}

			cv::Mat cap_frame = detection_data.cap_frame;
			cv::Mat draw_frame = detection_data.cap_frame.clone();
			std::vector<bbox_t> result_vec = detection_data.result_vec;



			//// track ID by using kalman filter
			//if (use_kalman_filter) {
			//	if (detection_data.new_detection) {
			//		result_vec = track_kalman.correct(result_vec);
			//	}
			//	else {
			//		result_vec = track_kalman.predict();
			//	}
			//}
			//// track ID by using custom function
			//else {
			//	int frame_story = std::max(5, current_fps_cap.load());
			//	result_vec = detector.tracking_id(result_vec, true, frame_story, 40);
			//}



			//small_preview.set(draw_frame, result_vec);
			//large_preview.set(draw_frame, result_vec);
			draw_boxes(draw_frame, result_vec, obj_names, 0, 0);
			
			// 是否在控制台显示信息
			//show_console_result(result_vec, obj_names, detection_data.frame_id);
			//large_preview.draw(draw_frame);
			//small_preview.draw(draw_frame, true);

			detection_data.result_vec = result_vec;
			detection_data.draw_frame = draw_frame;
			draw2show.send(detection_data);
		} while (!detection_data.exit_flag);
		std::cout << " t_draw exit \n";
	});

	// show detection
	detection_data_t detection_data;
	do {

		//steady_end = std::chrono::steady_clock::now();
		//float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
		//if (time_sec >= 1) {
		//	//current_fps_det = fps_det_counter.load() / time_sec;
		//	current_fps_cap = fps_cap_counter.load() / time_sec;
		//	steady_start = steady_end;
		//	fps_det_counter = 0;
		//	fps_cap_counter = 0;
		//}

		detection_data = draw2show.receive();
		cv::Mat draw_frame = detection_data.draw_frame;

		

		polylines(draw_frame, vec, true, Scalar(0, 0, 255), 1, LINE_8, 0);
		QImage image(draw_frame.data, draw_frame.cols, draw_frame.rows, draw_frame.step, QImage::Format_RGB888);
		ui.videoLabel->setPixmap(QPixmap::fromImage(image.rgbSwapped()));

		qApp->processEvents();

		//cv::imshow("window name", draw_frame);
		//int key = cv::waitKey(3);    // 3 or 16ms
		////if (key == 'f') show_small_boxes = !show_small_boxes;
		//if (key == 'p') while (true) if (cv::waitKey(100) == 'p') break;
		////if (key == 'e') extrapolate_flag = !extrapolate_flag;
		//if (key == 27) { exit_flag = true; }

		//std::cout << " current_fps_det = " << current_fps_det << ", current_fps_cap = " << current_fps_cap << std::endl;
	} while (!detection_data.exit_flag);

	// wait for all threads
	if (t_cap.joinable()) t_cap.join();
	if (t_prepare.joinable()) t_prepare.join();
	if (t_detect.joinable()) t_detect.join();
	if (t_draw.joinable()) t_draw.join();

	// 重绘界面
	//qApp->processEvents();


	//vcap >> frame;
	//blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
	//net.setInput(blob);
	//vector<Mat> outs;
	//net.forward(outs, getOutputsNames(net));
	//postprocess(frame, outs);
	//vector<double> layersTimes;
	//double freq = getTickFrequency() / 1000;
	//double t = net.getPerfProfile(layersTimes) / freq;
	//string label = format("Inference time for a frame : %.2f ms", t);
	//putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
	//Mat detectedFrame;
	//frame.convertTo(detectedFrame, CV_8U);
	//polylines(detectedFrame, vec, true, Scalar(0, 0, 255), 1, LINE_8,0);
	//QImage image(detectedFrame.data, detectedFrame.cols, detectedFrame.rows, detectedFrame.step, QImage::Format_RGB888);
	//ui.videoLabel->setPixmap(QPixmap::fromImage(image.rgbSwapped()));

	//// 重绘界面
	//qApp->processEvents();
}








//
//// cpu
//// Remove the bounding boxes with low confidence using non-maxima suppression
//void postprocess(Mat& frame, const vector<Mat>& outs)
//{
//	vector<int> classIds;
//	vector<float> confidences;
//	vector<Rect> boxes;
//
//	for (size_t i = 0; i < outs.size(); ++i)
//	{
//		// Scan through all the bounding boxes output from the network and keep only the
//		// ones with high confidence scores. Assign the box's class label as the class
//		// with the highest score for the box.
//		float* data = (float*)outs[i].data;
//		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
//		{
//			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
//			Point classIdPoint;
//			double confidence;
//			// Get the value and location of the maximum score
//			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
//			if (confidence > confThreshold)
//			{
//				int centerX = (int)(data[0] * frame.cols);
//				int centerY = (int)(data[1] * frame.rows);
//				int width = (int)(data[2] * frame.cols);
//				int height = (int)(data[3] * frame.rows);
//				int left = centerX - width / 2;
//				int top = centerY - height / 2;
//
//				classIds.push_back(classIdPoint.x);
//				confidences.push_back((float)confidence);
//				boxes.push_back(Rect(left, top, width, height));
//			}
//		}
//	}
//
//	// Perform non maximum suppression to eliminate redundant overlapping boxes with
//	// lower confidences
//	vector<int> indices;
//	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
//	for (size_t i = 0; i < indices.size(); ++i)
//	{
//		int idx = indices[i];
//		Rect box = boxes[idx];
//
//		//越界逻辑
//		//
//
//
//
//		drawPred(classIds[idx], confidences[idx], box.x, box.y,
//			box.x + box.width, box.y + box.height, frame);
//	}
//}
//
//// cpu
//// Draw the predicted bounding box
//void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
//{
//	//Draw a rectangle displaying the bounding box
//	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
//
//	//Get the label for the class name and its confidence
//	string label = format("%.2f", conf);
//	if (!classes.empty())
//	{
//		CV_Assert(classId < (int)classes.size());
//		label = classes[classId] + ":" + label;
//	}
//
//	//Display the label at the top of the bounding box
//	int baseLine;
//	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//	top = max(top, labelSize.height);
//	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
//	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
//}
//
//// cpu
//// Get the names of the output layers
//vector<String> getOutputsNames(const Net& net)
//{
//	static vector<String> names;
//	if (names.empty())
//	{
//		//Get the indices of the output layers, i.e. the layers with unconnected outputs
//		vector<int> outLayers = net.getUnconnectedOutLayers();
//
//		//get the names of all the layers in the network
//		vector<String> layersNames = net.getLayerNames();
//
//		// Get the names of the output layers in names
//		names.resize(outLayers.size());
//		for (size_t i = 0; i < outLayers.size(); ++i)
//			names[i] = layersNames[outLayers[i] - 1];
//	}
//	return names;
//}

