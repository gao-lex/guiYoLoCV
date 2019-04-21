#include "guiYoLoCV.h"

std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba) {
	return bbox_vect;
}

#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>


#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)

#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
	int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
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
	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
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
float confThreshold = 0.3; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<std::string> classes;
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);
// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);



guiYoLoCV::guiYoLoCV(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);


	// 定时刷新视频
	connect(&theTimer, &QTimer::timeout, this, &guiYoLoCV::updateImage);

	//vcap.open("VIRAT_S_010201_00_000000_000053.mp4");
	vcap.open("rtmp://127.0.0.1:1935/liveRaw");
	//vcap.set(CAP_PROP_BUFFERSIZE, 1);

	// 自适应视频窗口大小
	ui.videoLabel->resize(frame.cols, frame.rows);

	// 定时开始
	theTimer.start(33);

	string classesFile = "obj.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
		classes.push_back(line);
	cv::String modelConfiguration = "yolov3-tiny-me.cfg";
	cv::String modelWeights = "yolov3-tiny-me_2000.weights";
	// Load the network
	net = readNetFromDarknet(modelConfiguration, modelWeights);
	//net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	//net.setPreferableTarget(DNN_TARGET_CPU);
	net.setPreferableTarget(DNN_TARGET_OPENCL);



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


void guiYoLoCV::updateImage()
{

	vcap >> frame;
	blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));
	postprocess(frame, outs);
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
	Mat detectedFrame;
	frame.convertTo(detectedFrame, CV_8U);
	polylines(detectedFrame, vec, true, Scalar(0, 0, 255), 1, LINE_8,0);
	QImage image(detectedFrame.data, detectedFrame.cols, detectedFrame.rows, detectedFrame.step, QImage::Format_RGB888);
	ui.videoLabel->setPixmap(QPixmap::fromImage(image.rgbSwapped()));

	// 重绘界面
	qApp->processEvents();
}










// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];

		//越界逻辑
		//



		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

