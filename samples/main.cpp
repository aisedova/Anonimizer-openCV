/*#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/timer/timer.hpp>
*/
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <detector.h>

//namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv::dnn;
//using namespace cv::tbm;
using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

const char* cmdOptions =
{ "{video_name       | | video name                       }"
"{start_frame      |0| Start frame                      }"
"{frame_step       |1| Frame step                       }"
"{proposal_model   | | Path to detector's Caffe model   }"
"{proposal_weights | | Path to detector's Caffe weights }"
"{Refine_model   | | Path to Refine's Caffe model   }"
"{Refine_weights | | Path to Refine's Caffe weights }"
"{Output_model   | | Path to Output's Caffe model   }"
"{Output_weights | | Path to Output's Caffe weights }"
"{desired_class_id |-1| The desired class that should be tracked }"
"{desired_class_id |-1| The desired class that should be tracked }"
"{classificator_model   | | Path to classificator's Caffe model   }"
"{classificator_weights | | Path to classificator's Caffe weights }"
};
/*
-start_frame=200
-frame_step=4
-classificator_model="..\..\CV-SUMMER-CAMP\data\net\classification\squeezenet\1.1\caffe\squeezenet1.1.caffemodel"
-classificator_weights="..\..\CV-SUMMER-CAMP\data\net\classification\squeezenet\1.1\caffe\squeezenet1.1.prototxt"
-detector_model="C:\Users\temp2019\Desktop\CV-SUMMER-CAMP\data\object_detection\common\mobilenet-ssd\caffe\mobilenet-ssd.prototxt"
-detector_weights="C:\Users\temp2019\Desktop\CV-SUMMER-CAMP\data\object_detection\common\mobilenet-ssd\caffe\mobilenet-ssd.caffemodel"
-video_name="..\..\CV-SUMMER-CAMP\data\topdogs.mp4"
-desired_class_id=[12]
*/


static cv::Mat drawRectsAndPoints(const cv::Mat &img,
                                  const std::vector<rectPoints> data) {
  cv::Mat outImg;
  img.convertTo(outImg, CV_8UC3);

  for (auto &d : data) {
    cv::rectangle(outImg, d.first, cv::Scalar(0, 0, 255));
    auto pts = d.second;
    for (size_t i = 0; i < pts.size(); ++i) {
      cv::circle(outImg, pts[i], 3, cv::Scalar(0, 0, 255));
    }
  }
  return outImg;
}

int main(int argc, char **argv) {

	/*if (argc < 3) {
	std::cerr << "Usage " << argv[0] << ": "
				<< "<model-dir> "
				<< " "
				<< "<test-image>\n";
	return -1;
	}*/
	CommandLineParser parser(argc, argv, cmdOptions);

	//cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();

	String video_name = parser.get<String>("video_name");
	int start_frame = parser.get<int>("start_frame");
	int frame_step = parser.get<int>("frame_step");
	String detector_model = parser.get<String>("detector_model");
	String detector_weights = parser.get<String>("detector_weights");

	String classificator_model = parser.get<string>("classificator_model");
	String classificator_weights = parser.get<String>("classificator_weights");
	string class_ids = parser.get<string>("desired_class_id");
	vector<int> desired_class_id;
	//fs::path modelDir = fs::path(argv[1]);

	ProposalNetwork::Config pConfig;
	pConfig.caffeModel = parser.get<String>("Proposal_model");
	pConfig.protoText = parser.get<String>("Proposal_weights");
	pConfig.threshold = 0.6f;

	RefineNetwork::Config rConfig;
	rConfig.caffeModel = parser.get<String>("Refine_model");
	rConfig.protoText = parser.get<String>("Refine_weights");
	rConfig.threshold = 0.7f;

	OutputNetwork::Config oConfig;
	oConfig.caffeModel = parser.get<String>("Output_model");
	oConfig.protoText = parser.get<String>("Output_weights");
	oConfig.threshold = 0.7f;
  
  
	cv::Mat img = cv::imread(argv[2]);
	
	std::vector<Face> faces;
	MTCNNDetector detector(pConfig, rConfig, oConfig);
	
	//boost::timer::auto_cpu_timer t(3, "%w seconds\n");
	faces = detector.detect(img, 20.f, 0.709f);
	

	std::cout << "Number of faces found in the supplied image - " << faces.size()
			<< std::endl;

	std::vector<rectPoints> data;

	// show the image with faces in it
	for (size_t i = 0; i < faces.size(); ++i) {
	std::vector<cv::Point> pts;
	for (int p = 0; p < NUM_PTS; ++p) {
		pts.push_back(
			cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
	}

	auto rect = faces[i].bbox.getRect();
	auto d = std::make_pair(rect, pts);
	data.push_back(d);
	}

	auto resultImg = drawRectsAndPoints(img, data);
	cv::imshow("test-oc", resultImg);
	cv::waitKey(0);

	return 0;
}
