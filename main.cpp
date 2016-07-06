#include "imgHelp.h"

bool myDebug = false;
bool mySave = true;

// just do gamma correction
Mat& doGammaCorrection(Mat&, Mat&, double);
// find and display the max min point
void displayMaxMinPoint(Mat&);


// main entry
int main(int argc, char* argv[])
{
    const string path = "D:/GitHub/hdrs/src.jpg";
    Mat img = imread(path, 1);
    imgHelp test(img, false);
    Mat res;
    test.hpHdrByInvertedLocalPatterns(res);
    if (mySave) {
        imwrite("D:/GitHub/hdrs/res.jpg", res);
    }
	return 0;
}


/*
*  Birght or dark image by Gamma correction.
* comment:
* 1. 0 < gamma < 1, bright image;
* 2. gamma > 1, dark image;
* 3. gamma = 1, nothing done!
*/
Mat& doGammaCorrection(Mat& img, Mat& dst, double gamma = 0.0) {
	if (img.empty())
	{
		cerr << "Please check the input image!" << endl;
		return dst;
	}
	imgHelp iH(img, true);
	// setting gamma
	iH.setGamma(gamma);
	// cout << iH.getGamma() << endl;
	// do gamma correction
	iH.gammaCorrection(dst);
	return dst;
}


// find and display the max min point
void displayMaxMinPoint(Mat& img) {
	if (img.empty())
	{
		cout << "Please check the input image!" << endl;
		return;
	}
	if (img.channels() != 1) {
		cout << "Sorry, only handle one channel image!" << endl;
		return;
	}
	// (x,y)
	double minVal = 0.0, maxVal = 0.0;
	Point minLoc(0, 0), maxLoc(0, 0);
	double* p_minVal = &minVal;
	double* p_maxVal = &maxVal;
	Point* p_minLoc = &minLoc;
	Point* p_maxLoc = &maxLoc;

	Mat tmp = img.clone();

	switch (tmp.channels())
	{
		case 1:
			minMaxLoc(tmp, p_minVal, p_maxVal, p_minLoc, p_maxLoc);
			break;
		default:
			//tmp.reshape(0, 1);
			//minMaxLoc(tmp, p_minVal, p_maxVal, p_minLoc, p_maxLoc);
			break;
	}

	cout << "maxVal=" << maxVal << "; maxLoc=" << maxLoc << endl;
	cout << "minVal=" << minVal << "; minLoc=" << minLoc << endl;

	cv::cvtColor(tmp, tmp, COLOR_GRAY2BGR);
	circle(tmp, maxLoc, 3, Scalar(0, 0, 255), 1, 8, 0);
	circle(tmp, minLoc, 3, Scalar(0, 255, 0), 1, 8, 0);

	imgHelp disImg(tmp, true);
	disImg.displayImage("maxMinImg", tmp, 0);
}
