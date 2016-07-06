#ifndef _IMGHELP_H_
#define _IMGHELP_H_

#include "header.h"


class imgHelp
{
public:
    // Construct function
    imgHelp() = default;
    imgHelp(Mat img, bool mydebug);

    // Gamma correction
    void setGamma(double gamma = 0.0);
    double getGamma();
    void gammaCorrection(Mat& dst);

    // Abstarct patchs
    void abstractTopBottomPatch(vector<Mat>& res, int factor = 5);

    // Compute hist
    void computeGrayHistogram(MatND& hist);
    void drawDisplayGrayHistogram(MatND& hist);
    // Analysis Color hist
    void computeAndDisplayColorHistogram(MatND& hist);

    /*
    * 2015_CVPR_High - performance high dynamic range image generation by inverted local patterns
    * I modified some thresholds' value and this complementation do not include AGC for post processiong!
    */
    void hpHdrByInvertedLocalPatterns(Mat& img);

    void MyAutoAdjustBright(Mat& img);

    // display image
    void displayImage(const string name, int waittime = 1000);
    void displayImage(const string name, Mat& img, int waittime = 1000);
private:
    Mat src;
    bool debug = false;
    double pri_gamma = 0.0;
};


#endif // !< _IMGHELP_H_
