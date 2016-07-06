#include "imgHelp.h"


//Construct Fuction
imgHelp::imgHelp(Mat img, bool mydebug = false)
{
    if (img.empty()){
        cerr << " Construct Error! \n"
            << "Please check the instance img!"
            << endl;
        return;
    } else if ((img.channels() != 1) && (img.channels() != 3)) {
        cerr << "Construct Error! \n"
            << "The input image's channels only can be 1 or 3!"
            << endl;
        return;
    } else {
        src = img;
        debug = mydebug;
    }
}

/************************************************************************/
/*
* Setting gamma value for Gamma corrrection.
* default gamma = otsu threshold of src.
*/
/************************************************************************/
void imgHelp::setGamma(double gamma)
{
    if (gamma == 0.0)
    {
        Mat gray = src.clone();
        if (gray.channels()==3)
        {
            cv::cvtColor(gray, gray, COLOR_BGR2GRAY);
        }
        Mat dst;
        double ret = 0.0;
        ret = threshold(gray, dst, 0, 255, THRESH_BINARY + THRESH_OTSU);
        pri_gamma = ret / 255;
    }
    else
    {
        pri_gamma = gamma;
    }
}


// Get gamma value
double imgHelp::getGamma()
{
    return pri_gamma;
}


/************************************************************************/
/*
* Gamma correction
* comment：
* 1. 0 < gamma < 1, bright src；
* 2. gamma > 1, dark src；
* 3. gamma = 1, nothing done!
*/
/************************************************************************/
void imgHelp::gammaCorrection(Mat& dst)
{
    // build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), pri_gamma) * 255.0f);
    }
    // case 1 and 3 for different channels
    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
        case 1:
        {
            MatIterator_<uchar> it, end;
            for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
                *it = lut[(*it)];
            break;
        }
        case 3:
        {
            MatIterator_<Vec3b> it, end;
            for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
            {
                (*it)[0] = lut[((*it)[0])]; // B
                (*it)[1] = lut[((*it)[1])]; // G
                (*it)[2] = lut[((*it)[2])]; // R
            }
            break;
        }
    } // end for switch
}


// abstact patchs of src
void imgHelp::abstractTopBottomPatch(vector<Mat>& res, int factor)
{
    int rows = src.rows;
    int h = rows / factor;
    int w = src.cols;

    // Rect tl(x1, y1, w, h)
    const int x1 = 0;
    const int y1 = 0;
    Rect tl(x1, y1, w, h);

    // Rect bl(x2, y2, w, h)
    const int x2 = 0;
    const int y2 = rows * (factor - 1) / factor;
    Rect bl(x2, y2, w, h);

    // Mat roi(src t)
    Mat topLight(src, tl);
    Mat bottomLight(src, bl);

    // store to vector<Mat>
    res.push_back(topLight);
    res.push_back(bottomLight);
}


// compute hist of src in Gray level
void imgHelp::computeGrayHistogram(MatND& hist)
{
    Mat gray;
    if (src.channels() != 1) {
        cv::cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = src.clone();
    }
    const int histSize = 256;
    float range[2] = { 0, 255 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
}


// draw and display gray hist
void imgHelp::drawDisplayGrayHistogram(MatND& hist)
{
    double maxVal = 0;
    double minVal = 0;
    // Find maxVal and minVal in hist
    minMaxLoc(hist, &minVal, &maxVal, 0, 0);
    int histSize = hist.rows;
    Mat histImg(histSize, histSize, CV_8U, Scalar(255));
    // Setting maximum peak value to 90%*histSize
    int hpt = static_cast<int>(0.9*histSize);
    // Draw the histImage
    for (int h = 0; h < histSize; h++)
    {
        float binVal = hist.at<float>(h);
        int intensity = static_cast<int>(binVal*hpt / maxVal);
        line(histImg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
    }
    displayImage("single channel image's histogram", histImg);
}


// compute and display hist of color
void imgHelp::computeAndDisplayColorHistogram(MatND& hist)
{
    Mat srcColor;
    if (src.channels() != 3 ) {
        cv::cvtColor(src, srcColor, COLOR_GRAY2BGR);
    } else {
        srcColor = src.clone();
    }

    //split：B, G and R
    vector<Mat> bgr_planes;
    split(srcColor, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // draw histImage
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }
    displayImage("three channels image's histogram", histImage);
}


/*
* 2015_IET_High-performance high dynamic range image generation by inverted local patterns.pdf
* Without AGC in the following literature
* 2011_Low-complexity camera digital signal imaging for videodocument projection system.pdf
*/
void imgHelp::hpHdrByInvertedLocalPatterns(Mat& img)
{
    img = src.clone();
    if (img.channels() == 1)
    {
        cv::cvtColor(img, img, COLOR_GRAY2BGR);
    }

    // step1: convert BGR format To YCrCb
    cv::cvtColor(img, img, COLOR_BGR2YCrCb);
    // split channels
    vector<Mat> yCrCb(img.channels());
    split(img, yCrCb);
    Mat Y = yCrCb[0];
    int rows = Y.rows;
    int cols = Y.cols;

    // step2: detect the type of image("dark", "bright", "extreme", "normal")
    int lchannels = Y.channels();
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    MatND l_hist;
    cv::calcHist(&Y, 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate);

    double Hi = 0, Lo = 0;
    const int Th1 = 15;
    const int Th2 = 50;
    const int Th3 = 205;
    const int Th4 = 240;
    /*
    * Formula (1)
    */
    int histVal = 0;
    for (int i = 0; i < 256; ++i) {
        histVal = l_hist.at<int>(i);
        if ((i >= Th2) && (i < Th3))
        {
            continue;
        }

        // low gray level
        if ((i >= 0) && (i < Th1))
        {
            Lo += (histVal * 2);
        }
        if ((i >= Th1) && (i < Th2))
        {
            Lo += histVal;
        }

        // high gray level
        if ((i >= Th3) && (i < Th4))
        {
            Hi += histVal;
        }
        if ((i >= Th4) && (i <= 255))
        {
            Hi += (histVal * 2);
        }
    }


    /*
    * default image Type is "normal"
    * Table 1 Condition of three types
    */
    string Type = "normal";
    double HiRitoLo = 1.0 * Hi / Lo;
    int Ptotal = rows * cols / 4;
    if (Lo > (2 * Hi))
    {
        Type = "dark";
    }
    if (Hi > (2 * Lo))
    {
        Type = "bright";
    }
    if (((0.5 <= HiRitoLo) && (HiRitoLo <= 2)) &&
        (Hi > Ptotal) && (Lo > Ptotal))
    {
        Type = "extreme";
    }

    /*
    *  Type is "normal" nothing to do. just return.
    */
    if (Type == "normal")
    {
        // Merge vector<Mat> yCrCb to img(YCrYb format)
        cv::merge(yCrCb, img);
        // convert YCrCb to BGR format
        cv::cvtColor(img, img, COLOR_YCrCb2BGR);
        return;
    }

    /*
    * Inverse pattern generation kernel
    * Formula (2), (3)
    */
    const int w_mf = 2;
    Mat Ymax = Y.clone();
    // define the inner-loop variants.
    int idx, idy;
    uchar max_mf = 0, val_mf = 0;
    for (int j = 0; j < rows / w_mf; j++)
    {
        for (int i = 0; i < rows / w_mf; i++)
        {
            // Find max intensity of w_mf * w_mf window
            max_mf = 0;
            val_mf = 0;
            for (int m = 0; m < w_mf; m++)
            {
                idy = j*w_mf + m;
                idy = idy >= 0 ? (idy < rows ? idy : rows - 1) : 0;
                for (int n = 0; n < w_mf; n++)
                {
                    idx = i*w_mf + n;
                    idx = idx >= 0 ? (idx < cols ? idx : cols - 1) : 0;
                    val_mf = Ymax.at<uchar>(idy, idx);
                    if (val_mf > max_mf)
                    {
                        max_mf = val_mf;
                    }
                }
            }
            // Setting every pixel's intensity to max_mf
            for (int m = 0; m < w_mf; m++)
            {
                idy = j*w_mf + m;
                idy = idy >= 0 ? (idy < rows ? idy : rows - 1) : 0;
                for (int n = 0; n < w_mf; n++)
                {
                    idx = i*w_mf + n;
                    idx = idx >= 0 ? (idx < cols ? idx : cols - 1) : 0;
                    Ymax.at<uchar>(idy, idx) = max_mf;
                }
            }
        }
    }
    /*
    * Formula (4) Gain ylpf from ymax image.
    */
    const int w_lf = 3;
    Mat Ylpf;
    cv::boxFilter(Ymax, Ylpf, -1, Size(w_lf, w_lf));

    /*
    * Formula (5) Gain Yinv from Ylpf image.
    * Formula (6) Gain Ydark with the help of Thdark, Y.
    * Formula (8) Gain Ymix with the help of Ydark, Yinv and k.
    * Formula (9)-(14) Gain k.
    * Formula (15) Gain the Yhdr image which comprising Ymix, Ydark, Thb
    * Thb is selected by experiment results. and set Thb = 50.
    */
    Mat Yhdr = Y.clone();
    double k = 1.82*(1e-13)*Lo + 0.009;
    int Thb = 50;

    /*
    * Formula (7) Setting Thdark with the help of light's Type.
    */
    int Thdark = 0;
    if (Type == "dark")
    {
        Thdark = 128;
    }
    else if (Type == "extreme")
    {
        Thdark = 50;
    }
    else if (Type == "bright")
    {
        Thdark = 0;
    }
    else {
        Thdark = 0; // This bad idea for now.
    }

    if (debug)
    {
        cout << "Hi   = " << Hi << endl;
        cout << "Lo   = " << Lo << endl;
        cout << "HiRitoLo = " << HiRitoLo << endl;
        cout << "Ptotal = " << Ptotal << endl;
        cout << "Type = " << Type << endl;
        cout << "k = " << k << endl;
        cout << "Thdark = " << Thdark << endl;
    }

    /*
    yin    : pixel intensity of Y
    yinvv  : pixel intensity of Yinv
    ydarkv : pixel intensity of Ydark
    yminxv : pixel intensity of Ymix
    yhdrv  : pixel intensity of Yhdr
    */
    uchar yin = 0;
    uchar yinvv = 0, ydarkv = 0;
    uchar ymixv = 0;
    double yhdrv = 0;
    for (int j = 0; j < rows; j++)
    {
        for (int i = 0; i < cols; i++)
        {
            // Formula (5)
            yinvv = saturate_cast<uchar>(pow(255 - Ylpf.at<uchar>(j, i), 2) / 255);
            // Formula (6)
            yin = Y.at<uchar>(j, i);
            if (yin < Thdark)
            {
                ydarkv = saturate_cast<uchar>(yin + (Thdark - yin) * 0.05);
            }
            else
            {
                ydarkv = yin;
            }
            // Formula (8)
            ymixv = saturate_cast<uchar>(ydarkv * yinvv * k);
            // Formula (15)
            if (ydarkv >= Thb)
            {
                yhdrv = ymixv + pow(ydarkv - Thb, 2)*0.005;
            }
            else
            {
                yhdrv = ymixv;
            }

            Yhdr.at<uchar>(j, i) = saturate_cast<uchar>(yhdrv);
        }
    }
    // TODO AGC

    // Setting Y channel to Yhdr.
    yCrCb[0] = Yhdr;
    // Merge vector<Mat> yCrCb to img(YCrYb format)
    cv::merge(yCrCb, img);
    // convert YCrCb to BGR format
    cv::cvtColor(img, img, COLOR_YCrCb2BGR);
}


//
void imgHelp::MyAutoAdjustBright(Mat& img)
{
    img = src.clone();
    if (img.empty())
    {
        cerr << "Oh no! The input image is empty!" << endl;
        return;
    }
    if (img.channels() != 3)
    {
        cerr << "Oh no! The input image only allows 3-channels BGR image!" << endl;
        return;
    }
    // abstarct Y
    cv::cvtColor(img, img, COLOR_BGR2YCrCb);
    vector<Mat> yCrCb(img.channels());
    cv::split(img, yCrCb);
    Mat Y = yCrCb[0];

    // construct lut
    int dim(256);
    Mat lut(1, &dim, CV_8U);
    double afa = 0.0, yhdrv = 0.0;
    for (int i = 0; i < 256; i++)
    {
        if (i < 128) {
            afa = 1.0*i / (255 - i);
        }
        else {
            afa = 1.0*(255 - i) / i;
        }
        yhdrv = (1 - afa)*i + afa*(255 - i);
        lut.at<uchar>(i) = static_cast<uchar>(yhdrv);
    }

    Mat Yhdr;
    cv::LUT(Y, lut, Yhdr);

    // Setting Y channel to Yhdr.
    yCrCb[0] = Yhdr;
    // Merge vector<Mat> yCrCb to img(YCrYb format)
    cv::merge(yCrCb, img);
    // convert YCrCb to BGR format
    cv::cvtColor(img, img, COLOR_YCrCb2BGR);
}


// display src image
void imgHelp::displayImage(const string name, int waittime)
{
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, src);
    waitKey(waittime);
    destroyWindow(name);
}


// display image
void imgHelp::displayImage(const string name, Mat& img, int waittime)
{
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, img);
    waitKey(waittime);
    destroyWindow(name);
}
