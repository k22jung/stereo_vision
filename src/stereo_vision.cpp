#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

const double DOWNSCALE = 0.5;
const int WIN_SIZE_SGBM = 1;
const int MAX_X_GRAD = 25;
const double SMOOTHING_FACTOR = 4;
const double LAMBDA = 8000.0;
const double SIGMA = 1.5;
const double VIS_MULT = 4.0;
const int MAX_DISPARITY = 16*5;


Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance);

const String keys =
    "{help h usage ? |                  | print this message                                                }"
    "{@left          |../data/aloeL.jpg | left view of the stereopair                                       }"
    "{@right         |../data/aloeR.jpg | right view of the stereopair                                      }"
    "{filter         |wls_conf          | used post-filtering (wls_conf or wls_no_conf)                     }"
    "{no-downscale   |                  | force stereo matching on full-sized views to improve quality      }"
    ;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv,keys);
    parser.about("Disparity Filtering Demo");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String left_im = parser.get<String>(0);
    String right_im = parser.get<String>(1);
    String filter = parser.get<String>("filter");
    bool no_downscale = parser.has("no-downscale");
    int max_disp = MAX_DISPARITY;

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    if(max_disp <= 0 || max_disp%16 != 0)
    {
        cout<<"Incorrect MAX_DISPARITY value: it should be positive and divisible by 16"<<endl;
        return -1;
    }

    if(WIN_SIZE_SGBM <= 0 || WIN_SIZE_SGBM%2 != 1)
    {
        cout<<"Incorrect window_size value: it should be positive and odd"<<endl;
        return -1;
    }

    Mat left  = imread(left_im,IMREAD_COLOR);
    if ( left.empty() )
    {
        cout<<"Cannot read image file: "<<left_im;
        return -1;
    }

    Mat right = imread(right_im,IMREAD_COLOR);
    if ( right.empty() )
    {
        cout<<"Cannot read image file: "<<right_im;
        return -1;
    }

    Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    Mat conf_map = Mat(left.rows,left.cols,CV_8U);
    conf_map = Scalar(255);
    Rect ROI;
    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;

    if(filter=="wls_conf")
    {
        if(!no_downscale)
        {
            max_disp *= DOWNSCALE;
            if(max_disp%16!=0)
                max_disp += 16-(max_disp%16);
            resize(left ,left_for_matcher ,Size(),DOWNSCALE,DOWNSCALE);
            resize(right,right_for_matcher,Size(),DOWNSCALE,DOWNSCALE);
        }
        else
        {
            left_for_matcher  = left.clone();
            right_for_matcher = right.clone();
        }

		Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,WIN_SIZE_SGBM);
		left_matcher->setP1(24*WIN_SIZE_SGBM*WIN_SIZE_SGBM*SMOOTHING_FACTOR);
		left_matcher->setP2(96*WIN_SIZE_SGBM*WIN_SIZE_SGBM*SMOOTHING_FACTOR);
		left_matcher->setPreFilterCap(MAX_X_GRAD);
		left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
		wls_filter = createDisparityWLSFilter(left_matcher);
		Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

		matching_time = (double)getTickCount();
		left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
		right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
		matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();


        wls_filter->setLambda(LAMBDA);
        wls_filter->setSigmaColor(SIGMA);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp,left,filtered_disp,right_disp);
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();

        conf_map = wls_filter->getConfidenceMap();

        // Get the ROI that was used in the last filter call:
        if(!no_downscale)
        {
            // upscale raw disparity and ROI back for a proper comparison:
            resize(left_disp,left_disp,Size(),1/DOWNSCALE,1/DOWNSCALE);
            left_disp = left_disp/DOWNSCALE;
        }
    }
    else if(filter=="wls_no_conf")
    {
        left_for_matcher  = left.clone();
        right_for_matcher = right.clone();

		Ptr<StereoSGBM> matcher  = StereoSGBM::create(0,max_disp,WIN_SIZE_SGBM);
		matcher->setUniquenessRatio(0);
		matcher->setDisp12MaxDiff(1000000);
		matcher->setSpeckleWindowSize(0);
		matcher->setP1(24*WIN_SIZE_SGBM*WIN_SIZE_SGBM*SMOOTHING_FACTOR);
		matcher->setP2(96*WIN_SIZE_SGBM*WIN_SIZE_SGBM*SMOOTHING_FACTOR);
		matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
		ROI = computeROI(left_for_matcher.size(),matcher);
		wls_filter = createDisparityWLSFilterGeneric(false);
		wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*WIN_SIZE_SGBM));

		matching_time = (double)getTickCount();
		matcher->compute(left_for_matcher,right_for_matcher,left_disp);
		matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();

        wls_filter->setLambda(LAMBDA);
        wls_filter->setSigmaColor(SIGMA);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp,left,filtered_disp,Mat(),ROI);
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    }
    else
    {
        cout<<"Unsupported filter";
        return -1;
    }

    cout.precision(2);
    cout<<"Matching time:  "<<matching_time<<"s"<<endl;
    cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
    cout<<endl;

	namedWindow("left", WINDOW_AUTOSIZE);
	imshow("left", left);

	namedWindow("right", WINDOW_AUTOSIZE);
	imshow("right", right);

	Mat raw_disp_vis;
	getDisparityVis(left_disp,raw_disp_vis,VIS_MULT);
	namedWindow("raw disparity", WINDOW_AUTOSIZE);
	imshow("raw disparity", raw_disp_vis);

	Mat filtered_disp_vis;
	getDisparityVis(filtered_disp,filtered_disp_vis,VIS_MULT);
	namedWindow("Filtered Disparity", WINDOW_AUTOSIZE);
	imshow("Filtered Disparity", filtered_disp_vis);

	imwrite( "../output/disparity_filtered.jpg", filtered_disp_vis );
	imwrite( "../output/disparity_raw.jpg", raw_disp_vis);
	imwrite("../output/conf_map.jpg",conf_map);
	imwrite("../output/left_image.jpg",left);
	waitKey(0);


    return 0;
}

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

