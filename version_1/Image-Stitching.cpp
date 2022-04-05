#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

//顶点结构体
typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

//计算顶点
void CalcCorners(const Mat& H, const Mat& src)
{
    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

    V1 = H * V2;
    //左上角(0,0,1)
    cout << "V2: " << V2 << endl;
    cout << "V1: " << V1 << endl;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    //左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    //右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    //右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];

}

//优化两图的连接处，使得拼接自然
//使用渐入渐出的拼接方式
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
    int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界

    double processWidth = (img1.cols - start);//重叠区域的宽度
    int rows = dst.rows;
    int cols = img1.cols; //注意，是列数*通道数
    double alpha = 1;//img1中像素的权重
    for (int i = 0; i < rows; i++)
    {
        uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
        uchar* t = trans.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = start; j < cols; j++)
        {
            //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
            if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
            {
                alpha = 1;
            }

            else if (p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0)
            {
                alpha = 0;
            }
            else
            {
                //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
                alpha = (processWidth - (j - start)) / processWidth;
            }

            d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
            d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
            d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

        }
    }

}

    int main(int argc, char **argv) {

        //-- 读取图像
        Mat img_1 = imread("1.jpg", CV_LOAD_IMAGE_COLOR);//big
        Mat img_2 = imread("2.jpg", CV_LOAD_IMAGE_COLOR);//small
        assert(img_1.data != nullptr && img_2.data != nullptr);

        //-- 初始化
        std::vector<KeyPoint> keypoints_1, keypoints_2;
        Mat descriptors_1, descriptors_2;
        Ptr<FeatureDetector> detector = ORB::create();
        Ptr<DescriptorExtractor> descriptor = ORB::create();
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

        //-- 第一步:检测 Oriented FAST 角点位置
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        detector->detect(img_1, keypoints_1);
        detector->detect(img_2, keypoints_2);

        //-- 第二步:根据角点位置计算 BRIEF 描述子
        descriptor->compute(img_1, keypoints_1, descriptors_1);
        descriptor->compute(img_2, keypoints_2, descriptors_2);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

        Mat outimg1;
        drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("ORB features", outimg1);

        //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
        vector<DMatch> matches;
        t1 = chrono::steady_clock::now();
        matcher->match(descriptors_1, descriptors_2, matches);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

        //-- 第四步:匹配点对筛选
        // 计算最小距离和最大距离
        auto min_max = minmax_element(matches.begin(), matches.end(),
                                      [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;

        printf("-- Max dist : %f \n", max_dist);
        printf("-- Min dist : %f \n", min_dist);

        //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        std::vector<DMatch> good_matches;
        for (int i = 0; i < descriptors_1.rows; i++) {
            if (matches[i].distance <= max(2 * min_dist, 30.0)) {
                good_matches.push_back(matches[i]);
            }
        }

        //-- 第五步:绘制匹配结果
        Mat img_match;
        Mat img_goodmatch;
        drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
        drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

        Mat img_match_o, img_goodmatch_o;

        resize(img_match,img_match_o,cv::Size(1000,720));
        resize( img_goodmatch, img_goodmatch_o,cv::Size(1000,720));

        imshow("all matches", img_match_o);
        imshow("good matches", img_goodmatch_o);
        //right

        vector<Point2f> imagePoints1, imagePoints2;
//??change to point2f
        for (int i = 0; i < good_matches.size(); i++)
        {
            imagePoints1.push_back(keypoints_1[good_matches[i].queryIdx].pt); //顺序增大
            imagePoints2.push_back(keypoints_2[good_matches[i].trainIdx].pt);  //跳变
        }

        Mat homo = findHomography(imagePoints2, imagePoints1, RANSAC);//计算单应性矩阵

        CalcCorners(homo, img_2); //计算转换后的图片的顶点

        cout << homo << endl;
        cout << "顶点坐标" << endl;
        cout << corners.right_top << endl;
        cout << corners.right_bottom << endl;
        cout << corners.left_top << endl;
        cout << corners.left_bottom << endl;

        //透视变换后的图像
        Mat imageTransform1;
        //变换图像
        warpPerspective(img_2, imageTransform1, homo, Size(MAX(MIN(corners.right_top.x, corners.right_bottom.x), MIN(corners.left_top.x, corners.left_bottom.x)), img_1.rows));

        Mat FinalImage(imageTransform1.rows, imageTransform1.cols, CV_8UC3); //最终图像的矩阵
        FinalImage.setTo(0);//初始清零

        //进行图像拼接
        imageTransform1.copyTo(FinalImage(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
        img_1.copyTo(FinalImage(Rect(0, 0, img_1.cols, img_1.rows)));

        //youhua
        OptimizeSeam(img_1, imageTransform1, FinalImage); //拼接缝优化

        Mat FinalImage_o;
        resize(FinalImage,FinalImage_o,cv::Size(1200,720));

        imshow("FinalImage", FinalImage_o);

        waitKey();
        return 0;
    }