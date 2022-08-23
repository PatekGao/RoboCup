//
// Created by bismarck on 2022/3/24.
//

#include "scheduler/measure.h"
#include "ros/ros.h"
#include <boost/geometry.hpp>
#include <cmath>
#include <cv_bridge/cv_bridge.h>

//#define ADD_SCREW

bool Measure::getGoals(detection& result, cv::Mat& img1, screw& s1) {
    using namespace std;
    using namespace cv;
    vector <Point> contour;
    for (int i = 1; i < img1.rows; i++)
    {
        for (int n = 1; n < img1.cols; n++)
        {
            int rr = (int)(img1.at<uchar>(i, n));
            int ss = rr;
            if (ss > 100) {
                contour.emplace_back(i,n);
            }
        }
    }
    int sz = (int) (contour.size());
    if (sz == 0) {
        return false;
    }
    cv::Mat data_pts(sz, 2, CV_64FC1);
    auto *_data_pts = (double *) data_pts.data;
    for (int i = 0; i < data_pts.rows; i++, _data_pts += 2) {
        _data_pts[0] = contour[i].x;
        _data_pts[1] = contour[i].y;
    }
    cv::PCA pca(data_pts, cv::Mat(), PCA::DATA_AS_ROW);
    Point2f dir1, dir2, center;
    center.x = (float)(pca.mean.at<double>(0, 0));
    center.y = (float)(pca.mean.at<double>(0, 1));
    dir1.x = (float)(pca.eigenvectors.at<double>(0, 0));
    dir1.y = (float)(pca.eigenvectors.at<double>(0, 1));
    dir2.x = (float)(pca.eigenvectors.at<double>(1, 0));
    dir2.y = (float)(pca.eigenvectors.at<double>(1, 1));
    Mat eig = pca.eigenvectors;
    Mat data_ptss = pca.project(data_pts);
    s1.eig = eig;

    if (result.label == 0) {
        double amin, amax, bmax, bmin;
        bmin = data_ptss.at<double>(0, 0);
        bmax = data_ptss.at<double>(0, 0);

        for (int i = 1; i < data_ptss.rows; i++) {
            if (data_ptss.at<double>(i, 0) < bmin) {
                bmin = data_ptss.at<double>(i, 0);
            }
            if (data_ptss.at<double>(i, 0) > bmax) {
                bmax = data_ptss.at<double>(i, 0);
            }
        }
        result.goalB = bmax - bmin;

        const int divide = 10;
        vector<double> amax_p(divide);
        vector<double> amin_p(divide);
        double tmp = (result.goalB + 1) / divide;
        for (int i = 1; i < data_ptss.rows; i++) {
            int index = (int)((data_ptss.at<double>(i, 0) - bmin) / tmp);
            if (data_ptss.at<double>(i, 1) > amax_p[index]) {
                amax_p[index] = data_ptss.at<double>(i, 1);
            }
            if (data_ptss.at<double>(i, 1) < amin_p[index]) {
                amin_p[index] = data_ptss.at<double>(i, 1);
            }
        }
        sort(amax_p.begin(), amax_p.end());
        sort(amin_p.begin(), amin_p.end());
        amax = amax_p[(int)(divide / 2)];
        amin = amin_p[(int)(divide / 2)];
        result.goalA = amax - amin;
        double maxLength = maxEdge(result.bbox);
        if (result.goalA < maxLength * 0.85) {
            result.goalA = maxLength * 0.85;
        }
   } else {
        double min, max;
        min = data_ptss.at<double>(0, 0);
        max = data_ptss.at<double>(0, 0);
        for (int i = 1; i < data_ptss.rows; i++) {
            if (data_ptss.at<double>(i, 0) < min) {
                min = data_ptss.at<double>(i, 0);
            }
            if (data_ptss.at<double>(i, 0) > max) {
                max = data_ptss.at<double>(i, 0);
            }
        }
        result.goalA = max - min;
        double maxLength = maxEdge(result.bbox);
        if (result.goalA < maxLength * 0.85) {
            result.goalA = maxLength * 0.85;
        }
        result.goalB = (max - min) * 0.6;
    }
    return true;
}

void Measure::calculateGoals(detections& res) {
    using namespace std;
    using namespace cv;
    if (!res.empty()) {
        vector<screw> screws;
        for (int k = 0; k < res.size(); k++) {
            detection& result = res[k];
            Mat img1 = Mat::zeros(2200, 2200, CV_8UC1);
            int npt[] = {(int)result.contours.size()};   //数组长度
            if (npt[0] <= 0) {
                continue;
            }
            auto *root_points = new Point[npt[0]];
            for (int i = 0; i < npt[0]; i++) {
                if ((int)result.contours[i].x < 0 || (int)result.contours[i].x >= img1.cols) {
                    continue;
                }
                if ((int)result.contours[i].y < 0 || (int)result.contours[i].y >= img1.rows) {
                    continue;
                }
                root_points[i].x = (int)result.contours[i].x;
                root_points[i].y = (int)result.contours[i].y;
            }
            const Point *ppt[1] = {root_points};       //标注点的二维坐标
            cv::fillPoly(img1, ppt, npt, 1, 255);
//            Mat img2;
//            resize(img1, img2, Size(600, 600));
//            imshow("test", img2);
//            waitKey(0);
            delete[] root_points;
            //imshow("sos",img1);
            screw s1;
            s1.index = k;
            bool ok = getGoals(result, img1, s1);
            if (result.label == 0 && ok) {
                screws.push_back(s1);
            }
            /*Mat img5(3,luosi_len,CV_8UC1,255);
            imshow("ror",img5);*/

        }
#ifdef ADD_SCREW
        cv::Mat eigCompare((int)screws.size(), (int)screws.size(), CV_64FC1, MEASURE_MIN_ANGLE);
        for (int i = 0; i + 1 < screws.size(); i++) {
            for (int j = i + 1; j < screws.size(); j++) {
                double angle = cv::norm(screws[i].eig - screws[j].eig);
                // 解除注释允许90度夹角螺丝匹配
//                if (angle > 1.14) {
//                    angle -= 1.414;
//                }
                eigCompare.at<double>(i, j) = angle;

            }
        }
        int n = (int)screws.size();
        while(n--) {
            double minValue;
            cv::Point minLoc;
            cv::minMaxLoc(eigCompare, &minValue, nullptr, &minLoc, nullptr);
            if (minValue >= MEASURE_MIN_ANGLE) {
                break;
            }
            double shortest = getShortest(res[screws[minLoc.x].index].contours, res[screws[minLoc.y].index].contours);
            if (shortest < MEASURE_MIN_SCREW_LEN) {
                for (int i = 0; i < eigCompare.cols; i++) {
                    eigCompare.at<double>(minLoc.x, i) = MEASURE_MIN_ANGLE;
                    eigCompare.at<double>(i, minLoc.y) = MEASURE_MIN_ANGLE;
                }
                res[screws[minLoc.x].index].goalA += res[screws[minLoc.y].index].goalA + shortest;
                res[screws[minLoc.x].index].goalB += res[screws[minLoc.y].index].goalB;
                res[screws[minLoc.x].index].goalB /= 2;
                res[screws[minLoc.x].index].score = std::max(res[screws[minLoc.x].index].score, res[screws[minLoc.y].index].score);
                res[screws[minLoc.y].index].score = 0;
            }
        }
#endif
    }
}

void getContours(const cv::Mat& image, std::vector<cv::Point2f>& _contours, double x, double y) {
    cv::Mat gray_img;
    cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
    cv::Scalar mea = mean(gray_img);
    threshold(gray_img, gray_img, mea[0], 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat kernel1 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    morphologyEx(gray_img, gray_img, cv::MORPH_OPEN, kernel1);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(gray_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE); //提取轮廓

    if (contours.empty()) {
        return;
    }
    double max_area = 0;
    double image_area = gray_img.rows * gray_img.cols * 0.98;
    int id = 0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if (area < image_area && area > max_area) {
            id = (int)i;
            max_area = area;
        }
    }
    for (const auto& p: contours[id]) {
        _contours.emplace_back(p.x + x, p.y + y);
    }
}



void Measure::addDetection(const rc_msgs::results &in) {
    if (!in.results.empty()) {
        detections detects;
        cv::Mat image = cv_bridge::toCvCopy(in.color, sensor_msgs::image_encodings::BGR8)->image;
        for (const auto &it: in.results) {
            detection detect;
            detect.label = it.label;
            detect.score = it.score;
            int x, y, w, h;
            x = (int)it.contours[0].x;
            y = (int)it.contours[0].y;
            w = (int)it.contours[2].x - x;
            h = (int)it.contours[2].y - y;
            x -= (int)(w * 0.1);
            y -= (int)(h * 0.1);
            w = (int)(w * 1.2);
            h = (int)(h * 1.2);
            if (x < 0) {
                x = 0;
            }
            if (x > image.cols) {
                x = image.cols - 1;
            }
            if (x + w > image.cols) {
                w = image.cols - x;
            }
            if (y < 0) {
                y = 0;
            }
            if (y > image.rows) {
                y = image.rows - 1;
            }
            if (y + h > image.rows) {
                h = image.rows - y;
            }

            cv::Rect roi(x, y, w, h);
            cv::Mat clip = image(roi);
            std::vector<cv::Point2f> temp;
            for (const auto& itp: it.contours) {
                temp.emplace_back(itp.x, itp.y);
            }
            detect.bbox.swap(temp);
            getContours(clip, detect.contours, x, y);
            detects.push_back(detect);
        }
        info.push_back(detects);
    }
}

void Measure::calcResult() {
    calAvgPoints();
//    cv::Mat test(640, 480, CV_8UC1, 0);
//    cv::Mat test = cv::Mat::zeros(640, 480, CV_8UC1);
//    std::vector<cv::Point2f> pps;
//    for (const auto& it: res) {
//        pps.emplace_back(it.x, it.y);
//    }
//    cv::fillPoly(test, pps, cv::Scalar(255));
//    cv::imshow("test", test);
//    cv::waitKey(1);
    std::vector<cv::Point2f> src, dst;
    for (const auto& it: res) {
        src.emplace_back(it.x, it.y);
    }
    if (dt == SQUIRE55) {
        dst = std::vector<cv::Point2f>{{0, 0}, {2200, 0}, {2200, 2200}, {0, 2200}};
    } else if (dt == SQUIRE45) {
        dst = std::vector<cv::Point2f>{{0, 0}, {1800, 0}, {1800, 1800}, {0, 1800}};
    } else if (dt == CIRCLE60) {
        dst = std::vector<cv::Point2f>{{0, 1200}, {1200, 0}, {2400, 1200}, {1200, 2400}};
    }
    cv::Mat warpMatrix = getPerspectiveTransform(src, dst);
    int kt = 0;
    for (auto& detect: info) {
        for (auto& it: detect) {
            std::vector<cv::Point2f> temp, temp2;
            cv::perspectiveTransform(it.contours, temp, warpMatrix);
            cv::perspectiveTransform(it.bbox, temp2, warpMatrix);
            it.contours.swap(temp);
            it.bbox.swap(temp2);
        }
        calculateGoals(detect);
        if (result.empty()) {
            result = detect;
        } else {
            cv::Mat comp((int)result.size(), (int)detect.size(), CV_32FC1, MEASURE_MIN_GOAL_LEN);
            for (int i = 0; i < result.size(); i++) {
                for (int j = 0; j < detect.size(); j++) {
                    comp.at<float>(i, j) = (float)getGoalLen(result[i], detect[j]);
                }
            }
            int n = std::min((int)result.size(), (int)detect.size());
            while(n--) {
                double minValue;
                cv::Point minPoint;
                cv::minMaxLoc(comp, &minValue, nullptr, &minPoint, nullptr);
                if (minValue >= MEASURE_MIN_GOAL_LEN) {
                    break;
                } else {
                    for (int i = 0; i < comp.rows; i++) {
                        comp.at<float>(i, minPoint.y) = MEASURE_MIN_GOAL_LEN;
                    }
                    for (int i = 0; i < comp.cols; i++) {
                        comp.at<float>(minPoint.x, i) = MEASURE_MIN_GOAL_LEN;
                    }
                    result[minPoint.x].score = std::max(result[minPoint.x].score, detect[minPoint.y].score);
                }
            }
            for (int j = 0; j < comp.cols; j++) {
                float minValue;
                for (int i = 0; i < comp.rows; i++) {
                    if (comp.at<float>(i, j) < minValue) {
                        minValue = comp.at<float>(i, j);
                        minValue = comp.at<float>(i, j);
                    }
                }
                if (minValue > MEASURE_MIN_GOAL_LEN) {
                    result.push_back(detect[j]);
                }
            }
        }
        kt++;
        ROS_WARN("OK: %i/%i\n", kt, (int)info.size());
    }
    std::sort(result.begin(), result.end(), compareScore);
}

double Measure::getGoalLen(const Measure::detection &d1, const Measure::detection &d2) {
    return sqrt(pow(d1.goalA - d2.goalA, 2) + pow(d1.goalB - d2.goalB, 2));
}

bool Measure::compareScore(const Measure::detection &d1, const Measure::detection &d2) {
    return d1.score > d2.score;
}

double Measure::getShortest(const std::vector<cv::Point2f> &c1, const std::vector<cv::Point2f> &c2) {
    typedef boost::geometry::model::point<float, 2, boost::geometry::cs::cartesian> point_t;
    typedef boost::geometry::model::polygon<point_t> polygon_t;
    polygon_t poly1, poly2;
    for (const auto& it: c1) {
        poly1.outer().push_back(point_t(it.x, it.y));
    }
    for (const auto& it: c2) {
        poly2.outer().push_back(point_t(it.x, it.y));
    }
    return boost::geometry::distance(poly1, poly2);
}

inline double getLength(const cv::Point2f& p1, const cv::Point2f& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

double Measure::maxEdge(const std::vector<cv::Point2f>& ps) {
    double l1 = getLength(ps[0], ps[1]);
    double l2 = getLength(ps[1], ps[2]);
    double l3 = getLength(ps[2], ps[3]);
    double l4 = getLength(ps[3], ps[0]);
    double l13 = (l1 + l3) / 2;
    double l24 = (l2 + l4) / 2;
    return fmax(l13, l24);
}

void Measure::clear() {
    clearCalibrate();
    info.clear();
}

std::string Measure::getResult() {
    std::stringstream ss;
    ss << "START" << std::endl << std::setiosflags(std::ios::fixed) << std::setprecision(1);
    for (const auto& it: result) {
        if (it.goalA < 0.1 && it.goalB < 0.1) {
            continue;
        }
        ss << "Goal_ID=" << it.label + 1 << ";";
        ss << "Goal_A=" << round(it.goalA * 2.5) / 10 << ";";
        ss << "Goal_B=" << round(it.goalB * 2.5) / 10 << std::endl;
    }
    ss << "END" << std::endl;
    return ss.str();
}
