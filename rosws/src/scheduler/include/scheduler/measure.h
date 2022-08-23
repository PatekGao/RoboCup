//
// Created by bismarck on 2022/3/24.
//

#ifndef SRC_MEASURE_H
#define SRC_MEASURE_H

#define MEASURE_MIN_GOAL_LEN 20
#define MEASURE_MIN_SCREW_LEN 20
#define MEASURE_MIN_ANGLE 0.4       // 这是25度，计算公式: 2*sin(angle/2)

#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include "scheduler/interface.h"


class Measure: public Interface {
public:
    struct detection {
        std::vector<cv::Point2f> contours;
        std::vector<cv::Point2f> bbox;
        float score;
        int label;
        double goalA=-1;
        double goalB=-1;
    };
    typedef std::vector<detection> detections;
    detections result;

    void addDetection(const rc_msgs::results& in) override;
    void calcResult() override;
    void clear() override;
    std::string getResult() override;

protected:
    struct screw {
        int index;
        cv::Mat eig;
    };
    std::vector<detections> info;

    static void calculateGoals(detections& res);
    static inline double getGoalLen(const detection& d1, const detection& d2);
    static bool compareScore(const detection& d1, const detection& d2);
    static bool getGoals(detection& result, cv::Mat& img1, screw& s1);
    static double getShortest(const std::vector<cv::Point2f>& c1, const std::vector<cv::Point2f>& c2);
    static double maxEdge(const std::vector<cv::Point2f>& ps);
};

#endif //SRC_MEASURE_H
