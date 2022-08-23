//
// Created by bismarck on 2022/3/24.
//

#ifndef SRC_IDENTIFY_H
#define SRC_IDENTIFY_H

#include "scheduler/interface.h"
#include "scheduler/clipper.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

class Identify: public Interface {
public:
    struct detection {
        int label;
        ClipperLib::Path path;
    };
    typedef std::vector<detection> detections;
    typedef std::vector<int> labelCount;

    double AREA_THRESH = 10;
    labelCount result = labelCount(16, 0);

    void addDetection(const rc_msgs::results &in) override;
    void calcResult() override;
    void clear() override;
    std::string getResult() override;

private:
    std::vector<detections> identifyResult;
    std::vector<std::string> label2str{{"CA001", "CA002", "CA003", "CA004", "CB001", "CB002", "CB003", "CB004",
                                        "CC001", "CC002", "CC003", "CC004", "CD001", "CD002", "CD003", "CD004"}};
};

#endif //SRC_IDENTIFY_H
