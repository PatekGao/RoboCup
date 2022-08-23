//
// Created by bismarck on 2022/3/24.
//

#include "scheduler/identify.h"
#include <cmath>

void Identify::calcResult() {
    calAvgPoints();
    ClipperLib::Path desk;
    for (const auto& it: res) {
        desk.push_back({(ClipperLib::cInt)(it.x * 1000), (ClipperLib::cInt)(it.y * 1000)});
    }
    std::vector<labelCount> labelCounts;
    for (const auto& det : identifyResult) {
        labelCount count = labelCount(16);
        for (const auto& label : det) {
            ClipperLib::Clipper clipper;
            clipper.AddPath(desk, ClipperLib::ptSubject, true);
            clipper.AddPath(label.path, ClipperLib::ptClip, true);
            ClipperLib::Paths solution;
            clipper.Execute(ClipperLib::ctIntersection, solution, ClipperLib::pftNonZero, ClipperLib::pftNonZero);
            if (!solution.empty()) {
                double area = 0;
                for (const auto& p : solution) {
                    area += std::abs(ClipperLib::Area(p) / 1000000);
                }
                if (area > AREA_THRESH) {
                    count[label.label] += 1;
                }
            }
        }
        labelCounts.push_back(count);
    }
    for (int i = 0; i < result.size(); i++) {
        int temp = 0;
        for (const auto& count : labelCounts) {
            if (count[i] > temp) {
                temp = count[i];
            }
        }
        result[i] += temp;
    }
    for (auto& count : result) {
        if (count > 5) {
            count = 5;
        }
    }
}

void Identify::addDetection(const rc_msgs::results &in) {
    if (!in.results.empty()) {
        Identify::detections det;
        for (const auto& it: in.results) {
            ClipperLib::Path pa;
            for (const auto& contour: it.contours) {
                pa.push_back({(ClipperLib::cInt)(contour.x * 1000), (ClipperLib::cInt)(contour.y * 1000)});
            }
            det.push_back({it.label, pa});
        }
        identifyResult.push_back(det);
    }
}

void Identify::clear() {
    clearCalibrate();
    identifyResult.clear();
}

std::string Identify::getResult() {
    std::stringstream ss;
    ss << "START" << std::endl;
    for (int i = 0; i < result.size(); i++) {
        if (result[i] == 0) {
            continue;
        }
        ss << "Goal_ID=" << label2str[i] << ";";
        ss << "Num=" << result[i] << std::endl;
    }
    ss << "END" << std::endl;
    return ss.str();
}
