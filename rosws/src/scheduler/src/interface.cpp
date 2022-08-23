//
// Created by bismarck on 2022/3/24.
//

#include "scheduler/interface.h"
#include "ros/ros.h"

void Interface::calAvgPoints() {
    res.clear();
    if(!pps[0].empty()) {
        for(int i = 0; i < 4; i++) {
            float xSum = 0, ySum = 0;
            int count = 0;
            for (const auto& it: pps[i]) {
                xSum += it.x;
                ySum += it.y;
                count++;
            }
            res.emplace_back(xSum / (float)count, ySum / (float)count);;
            std::stringstream ss;
        }
    } else {
        res.emplace_back(0, 0);
        res.emplace_back(600, 0);
        res.emplace_back(600, 400);
        res.emplace_back(0, 400);
    }
}

void Interface::addPoints(Interface::points in) {
    pps[0].push_back(in[1]);
    pps[1].push_back(in[0]);
    pps[2].push_back(in[2]);
    pps[3].push_back(in[3]);
}

void Interface::clearCalibrate() {
    pps.clear();
}

bool Interface::empty() {
    return pps.empty();
}
