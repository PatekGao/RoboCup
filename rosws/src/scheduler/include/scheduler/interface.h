//
// Created by bismarck on 2022/3/24.
//

#ifndef SRC_INTERFACE_H
#define SRC_INTERFACE_H

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include "rc_msgs/results.h"

class Interface {
public:
    struct point_t {
        float x=0, y=0;
        point_t(float a,float b){
            x = a;
            y = b;
        }
    };
    typedef std::vector<point_t> points;
    points res;
    enum deskType {
        SQUIRE55,
        SQUIRE45,
        CIRCLE60
    };
    deskType dt;

    void addPoints(points);
    bool empty();
    virtual ~Interface() = default;
    virtual void calcResult() = 0;
    virtual void addDetection(const rc_msgs::results& in) = 0;
    virtual void clear() = 0;
    virtual std::string getResult() = 0;

protected:
    std::vector<points> pps = std::vector<points>(4);
    void clearCalibrate();
    void calAvgPoints();
};

#endif //SRC_INTERFACE_H
