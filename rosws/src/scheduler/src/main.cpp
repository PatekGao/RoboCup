//
// Created by bismarck on 2022/3/24.
//

#include "scheduler/measure.h"
#include "scheduler/identify.h"
#include "scheduler/tcpClient.h"
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include "rc_msgs/step.h"
#include "rc_msgs/results.h"
#include "rc_msgs/calibrateResult.h"
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>

#define JUDGE_SYSTEM

std::string mode = "None";
std::string filename = std::string(OUTPUT_PATH) + "NUAA-ZSWW-R";
std::atomic<int> step(0);
std::atomic<bool> finish(false);
std::atomic<bool> is_running(true);
std::atomic<bool> isCalibrate(false);
std::mutex mtx;
std::mutex modeMtx;
std::thread controller;
ros::Publisher beatPub;
ros::Publisher endPub;
#ifdef JUDGE_SYSTEM
tcpClient client("192.168.1.66", 6666);
#endif

void beatSend();
void stepCallback(const rc_msgs::step::ConstPtr &msg);
void isIdentifyCallback(const std_msgs::Bool::ConstPtr &msg);
void calibrateCallback(const rc_msgs::calibrateResult &msg);
void resultCallback(const rc_msgs::results &msg);

Interface* process = nullptr;

int main (int argc, char **argv) {
    ros::init(argc, argv, "scheduler");
    ros::NodeHandle nh;

    endPub = nh.advertise<std_msgs::Bool>("/ifend", 1);
    beatPub = nh.advertise<std_msgs::Bool>("/main_beat", 1);

    ros::Subscriber stepSub = nh.subscribe("/step", 1, stepCallback);
    ros::Subscriber isIdentifySub = nh.subscribe("/isIdentify", 1, isIdentifyCallback);
    ros::Subscriber calibrateSub = nh.subscribe("/calibrateResult", 3, calibrateCallback);
    ros::Subscriber resultSub = nh.subscribe("/rcnn_results", 3, resultCallback);

    std::thread beatThread(beatSend);

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    is_running = false;
    beatThread.join();
    controller.join();
    delete process;
    return 0;
}

inline void startTimer() {
#ifdef JUDGE_SYSTEM
    client.send(teamId, "NUAA-ZSWW");
#endif
}

void saveResult(const std::string& res) {
    modeMtx.lock();
    if (mode == "identify") {
        modeMtx.unlock();
#ifdef JUDGE_SYSTEM
        client.send(shibie, res);
#endif
    } else {
        modeMtx.unlock();
#ifdef JUDGE_SYSTEM
        client.send(celiang, res);
#endif
    }
    std::ofstream result(filename, std::ios::trunc);
    result << res;
}

void endProcess() {
    finish = true;
    std_msgs::Bool msg;
    mtx.lock();
    process->calcResult();
    std::string res = process->getResult();
    mtx.unlock();
    saveResult(res);
    msg.data = true;
    endPub.publish(msg);
    mtx.lock();
    delete process;
    process = nullptr;
    mtx.unlock();
    step = 0;
    isCalibrate = false;
    modeMtx.lock();
    mode = "None";
    modeMtx.unlock();
}

void timeoutControl(const std::string& _mode, int _step) {
    startTimer();
    if (_mode == "identify") {
        if (_step == 1) {
            std_msgs::Bool msg;
            filename += "2.txt";
            process->dt = Interface::SQUIRE55;
            ros::Duration(15).sleep();
            mtx.lock();
            process->calcResult();
            mtx.unlock();
            msg.data = false;
            endPub.publish(msg);
            ros::Duration(20).sleep();
            mtx.lock();
            process->calcResult();
            mtx.unlock();
            msg.data = false;
            endPub.publish(msg);
            ros::Duration(35).sleep();
            endProcess();
        } if (_step == 7) {
            process->dt = Interface::CIRCLE60;
            filename += "1.txt";
            ros::Duration(20).sleep();
            endProcess();
        }
    } else if (_mode == "measure1") {
        process->dt = Interface::SQUIRE55;
        filename += "1.txt";
        ros::Duration(15).sleep();
        std::cout << "identify finish\n";
        endProcess();
    } else if (_mode == "measure2") {
        process->dt = Interface::SQUIRE45;
        filename += "2.txt";
        ros::Duration(20).sleep();
        std::cout << "identify finish\n";
        endProcess();
    }
}

void beatSend() {
    while(is_running) {
        std_msgs::Bool beat;
        beat.data = true;
        beatPub.publish(beat);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void stepCallback(const rc_msgs::step::ConstPtr &msg) {
    modeMtx.lock();
    if (mode == "None") {
        modeMtx.unlock();
        if (msg->mode == "identify") {
            process = new Identify();
        } else {
            process = new Measure();
        }
        controller = std::thread(timeoutControl, msg->mode, msg->data);
    } else {
        modeMtx.unlock();
    }
    modeMtx.lock();
    mode = msg->mode;
    modeMtx.unlock();
    if (msg->data == 1 || msg->data == 4 || msg->data == 7 || msg->data == 8) {
        step = msg->data;
    }
}

void isIdentifyCallback(const std_msgs::Bool::ConstPtr &msg) {
    if (msg->data != isCalibrate) {
        if (msg->data) {
            ros::Duration(1).sleep();
            mtx.lock();
            if (!process->empty()) {
                process->clear();
            }
            if (step == 7) {
                process->dt = Interface::CIRCLE60;
            }
            mtx.unlock();
            isCalibrate = true;
        } else {
            isCalibrate = false;
        }
    }
}

void calibrateCallback(const rc_msgs::calibrateResult &msg) {
    modeMtx.lock();
    if (isCalibrate && mode != "None" && !finish) {
        modeMtx.unlock();
        Interface::points pos;
        for (const auto &point : msg.data) {
            pos.push_back(Interface::point_t(point.x, point.y));
        }
        mtx.lock();
        process->addPoints(pos);
        mtx.unlock();
    } else {
        modeMtx.unlock();
    }
}

void resultCallback(const rc_msgs::results &msg) {
    modeMtx.lock();
    if (msg.step == step && mode != "None" && !finish) {
        modeMtx.unlock();
        mtx.lock();
        process->addDetection(msg);
        mtx.unlock();
    } else {
        modeMtx.unlock();
    }
}
