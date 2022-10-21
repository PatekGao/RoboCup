#include "ros/ros.h"
#include"rc_msgs/calibrateResult.h"
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <librealsense2/rs.hpp>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>

//lib
#include<librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_processing.hpp>
#include <librealsense2/hpp/rs_types.hpp>
#include <librealsense2/hpp/rs_sensor.hpp>

#include <opencv2/opencv.hpp>


//pcl
// PCL 库


using namespace std;

using namespace cv;

using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

// 相机内参
const double camera_factor = 1000;
const double camera_cx = 311.704;
const double camera_cy = 245.82;
const double camera_fx = 474.055;
const double camera_fy = 474.055;

//
ros::Publisher img_pub;
ros::Publisher cloud_pub;
ros::Publisher img_depth_pub;
sensor_msgs::Image img_msg;
sensor_msgs::Image img_msg_depth;
sensor_msgs::PointCloud2 cloudmsg;
Mat depth;

//rs2::frameset frame_set;



static void change_sensor_option(const rs2::sensor &sensor, float value = 24, rs2_option option_type = rs2_option(15)) {
    if (!sensor.supports(option_type)) {
        std::cerr << "This option is not supported by this sensor" << std::endl;
        return;
    }
    try {
        sensor.set_option(option_type, value);
    }
    catch (const rs2::error &e) {
        std::cerr << "Failed to set option " << option_type << ". (" << e.what() << ")" << std::endl;
    }
}


pcl_ptr points_to_pcl(const rs2::points &points) {
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto &p: cloud->points) {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}

void setToDefault(const rs2::sensor &sensor) {
    for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++) {
        rs2_option option_type = static_cast<rs2_option>(i);
        if (sensor.supports(option_type)) {
            rs2::option_range range = sensor.get_option_range(option_type);
            float default_value = range.def;
            try {
                sensor.set_option(option_type, default_value);
            }
            catch (const rs2::error &e) {
                std::cerr << "Failed to set option " << option_type << ". (" << e.what() << ")" << std::endl;
            }
        }
    }
}

void get_img(ros::NodeHandle nh) {
    rs2::context ctx;
    auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
    if (list.size() == 0)
        throw std::runtime_error("No device detected. Is it plugged in?");
    rs2::device dev = list.front();

    auto sensors = dev.query_sensors();
    for (const auto &sensor: sensors) {
        setToDefault(sensor);
    }
    change_sensor_option(sensors[0]);

    rs2::pipeline pipe;
    //Construct a pipeline which abstracts the device
    rs2::config cfg;
    //Create a configuration for configuring the pipeline with a non default profile

    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    rs2::pipeline_profile selection = pipe.start(cfg);
//    auto depth_stream=selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
//    auto i=depth_stream.get_intrinsics();
    rs2_stream align_to = RS2_STREAM_COLOR;
    rs2::align align(align_to);
    //cout<<"width:"<<i.width<<"height:"<<i.height<<"ppx:"<<i.ppx<<"ppy:"<<i.ppy<<"fx:"<<i.fx<<"fy:"<<i.fy<<"dis model:"<<i.model<<endl;


    while (ros::ok()) {
        rs2::frameset frames;
        frames = pipe.wait_for_frames();
        rs2::pointcloud pc;
        rs2::points points;

        auto processed = align.process(frames);

        //Get each frame
        auto color_frame = processed.get_color_frame();
        auto depth_frame = processed.get_depth_frame();
        points = pc.calculate(depth_frame);
        auto prof = depth_frame.get_profile().as<rs2::video_stream_profile>();
        auto i = prof.get_intrinsics();


        auto pcl_points = points_to_pcl(points);


        //create cv::Mat from rs2::frame
        Mat depth_;
        Mat color(Size(640, 480), CV_8UC3);
        memcpy(color.data, color_frame.get_data(), 921600);
        Mat deep(Size(640, 480), CV_16UC1);
        memcpy(deep.data, depth_frame.get_data(), 614400);

        Mat dst3(Size(640, 480), CV_16UC1);
        ushort *p;
        ushort *q;

        //float scale = frames.get_depth_scale();
        for (int y = 0; y < 480; y++) {
            q = deep.ptr<ushort>(y);
            for (int x = 0; x < 640; x++) {
                /*dst->imageData[y * depth_info.height + x]
                = depth__data[y * depth_info.height + x]*/;
                ushort d = 0.125 * q[x];
                //cout << "d:  " << d << endl;
                p = dst3.ptr<ushort>(y);

                //距离在0.2m至1.2	int k = 0;m之间

                if (d > 0) {
                    p[x] = 255 - 0.255 * (d - 200);
                    //cout << "p:  " << p[x] << endl;
                } else
                    p[x] = 0;
            }
        }
        dst3.convertTo(depth_, CV_8UC1, 1);


        toROSMsg(*pcl_points, cloudmsg);
        sensor_msgs::ImagePtr color_msg =
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", color).toImageMsg();
        color_msg->header.stamp = ros::Time::now();
        sensor_msgs::ImagePtr depth_msg =
                cv_bridge::CvImage(std_msgs::Header(), "mono8", depth_).toImageMsg();
        depth_msg->header.stamp = ros::Time::now();

        img_msg = *color_msg;
        img_msg_depth = *depth_msg;

        img_depth_pub.publish(img_msg_depth);
        img_pub.publish(img_msg);

        cloud_pub.publish(cloudmsg);

    }
    change_sensor_option(sensors[0], 0);
}

int main(int argc, char **argv) {
    //初始化节点
    ros::init(argc, argv, "mv_driver_node");
    //声明节点句柄
    ros::NodeHandle nh;
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud", 10);
    img_pub = nh.advertise<sensor_msgs::Image>("/raw_img", 10);
    img_depth_pub = nh.advertise<sensor_msgs::Image>("/raw_img_depth", 10);
    ros::Duration(1).sleep();
    while (ros::ok())
        get_img(nh);
    nh.shutdown();
}
