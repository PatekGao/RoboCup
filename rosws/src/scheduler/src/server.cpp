#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <dynamic_reconfigure/client.h>
#include "rc_msgs/My_cfgConfig.h"

using namespace dynamic_reconfigure;

void callback(dynamic_cup::My_cfgConfig &config, uint32_t level) {
  ROS_INFO("Reconfigure Request: %d %s", 
            config.int_param,  
            config.str_param.c_str());
}

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "");

    dynamic_reconfigure::Server<dynamic_cup::My_cfgConfig> server;
    dynamic_reconfigure::Server<dynamic_cup::My_cfgConfig>::CallbackType f;


    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);

    ROS_INFO("Spinning node");
    ros::spin();

    return 0;
}
