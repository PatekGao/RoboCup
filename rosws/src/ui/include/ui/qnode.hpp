/**
 * @file /include/ui/qnode.hpp
 *
 * @brief Communications central!
 *
 * @date February 2011
 **/
/*****************************************************************************
** Ifdefs
*****************************************************************************/

#ifndef ui_QNODE_HPP_
#define ui_QNODE_HPP_

/*****************************************************************************
** Includes
*****************************************************************************/

// To workaround boost/qt4 problems that won't be bugfixed. Refer to
//    https://bugreports.qt.io/browse/QTBUG-22829
#ifndef Q_MOC_RUN
#include <ros/ros.h>
#endif
#include <string>
#include <QThread>
#include <QStringListModel>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "ui/rotation_recongnition.hpp"
#include "rc_msgs/raw_img.h"
#include "rc_msgs/results.h"
#include "rc_msgs/step.h"
#include "rc_msgs/calibrateResult.h"
#include "std_msgs/Bool.h"
#include <mutex>
#include "rc_msgs/My_cfgConfig.h"
#include <dynamic_reconfigure/server.h>
#include <dynamic_reconfigure/client.h>
/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace ui {

/*****************************************************************************
** Class
*****************************************************************************/

class QNode : public QThread {
    Q_OBJECT
public:
	cv::Mat colorImg, depthImg;
	turn2 rotate;		// 修改此处切换旋转判定方案，turn为连续判定，turn2为MSE和判定
    std::mutex imageMtx;
    dynamic_reconfigure::Client<dynamic_cup::My_cfgConfig> client;
    dynamic_cup::My_cfgConfig config;
	QNode(int argc, char** argv );
	virtual ~QNode();
    bool init();
	void run();
	void rawImageCallback(const rc_msgs::raw_imgConstPtr& msg);
    void resultImageCallback(const rc_msgs::resultsConstPtr& msg);
	void beatCallback(const std_msgs::Bool::ConstPtr& msg);
	void nnBeatCallback(const std_msgs::Bool::ConstPtr& msg);
	void endCallback(const std_msgs::Bool::ConstPtr& msg);
    void deskCallback(const rc_msgs::calibrateResult::ConstPtr& msg);
    void callback(dynamic_cup::My_cfgConfig &config, uint32_t level);
    /*********************
	** Logging
	**********************/
	enum LogLevel {
		Debug,
		Info,
		Warn,
		Error,
		Fatal
	 };

	QStringListModel* loggingModel() { return &logging_model; }
	void log( const LogLevel &level, const std::string &msg);
	void startIdentify(bool isRoundOne, std::string _mode);

Q_SIGNALS:
	void loggingUpdated();
    void rosShutdown();
	void complete();
	void updateStatus(int, bool*);

private:
	int init_argc;
	char** init_argv;

	ros::Publisher stepPublisher;
    ros::Publisher indentifyControler;
    QStringListModel logging_model;
    std::vector<cv::Point> lastDesk;
    cv::Mat rotateImg, rawImage;
};

}  // namespace ui

#endif /* ui_QNODE_HPP_ */
