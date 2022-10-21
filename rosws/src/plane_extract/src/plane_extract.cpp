#include<ros/ros.h>
#include"rc_msgs/raw_img.h"
#include "rc_msgs/stepConfig.h"
#include "rc_msgs/point.h"
#include "rc_msgs/step.h"
#include"rc_msgs/calibrateResult.h"
#include"std_msgs/Bool.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/distances.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <dynamic_reconfigure/client.h>
#include <iostream>
#include<vector>

using namespace std;


typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

void identifyCallback(const std_msgs::Bool::ConstPtr &msg);

void callback(const rc_msgs::stepConfig &config);

ros::Publisher res_pub;
bool isIdentify = false;
int16_t step;
string mode;
double x11, x22, y11, y22;
rc_msgs::calibrateResult res;


void identifyCallback(const std_msgs::Bool::ConstPtr &msg) {
    isIdentify = msg->data;
}

void callback(const rc_msgs::stepConfig &config) {
    step=config.step;
    mode=config.mode;

}

const double camera_factor = 5000;
const double cx = 320;
const double cy = 240;
const double fx = 1395.56689453125;
const double fy = 1395.567138671875;


void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    rs2_intrinsics intrin;
    intrin.width=640;
    intrin.height=480;
    intrin.ppx=308.783;
    intrin.ppy=237.412;
    intrin.fx=620.252;
    intrin.fy=620.252;
    intrin.model=RS2_DISTORTION_NONE;
    for(int i=0;i<5;i++){intrin.coeffs[i]=0;}
    if (isIdentify) {
        //plane_extract
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PCLPointCloud2::Ptr cloud_blob(new pcl::PCLPointCloud2), cloud_filtered_blob(new pcl::PCLPointCloud2);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(
                new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        // Create the filtering object: downsample the dataset using a leaf size of 1cm
        fromROSMsg(*msg,*cloud);
        pcl::toPCLPointCloud2(*cloud, *cloud_blob);
        pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
        sor.setInputCloud(cloud_blob);
        sor.setLeafSize(0.01f, 0.01f, 0.01f);
        sor.filter(*cloud_filtered_blob);
        pcl::fromPCLPointCloud2(*cloud_filtered_blob, *cloud_filtered);


        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(800);
        seg.setDistanceThreshold(0.01);

        // Create the filtering object
        pcl::ExtractIndices<pcl::PointXYZ> extract;


        int i = 0, nr_points = (int) cloud_filtered->size();
        // While 30% of the original cloud is still there
        while (cloud_filtered->size() > 0.3 * nr_points) {
            // Segment the largest planar component from the remaining cloud
            seg.setInputCloud(cloud_filtered);
            seg.segment(*inliers, *coefficients);
            if (inliers->indices.size() <= 0.3 * nr_points) {
                std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
                break;
            }

            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

            // Extract the inliers
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*cloud_p);
            ne.setInputCloud(cloud_p);
            ne.setRadiusSearch(0.4);
            ne.compute(*cloud_normals);
            extract.setNegative(true);
            extract.filter(*cloud_f);
            cloud_filtered.swap(cloud_f);

            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud (cloud_filtered);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance (0.03); //设置近邻搜索的搜索半径为2cm
            ec.setMinClusterSize (1000);    //设置一个聚类需要的最少点数目为100
            ec.setMaxClusterSize (35000);  //设置一个聚类需要的最大点数目为25000
            ec.setSearchMethod (tree);     //设置点云的搜索机制
            ec.setInputCloud (cloud_p); //设置原始点云
            ec.extract (cluster_indices);  //从点云中提取聚类

                cout<<cloud_p->size()<<endl;
            // 可视化部分
            // pcl::visualization::PCLVisualizer viewer("segmention");
            int num = cluster_indices.size();

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator pit = cluster_indices.begin ()->indices.begin (); pit != cluster_indices.begin ()->indices.end (); pit++)
                cloud_cluster->points.push_back (cloud_p->points[*pit]); //*
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            cloud_p=cloud_cluster;

            double x, y, z;
            for (int i =cloud_p->size()/20; i < cloud_p->size(); i++) {
                if (isnan(cloud_normals->points[i].normal_x) || isnan(cloud_normals->points[i].normal_y) ||
                    isnan(cloud_normals->points[i].normal_z)) { continue; }
                else {
                    x = cloud_normals->points[i].normal_x;
                    y = cloud_normals->points[i].normal_y;
                    z = cloud_normals->points[i].normal_z;
                    break;
                }

            }
            if(abs(x)>0.2)
                continue;

            pcl::PointXYZ min, max, pmin, pmax, p;
            pcl::getMinMax3D(*cloud_p, min, max);
            int flaga = 0, flagb = 0, flagc = 0, left, front, right;
            for (int i = 0; i < cloud_p->size(); i++) {
                p = cloud_p->points[i];
                if (p.z == min.z) {
                    front = i;
                    flaga = 1;
                }
                if (abs(p.x - max.x) < 0.001) {
                    right = i;
                    flagb = 1;
                }
                if (abs(p.x - min.x) < 0.001) {
                    left = i;
                    flagc = 1;
                }
                if (flaga == 1 && flagb == 1 && flagc == 1) { break; }
            }
            pmin = cloud_p->points[left];
            if (abs(cloud_p->points[left].z - cloud_p->points[front].z) < 0.001) {
                pmax = cloud_p->points[right];
            } else {
                pmax = cloud_p->points[front];
            }

            // Create the filtering object
            double x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4;
            x0 = x;
            y0 = y;
            z0 = z;
            x1 = pmin.x;
            y1 = pmin.y;
            z1 = pmin.z;
            x2 = pmax.x;
            y2 = pmax.y;
            z2 = (x0 * x1 - x0 * x2 + y0 * y1 - y0 * y2 + z0 * z1) / z0;
            x3 = (pow(x0, 2) * x1 * pow(y1, 2) - 2 * pow(x0, 2) * x1 * y1 * y2 + pow(x0, 2) * x1 * pow(y2, 2) +
                  pow(x0, 2) * x1 * pow(z1, 2) - 2 * pow(x0, 2) * x1 * z1 * z2 + pow(x0, 2) * x1 * pow(z2, 2) -
                  2 * x0 * pow(x1, 2) * y0 * y1 + 2 * x0 * pow(x1, 2) * y0 * y2 - 2 * x0 * pow(x1, 2) * z0 * z1 +
                  2 * x0 * pow(x1, 2) * z0 * z2 + 2 * x0 * x1 * x2 * y0 * y1 - 2 * x0 * x1 * x2 * y0 * y2 +
                  2 * x0 * x1 * x2 * z0 * z1 - 2 * x0 * x1 * x2 * z0 * z2 + pow(x1, 3) * pow(y0, 2) +
                  pow(x1, 3) * pow(z0, 2) - 2 * pow(x1, 2) * x2 * pow(y0, 2) - 2 * pow(x1, 2) * x2 * pow(z0, 2) +
                  x1 * pow(x2, 2) * pow(y0, 2) + x1 * pow(x2, 2) * pow(z0, 2) + x1 * pow(y0, 2) * pow(z1, 2) -
                  2 * x1 * pow(y0, 2) * z1 * z2 + x1 * pow(y0, 2) * pow(z2, 2) - 2 * x1 * y0 * y1 * z0 * z1 +
                  2 * x1 * y0 * y1 * z0 * z2 + 2 * x1 * y0 * y2 * z0 * z1 - 2 * x1 * y0 * y2 * z0 * z2 +
                  x1 * pow(y1, 2) * pow(z0, 2) - 2 * x1 * y1 * y2 * pow(z0, 2) + x1 * pow(y2, 2) * pow(z0, 2) -
                  y0 * z1 * sqrt(pow(x0, 2) * pow(x1, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x1, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x1, 2) * pow(y2, 2) + pow(x0, 2) * pow(x1, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x1, 2) * z1 * z2 + pow(x0, 2) * pow(x1, 2) * pow(z2, 2) -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y1, 2) + 4 * pow(x0, 2) * x1 * x2 * y1 * y2 -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y2, 2) - 2 * pow(x0, 2) * x1 * x2 * pow(z1, 2) +
                                 4 * pow(x0, 2) * x1 * x2 * z1 * z2 - 2 * pow(x0, 2) * x1 * x2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(x2, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x2, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x2, 2) * pow(y2, 2) + pow(x0, 2) * pow(x2, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x2, 2) * z1 * z2 + pow(x0, 2) * pow(x2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 2 * x0 * pow(x1, 3) * y0 * y1 +
                                 2 * x0 * pow(x1, 3) * y0 * y2 - 2 * x0 * pow(x1, 3) * z0 * z1 +
                                 2 * x0 * pow(x1, 3) * z0 * z2 + 6 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 6 * x0 * pow(x1, 2) * x2 * y0 * y2 + 6 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 6 * x0 * pow(x1, 2) * x2 * z0 * z2 - 6 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 6 * x0 * x1 * pow(x2, 2) * y0 * y2 - 6 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 6 * x0 * x1 * pow(x2, 2) * z0 * z2 - 2 * x0 * x1 * y0 * pow(y1, 3) +
                                 6 * x0 * x1 * y0 * pow(y1, 2) * y2 - 6 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 2 * x0 * x1 * y0 * y1 * pow(z1, 2) + 4 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 2 * x0 * x1 * y0 * y1 * pow(z2, 2) + 2 * x0 * x1 * y0 * pow(y2, 3) +
                                 2 * x0 * x1 * y0 * y2 * pow(z1, 2) - 4 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 2 * x0 * x1 * y0 * y2 * pow(z2, 2) - 2 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y1, 2) * z0 * z2 + 4 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 4 * x0 * x1 * y1 * y2 * z0 * z2 - 2 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y2, 2) * z0 * z2 - 2 * x0 * x1 * z0 * pow(z1, 3) +
                                 6 * x0 * x1 * z0 * pow(z1, 2) * z2 - 6 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 2 * x0 * x1 * z0 * pow(z2, 3) + 2 * x0 * pow(x2, 3) * y0 * y1 -
                                 2 * x0 * pow(x2, 3) * y0 * y2 + 2 * x0 * pow(x2, 3) * z0 * z1 -
                                 2 * x0 * pow(x2, 3) * z0 * z2 + 2 * x0 * x2 * y0 * pow(y1, 3) -
                                 6 * x0 * x2 * y0 * pow(y1, 2) * y2 + 6 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 2 * x0 * x2 * y0 * y1 * pow(z1, 2) - 4 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 2 * x0 * x2 * y0 * y1 * pow(z2, 2) - 2 * x0 * x2 * y0 * pow(y2, 3) -
                                 2 * x0 * x2 * y0 * y2 * pow(z1, 2) + 4 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 2 * x0 * x2 * y0 * y2 * pow(z2, 2) + 2 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y1, 2) * z0 * z2 - 4 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 4 * x0 * x2 * y1 * y2 * z0 * z2 + 2 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y2, 2) * z0 * z2 + 2 * x0 * x2 * z0 * pow(z1, 3) -
                                 6 * x0 * x2 * z0 * pow(z1, 2) * z2 + 6 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 2 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 pow(x1, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x1, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x1, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x1, 2) * y0 * y1 * z0 * z1 + 2 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x1, 2) * y0 * y2 * z0 * z1 - 2 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) + pow(x1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x1, 2) * pow(z0, 2) * z1 * z2 + pow(x1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y1, 2) + 4 * x1 * x2 * pow(y0, 2) * y1 * y2 -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y2, 2) - 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                 8 * x1 * x2 * pow(y0, 2) * z1 * z2 - 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                 4 * x1 * x2 * y0 * y1 * z0 * z1 - 4 * x1 * x2 * y0 * y1 * z0 * z2 -
                                 4 * x1 * x2 * y0 * y2 * z0 * z1 + 4 * x1 * x2 * y0 * y2 * z0 * z2 -
                                 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) + 8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * x1 * x2 * pow(z0, 2) * z1 * z2 - 2 * x1 * x2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 pow(x2, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x2, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x2, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x2, 2) * y0 * y1 * z0 * z1 + 2 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x2, 2) * y0 * y2 * z0 * z1 - 2 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) + pow(x2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x2, 2) * pow(z0, 2) * z1 * z2 + pow(x2, 2) * pow(z0, 2) * pow(z2, 2) +
                                 pow(y0, 2) * pow(y1, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y1, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y1, 2) * pow(z2, 2) - 2 * pow(y0, 2) * y1 * y2 * pow(z1, 2) +
                                 4 * pow(y0, 2) * y1 * y2 * z1 * z2 - 2 * pow(y0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(y0, 2) * pow(y2, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y2, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y2, 2) * pow(z2, 2) + pow(y0, 2) * pow(z1, 4) -
                                 4 * pow(y0, 2) * pow(z1, 3) * z2 + 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) -
                                 4 * pow(y0, 2) * z1 * pow(z2, 3) + pow(y0, 2) * pow(z2, 4) -
                                 2 * y0 * pow(y1, 3) * z0 * z1 + 2 * y0 * pow(y1, 3) * z0 * z2 +
                                 6 * y0 * pow(y1, 2) * y2 * z0 * z1 - 6 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                 6 * y0 * y1 * pow(y2, 2) * z0 * z1 + 6 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                 2 * y0 * y1 * z0 * pow(z1, 3) + 6 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                 6 * y0 * y1 * z0 * z1 * pow(z2, 2) + 2 * y0 * y1 * z0 * pow(z2, 3) +
                                 2 * y0 * pow(y2, 3) * z0 * z1 - 2 * y0 * pow(y2, 3) * z0 * z2 +
                                 2 * y0 * y2 * z0 * pow(z1, 3) - 6 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                 6 * y0 * y2 * z0 * z1 * pow(z2, 2) - 2 * y0 * y2 * z0 * pow(z2, 3) +
                                 pow(y1, 4) * pow(z0, 2) - 4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) + pow(y1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y1, 2) * pow(z0, 2) * z1 * z2 + pow(y1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * y1 * y2 * pow(z0, 2) * z1 * z2 - 2 * y1 * y2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(y2, 4) * pow(z0, 2) + pow(y2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y2, 2) * pow(z0, 2) * z1 * z2 + pow(y2, 2) * pow(z0, 2) * pow(z2, 2)) +
                  y0 * z2 * sqrt(pow(x0, 2) * pow(x1, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x1, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x1, 2) * pow(y2, 2) + pow(x0, 2) * pow(x1, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x1, 2) * z1 * z2 + pow(x0, 2) * pow(x1, 2) * pow(z2, 2) -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y1, 2) + 4 * pow(x0, 2) * x1 * x2 * y1 * y2 -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y2, 2) - 2 * pow(x0, 2) * x1 * x2 * pow(z1, 2) +
                                 4 * pow(x0, 2) * x1 * x2 * z1 * z2 - 2 * pow(x0, 2) * x1 * x2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(x2, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x2, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x2, 2) * pow(y2, 2) + pow(x0, 2) * pow(x2, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x2, 2) * z1 * z2 + pow(x0, 2) * pow(x2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 2 * x0 * pow(x1, 3) * y0 * y1 +
                                 2 * x0 * pow(x1, 3) * y0 * y2 - 2 * x0 * pow(x1, 3) * z0 * z1 +
                                 2 * x0 * pow(x1, 3) * z0 * z2 + 6 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 6 * x0 * pow(x1, 2) * x2 * y0 * y2 + 6 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 6 * x0 * pow(x1, 2) * x2 * z0 * z2 - 6 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 6 * x0 * x1 * pow(x2, 2) * y0 * y2 - 6 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 6 * x0 * x1 * pow(x2, 2) * z0 * z2 - 2 * x0 * x1 * y0 * pow(y1, 3) +
                                 6 * x0 * x1 * y0 * pow(y1, 2) * y2 - 6 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 2 * x0 * x1 * y0 * y1 * pow(z1, 2) + 4 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 2 * x0 * x1 * y0 * y1 * pow(z2, 2) + 2 * x0 * x1 * y0 * pow(y2, 3) +
                                 2 * x0 * x1 * y0 * y2 * pow(z1, 2) - 4 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 2 * x0 * x1 * y0 * y2 * pow(z2, 2) - 2 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y1, 2) * z0 * z2 + 4 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 4 * x0 * x1 * y1 * y2 * z0 * z2 - 2 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y2, 2) * z0 * z2 - 2 * x0 * x1 * z0 * pow(z1, 3) +
                                 6 * x0 * x1 * z0 * pow(z1, 2) * z2 - 6 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 2 * x0 * x1 * z0 * pow(z2, 3) + 2 * x0 * pow(x2, 3) * y0 * y1 -
                                 2 * x0 * pow(x2, 3) * y0 * y2 + 2 * x0 * pow(x2, 3) * z0 * z1 -
                                 2 * x0 * pow(x2, 3) * z0 * z2 + 2 * x0 * x2 * y0 * pow(y1, 3) -
                                 6 * x0 * x2 * y0 * pow(y1, 2) * y2 + 6 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 2 * x0 * x2 * y0 * y1 * pow(z1, 2) - 4 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 2 * x0 * x2 * y0 * y1 * pow(z2, 2) - 2 * x0 * x2 * y0 * pow(y2, 3) -
                                 2 * x0 * x2 * y0 * y2 * pow(z1, 2) + 4 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 2 * x0 * x2 * y0 * y2 * pow(z2, 2) + 2 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y1, 2) * z0 * z2 - 4 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 4 * x0 * x2 * y1 * y2 * z0 * z2 + 2 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y2, 2) * z0 * z2 + 2 * x0 * x2 * z0 * pow(z1, 3) -
                                 6 * x0 * x2 * z0 * pow(z1, 2) * z2 + 6 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 2 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 pow(x1, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x1, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x1, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x1, 2) * y0 * y1 * z0 * z1 + 2 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x1, 2) * y0 * y2 * z0 * z1 - 2 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) + pow(x1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x1, 2) * pow(z0, 2) * z1 * z2 + pow(x1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y1, 2) + 4 * x1 * x2 * pow(y0, 2) * y1 * y2 -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y2, 2) - 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                 8 * x1 * x2 * pow(y0, 2) * z1 * z2 - 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                 4 * x1 * x2 * y0 * y1 * z0 * z1 - 4 * x1 * x2 * y0 * y1 * z0 * z2 -
                                 4 * x1 * x2 * y0 * y2 * z0 * z1 + 4 * x1 * x2 * y0 * y2 * z0 * z2 -
                                 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) + 8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * x1 * x2 * pow(z0, 2) * z1 * z2 - 2 * x1 * x2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 pow(x2, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x2, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x2, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x2, 2) * y0 * y1 * z0 * z1 + 2 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x2, 2) * y0 * y2 * z0 * z1 - 2 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) + pow(x2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x2, 2) * pow(z0, 2) * z1 * z2 + pow(x2, 2) * pow(z0, 2) * pow(z2, 2) +
                                 pow(y0, 2) * pow(y1, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y1, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y1, 2) * pow(z2, 2) - 2 * pow(y0, 2) * y1 * y2 * pow(z1, 2) +
                                 4 * pow(y0, 2) * y1 * y2 * z1 * z2 - 2 * pow(y0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(y0, 2) * pow(y2, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y2, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y2, 2) * pow(z2, 2) + pow(y0, 2) * pow(z1, 4) -
                                 4 * pow(y0, 2) * pow(z1, 3) * z2 + 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) -
                                 4 * pow(y0, 2) * z1 * pow(z2, 3) + pow(y0, 2) * pow(z2, 4) -
                                 2 * y0 * pow(y1, 3) * z0 * z1 + 2 * y0 * pow(y1, 3) * z0 * z2 +
                                 6 * y0 * pow(y1, 2) * y2 * z0 * z1 - 6 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                 6 * y0 * y1 * pow(y2, 2) * z0 * z1 + 6 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                 2 * y0 * y1 * z0 * pow(z1, 3) + 6 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                 6 * y0 * y1 * z0 * z1 * pow(z2, 2) + 2 * y0 * y1 * z0 * pow(z2, 3) +
                                 2 * y0 * pow(y2, 3) * z0 * z1 - 2 * y0 * pow(y2, 3) * z0 * z2 +
                                 2 * y0 * y2 * z0 * pow(z1, 3) - 6 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                 6 * y0 * y2 * z0 * z1 * pow(z2, 2) - 2 * y0 * y2 * z0 * pow(z2, 3) +
                                 pow(y1, 4) * pow(z0, 2) - 4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) + pow(y1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y1, 2) * pow(z0, 2) * z1 * z2 + pow(y1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * y1 * y2 * pow(z0, 2) * z1 * z2 - 2 * y1 * y2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(y2, 4) * pow(z0, 2) + pow(y2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y2, 2) * pow(z0, 2) * z1 * z2 + pow(y2, 2) * pow(z0, 2) * pow(z2, 2)) +
                  y1 * z0 * sqrt(pow(x0, 2) * pow(x1, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x1, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x1, 2) * pow(y2, 2) + pow(x0, 2) * pow(x1, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x1, 2) * z1 * z2 + pow(x0, 2) * pow(x1, 2) * pow(z2, 2) -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y1, 2) + 4 * pow(x0, 2) * x1 * x2 * y1 * y2 -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y2, 2) - 2 * pow(x0, 2) * x1 * x2 * pow(z1, 2) +
                                 4 * pow(x0, 2) * x1 * x2 * z1 * z2 - 2 * pow(x0, 2) * x1 * x2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(x2, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x2, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x2, 2) * pow(y2, 2) + pow(x0, 2) * pow(x2, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x2, 2) * z1 * z2 + pow(x0, 2) * pow(x2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 2 * x0 * pow(x1, 3) * y0 * y1 +
                                 2 * x0 * pow(x1, 3) * y0 * y2 - 2 * x0 * pow(x1, 3) * z0 * z1 +
                                 2 * x0 * pow(x1, 3) * z0 * z2 + 6 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 6 * x0 * pow(x1, 2) * x2 * y0 * y2 + 6 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 6 * x0 * pow(x1, 2) * x2 * z0 * z2 - 6 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 6 * x0 * x1 * pow(x2, 2) * y0 * y2 - 6 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 6 * x0 * x1 * pow(x2, 2) * z0 * z2 - 2 * x0 * x1 * y0 * pow(y1, 3) +
                                 6 * x0 * x1 * y0 * pow(y1, 2) * y2 - 6 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 2 * x0 * x1 * y0 * y1 * pow(z1, 2) + 4 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 2 * x0 * x1 * y0 * y1 * pow(z2, 2) + 2 * x0 * x1 * y0 * pow(y2, 3) +
                                 2 * x0 * x1 * y0 * y2 * pow(z1, 2) - 4 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 2 * x0 * x1 * y0 * y2 * pow(z2, 2) - 2 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y1, 2) * z0 * z2 + 4 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 4 * x0 * x1 * y1 * y2 * z0 * z2 - 2 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y2, 2) * z0 * z2 - 2 * x0 * x1 * z0 * pow(z1, 3) +
                                 6 * x0 * x1 * z0 * pow(z1, 2) * z2 - 6 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 2 * x0 * x1 * z0 * pow(z2, 3) + 2 * x0 * pow(x2, 3) * y0 * y1 -
                                 2 * x0 * pow(x2, 3) * y0 * y2 + 2 * x0 * pow(x2, 3) * z0 * z1 -
                                 2 * x0 * pow(x2, 3) * z0 * z2 + 2 * x0 * x2 * y0 * pow(y1, 3) -
                                 6 * x0 * x2 * y0 * pow(y1, 2) * y2 + 6 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 2 * x0 * x2 * y0 * y1 * pow(z1, 2) - 4 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 2 * x0 * x2 * y0 * y1 * pow(z2, 2) - 2 * x0 * x2 * y0 * pow(y2, 3) -
                                 2 * x0 * x2 * y0 * y2 * pow(z1, 2) + 4 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 2 * x0 * x2 * y0 * y2 * pow(z2, 2) + 2 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y1, 2) * z0 * z2 - 4 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 4 * x0 * x2 * y1 * y2 * z0 * z2 + 2 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y2, 2) * z0 * z2 + 2 * x0 * x2 * z0 * pow(z1, 3) -
                                 6 * x0 * x2 * z0 * pow(z1, 2) * z2 + 6 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 2 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 pow(x1, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x1, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x1, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x1, 2) * y0 * y1 * z0 * z1 + 2 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x1, 2) * y0 * y2 * z0 * z1 - 2 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) + pow(x1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x1, 2) * pow(z0, 2) * z1 * z2 + pow(x1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y1, 2) + 4 * x1 * x2 * pow(y0, 2) * y1 * y2 -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y2, 2) - 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                 8 * x1 * x2 * pow(y0, 2) * z1 * z2 - 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                 4 * x1 * x2 * y0 * y1 * z0 * z1 - 4 * x1 * x2 * y0 * y1 * z0 * z2 -
                                 4 * x1 * x2 * y0 * y2 * z0 * z1 + 4 * x1 * x2 * y0 * y2 * z0 * z2 -
                                 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) + 8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * x1 * x2 * pow(z0, 2) * z1 * z2 - 2 * x1 * x2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 pow(x2, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x2, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x2, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x2, 2) * y0 * y1 * z0 * z1 + 2 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x2, 2) * y0 * y2 * z0 * z1 - 2 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) + pow(x2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x2, 2) * pow(z0, 2) * z1 * z2 + pow(x2, 2) * pow(z0, 2) * pow(z2, 2) +
                                 pow(y0, 2) * pow(y1, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y1, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y1, 2) * pow(z2, 2) - 2 * pow(y0, 2) * y1 * y2 * pow(z1, 2) +
                                 4 * pow(y0, 2) * y1 * y2 * z1 * z2 - 2 * pow(y0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(y0, 2) * pow(y2, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y2, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y2, 2) * pow(z2, 2) + pow(y0, 2) * pow(z1, 4) -
                                 4 * pow(y0, 2) * pow(z1, 3) * z2 + 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) -
                                 4 * pow(y0, 2) * z1 * pow(z2, 3) + pow(y0, 2) * pow(z2, 4) -
                                 2 * y0 * pow(y1, 3) * z0 * z1 + 2 * y0 * pow(y1, 3) * z0 * z2 +
                                 6 * y0 * pow(y1, 2) * y2 * z0 * z1 - 6 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                 6 * y0 * y1 * pow(y2, 2) * z0 * z1 + 6 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                 2 * y0 * y1 * z0 * pow(z1, 3) + 6 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                 6 * y0 * y1 * z0 * z1 * pow(z2, 2) + 2 * y0 * y1 * z0 * pow(z2, 3) +
                                 2 * y0 * pow(y2, 3) * z0 * z1 - 2 * y0 * pow(y2, 3) * z0 * z2 +
                                 2 * y0 * y2 * z0 * pow(z1, 3) - 6 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                 6 * y0 * y2 * z0 * z1 * pow(z2, 2) - 2 * y0 * y2 * z0 * pow(z2, 3) +
                                 pow(y1, 4) * pow(z0, 2) - 4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) + pow(y1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y1, 2) * pow(z0, 2) * z1 * z2 + pow(y1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * y1 * y2 * pow(z0, 2) * z1 * z2 - 2 * y1 * y2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(y2, 4) * pow(z0, 2) + pow(y2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y2, 2) * pow(z0, 2) * z1 * z2 + pow(y2, 2) * pow(z0, 2) * pow(z2, 2)) -
                  y2 * z0 * sqrt(pow(x0, 2) * pow(x1, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x1, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x1, 2) * pow(y2, 2) + pow(x0, 2) * pow(x1, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x1, 2) * z1 * z2 + pow(x0, 2) * pow(x1, 2) * pow(z2, 2) -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y1, 2) + 4 * pow(x0, 2) * x1 * x2 * y1 * y2 -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y2, 2) - 2 * pow(x0, 2) * x1 * x2 * pow(z1, 2) +
                                 4 * pow(x0, 2) * x1 * x2 * z1 * z2 - 2 * pow(x0, 2) * x1 * x2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(x2, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x2, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x2, 2) * pow(y2, 2) + pow(x0, 2) * pow(x2, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x2, 2) * z1 * z2 + pow(x0, 2) * pow(x2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 2 * x0 * pow(x1, 3) * y0 * y1 +
                                 2 * x0 * pow(x1, 3) * y0 * y2 - 2 * x0 * pow(x1, 3) * z0 * z1 +
                                 2 * x0 * pow(x1, 3) * z0 * z2 + 6 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 6 * x0 * pow(x1, 2) * x2 * y0 * y2 + 6 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 6 * x0 * pow(x1, 2) * x2 * z0 * z2 - 6 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 6 * x0 * x1 * pow(x2, 2) * y0 * y2 - 6 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 6 * x0 * x1 * pow(x2, 2) * z0 * z2 - 2 * x0 * x1 * y0 * pow(y1, 3) +
                                 6 * x0 * x1 * y0 * pow(y1, 2) * y2 - 6 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 2 * x0 * x1 * y0 * y1 * pow(z1, 2) + 4 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 2 * x0 * x1 * y0 * y1 * pow(z2, 2) + 2 * x0 * x1 * y0 * pow(y2, 3) +
                                 2 * x0 * x1 * y0 * y2 * pow(z1, 2) - 4 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 2 * x0 * x1 * y0 * y2 * pow(z2, 2) - 2 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y1, 2) * z0 * z2 + 4 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 4 * x0 * x1 * y1 * y2 * z0 * z2 - 2 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y2, 2) * z0 * z2 - 2 * x0 * x1 * z0 * pow(z1, 3) +
                                 6 * x0 * x1 * z0 * pow(z1, 2) * z2 - 6 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 2 * x0 * x1 * z0 * pow(z2, 3) + 2 * x0 * pow(x2, 3) * y0 * y1 -
                                 2 * x0 * pow(x2, 3) * y0 * y2 + 2 * x0 * pow(x2, 3) * z0 * z1 -
                                 2 * x0 * pow(x2, 3) * z0 * z2 + 2 * x0 * x2 * y0 * pow(y1, 3) -
                                 6 * x0 * x2 * y0 * pow(y1, 2) * y2 + 6 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 2 * x0 * x2 * y0 * y1 * pow(z1, 2) - 4 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 2 * x0 * x2 * y0 * y1 * pow(z2, 2) - 2 * x0 * x2 * y0 * pow(y2, 3) -
                                 2 * x0 * x2 * y0 * y2 * pow(z1, 2) + 4 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 2 * x0 * x2 * y0 * y2 * pow(z2, 2) + 2 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y1, 2) * z0 * z2 - 4 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 4 * x0 * x2 * y1 * y2 * z0 * z2 + 2 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y2, 2) * z0 * z2 + 2 * x0 * x2 * z0 * pow(z1, 3) -
                                 6 * x0 * x2 * z0 * pow(z1, 2) * z2 + 6 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 2 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 pow(x1, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x1, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x1, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x1, 2) * y0 * y1 * z0 * z1 + 2 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x1, 2) * y0 * y2 * z0 * z1 - 2 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) + pow(x1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x1, 2) * pow(z0, 2) * z1 * z2 + pow(x1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y1, 2) + 4 * x1 * x2 * pow(y0, 2) * y1 * y2 -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y2, 2) - 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                 8 * x1 * x2 * pow(y0, 2) * z1 * z2 - 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                 4 * x1 * x2 * y0 * y1 * z0 * z1 - 4 * x1 * x2 * y0 * y1 * z0 * z2 -
                                 4 * x1 * x2 * y0 * y2 * z0 * z1 + 4 * x1 * x2 * y0 * y2 * z0 * z2 -
                                 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) + 8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * x1 * x2 * pow(z0, 2) * z1 * z2 - 2 * x1 * x2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 pow(x2, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x2, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x2, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x2, 2) * y0 * y1 * z0 * z1 + 2 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x2, 2) * y0 * y2 * z0 * z1 - 2 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) + pow(x2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x2, 2) * pow(z0, 2) * z1 * z2 + pow(x2, 2) * pow(z0, 2) * pow(z2, 2) +
                                 pow(y0, 2) * pow(y1, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y1, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y1, 2) * pow(z2, 2) - 2 * pow(y0, 2) * y1 * y2 * pow(z1, 2) +
                                 4 * pow(y0, 2) * y1 * y2 * z1 * z2 - 2 * pow(y0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(y0, 2) * pow(y2, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y2, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y2, 2) * pow(z2, 2) + pow(y0, 2) * pow(z1, 4) -
                                 4 * pow(y0, 2) * pow(z1, 3) * z2 + 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) -
                                 4 * pow(y0, 2) * z1 * pow(z2, 3) + pow(y0, 2) * pow(z2, 4) -
                                 2 * y0 * pow(y1, 3) * z0 * z1 + 2 * y0 * pow(y1, 3) * z0 * z2 +
                                 6 * y0 * pow(y1, 2) * y2 * z0 * z1 - 6 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                 6 * y0 * y1 * pow(y2, 2) * z0 * z1 + 6 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                 2 * y0 * y1 * z0 * pow(z1, 3) + 6 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                 6 * y0 * y1 * z0 * z1 * pow(z2, 2) + 2 * y0 * y1 * z0 * pow(z2, 3) +
                                 2 * y0 * pow(y2, 3) * z0 * z1 - 2 * y0 * pow(y2, 3) * z0 * z2 +
                                 2 * y0 * y2 * z0 * pow(z1, 3) - 6 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                 6 * y0 * y2 * z0 * z1 * pow(z2, 2) - 2 * y0 * y2 * z0 * pow(z2, 3) +
                                 pow(y1, 4) * pow(z0, 2) - 4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) + pow(y1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y1, 2) * pow(z0, 2) * z1 * z2 + pow(y1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * y1 * y2 * pow(z0, 2) * z1 * z2 - 2 * y1 * y2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(y2, 4) * pow(z0, 2) + pow(y2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y2, 2) * pow(z0, 2) * z1 * z2 + pow(y2, 2) * pow(z0, 2) * pow(z2, 2))) /
                 (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                  pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) - 2 * x0 * x1 * y0 * y1 +
                  2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 + 2 * x0 * x1 * z0 * z2 + 2 * x0 * x2 * y0 * y1 -
                  2 * x0 * x2 * y0 * y2 + 2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 + pow(x1, 2) * pow(y0, 2) +
                  pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) - 2 * x1 * x2 * pow(z0, 2) +
                  pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) + pow(y0, 2) * pow(z1, 2) -
                  2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) - 2 * y0 * y1 * z0 * z1 + 2 * y0 * y1 * z0 * z2 +
                  2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 + pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) +
                  pow(y2, 2) * pow(z0, 2));
            y3 = (pow(x0, 2) * pow(y1, 3) - 2 * pow(x0, 2) * pow(y1, 2) * y2 + pow(x0, 2) * y1 * pow(y2, 2) +
                  pow(x0, 2) * y1 * pow(z1, 2) - 2 * pow(x0, 2) * y1 * z1 * z2 + pow(x0, 2) * y1 * pow(z2, 2) -
                  2 * x0 * x1 * y0 * pow(y1, 2) + 2 * x0 * x1 * y0 * y1 * y2 - 2 * x0 * x1 * y1 * z0 * z1 +
                  2 * x0 * x1 * y1 * z0 * z2 + 2 * x0 * x2 * y0 * pow(y1, 2) - 2 * x0 * x2 * y0 * y1 * y2 +
                  2 * x0 * x2 * y1 * z0 * z1 - 2 * x0 * x2 * y1 * z0 * z2 + x0 * z1 *
                                                                            sqrt(pow(x0, 2) * pow(x1, 2) * pow(y1, 2) -
                                                                                 2 * pow(x0, 2) * pow(x1, 2) * y1 * y2 +
                                                                                 pow(x0, 2) * pow(x1, 2) * pow(y2, 2) +
                                                                                 pow(x0, 2) * pow(x1, 2) * pow(z1, 2) -
                                                                                 2 * pow(x0, 2) * pow(x1, 2) * z1 * z2 +
                                                                                 pow(x0, 2) * pow(x1, 2) * pow(z2, 2) -
                                                                                 2 * pow(x0, 2) * x1 * x2 * pow(y1, 2) +
                                                                                 4 * pow(x0, 2) * x1 * x2 * y1 * y2 -
                                                                                 2 * pow(x0, 2) * x1 * x2 * pow(y2, 2) -
                                                                                 2 * pow(x0, 2) * x1 * x2 * pow(z1, 2) +
                                                                                 4 * pow(x0, 2) * x1 * x2 * z1 * z2 -
                                                                                 2 * pow(x0, 2) * x1 * x2 * pow(z2, 2) +
                                                                                 pow(x0, 2) * pow(x2, 2) * pow(y1, 2) -
                                                                                 2 * pow(x0, 2) * pow(x2, 2) * y1 * y2 +
                                                                                 pow(x0, 2) * pow(x2, 2) * pow(y2, 2) +
                                                                                 pow(x0, 2) * pow(x2, 2) * pow(z1, 2) -
                                                                                 2 * pow(x0, 2) * pow(x2, 2) * z1 * z2 +
                                                                                 pow(x0, 2) * pow(x2, 2) * pow(z2, 2) +
                                                                                 pow(x0, 2) * pow(y1, 4) -
                                                                                 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                                                                 6 * pow(x0, 2) * pow(y1, 2) *
                                                                                 pow(y2, 2) +
                                                                                 2 * pow(x0, 2) * pow(y1, 2) *
                                                                                 pow(z1, 2) -
                                                                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 +
                                                                                 2 * pow(x0, 2) * pow(y1, 2) *
                                                                                 pow(z2, 2) -
                                                                                 4 * pow(x0, 2) * y1 * pow(y2, 3) -
                                                                                 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                                                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 -
                                                                                 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                                                                 pow(x0, 2) * pow(y2, 4) +
                                                                                 2 * pow(x0, 2) * pow(y2, 2) *
                                                                                 pow(z1, 2) -
                                                                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 +
                                                                                 2 * pow(x0, 2) * pow(y2, 2) *
                                                                                 pow(z2, 2) + pow(x0, 2) * pow(z1, 4) -
                                                                                 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                                                                 6 * pow(x0, 2) * pow(z1, 2) *
                                                                                 pow(z2, 2) -
                                                                                 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                                                                 pow(x0, 2) * pow(z2, 4) -
                                                                                 2 * x0 * pow(x1, 3) * y0 * y1 +
                                                                                 2 * x0 * pow(x1, 3) * y0 * y2 -
                                                                                 2 * x0 * pow(x1, 3) * z0 * z1 +
                                                                                 2 * x0 * pow(x1, 3) * z0 * z2 +
                                                                                 6 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                                                                 6 * x0 * pow(x1, 2) * x2 * y0 * y2 +
                                                                                 6 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                                                                 6 * x0 * pow(x1, 2) * x2 * z0 * z2 -
                                                                                 6 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                                                                 6 * x0 * x1 * pow(x2, 2) * y0 * y2 -
                                                                                 6 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                                                                 6 * x0 * x1 * pow(x2, 2) * z0 * z2 -
                                                                                 2 * x0 * x1 * y0 * pow(y1, 3) +
                                                                                 6 * x0 * x1 * y0 * pow(y1, 2) * y2 -
                                                                                 6 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                                                                 2 * x0 * x1 * y0 * y1 * pow(z1, 2) +
                                                                                 4 * x0 * x1 * y0 * y1 * z1 * z2 -
                                                                                 2 * x0 * x1 * y0 * y1 * pow(z2, 2) +
                                                                                 2 * x0 * x1 * y0 * pow(y2, 3) +
                                                                                 2 * x0 * x1 * y0 * y2 * pow(z1, 2) -
                                                                                 4 * x0 * x1 * y0 * y2 * z1 * z2 +
                                                                                 2 * x0 * x1 * y0 * y2 * pow(z2, 2) -
                                                                                 2 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                                                                 2 * x0 * x1 * pow(y1, 2) * z0 * z2 +
                                                                                 4 * x0 * x1 * y1 * y2 * z0 * z1 -
                                                                                 4 * x0 * x1 * y1 * y2 * z0 * z2 -
                                                                                 2 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                                                                 2 * x0 * x1 * pow(y2, 2) * z0 * z2 -
                                                                                 2 * x0 * x1 * z0 * pow(z1, 3) +
                                                                                 6 * x0 * x1 * z0 * pow(z1, 2) * z2 -
                                                                                 6 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                                                                 2 * x0 * x1 * z0 * pow(z2, 3) +
                                                                                 2 * x0 * pow(x2, 3) * y0 * y1 -
                                                                                 2 * x0 * pow(x2, 3) * y0 * y2 +
                                                                                 2 * x0 * pow(x2, 3) * z0 * z1 -
                                                                                 2 * x0 * pow(x2, 3) * z0 * z2 +
                                                                                 2 * x0 * x2 * y0 * pow(y1, 3) -
                                                                                 6 * x0 * x2 * y0 * pow(y1, 2) * y2 +
                                                                                 6 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                                                                 2 * x0 * x2 * y0 * y1 * pow(z1, 2) -
                                                                                 4 * x0 * x2 * y0 * y1 * z1 * z2 +
                                                                                 2 * x0 * x2 * y0 * y1 * pow(z2, 2) -
                                                                                 2 * x0 * x2 * y0 * pow(y2, 3) -
                                                                                 2 * x0 * x2 * y0 * y2 * pow(z1, 2) +
                                                                                 4 * x0 * x2 * y0 * y2 * z1 * z2 -
                                                                                 2 * x0 * x2 * y0 * y2 * pow(z2, 2) +
                                                                                 2 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                                                                 2 * x0 * x2 * pow(y1, 2) * z0 * z2 -
                                                                                 4 * x0 * x2 * y1 * y2 * z0 * z1 +
                                                                                 4 * x0 * x2 * y1 * y2 * z0 * z2 +
                                                                                 2 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                                                                 2 * x0 * x2 * pow(y2, 2) * z0 * z2 +
                                                                                 2 * x0 * x2 * z0 * pow(z1, 3) -
                                                                                 6 * x0 * x2 * z0 * pow(z1, 2) * z2 +
                                                                                 6 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                                                                 2 * x0 * x2 * z0 * pow(z2, 3) +
                                                                                 pow(x1, 4) * pow(y0, 2) +
                                                                                 pow(x1, 4) * pow(z0, 2) -
                                                                                 4 * pow(x1, 3) * x2 * pow(y0, 2) -
                                                                                 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                                                                 6 * pow(x1, 2) * pow(x2, 2) *
                                                                                 pow(y0, 2) +
                                                                                 6 * pow(x1, 2) * pow(x2, 2) *
                                                                                 pow(z0, 2) +
                                                                                 pow(x1, 2) * pow(y0, 2) * pow(y1, 2) -
                                                                                 2 * pow(x1, 2) * pow(y0, 2) * y1 * y2 +
                                                                                 pow(x1, 2) * pow(y0, 2) * pow(y2, 2) +
                                                                                 2 * pow(x1, 2) * pow(y0, 2) *
                                                                                 pow(z1, 2) -
                                                                                 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 +
                                                                                 2 * pow(x1, 2) * pow(y0, 2) *
                                                                                 pow(z2, 2) -
                                                                                 2 * pow(x1, 2) * y0 * y1 * z0 * z1 +
                                                                                 2 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                                                                 2 * pow(x1, 2) * y0 * y2 * z0 * z1 -
                                                                                 2 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                                                                 2 * pow(x1, 2) * pow(y1, 2) *
                                                                                 pow(z0, 2) -
                                                                                 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                                                                 2 * pow(x1, 2) * pow(y2, 2) *
                                                                                 pow(z0, 2) +
                                                                                 pow(x1, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                                 2 * pow(x1, 2) * pow(z0, 2) * z1 * z2 +
                                                                                 pow(x1, 2) * pow(z0, 2) * pow(z2, 2) -
                                                                                 4 * x1 * pow(x2, 3) * pow(y0, 2) -
                                                                                 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                                                                 2 * x1 * x2 * pow(y0, 2) * pow(y1, 2) +
                                                                                 4 * x1 * x2 * pow(y0, 2) * y1 * y2 -
                                                                                 2 * x1 * x2 * pow(y0, 2) * pow(y2, 2) -
                                                                                 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                                                                 8 * x1 * x2 * pow(y0, 2) * z1 * z2 -
                                                                                 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                                                                 4 * x1 * x2 * y0 * y1 * z0 * z1 -
                                                                                 4 * x1 * x2 * y0 * y1 * z0 * z2 -
                                                                                 4 * x1 * x2 * y0 * y2 * z0 * z1 +
                                                                                 4 * x1 * x2 * y0 * y2 * z0 * z2 -
                                                                                 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) +
                                                                                 8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                                                                 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) -
                                                                                 2 * x1 * x2 * pow(z0, 2) * pow(z1, 2) +
                                                                                 4 * x1 * x2 * pow(z0, 2) * z1 * z2 -
                                                                                 2 * x1 * x2 * pow(z0, 2) * pow(z2, 2) +
                                                                                 pow(x2, 4) * pow(y0, 2) +
                                                                                 pow(x2, 4) * pow(z0, 2) +
                                                                                 pow(x2, 2) * pow(y0, 2) * pow(y1, 2) -
                                                                                 2 * pow(x2, 2) * pow(y0, 2) * y1 * y2 +
                                                                                 pow(x2, 2) * pow(y0, 2) * pow(y2, 2) +
                                                                                 2 * pow(x2, 2) * pow(y0, 2) *
                                                                                 pow(z1, 2) -
                                                                                 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 +
                                                                                 2 * pow(x2, 2) * pow(y0, 2) *
                                                                                 pow(z2, 2) -
                                                                                 2 * pow(x2, 2) * y0 * y1 * z0 * z1 +
                                                                                 2 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                                                                 2 * pow(x2, 2) * y0 * y2 * z0 * z1 -
                                                                                 2 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                                                                 2 * pow(x2, 2) * pow(y1, 2) *
                                                                                 pow(z0, 2) -
                                                                                 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                                                                 2 * pow(x2, 2) * pow(y2, 2) *
                                                                                 pow(z0, 2) +
                                                                                 pow(x2, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                                 2 * pow(x2, 2) * pow(z0, 2) * z1 * z2 +
                                                                                 pow(x2, 2) * pow(z0, 2) * pow(z2, 2) +
                                                                                 pow(y0, 2) * pow(y1, 2) * pow(z1, 2) -
                                                                                 2 * pow(y0, 2) * pow(y1, 2) * z1 * z2 +
                                                                                 pow(y0, 2) * pow(y1, 2) * pow(z2, 2) -
                                                                                 2 * pow(y0, 2) * y1 * y2 * pow(z1, 2) +
                                                                                 4 * pow(y0, 2) * y1 * y2 * z1 * z2 -
                                                                                 2 * pow(y0, 2) * y1 * y2 * pow(z2, 2) +
                                                                                 pow(y0, 2) * pow(y2, 2) * pow(z1, 2) -
                                                                                 2 * pow(y0, 2) * pow(y2, 2) * z1 * z2 +
                                                                                 pow(y0, 2) * pow(y2, 2) * pow(z2, 2) +
                                                                                 pow(y0, 2) * pow(z1, 4) -
                                                                                 4 * pow(y0, 2) * pow(z1, 3) * z2 +
                                                                                 6 * pow(y0, 2) * pow(z1, 2) *
                                                                                 pow(z2, 2) -
                                                                                 4 * pow(y0, 2) * z1 * pow(z2, 3) +
                                                                                 pow(y0, 2) * pow(z2, 4) -
                                                                                 2 * y0 * pow(y1, 3) * z0 * z1 +
                                                                                 2 * y0 * pow(y1, 3) * z0 * z2 +
                                                                                 6 * y0 * pow(y1, 2) * y2 * z0 * z1 -
                                                                                 6 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                                                                 6 * y0 * y1 * pow(y2, 2) * z0 * z1 +
                                                                                 6 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                                                                 2 * y0 * y1 * z0 * pow(z1, 3) +
                                                                                 6 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                                                                 6 * y0 * y1 * z0 * z1 * pow(z2, 2) +
                                                                                 2 * y0 * y1 * z0 * pow(z2, 3) +
                                                                                 2 * y0 * pow(y2, 3) * z0 * z1 -
                                                                                 2 * y0 * pow(y2, 3) * z0 * z2 +
                                                                                 2 * y0 * y2 * z0 * pow(z1, 3) -
                                                                                 6 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                                                                 6 * y0 * y2 * z0 * z1 * pow(z2, 2) -
                                                                                 2 * y0 * y2 * z0 * pow(z2, 3) +
                                                                                 pow(y1, 4) * pow(z0, 2) -
                                                                                 4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                                                                 6 * pow(y1, 2) * pow(y2, 2) *
                                                                                 pow(z0, 2) +
                                                                                 pow(y1, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                                 2 * pow(y1, 2) * pow(z0, 2) * z1 * z2 +
                                                                                 pow(y1, 2) * pow(z0, 2) * pow(z2, 2) -
                                                                                 4 * y1 * pow(y2, 3) * pow(z0, 2) -
                                                                                 2 * y1 * y2 * pow(z0, 2) * pow(z1, 2) +
                                                                                 4 * y1 * y2 * pow(z0, 2) * z1 * z2 -
                                                                                 2 * y1 * y2 * pow(z0, 2) * pow(z2, 2) +
                                                                                 pow(y2, 4) * pow(z0, 2) +
                                                                                 pow(y2, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                                 2 * pow(y2, 2) * pow(z0, 2) * z1 * z2 +
                                                                                 pow(y2, 2) * pow(z0, 2) * pow(z2, 2)) -
                  x0 * z2 * sqrt(pow(x0, 2) * pow(x1, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x1, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x1, 2) * pow(y2, 2) + pow(x0, 2) * pow(x1, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x1, 2) * z1 * z2 + pow(x0, 2) * pow(x1, 2) * pow(z2, 2) -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y1, 2) + 4 * pow(x0, 2) * x1 * x2 * y1 * y2 -
                                 2 * pow(x0, 2) * x1 * x2 * pow(y2, 2) - 2 * pow(x0, 2) * x1 * x2 * pow(z1, 2) +
                                 4 * pow(x0, 2) * x1 * x2 * z1 * z2 - 2 * pow(x0, 2) * x1 * x2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(x2, 2) * pow(y1, 2) - 2 * pow(x0, 2) * pow(x2, 2) * y1 * y2 +
                                 pow(x0, 2) * pow(x2, 2) * pow(y2, 2) + pow(x0, 2) * pow(x2, 2) * pow(z1, 2) -
                                 2 * pow(x0, 2) * pow(x2, 2) * z1 * z2 + pow(x0, 2) * pow(x2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 2 * x0 * pow(x1, 3) * y0 * y1 +
                                 2 * x0 * pow(x1, 3) * y0 * y2 - 2 * x0 * pow(x1, 3) * z0 * z1 +
                                 2 * x0 * pow(x1, 3) * z0 * z2 + 6 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 6 * x0 * pow(x1, 2) * x2 * y0 * y2 + 6 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 6 * x0 * pow(x1, 2) * x2 * z0 * z2 - 6 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 6 * x0 * x1 * pow(x2, 2) * y0 * y2 - 6 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 6 * x0 * x1 * pow(x2, 2) * z0 * z2 - 2 * x0 * x1 * y0 * pow(y1, 3) +
                                 6 * x0 * x1 * y0 * pow(y1, 2) * y2 - 6 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 2 * x0 * x1 * y0 * y1 * pow(z1, 2) + 4 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 2 * x0 * x1 * y0 * y1 * pow(z2, 2) + 2 * x0 * x1 * y0 * pow(y2, 3) +
                                 2 * x0 * x1 * y0 * y2 * pow(z1, 2) - 4 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 2 * x0 * x1 * y0 * y2 * pow(z2, 2) - 2 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y1, 2) * z0 * z2 + 4 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 4 * x0 * x1 * y1 * y2 * z0 * z2 - 2 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 2 * x0 * x1 * pow(y2, 2) * z0 * z2 - 2 * x0 * x1 * z0 * pow(z1, 3) +
                                 6 * x0 * x1 * z0 * pow(z1, 2) * z2 - 6 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 2 * x0 * x1 * z0 * pow(z2, 3) + 2 * x0 * pow(x2, 3) * y0 * y1 -
                                 2 * x0 * pow(x2, 3) * y0 * y2 + 2 * x0 * pow(x2, 3) * z0 * z1 -
                                 2 * x0 * pow(x2, 3) * z0 * z2 + 2 * x0 * x2 * y0 * pow(y1, 3) -
                                 6 * x0 * x2 * y0 * pow(y1, 2) * y2 + 6 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 2 * x0 * x2 * y0 * y1 * pow(z1, 2) - 4 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 2 * x0 * x2 * y0 * y1 * pow(z2, 2) - 2 * x0 * x2 * y0 * pow(y2, 3) -
                                 2 * x0 * x2 * y0 * y2 * pow(z1, 2) + 4 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 2 * x0 * x2 * y0 * y2 * pow(z2, 2) + 2 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y1, 2) * z0 * z2 - 4 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 4 * x0 * x2 * y1 * y2 * z0 * z2 + 2 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 2 * x0 * x2 * pow(y2, 2) * z0 * z2 + 2 * x0 * x2 * z0 * pow(z1, 3) -
                                 6 * x0 * x2 * z0 * pow(z1, 2) * z2 + 6 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 2 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 pow(x1, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x1, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x1, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x1, 2) * y0 * y1 * z0 * z1 + 2 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x1, 2) * y0 * y2 * z0 * z1 - 2 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) + pow(x1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x1, 2) * pow(z0, 2) * z1 * z2 + pow(x1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y1, 2) + 4 * x1 * x2 * pow(y0, 2) * y1 * y2 -
                                 2 * x1 * x2 * pow(y0, 2) * pow(y2, 2) - 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                 8 * x1 * x2 * pow(y0, 2) * z1 * z2 - 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                 4 * x1 * x2 * y0 * y1 * z0 * z1 - 4 * x1 * x2 * y0 * y1 * z0 * z2 -
                                 4 * x1 * x2 * y0 * y2 * z0 * z1 + 4 * x1 * x2 * y0 * y2 * z0 * z2 -
                                 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) + 8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * x1 * x2 * pow(z0, 2) * z1 * z2 - 2 * x1 * x2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 pow(x2, 2) * pow(y0, 2) * pow(y1, 2) - 2 * pow(x2, 2) * pow(y0, 2) * y1 * y2 +
                                 pow(x2, 2) * pow(y0, 2) * pow(y2, 2) + 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) -
                                 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 + 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) -
                                 2 * pow(x2, 2) * y0 * y1 * z0 * z1 + 2 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                 2 * pow(x2, 2) * y0 * y2 * z0 * z1 - 2 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) - 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) + pow(x2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(x2, 2) * pow(z0, 2) * z1 * z2 + pow(x2, 2) * pow(z0, 2) * pow(z2, 2) +
                                 pow(y0, 2) * pow(y1, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y1, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y1, 2) * pow(z2, 2) - 2 * pow(y0, 2) * y1 * y2 * pow(z1, 2) +
                                 4 * pow(y0, 2) * y1 * y2 * z1 * z2 - 2 * pow(y0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(y0, 2) * pow(y2, 2) * pow(z1, 2) - 2 * pow(y0, 2) * pow(y2, 2) * z1 * z2 +
                                 pow(y0, 2) * pow(y2, 2) * pow(z2, 2) + pow(y0, 2) * pow(z1, 4) -
                                 4 * pow(y0, 2) * pow(z1, 3) * z2 + 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) -
                                 4 * pow(y0, 2) * z1 * pow(z2, 3) + pow(y0, 2) * pow(z2, 4) -
                                 2 * y0 * pow(y1, 3) * z0 * z1 + 2 * y0 * pow(y1, 3) * z0 * z2 +
                                 6 * y0 * pow(y1, 2) * y2 * z0 * z1 - 6 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                 6 * y0 * y1 * pow(y2, 2) * z0 * z1 + 6 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                 2 * y0 * y1 * z0 * pow(z1, 3) + 6 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                 6 * y0 * y1 * z0 * z1 * pow(z2, 2) + 2 * y0 * y1 * z0 * pow(z2, 3) +
                                 2 * y0 * pow(y2, 3) * z0 * z1 - 2 * y0 * pow(y2, 3) * z0 * z2 +
                                 2 * y0 * y2 * z0 * pow(z1, 3) - 6 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                 6 * y0 * y2 * z0 * z1 * pow(z2, 2) - 2 * y0 * y2 * z0 * pow(z2, 3) +
                                 pow(y1, 4) * pow(z0, 2) - 4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) + pow(y1, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y1, 2) * pow(z0, 2) * z1 * z2 + pow(y1, 2) * pow(z0, 2) * pow(z2, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) * pow(z1, 2) +
                                 4 * y1 * y2 * pow(z0, 2) * z1 * z2 - 2 * y1 * y2 * pow(z0, 2) * pow(z2, 2) +
                                 pow(y2, 4) * pow(z0, 2) + pow(y2, 2) * pow(z0, 2) * pow(z1, 2) -
                                 2 * pow(y2, 2) * pow(z0, 2) * z1 * z2 + pow(y2, 2) * pow(z0, 2) * pow(z2, 2)) +
                  pow(x1, 2) * pow(y0, 2) * y1 + pow(x1, 2) * y1 * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) * y1 -
                  2 * x1 * x2 * y1 * pow(z0, 2) - x1 * z0 * sqrt(pow(x0, 2) * pow(x1, 2) * pow(y1, 2) -
                                                                 2 * pow(x0, 2) * pow(x1, 2) * y1 * y2 +
                                                                 pow(x0, 2) * pow(x1, 2) * pow(y2, 2) +
                                                                 pow(x0, 2) * pow(x1, 2) * pow(z1, 2) -
                                                                 2 * pow(x0, 2) * pow(x1, 2) * z1 * z2 +
                                                                 pow(x0, 2) * pow(x1, 2) * pow(z2, 2) -
                                                                 2 * pow(x0, 2) * x1 * x2 * pow(y1, 2) +
                                                                 4 * pow(x0, 2) * x1 * x2 * y1 * y2 -
                                                                 2 * pow(x0, 2) * x1 * x2 * pow(y2, 2) -
                                                                 2 * pow(x0, 2) * x1 * x2 * pow(z1, 2) +
                                                                 4 * pow(x0, 2) * x1 * x2 * z1 * z2 -
                                                                 2 * pow(x0, 2) * x1 * x2 * pow(z2, 2) +
                                                                 pow(x0, 2) * pow(x2, 2) * pow(y1, 2) -
                                                                 2 * pow(x0, 2) * pow(x2, 2) * y1 * y2 +
                                                                 pow(x0, 2) * pow(x2, 2) * pow(y2, 2) +
                                                                 pow(x0, 2) * pow(x2, 2) * pow(z1, 2) -
                                                                 2 * pow(x0, 2) * pow(x2, 2) * z1 * z2 +
                                                                 pow(x0, 2) * pow(x2, 2) * pow(z2, 2) +
                                                                 pow(x0, 2) * pow(y1, 4) -
                                                                 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) +
                                                                 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 +
                                                                 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                                                 4 * pow(x0, 2) * y1 * pow(y2, 3) -
                                                                 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 -
                                                                 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                                                 pow(x0, 2) * pow(y2, 4) +
                                                                 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 +
                                                                 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                                                 pow(x0, 2) * pow(z1, 4) -
                                                                 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) -
                                                                 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                                                 pow(x0, 2) * pow(z2, 4) -
                                                                 2 * x0 * pow(x1, 3) * y0 * y1 +
                                                                 2 * x0 * pow(x1, 3) * y0 * y2 -
                                                                 2 * x0 * pow(x1, 3) * z0 * z1 +
                                                                 2 * x0 * pow(x1, 3) * z0 * z2 +
                                                                 6 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                                                 6 * x0 * pow(x1, 2) * x2 * y0 * y2 +
                                                                 6 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                                                 6 * x0 * pow(x1, 2) * x2 * z0 * z2 -
                                                                 6 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                                                 6 * x0 * x1 * pow(x2, 2) * y0 * y2 -
                                                                 6 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                                                 6 * x0 * x1 * pow(x2, 2) * z0 * z2 -
                                                                 2 * x0 * x1 * y0 * pow(y1, 3) +
                                                                 6 * x0 * x1 * y0 * pow(y1, 2) * y2 -
                                                                 6 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                                                 2 * x0 * x1 * y0 * y1 * pow(z1, 2) +
                                                                 4 * x0 * x1 * y0 * y1 * z1 * z2 -
                                                                 2 * x0 * x1 * y0 * y1 * pow(z2, 2) +
                                                                 2 * x0 * x1 * y0 * pow(y2, 3) +
                                                                 2 * x0 * x1 * y0 * y2 * pow(z1, 2) -
                                                                 4 * x0 * x1 * y0 * y2 * z1 * z2 +
                                                                 2 * x0 * x1 * y0 * y2 * pow(z2, 2) -
                                                                 2 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                                                 2 * x0 * x1 * pow(y1, 2) * z0 * z2 +
                                                                 4 * x0 * x1 * y1 * y2 * z0 * z1 -
                                                                 4 * x0 * x1 * y1 * y2 * z0 * z2 -
                                                                 2 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                                                 2 * x0 * x1 * pow(y2, 2) * z0 * z2 -
                                                                 2 * x0 * x1 * z0 * pow(z1, 3) +
                                                                 6 * x0 * x1 * z0 * pow(z1, 2) * z2 -
                                                                 6 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                                                 2 * x0 * x1 * z0 * pow(z2, 3) +
                                                                 2 * x0 * pow(x2, 3) * y0 * y1 -
                                                                 2 * x0 * pow(x2, 3) * y0 * y2 +
                                                                 2 * x0 * pow(x2, 3) * z0 * z1 -
                                                                 2 * x0 * pow(x2, 3) * z0 * z2 +
                                                                 2 * x0 * x2 * y0 * pow(y1, 3) -
                                                                 6 * x0 * x2 * y0 * pow(y1, 2) * y2 +
                                                                 6 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                                                 2 * x0 * x2 * y0 * y1 * pow(z1, 2) -
                                                                 4 * x0 * x2 * y0 * y1 * z1 * z2 +
                                                                 2 * x0 * x2 * y0 * y1 * pow(z2, 2) -
                                                                 2 * x0 * x2 * y0 * pow(y2, 3) -
                                                                 2 * x0 * x2 * y0 * y2 * pow(z1, 2) +
                                                                 4 * x0 * x2 * y0 * y2 * z1 * z2 -
                                                                 2 * x0 * x2 * y0 * y2 * pow(z2, 2) +
                                                                 2 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                                                 2 * x0 * x2 * pow(y1, 2) * z0 * z2 -
                                                                 4 * x0 * x2 * y1 * y2 * z0 * z1 +
                                                                 4 * x0 * x2 * y1 * y2 * z0 * z2 +
                                                                 2 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                                                 2 * x0 * x2 * pow(y2, 2) * z0 * z2 +
                                                                 2 * x0 * x2 * z0 * pow(z1, 3) -
                                                                 6 * x0 * x2 * z0 * pow(z1, 2) * z2 +
                                                                 6 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                                                 2 * x0 * x2 * z0 * pow(z2, 3) +
                                                                 pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                                                 4 * pow(x1, 3) * x2 * pow(y0, 2) -
                                                                 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) +
                                                                 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                                                 pow(x1, 2) * pow(y0, 2) * pow(y1, 2) -
                                                                 2 * pow(x1, 2) * pow(y0, 2) * y1 * y2 +
                                                                 pow(x1, 2) * pow(y0, 2) * pow(y2, 2) +
                                                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) -
                                                                 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 +
                                                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) -
                                                                 2 * pow(x1, 2) * y0 * y1 * z0 * z1 +
                                                                 2 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                                                 2 * pow(x1, 2) * y0 * y2 * z0 * z1 -
                                                                 2 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                                                 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) -
                                                                 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                                                 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) +
                                                                 pow(x1, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                 2 * pow(x1, 2) * pow(z0, 2) * z1 * z2 +
                                                                 pow(x1, 2) * pow(z0, 2) * pow(z2, 2) -
                                                                 4 * x1 * pow(x2, 3) * pow(y0, 2) -
                                                                 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                                                 2 * x1 * x2 * pow(y0, 2) * pow(y1, 2) +
                                                                 4 * x1 * x2 * pow(y0, 2) * y1 * y2 -
                                                                 2 * x1 * x2 * pow(y0, 2) * pow(y2, 2) -
                                                                 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                                                 8 * x1 * x2 * pow(y0, 2) * z1 * z2 -
                                                                 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                                                 4 * x1 * x2 * y0 * y1 * z0 * z1 -
                                                                 4 * x1 * x2 * y0 * y1 * z0 * z2 -
                                                                 4 * x1 * x2 * y0 * y2 * z0 * z1 +
                                                                 4 * x1 * x2 * y0 * y2 * z0 * z2 -
                                                                 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) +
                                                                 8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                                                 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) -
                                                                 2 * x1 * x2 * pow(z0, 2) * pow(z1, 2) +
                                                                 4 * x1 * x2 * pow(z0, 2) * z1 * z2 -
                                                                 2 * x1 * x2 * pow(z0, 2) * pow(z2, 2) +
                                                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                                                 pow(x2, 2) * pow(y0, 2) * pow(y1, 2) -
                                                                 2 * pow(x2, 2) * pow(y0, 2) * y1 * y2 +
                                                                 pow(x2, 2) * pow(y0, 2) * pow(y2, 2) +
                                                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) -
                                                                 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 +
                                                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) -
                                                                 2 * pow(x2, 2) * y0 * y1 * z0 * z1 +
                                                                 2 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                                                 2 * pow(x2, 2) * y0 * y2 * z0 * z1 -
                                                                 2 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                                                 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) -
                                                                 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                                                 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) +
                                                                 pow(x2, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                 2 * pow(x2, 2) * pow(z0, 2) * z1 * z2 +
                                                                 pow(x2, 2) * pow(z0, 2) * pow(z2, 2) +
                                                                 pow(y0, 2) * pow(y1, 2) * pow(z1, 2) -
                                                                 2 * pow(y0, 2) * pow(y1, 2) * z1 * z2 +
                                                                 pow(y0, 2) * pow(y1, 2) * pow(z2, 2) -
                                                                 2 * pow(y0, 2) * y1 * y2 * pow(z1, 2) +
                                                                 4 * pow(y0, 2) * y1 * y2 * z1 * z2 -
                                                                 2 * pow(y0, 2) * y1 * y2 * pow(z2, 2) +
                                                                 pow(y0, 2) * pow(y2, 2) * pow(z1, 2) -
                                                                 2 * pow(y0, 2) * pow(y2, 2) * z1 * z2 +
                                                                 pow(y0, 2) * pow(y2, 2) * pow(z2, 2) +
                                                                 pow(y0, 2) * pow(z1, 4) -
                                                                 4 * pow(y0, 2) * pow(z1, 3) * z2 +
                                                                 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) -
                                                                 4 * pow(y0, 2) * z1 * pow(z2, 3) +
                                                                 pow(y0, 2) * pow(z2, 4) -
                                                                 2 * y0 * pow(y1, 3) * z0 * z1 +
                                                                 2 * y0 * pow(y1, 3) * z0 * z2 +
                                                                 6 * y0 * pow(y1, 2) * y2 * z0 * z1 -
                                                                 6 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                                                 6 * y0 * y1 * pow(y2, 2) * z0 * z1 +
                                                                 6 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                                                 2 * y0 * y1 * z0 * pow(z1, 3) +
                                                                 6 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                                                 6 * y0 * y1 * z0 * z1 * pow(z2, 2) +
                                                                 2 * y0 * y1 * z0 * pow(z2, 3) +
                                                                 2 * y0 * pow(y2, 3) * z0 * z1 -
                                                                 2 * y0 * pow(y2, 3) * z0 * z2 +
                                                                 2 * y0 * y2 * z0 * pow(z1, 3) -
                                                                 6 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                                                 6 * y0 * y2 * z0 * z1 * pow(z2, 2) -
                                                                 2 * y0 * y2 * z0 * pow(z2, 3) +
                                                                 pow(y1, 4) * pow(z0, 2) -
                                                                 4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                                                 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) +
                                                                 pow(y1, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                 2 * pow(y1, 2) * pow(z0, 2) * z1 * z2 +
                                                                 pow(y1, 2) * pow(z0, 2) * pow(z2, 2) -
                                                                 4 * y1 * pow(y2, 3) * pow(z0, 2) -
                                                                 2 * y1 * y2 * pow(z0, 2) * pow(z1, 2) +
                                                                 4 * y1 * y2 * pow(z0, 2) * z1 * z2 -
                                                                 2 * y1 * y2 * pow(z0, 2) * pow(z2, 2) +
                                                                 pow(y2, 4) * pow(z0, 2) +
                                                                 pow(y2, 2) * pow(z0, 2) * pow(z1, 2) -
                                                                 2 * pow(y2, 2) * pow(z0, 2) * z1 * z2 +
                                                                 pow(y2, 2) * pow(z0, 2) * pow(z2, 2)) +
                  pow(x2, 2) * pow(y0, 2) * y1 + pow(x2, 2) * y1 * pow(z0, 2) + x2 * z0 * sqrt(pow(x0, 2) * pow(x1, 2) *
                                                                                               pow(y1, 2) -
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(x1, 2) * y1 * y2 +
                                                                                               pow(x0, 2) * pow(x1, 2) *
                                                                                               pow(y2, 2) +
                                                                                               pow(x0, 2) * pow(x1, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(x1, 2) * z1 * z2 +
                                                                                               pow(x0, 2) * pow(x1, 2) *
                                                                                               pow(z2, 2) -
                                                                                               2 * pow(x0, 2) * x1 *
                                                                                               x2 * pow(y1, 2) +
                                                                                               4 * pow(x0, 2) * x1 *
                                                                                               x2 * y1 * y2 -
                                                                                               2 * pow(x0, 2) * x1 *
                                                                                               x2 * pow(y2, 2) -
                                                                                               2 * pow(x0, 2) * x1 *
                                                                                               x2 * pow(z1, 2) +
                                                                                               4 * pow(x0, 2) * x1 *
                                                                                               x2 * z1 * z2 -
                                                                                               2 * pow(x0, 2) * x1 *
                                                                                               x2 * pow(z2, 2) +
                                                                                               pow(x0, 2) * pow(x2, 2) *
                                                                                               pow(y1, 2) -
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(x2, 2) * y1 * y2 +
                                                                                               pow(x0, 2) * pow(x2, 2) *
                                                                                               pow(y2, 2) +
                                                                                               pow(x0, 2) * pow(x2, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(x2, 2) * z1 * z2 +
                                                                                               pow(x0, 2) * pow(x2, 2) *
                                                                                               pow(z2, 2) +
                                                                                               pow(x0, 2) * pow(y1, 4) -
                                                                                               4 * pow(x0, 2) *
                                                                                               pow(y1, 3) * y2 +
                                                                                               6 * pow(x0, 2) *
                                                                                               pow(y1, 2) * pow(y2, 2) +
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(y1, 2) * pow(z1, 2) -
                                                                                               4 * pow(x0, 2) *
                                                                                               pow(y1, 2) * z1 * z2 +
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(y1, 2) * pow(z2, 2) -
                                                                                               4 * pow(x0, 2) * y1 *
                                                                                               pow(y2, 3) -
                                                                                               4 * pow(x0, 2) * y1 *
                                                                                               y2 * pow(z1, 2) +
                                                                                               8 * pow(x0, 2) * y1 *
                                                                                               y2 * z1 * z2 -
                                                                                               4 * pow(x0, 2) * y1 *
                                                                                               y2 * pow(z2, 2) +
                                                                                               pow(x0, 2) * pow(y2, 4) +
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(y2, 2) * pow(z1, 2) -
                                                                                               4 * pow(x0, 2) *
                                                                                               pow(y2, 2) * z1 * z2 +
                                                                                               2 * pow(x0, 2) *
                                                                                               pow(y2, 2) * pow(z2, 2) +
                                                                                               pow(x0, 2) * pow(z1, 4) -
                                                                                               4 * pow(x0, 2) *
                                                                                               pow(z1, 3) * z2 +
                                                                                               6 * pow(x0, 2) *
                                                                                               pow(z1, 2) * pow(z2, 2) -
                                                                                               4 * pow(x0, 2) * z1 *
                                                                                               pow(z2, 3) +
                                                                                               pow(x0, 2) * pow(z2, 4) -
                                                                                               2 * x0 * pow(x1, 3) *
                                                                                               y0 * y1 +
                                                                                               2 * x0 * pow(x1, 3) *
                                                                                               y0 * y2 -
                                                                                               2 * x0 * pow(x1, 3) *
                                                                                               z0 * z1 +
                                                                                               2 * x0 * pow(x1, 3) *
                                                                                               z0 * z2 +
                                                                                               6 * x0 * pow(x1, 2) *
                                                                                               x2 * y0 * y1 -
                                                                                               6 * x0 * pow(x1, 2) *
                                                                                               x2 * y0 * y2 +
                                                                                               6 * x0 * pow(x1, 2) *
                                                                                               x2 * z0 * z1 -
                                                                                               6 * x0 * pow(x1, 2) *
                                                                                               x2 * z0 * z2 -
                                                                                               6 * x0 * x1 *
                                                                                               pow(x2, 2) * y0 * y1 +
                                                                                               6 * x0 * x1 *
                                                                                               pow(x2, 2) * y0 * y2 -
                                                                                               6 * x0 * x1 *
                                                                                               pow(x2, 2) * z0 * z1 +
                                                                                               6 * x0 * x1 *
                                                                                               pow(x2, 2) * z0 * z2 -
                                                                                               2 * x0 * x1 * y0 *
                                                                                               pow(y1, 3) +
                                                                                               6 * x0 * x1 * y0 *
                                                                                               pow(y1, 2) * y2 -
                                                                                               6 * x0 * x1 * y0 * y1 *
                                                                                               pow(y2, 2) -
                                                                                               2 * x0 * x1 * y0 * y1 *
                                                                                               pow(z1, 2) +
                                                                                               4 * x0 * x1 * y0 * y1 *
                                                                                               z1 * z2 -
                                                                                               2 * x0 * x1 * y0 * y1 *
                                                                                               pow(z2, 2) +
                                                                                               2 * x0 * x1 * y0 *
                                                                                               pow(y2, 3) +
                                                                                               2 * x0 * x1 * y0 * y2 *
                                                                                               pow(z1, 2) -
                                                                                               4 * x0 * x1 * y0 * y2 *
                                                                                               z1 * z2 +
                                                                                               2 * x0 * x1 * y0 * y2 *
                                                                                               pow(z2, 2) -
                                                                                               2 * x0 * x1 *
                                                                                               pow(y1, 2) * z0 * z1 +
                                                                                               2 * x0 * x1 *
                                                                                               pow(y1, 2) * z0 * z2 +
                                                                                               4 * x0 * x1 * y1 * y2 *
                                                                                               z0 * z1 -
                                                                                               4 * x0 * x1 * y1 * y2 *
                                                                                               z0 * z2 - 2 * x0 * x1 *
                                                                                                         pow(y2, 2) *
                                                                                                         z0 * z1 +
                                                                                               2 * x0 * x1 *
                                                                                               pow(y2, 2) * z0 * z2 -
                                                                                               2 * x0 * x1 * z0 *
                                                                                               pow(z1, 3) +
                                                                                               6 * x0 * x1 * z0 *
                                                                                               pow(z1, 2) * z2 -
                                                                                               6 * x0 * x1 * z0 * z1 *
                                                                                               pow(z2, 2) +
                                                                                               2 * x0 * x1 * z0 *
                                                                                               pow(z2, 3) +
                                                                                               2 * x0 * pow(x2, 3) *
                                                                                               y0 * y1 -
                                                                                               2 * x0 * pow(x2, 3) *
                                                                                               y0 * y2 +
                                                                                               2 * x0 * pow(x2, 3) *
                                                                                               z0 * z1 -
                                                                                               2 * x0 * pow(x2, 3) *
                                                                                               z0 * z2 +
                                                                                               2 * x0 * x2 * y0 *
                                                                                               pow(y1, 3) -
                                                                                               6 * x0 * x2 * y0 *
                                                                                               pow(y1, 2) * y2 +
                                                                                               6 * x0 * x2 * y0 * y1 *
                                                                                               pow(y2, 2) +
                                                                                               2 * x0 * x2 * y0 * y1 *
                                                                                               pow(z1, 2) -
                                                                                               4 * x0 * x2 * y0 * y1 *
                                                                                               z1 * z2 +
                                                                                               2 * x0 * x2 * y0 * y1 *
                                                                                               pow(z2, 2) -
                                                                                               2 * x0 * x2 * y0 *
                                                                                               pow(y2, 3) -
                                                                                               2 * x0 * x2 * y0 * y2 *
                                                                                               pow(z1, 2) +
                                                                                               4 * x0 * x2 * y0 * y2 *
                                                                                               z1 * z2 -
                                                                                               2 * x0 * x2 * y0 * y2 *
                                                                                               pow(z2, 2) +
                                                                                               2 * x0 * x2 *
                                                                                               pow(y1, 2) * z0 * z1 -
                                                                                               2 * x0 * x2 *
                                                                                               pow(y1, 2) * z0 * z2 -
                                                                                               4 * x0 * x2 * y1 * y2 *
                                                                                               z0 * z1 +
                                                                                               4 * x0 * x2 * y1 * y2 *
                                                                                               z0 * z2 + 2 * x0 * x2 *
                                                                                                         pow(y2, 2) *
                                                                                                         z0 * z1 -
                                                                                               2 * x0 * x2 *
                                                                                               pow(y2, 2) * z0 * z2 +
                                                                                               2 * x0 * x2 * z0 *
                                                                                               pow(z1, 3) -
                                                                                               6 * x0 * x2 * z0 *
                                                                                               pow(z1, 2) * z2 +
                                                                                               6 * x0 * x2 * z0 * z1 *
                                                                                               pow(z2, 2) -
                                                                                               2 * x0 * x2 * z0 *
                                                                                               pow(z2, 3) +
                                                                                               pow(x1, 4) * pow(y0, 2) +
                                                                                               pow(x1, 4) * pow(z0, 2) -
                                                                                               4 * pow(x1, 3) * x2 *
                                                                                               pow(y0, 2) -
                                                                                               4 * pow(x1, 3) * x2 *
                                                                                               pow(z0, 2) +
                                                                                               6 * pow(x1, 2) *
                                                                                               pow(x2, 2) * pow(y0, 2) +
                                                                                               6 * pow(x1, 2) *
                                                                                               pow(x2, 2) * pow(z0, 2) +
                                                                                               pow(x1, 2) * pow(y0, 2) *
                                                                                               pow(y1, 2) -
                                                                                               2 * pow(x1, 2) *
                                                                                               pow(y0, 2) * y1 * y2 +
                                                                                               pow(x1, 2) * pow(y0, 2) *
                                                                                               pow(y2, 2) +
                                                                                               2 * pow(x1, 2) *
                                                                                               pow(y0, 2) * pow(z1, 2) -
                                                                                               4 * pow(x1, 2) *
                                                                                               pow(y0, 2) * z1 * z2 +
                                                                                               2 * pow(x1, 2) *
                                                                                               pow(y0, 2) * pow(z2, 2) -
                                                                                               2 * pow(x1, 2) * y0 *
                                                                                               y1 * z0 * z1 +
                                                                                               2 * pow(x1, 2) * y0 *
                                                                                               y1 * z0 * z2 +
                                                                                               2 * pow(x1, 2) * y0 *
                                                                                               y2 * z0 * z1 -
                                                                                               2 * pow(x1, 2) * y0 *
                                                                                               y2 * z0 * z2 +
                                                                                               2 * pow(x1, 2) *
                                                                                               pow(y1, 2) * pow(z0, 2) -
                                                                                               4 * pow(x1, 2) * y1 *
                                                                                               y2 * pow(z0, 2) +
                                                                                               2 * pow(x1, 2) *
                                                                                               pow(y2, 2) * pow(z0, 2) +
                                                                                               pow(x1, 2) * pow(z0, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(x1, 2) *
                                                                                               pow(z0, 2) * z1 * z2 +
                                                                                               pow(x1, 2) * pow(z0, 2) *
                                                                                               pow(z2, 2) -
                                                                                               4 * x1 * pow(x2, 3) *
                                                                                               pow(y0, 2) -
                                                                                               4 * x1 * pow(x2, 3) *
                                                                                               pow(z0, 2) -
                                                                                               2 * x1 * x2 *
                                                                                               pow(y0, 2) * pow(y1, 2) +
                                                                                               4 * x1 * x2 *
                                                                                               pow(y0, 2) * y1 * y2 -
                                                                                               2 * x1 * x2 *
                                                                                               pow(y0, 2) * pow(y2, 2) -
                                                                                               4 * x1 * x2 *
                                                                                               pow(y0, 2) * pow(z1, 2) +
                                                                                               8 * x1 * x2 *
                                                                                               pow(y0, 2) * z1 * z2 -
                                                                                               4 * x1 * x2 *
                                                                                               pow(y0, 2) * pow(z2, 2) +
                                                                                               4 * x1 * x2 * y0 * y1 *
                                                                                               z0 * z1 -
                                                                                               4 * x1 * x2 * y0 * y1 *
                                                                                               z0 * z2 -
                                                                                               4 * x1 * x2 * y0 * y2 *
                                                                                               z0 * z1 +
                                                                                               4 * x1 * x2 * y0 * y2 *
                                                                                               z0 * z2 - 4 * x1 * x2 *
                                                                                                         pow(y1, 2) *
                                                                                                         pow(z0, 2) +
                                                                                               8 * x1 * x2 * y1 * y2 *
                                                                                               pow(z0, 2) -
                                                                                               4 * x1 * x2 *
                                                                                               pow(y2, 2) * pow(z0, 2) -
                                                                                               2 * x1 * x2 *
                                                                                               pow(z0, 2) * pow(z1, 2) +
                                                                                               4 * x1 * x2 *
                                                                                               pow(z0, 2) * z1 * z2 -
                                                                                               2 * x1 * x2 *
                                                                                               pow(z0, 2) * pow(z2, 2) +
                                                                                               pow(x2, 4) * pow(y0, 2) +
                                                                                               pow(x2, 4) * pow(z0, 2) +
                                                                                               pow(x2, 2) * pow(y0, 2) *
                                                                                               pow(y1, 2) -
                                                                                               2 * pow(x2, 2) *
                                                                                               pow(y0, 2) * y1 * y2 +
                                                                                               pow(x2, 2) * pow(y0, 2) *
                                                                                               pow(y2, 2) +
                                                                                               2 * pow(x2, 2) *
                                                                                               pow(y0, 2) * pow(z1, 2) -
                                                                                               4 * pow(x2, 2) *
                                                                                               pow(y0, 2) * z1 * z2 +
                                                                                               2 * pow(x2, 2) *
                                                                                               pow(y0, 2) * pow(z2, 2) -
                                                                                               2 * pow(x2, 2) * y0 *
                                                                                               y1 * z0 * z1 +
                                                                                               2 * pow(x2, 2) * y0 *
                                                                                               y1 * z0 * z2 +
                                                                                               2 * pow(x2, 2) * y0 *
                                                                                               y2 * z0 * z1 -
                                                                                               2 * pow(x2, 2) * y0 *
                                                                                               y2 * z0 * z2 +
                                                                                               2 * pow(x2, 2) *
                                                                                               pow(y1, 2) * pow(z0, 2) -
                                                                                               4 * pow(x2, 2) * y1 *
                                                                                               y2 * pow(z0, 2) +
                                                                                               2 * pow(x2, 2) *
                                                                                               pow(y2, 2) * pow(z0, 2) +
                                                                                               pow(x2, 2) * pow(z0, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(x2, 2) *
                                                                                               pow(z0, 2) * z1 * z2 +
                                                                                               pow(x2, 2) * pow(z0, 2) *
                                                                                               pow(z2, 2) +
                                                                                               pow(y0, 2) * pow(y1, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(y0, 2) *
                                                                                               pow(y1, 2) * z1 * z2 +
                                                                                               pow(y0, 2) * pow(y1, 2) *
                                                                                               pow(z2, 2) -
                                                                                               2 * pow(y0, 2) * y1 *
                                                                                               y2 * pow(z1, 2) +
                                                                                               4 * pow(y0, 2) * y1 *
                                                                                               y2 * z1 * z2 -
                                                                                               2 * pow(y0, 2) * y1 *
                                                                                               y2 * pow(z2, 2) +
                                                                                               pow(y0, 2) * pow(y2, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(y0, 2) *
                                                                                               pow(y2, 2) * z1 * z2 +
                                                                                               pow(y0, 2) * pow(y2, 2) *
                                                                                               pow(z2, 2) +
                                                                                               pow(y0, 2) * pow(z1, 4) -
                                                                                               4 * pow(y0, 2) *
                                                                                               pow(z1, 3) * z2 +
                                                                                               6 * pow(y0, 2) *
                                                                                               pow(z1, 2) * pow(z2, 2) -
                                                                                               4 * pow(y0, 2) * z1 *
                                                                                               pow(z2, 3) +
                                                                                               pow(y0, 2) * pow(z2, 4) -
                                                                                               2 * y0 * pow(y1, 3) *
                                                                                               z0 * z1 +
                                                                                               2 * y0 * pow(y1, 3) *
                                                                                               z0 * z2 +
                                                                                               6 * y0 * pow(y1, 2) *
                                                                                               y2 * z0 * z1 -
                                                                                               6 * y0 * pow(y1, 2) *
                                                                                               y2 * z0 * z2 -
                                                                                               6 * y0 * y1 *
                                                                                               pow(y2, 2) * z0 * z1 +
                                                                                               6 * y0 * y1 *
                                                                                               pow(y2, 2) * z0 * z2 -
                                                                                               2 * y0 * y1 * z0 *
                                                                                               pow(z1, 3) +
                                                                                               6 * y0 * y1 * z0 *
                                                                                               pow(z1, 2) * z2 -
                                                                                               6 * y0 * y1 * z0 * z1 *
                                                                                               pow(z2, 2) +
                                                                                               2 * y0 * y1 * z0 *
                                                                                               pow(z2, 3) +
                                                                                               2 * y0 * pow(y2, 3) *
                                                                                               z0 * z1 -
                                                                                               2 * y0 * pow(y2, 3) *
                                                                                               z0 * z2 +
                                                                                               2 * y0 * y2 * z0 *
                                                                                               pow(z1, 3) -
                                                                                               6 * y0 * y2 * z0 *
                                                                                               pow(z1, 2) * z2 +
                                                                                               6 * y0 * y2 * z0 * z1 *
                                                                                               pow(z2, 2) -
                                                                                               2 * y0 * y2 * z0 *
                                                                                               pow(z2, 3) +
                                                                                               pow(y1, 4) * pow(z0, 2) -
                                                                                               4 * pow(y1, 3) * y2 *
                                                                                               pow(z0, 2) +
                                                                                               6 * pow(y1, 2) *
                                                                                               pow(y2, 2) * pow(z0, 2) +
                                                                                               pow(y1, 2) * pow(z0, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(y1, 2) *
                                                                                               pow(z0, 2) * z1 * z2 +
                                                                                               pow(y1, 2) * pow(z0, 2) *
                                                                                               pow(z2, 2) -
                                                                                               4 * y1 * pow(y2, 3) *
                                                                                               pow(z0, 2) -
                                                                                               2 * y1 * y2 *
                                                                                               pow(z0, 2) * pow(z1, 2) +
                                                                                               4 * y1 * y2 *
                                                                                               pow(z0, 2) * z1 * z2 -
                                                                                               2 * y1 * y2 *
                                                                                               pow(z0, 2) * pow(z2, 2) +
                                                                                               pow(y2, 4) * pow(z0, 2) +
                                                                                               pow(y2, 2) * pow(z0, 2) *
                                                                                               pow(z1, 2) -
                                                                                               2 * pow(y2, 2) *
                                                                                               pow(z0, 2) * z1 * z2 +
                                                                                               pow(y2, 2) * pow(z0, 2) *
                                                                                               pow(z2, 2)) +
                  pow(y0, 2) * y1 * pow(z1, 2) - 2 * pow(y0, 2) * y1 * z1 * z2 + pow(y0, 2) * y1 * pow(z2, 2) -
                  2 * y0 * pow(y1, 2) * z0 * z1 + 2 * y0 * pow(y1, 2) * z0 * z2 + 2 * y0 * y1 * y2 * z0 * z1 -
                  2 * y0 * y1 * y2 * z0 * z2 + pow(y1, 3) * pow(z0, 2) - 2 * pow(y1, 2) * y2 * pow(z0, 2) +
                  y1 * pow(y2, 2) * pow(z0, 2)) /
                 (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                  pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) - 2 * x0 * x1 * y0 * y1 +
                  2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 + 2 * x0 * x1 * z0 * z2 + 2 * x0 * x2 * y0 * y1 -
                  2 * x0 * x2 * y0 * y2 + 2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 + pow(x1, 2) * pow(y0, 2) +
                  pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) - 2 * x1 * x2 * pow(z0, 2) +
                  pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) + pow(y0, 2) * pow(z1, 2) -
                  2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) - 2 * y0 * y1 * z0 * z1 + 2 * y0 * y1 * z0 * z2 +
                  2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 + pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) +
                  pow(y2, 2) * pow(z0, 2));
            z3 = z1 - sqrt((pow(x1, 2) - 2 * x1 * x2 + pow(x2, 2) + pow(y1, 2) - 2 * y1 * y2 + pow(y2, 2) + pow(z1, 2) -
                            2 * z1 * z2 + pow(z2, 2)) *
                           (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                            pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) -
                            2 * x0 * x1 * y0 * y1 + 2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 +
                            2 * x0 * x1 * z0 * z2 + 2 * x0 * x2 * y0 * y1 - 2 * x0 * x2 * y0 * y2 +
                            2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 + pow(x1, 2) * pow(y0, 2) +
                            pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) - 2 * x1 * x2 * pow(z0, 2) +
                            pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) + pow(y0, 2) * pow(z1, 2) -
                            2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) - 2 * y0 * y1 * z0 * z1 +
                            2 * y0 * y1 * z0 * z2 + 2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 +
                            pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) + pow(y2, 2) * pow(z0, 2))) *
                      (x0 * y1 - x0 * y2 - x1 * y0 + x2 * y0) /
                      (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                       pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) -
                       2 * x0 * x1 * y0 * y1 + 2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 + 2 * x0 * x1 * z0 * z2 +
                       2 * x0 * x2 * y0 * y1 - 2 * x0 * x2 * y0 * y2 + 2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 +
                       pow(x1, 2) * pow(y0, 2) + pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) -
                       2 * x1 * x2 * pow(z0, 2) + pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) +
                       pow(y0, 2) * pow(z1, 2) - 2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) -
                       2 * y0 * y1 * z0 * z1 + 2 * y0 * y1 * z0 * z2 + 2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 +
                       pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) + pow(y2, 2) * pow(z0, 2));
            x4 = (pow(x0, 2) * x1 * pow(y1, 2) - 2 * pow(x0, 2) * x1 * y1 * y2 + pow(x0, 2) * x1 * pow(y2, 2) +
                  pow(x0, 2) * x1 * pow(z1, 2) - 2 * pow(x0, 2) * x1 * z1 * z2 + pow(x0, 2) * x1 * pow(z2, 2) -
                  x0 * pow(x1, 2) * y0 * y1 + x0 * pow(x1, 2) * y0 * y2 - x0 * pow(x1, 2) * z0 * z1 +
                  x0 * pow(x1, 2) * z0 * z2 + x0 * pow(x2, 2) * y0 * y1 - x0 * pow(x2, 2) * y0 * y2 +
                  x0 * pow(x2, 2) * z0 * z1 - x0 * pow(x2, 2) * z0 * z2 + x0 * y0 * pow(y1, 3) -
                  3 * x0 * y0 * pow(y1, 2) * y2 + 3 * x0 * y0 * y1 * pow(y2, 2) + x0 * y0 * y1 * pow(z1, 2) -
                  2 * x0 * y0 * y1 * z1 * z2 + x0 * y0 * y1 * pow(z2, 2) - x0 * y0 * pow(y2, 3) -
                  x0 * y0 * y2 * pow(z1, 2) + 2 * x0 * y0 * y2 * z1 * z2 - x0 * y0 * y2 * pow(z2, 2) +
                  x0 * pow(y1, 2) * z0 * z1 - x0 * pow(y1, 2) * z0 * z2 - 2 * x0 * y1 * y2 * z0 * z1 +
                  2 * x0 * y1 * y2 * z0 * z2 + x0 * pow(y2, 2) * z0 * z1 - x0 * pow(y2, 2) * z0 * z2 +
                  x0 * z0 * pow(z1, 3) - 3 * x0 * z0 * pow(z1, 2) * z2 + 3 * x0 * z0 * z1 * pow(z2, 2) -
                  x0 * z0 * pow(z2, 3) + pow(x1, 2) * x2 * pow(y0, 2) + pow(x1, 2) * x2 * pow(z0, 2) -
                  2 * x1 * pow(x2, 2) * pow(y0, 2) - 2 * x1 * pow(x2, 2) * pow(z0, 2) - x1 * pow(y0, 2) * pow(y1, 2) +
                  2 * x1 * pow(y0, 2) * y1 * y2 - x1 * pow(y0, 2) * pow(y2, 2) - 2 * x1 * y0 * y1 * z0 * z1 +
                  2 * x1 * y0 * y1 * z0 * z2 + 2 * x1 * y0 * y2 * z0 * z1 - 2 * x1 * y0 * y2 * z0 * z2 -
                  x1 * pow(z0, 2) * pow(z1, 2) + 2 * x1 * pow(z0, 2) * z1 * z2 - x1 * pow(z0, 2) * pow(z2, 2) +
                  pow(x2, 3) * pow(y0, 2) + pow(x2, 3) * pow(z0, 2) + x2 * pow(y0, 2) * pow(y1, 2) -
                  2 * x2 * pow(y0, 2) * y1 * y2 + x2 * pow(y0, 2) * pow(y2, 2) + x2 * pow(y0, 2) * pow(z1, 2) -
                  2 * x2 * pow(y0, 2) * z1 * z2 + x2 * pow(y0, 2) * pow(z2, 2) + x2 * pow(y1, 2) * pow(z0, 2) -
                  2 * x2 * y1 * y2 * pow(z0, 2) + x2 * pow(y2, 2) * pow(z0, 2) + x2 * pow(z0, 2) * pow(z1, 2) -
                  2 * x2 * pow(z0, 2) * z1 * z2 + x2 * pow(z0, 2) * pow(z2, 2) - y0 * z1 *
                                                                                 sqrt(-pow(x0, 2) * pow(x1, 4) +
                                                                                      4 * pow(x0, 2) * pow(x1, 3) * x2 -
                                                                                      6 * pow(x0, 2) * pow(x1, 2) *
                                                                                      pow(x2, 2) +
                                                                                      4 * pow(x0, 2) * x1 * pow(x2, 3) -
                                                                                      pow(x0, 2) * pow(x2, 4) +
                                                                                      pow(x0, 2) * pow(y1, 4) -
                                                                                      4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                                                                      6 * pow(x0, 2) * pow(y1, 2) *
                                                                                      pow(y2, 2) +
                                                                                      2 * pow(x0, 2) * pow(y1, 2) *
                                                                                      pow(z1, 2) -
                                                                                      4 * pow(x0, 2) * pow(y1, 2) * z1 *
                                                                                      z2 + 2 * pow(x0, 2) * pow(y1, 2) *
                                                                                           pow(z2, 2) -
                                                                                      4 * pow(x0, 2) * y1 * pow(y2, 3) -
                                                                                      4 * pow(x0, 2) * y1 * y2 *
                                                                                      pow(z1, 2) +
                                                                                      8 * pow(x0, 2) * y1 * y2 * z1 *
                                                                                      z2 - 4 * pow(x0, 2) * y1 * y2 *
                                                                                           pow(z2, 2) +
                                                                                      pow(x0, 2) * pow(y2, 4) +
                                                                                      2 * pow(x0, 2) * pow(y2, 2) *
                                                                                      pow(z1, 2) -
                                                                                      4 * pow(x0, 2) * pow(y2, 2) * z1 *
                                                                                      z2 + 2 * pow(x0, 2) * pow(y2, 2) *
                                                                                           pow(z2, 2) +
                                                                                      pow(x0, 2) * pow(z1, 4) -
                                                                                      4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                                                                      6 * pow(x0, 2) * pow(z1, 2) *
                                                                                      pow(z2, 2) -
                                                                                      4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                                                                      pow(x0, 2) * pow(z2, 4) -
                                                                                      4 * x0 * pow(x1, 3) * y0 * y1 +
                                                                                      4 * x0 * pow(x1, 3) * y0 * y2 -
                                                                                      4 * x0 * pow(x1, 3) * z0 * z1 +
                                                                                      4 * x0 * pow(x1, 3) * z0 * z2 +
                                                                                      12 * x0 * pow(x1, 2) * x2 * y0 *
                                                                                      y1 -
                                                                                      12 * x0 * pow(x1, 2) * x2 * y0 *
                                                                                      y2 +
                                                                                      12 * x0 * pow(x1, 2) * x2 * z0 *
                                                                                      z1 -
                                                                                      12 * x0 * pow(x1, 2) * x2 * z0 *
                                                                                      z2 -
                                                                                      12 * x0 * x1 * pow(x2, 2) * y0 *
                                                                                      y1 +
                                                                                      12 * x0 * x1 * pow(x2, 2) * y0 *
                                                                                      y2 -
                                                                                      12 * x0 * x1 * pow(x2, 2) * z0 *
                                                                                      z1 +
                                                                                      12 * x0 * x1 * pow(x2, 2) * z0 *
                                                                                      z2 -
                                                                                      4 * x0 * x1 * y0 * pow(y1, 3) +
                                                                                      12 * x0 * x1 * y0 * pow(y1, 2) *
                                                                                      y2 - 12 * x0 * x1 * y0 * y1 *
                                                                                           pow(y2, 2) -
                                                                                      4 * x0 * x1 * y0 * y1 *
                                                                                      pow(z1, 2) +
                                                                                      8 * x0 * x1 * y0 * y1 * z1 * z2 -
                                                                                      4 * x0 * x1 * y0 * y1 *
                                                                                      pow(z2, 2) +
                                                                                      4 * x0 * x1 * y0 * pow(y2, 3) +
                                                                                      4 * x0 * x1 * y0 * y2 *
                                                                                      pow(z1, 2) -
                                                                                      8 * x0 * x1 * y0 * y2 * z1 * z2 +
                                                                                      4 * x0 * x1 * y0 * y2 *
                                                                                      pow(z2, 2) -
                                                                                      4 * x0 * x1 * pow(y1, 2) * z0 *
                                                                                      z1 +
                                                                                      4 * x0 * x1 * pow(y1, 2) * z0 *
                                                                                      z2 +
                                                                                      8 * x0 * x1 * y1 * y2 * z0 * z1 -
                                                                                      8 * x0 * x1 * y1 * y2 * z0 * z2 -
                                                                                      4 * x0 * x1 * pow(y2, 2) * z0 *
                                                                                      z1 +
                                                                                      4 * x0 * x1 * pow(y2, 2) * z0 *
                                                                                      z2 -
                                                                                      4 * x0 * x1 * z0 * pow(z1, 3) +
                                                                                      12 * x0 * x1 * z0 * pow(z1, 2) *
                                                                                      z2 - 12 * x0 * x1 * z0 * z1 *
                                                                                           pow(z2, 2) +
                                                                                      4 * x0 * x1 * z0 * pow(z2, 3) +
                                                                                      4 * x0 * pow(x2, 3) * y0 * y1 -
                                                                                      4 * x0 * pow(x2, 3) * y0 * y2 +
                                                                                      4 * x0 * pow(x2, 3) * z0 * z1 -
                                                                                      4 * x0 * pow(x2, 3) * z0 * z2 +
                                                                                      4 * x0 * x2 * y0 * pow(y1, 3) -
                                                                                      12 * x0 * x2 * y0 * pow(y1, 2) *
                                                                                      y2 + 12 * x0 * x2 * y0 * y1 *
                                                                                           pow(y2, 2) +
                                                                                      4 * x0 * x2 * y0 * y1 *
                                                                                      pow(z1, 2) -
                                                                                      8 * x0 * x2 * y0 * y1 * z1 * z2 +
                                                                                      4 * x0 * x2 * y0 * y1 *
                                                                                      pow(z2, 2) -
                                                                                      4 * x0 * x2 * y0 * pow(y2, 3) -
                                                                                      4 * x0 * x2 * y0 * y2 *
                                                                                      pow(z1, 2) +
                                                                                      8 * x0 * x2 * y0 * y2 * z1 * z2 -
                                                                                      4 * x0 * x2 * y0 * y2 *
                                                                                      pow(z2, 2) +
                                                                                      4 * x0 * x2 * pow(y1, 2) * z0 *
                                                                                      z1 -
                                                                                      4 * x0 * x2 * pow(y1, 2) * z0 *
                                                                                      z2 -
                                                                                      8 * x0 * x2 * y1 * y2 * z0 * z1 +
                                                                                      8 * x0 * x2 * y1 * y2 * z0 * z2 +
                                                                                      4 * x0 * x2 * pow(y2, 2) * z0 *
                                                                                      z1 -
                                                                                      4 * x0 * x2 * pow(y2, 2) * z0 *
                                                                                      z2 +
                                                                                      4 * x0 * x2 * z0 * pow(z1, 3) -
                                                                                      12 * x0 * x2 * z0 * pow(z1, 2) *
                                                                                      z2 + 12 * x0 * x2 * z0 * z1 *
                                                                                           pow(z2, 2) -
                                                                                      4 * x0 * x2 * z0 * pow(z2, 3) +
                                                                                      pow(x1, 4) * pow(y0, 2) +
                                                                                      pow(x1, 4) * pow(z0, 2) -
                                                                                      4 * pow(x1, 3) * x2 * pow(y0, 2) -
                                                                                      4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                                                                      6 * pow(x1, 2) * pow(x2, 2) *
                                                                                      pow(y0, 2) +
                                                                                      6 * pow(x1, 2) * pow(x2, 2) *
                                                                                      pow(z0, 2) +
                                                                                      2 * pow(x1, 2) * pow(y0, 2) *
                                                                                      pow(z1, 2) -
                                                                                      4 * pow(x1, 2) * pow(y0, 2) * z1 *
                                                                                      z2 + 2 * pow(x1, 2) * pow(y0, 2) *
                                                                                           pow(z2, 2) -
                                                                                      4 * pow(x1, 2) * y0 * y1 * z0 *
                                                                                      z1 +
                                                                                      4 * pow(x1, 2) * y0 * y1 * z0 *
                                                                                      z2 +
                                                                                      4 * pow(x1, 2) * y0 * y2 * z0 *
                                                                                      z1 -
                                                                                      4 * pow(x1, 2) * y0 * y2 * z0 *
                                                                                      z2 + 2 * pow(x1, 2) * pow(y1, 2) *
                                                                                           pow(z0, 2) -
                                                                                      4 * pow(x1, 2) * y1 * y2 *
                                                                                      pow(z0, 2) +
                                                                                      2 * pow(x1, 2) * pow(y2, 2) *
                                                                                      pow(z0, 2) -
                                                                                      4 * x1 * pow(x2, 3) * pow(y0, 2) -
                                                                                      4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                                                                      4 * x1 * x2 * pow(y0, 2) *
                                                                                      pow(z1, 2) +
                                                                                      8 * x1 * x2 * pow(y0, 2) * z1 *
                                                                                      z2 - 4 * x1 * x2 * pow(y0, 2) *
                                                                                           pow(z2, 2) +
                                                                                      8 * x1 * x2 * y0 * y1 * z0 * z1 -
                                                                                      8 * x1 * x2 * y0 * y1 * z0 * z2 -
                                                                                      8 * x1 * x2 * y0 * y2 * z0 * z1 +
                                                                                      8 * x1 * x2 * y0 * y2 * z0 * z2 -
                                                                                      4 * x1 * x2 * pow(y1, 2) *
                                                                                      pow(z0, 2) +
                                                                                      8 * x1 * x2 * y1 * y2 *
                                                                                      pow(z0, 2) -
                                                                                      4 * x1 * x2 * pow(y2, 2) *
                                                                                      pow(z0, 2) +
                                                                                      pow(x2, 4) * pow(y0, 2) +
                                                                                      pow(x2, 4) * pow(z0, 2) +
                                                                                      2 * pow(x2, 2) * pow(y0, 2) *
                                                                                      pow(z1, 2) -
                                                                                      4 * pow(x2, 2) * pow(y0, 2) * z1 *
                                                                                      z2 + 2 * pow(x2, 2) * pow(y0, 2) *
                                                                                           pow(z2, 2) -
                                                                                      4 * pow(x2, 2) * y0 * y1 * z0 *
                                                                                      z1 +
                                                                                      4 * pow(x2, 2) * y0 * y1 * z0 *
                                                                                      z2 +
                                                                                      4 * pow(x2, 2) * y0 * y2 * z0 *
                                                                                      z1 -
                                                                                      4 * pow(x2, 2) * y0 * y2 * z0 *
                                                                                      z2 + 2 * pow(x2, 2) * pow(y1, 2) *
                                                                                           pow(z0, 2) -
                                                                                      4 * pow(x2, 2) * y1 * y2 *
                                                                                      pow(z0, 2) +
                                                                                      2 * pow(x2, 2) * pow(y2, 2) *
                                                                                      pow(z0, 2) -
                                                                                      pow(y0, 2) * pow(y1, 4) +
                                                                                      4 * pow(y0, 2) * pow(y1, 3) * y2 -
                                                                                      6 * pow(y0, 2) * pow(y1, 2) *
                                                                                      pow(y2, 2) +
                                                                                      4 * pow(y0, 2) * y1 * pow(y2, 3) -
                                                                                      pow(y0, 2) * pow(y2, 4) +
                                                                                      pow(y0, 2) * pow(z1, 4) -
                                                                                      4 * pow(y0, 2) * pow(z1, 3) * z2 +
                                                                                      6 * pow(y0, 2) * pow(z1, 2) *
                                                                                      pow(z2, 2) -
                                                                                      4 * pow(y0, 2) * z1 * pow(z2, 3) +
                                                                                      pow(y0, 2) * pow(z2, 4) -
                                                                                      4 * y0 * pow(y1, 3) * z0 * z1 +
                                                                                      4 * y0 * pow(y1, 3) * z0 * z2 +
                                                                                      12 * y0 * pow(y1, 2) * y2 * z0 *
                                                                                      z1 -
                                                                                      12 * y0 * pow(y1, 2) * y2 * z0 *
                                                                                      z2 -
                                                                                      12 * y0 * y1 * pow(y2, 2) * z0 *
                                                                                      z1 +
                                                                                      12 * y0 * y1 * pow(y2, 2) * z0 *
                                                                                      z2 -
                                                                                      4 * y0 * y1 * z0 * pow(z1, 3) +
                                                                                      12 * y0 * y1 * z0 * pow(z1, 2) *
                                                                                      z2 - 12 * y0 * y1 * z0 * z1 *
                                                                                           pow(z2, 2) +
                                                                                      4 * y0 * y1 * z0 * pow(z2, 3) +
                                                                                      4 * y0 * pow(y2, 3) * z0 * z1 -
                                                                                      4 * y0 * pow(y2, 3) * z0 * z2 +
                                                                                      4 * y0 * y2 * z0 * pow(z1, 3) -
                                                                                      12 * y0 * y2 * z0 * pow(z1, 2) *
                                                                                      z2 + 12 * y0 * y2 * z0 * z1 *
                                                                                           pow(z2, 2) -
                                                                                      4 * y0 * y2 * z0 * pow(z2, 3) +
                                                                                      pow(y1, 4) * pow(z0, 2) -
                                                                                      4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                                                                      6 * pow(y1, 2) * pow(y2, 2) *
                                                                                      pow(z0, 2) -
                                                                                      4 * y1 * pow(y2, 3) * pow(z0, 2) +
                                                                                      pow(y2, 4) * pow(z0, 2) -
                                                                                      pow(z0, 2) * pow(z1, 4) +
                                                                                      4 * pow(z0, 2) * pow(z1, 3) * z2 -
                                                                                      6 * pow(z0, 2) * pow(z1, 2) *
                                                                                      pow(z2, 2) +
                                                                                      4 * pow(z0, 2) * z1 * pow(z2, 3) -
                                                                                      pow(z0, 2) * pow(z2, 4)) +
                  y0 * z2 * sqrt(-pow(x0, 2) * pow(x1, 4) + 4 * pow(x0, 2) * pow(x1, 3) * x2 -
                                 6 * pow(x0, 2) * pow(x1, 2) * pow(x2, 2) + 4 * pow(x0, 2) * x1 * pow(x2, 3) -
                                 pow(x0, 2) * pow(x2, 4) + pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 4 * x0 * pow(x1, 3) * y0 * y1 +
                                 4 * x0 * pow(x1, 3) * y0 * y2 - 4 * x0 * pow(x1, 3) * z0 * z1 +
                                 4 * x0 * pow(x1, 3) * z0 * z2 + 12 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 12 * x0 * pow(x1, 2) * x2 * y0 * y2 + 12 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 12 * x0 * pow(x1, 2) * x2 * z0 * z2 - 12 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 12 * x0 * x1 * pow(x2, 2) * y0 * y2 - 12 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 12 * x0 * x1 * pow(x2, 2) * z0 * z2 - 4 * x0 * x1 * y0 * pow(y1, 3) +
                                 12 * x0 * x1 * y0 * pow(y1, 2) * y2 - 12 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 4 * x0 * x1 * y0 * y1 * pow(z1, 2) + 8 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 4 * x0 * x1 * y0 * y1 * pow(z2, 2) + 4 * x0 * x1 * y0 * pow(y2, 3) +
                                 4 * x0 * x1 * y0 * y2 * pow(z1, 2) - 8 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 4 * x0 * x1 * y0 * y2 * pow(z2, 2) - 4 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 4 * x0 * x1 * pow(y1, 2) * z0 * z2 + 8 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 8 * x0 * x1 * y1 * y2 * z0 * z2 - 4 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 4 * x0 * x1 * pow(y2, 2) * z0 * z2 - 4 * x0 * x1 * z0 * pow(z1, 3) +
                                 12 * x0 * x1 * z0 * pow(z1, 2) * z2 - 12 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 4 * x0 * x1 * z0 * pow(z2, 3) + 4 * x0 * pow(x2, 3) * y0 * y1 -
                                 4 * x0 * pow(x2, 3) * y0 * y2 + 4 * x0 * pow(x2, 3) * z0 * z1 -
                                 4 * x0 * pow(x2, 3) * z0 * z2 + 4 * x0 * x2 * y0 * pow(y1, 3) -
                                 12 * x0 * x2 * y0 * pow(y1, 2) * y2 + 12 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 4 * x0 * x2 * y0 * y1 * pow(z1, 2) - 8 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 4 * x0 * x2 * y0 * y1 * pow(z2, 2) - 4 * x0 * x2 * y0 * pow(y2, 3) -
                                 4 * x0 * x2 * y0 * y2 * pow(z1, 2) + 8 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 4 * x0 * x2 * y0 * y2 * pow(z2, 2) + 4 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 4 * x0 * x2 * pow(y1, 2) * z0 * z2 - 8 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 8 * x0 * x2 * y1 * y2 * z0 * z2 + 4 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 4 * x0 * x2 * pow(y2, 2) * z0 * z2 + 4 * x0 * x2 * z0 * pow(z1, 3) -
                                 12 * x0 * x2 * z0 * pow(z1, 2) * z2 + 12 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 4 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) - 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 +
                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) - 4 * pow(x1, 2) * y0 * y1 * z0 * z1 +
                                 4 * pow(x1, 2) * y0 * y1 * z0 * z2 + 4 * pow(x1, 2) * y0 * y2 * z0 * z1 -
                                 4 * pow(x1, 2) * y0 * y2 * z0 * z2 + 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) -
                                 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) + 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) + 8 * x1 * x2 * pow(y0, 2) * z1 * z2 -
                                 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) + 8 * x1 * x2 * y0 * y1 * z0 * z1 -
                                 8 * x1 * x2 * y0 * y1 * z0 * z2 - 8 * x1 * x2 * y0 * y2 * z0 * z1 +
                                 8 * x1 * x2 * y0 * y2 * z0 * z2 - 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) +
                                 8 * x1 * x2 * y1 * y2 * pow(z0, 2) - 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) - 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 +
                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) - 4 * pow(x2, 2) * y0 * y1 * z0 * z1 +
                                 4 * pow(x2, 2) * y0 * y1 * z0 * z2 + 4 * pow(x2, 2) * y0 * y2 * z0 * z1 -
                                 4 * pow(x2, 2) * y0 * y2 * z0 * z2 + 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) -
                                 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) + 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) -
                                 pow(y0, 2) * pow(y1, 4) + 4 * pow(y0, 2) * pow(y1, 3) * y2 -
                                 6 * pow(y0, 2) * pow(y1, 2) * pow(y2, 2) + 4 * pow(y0, 2) * y1 * pow(y2, 3) -
                                 pow(y0, 2) * pow(y2, 4) + pow(y0, 2) * pow(z1, 4) - 4 * pow(y0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(y0, 2) * z1 * pow(z2, 3) +
                                 pow(y0, 2) * pow(z2, 4) - 4 * y0 * pow(y1, 3) * z0 * z1 +
                                 4 * y0 * pow(y1, 3) * z0 * z2 + 12 * y0 * pow(y1, 2) * y2 * z0 * z1 -
                                 12 * y0 * pow(y1, 2) * y2 * z0 * z2 - 12 * y0 * y1 * pow(y2, 2) * z0 * z1 +
                                 12 * y0 * y1 * pow(y2, 2) * z0 * z2 - 4 * y0 * y1 * z0 * pow(z1, 3) +
                                 12 * y0 * y1 * z0 * pow(z1, 2) * z2 - 12 * y0 * y1 * z0 * z1 * pow(z2, 2) +
                                 4 * y0 * y1 * z0 * pow(z2, 3) + 4 * y0 * pow(y2, 3) * z0 * z1 -
                                 4 * y0 * pow(y2, 3) * z0 * z2 + 4 * y0 * y2 * z0 * pow(z1, 3) -
                                 12 * y0 * y2 * z0 * pow(z1, 2) * z2 + 12 * y0 * y2 * z0 * z1 * pow(z2, 2) -
                                 4 * y0 * y2 * z0 * pow(z2, 3) + pow(y1, 4) * pow(z0, 2) -
                                 4 * pow(y1, 3) * y2 * pow(z0, 2) + 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) + pow(y2, 4) * pow(z0, 2) - pow(z0, 2) * pow(z1, 4) +
                                 4 * pow(z0, 2) * pow(z1, 3) * z2 - 6 * pow(z0, 2) * pow(z1, 2) * pow(z2, 2) +
                                 4 * pow(z0, 2) * z1 * pow(z2, 3) - pow(z0, 2) * pow(z2, 4)) + y1 * z0 *
                                                                                               sqrt(-pow(x0, 2) *
                                                                                                    pow(x1, 4) +
                                                                                                    4 * pow(x0, 2) *
                                                                                                    pow(x1, 3) * x2 -
                                                                                                    6 * pow(x0, 2) *
                                                                                                    pow(x1, 2) *
                                                                                                    pow(x2, 2) +
                                                                                                    4 * pow(x0, 2) *
                                                                                                    x1 * pow(x2, 3) -
                                                                                                    pow(x0, 2) *
                                                                                                    pow(x2, 4) +
                                                                                                    pow(x0, 2) *
                                                                                                    pow(y1, 4) -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    pow(y1, 3) * y2 +
                                                                                                    6 * pow(x0, 2) *
                                                                                                    pow(y1, 2) *
                                                                                                    pow(y2, 2) +
                                                                                                    2 * pow(x0, 2) *
                                                                                                    pow(y1, 2) *
                                                                                                    pow(z1, 2) -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    pow(y1, 2) * z1 *
                                                                                                    z2 +
                                                                                                    2 * pow(x0, 2) *
                                                                                                    pow(y1, 2) *
                                                                                                    pow(z2, 2) -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    y1 * pow(y2, 3) -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    y1 * y2 *
                                                                                                    pow(z1, 2) +
                                                                                                    8 * pow(x0, 2) *
                                                                                                    y1 * y2 * z1 * z2 -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    y1 * y2 *
                                                                                                    pow(z2, 2) +
                                                                                                    pow(x0, 2) *
                                                                                                    pow(y2, 4) +
                                                                                                    2 * pow(x0, 2) *
                                                                                                    pow(y2, 2) *
                                                                                                    pow(z1, 2) -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    pow(y2, 2) * z1 *
                                                                                                    z2 +
                                                                                                    2 * pow(x0, 2) *
                                                                                                    pow(y2, 2) *
                                                                                                    pow(z2, 2) +
                                                                                                    pow(x0, 2) *
                                                                                                    pow(z1, 4) -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    pow(z1, 3) * z2 +
                                                                                                    6 * pow(x0, 2) *
                                                                                                    pow(z1, 2) *
                                                                                                    pow(z2, 2) -
                                                                                                    4 * pow(x0, 2) *
                                                                                                    z1 * pow(z2, 3) +
                                                                                                    pow(x0, 2) *
                                                                                                    pow(z2, 4) -
                                                                                                    4 * x0 *
                                                                                                    pow(x1, 3) * y0 *
                                                                                                    y1 + 4 * x0 *
                                                                                                         pow(x1, 3) *
                                                                                                         y0 * y2 -
                                                                                                    4 * x0 *
                                                                                                    pow(x1, 3) * z0 *
                                                                                                    z1 + 4 * x0 *
                                                                                                         pow(x1, 3) *
                                                                                                         z0 * z2 +
                                                                                                    12 * x0 *
                                                                                                    pow(x1, 2) * x2 *
                                                                                                    y0 * y1 - 12 * x0 *
                                                                                                              pow(x1,
                                                                                                                  2) *
                                                                                                              x2 * y0 *
                                                                                                              y2 +
                                                                                                    12 * x0 *
                                                                                                    pow(x1, 2) * x2 *
                                                                                                    z0 * z1 - 12 * x0 *
                                                                                                              pow(x1,
                                                                                                                  2) *
                                                                                                              x2 * z0 *
                                                                                                              z2 -
                                                                                                    12 * x0 * x1 *
                                                                                                    pow(x2, 2) * y0 *
                                                                                                    y1 + 12 * x0 * x1 *
                                                                                                         pow(x2, 2) *
                                                                                                         y0 * y2 -
                                                                                                    12 * x0 * x1 *
                                                                                                    pow(x2, 2) * z0 *
                                                                                                    z1 + 12 * x0 * x1 *
                                                                                                         pow(x2, 2) *
                                                                                                         z0 * z2 -
                                                                                                    4 * x0 * x1 * y0 *
                                                                                                    pow(y1, 3) +
                                                                                                    12 * x0 * x1 * y0 *
                                                                                                    pow(y1, 2) * y2 -
                                                                                                    12 * x0 * x1 * y0 *
                                                                                                    y1 * pow(y2, 2) -
                                                                                                    4 * x0 * x1 * y0 *
                                                                                                    y1 * pow(z1, 2) +
                                                                                                    8 * x0 * x1 * y0 *
                                                                                                    y1 * z1 * z2 -
                                                                                                    4 * x0 * x1 * y0 *
                                                                                                    y1 * pow(z2, 2) +
                                                                                                    4 * x0 * x1 * y0 *
                                                                                                    pow(y2, 3) +
                                                                                                    4 * x0 * x1 * y0 *
                                                                                                    y2 * pow(z1, 2) -
                                                                                                    8 * x0 * x1 * y0 *
                                                                                                    y2 * z1 * z2 +
                                                                                                    4 * x0 * x1 * y0 *
                                                                                                    y2 * pow(z2, 2) -
                                                                                                    4 * x0 * x1 *
                                                                                                    pow(y1, 2) * z0 *
                                                                                                    z1 + 4 * x0 * x1 *
                                                                                                         pow(y1, 2) *
                                                                                                         z0 * z2 +
                                                                                                    8 * x0 * x1 * y1 *
                                                                                                    y2 * z0 * z1 -
                                                                                                    8 * x0 * x1 * y1 *
                                                                                                    y2 * z0 * z2 -
                                                                                                    4 * x0 * x1 *
                                                                                                    pow(y2, 2) * z0 *
                                                                                                    z1 + 4 * x0 * x1 *
                                                                                                         pow(y2, 2) *
                                                                                                         z0 * z2 -
                                                                                                    4 * x0 * x1 * z0 *
                                                                                                    pow(z1, 3) +
                                                                                                    12 * x0 * x1 * z0 *
                                                                                                    pow(z1, 2) * z2 -
                                                                                                    12 * x0 * x1 * z0 *
                                                                                                    z1 * pow(z2, 2) +
                                                                                                    4 * x0 * x1 * z0 *
                                                                                                    pow(z2, 3) +
                                                                                                    4 * x0 *
                                                                                                    pow(x2, 3) * y0 *
                                                                                                    y1 - 4 * x0 *
                                                                                                         pow(x2, 3) *
                                                                                                         y0 * y2 +
                                                                                                    4 * x0 *
                                                                                                    pow(x2, 3) * z0 *
                                                                                                    z1 - 4 * x0 *
                                                                                                         pow(x2, 3) *
                                                                                                         z0 * z2 +
                                                                                                    4 * x0 * x2 * y0 *
                                                                                                    pow(y1, 3) -
                                                                                                    12 * x0 * x2 * y0 *
                                                                                                    pow(y1, 2) * y2 +
                                                                                                    12 * x0 * x2 * y0 *
                                                                                                    y1 * pow(y2, 2) +
                                                                                                    4 * x0 * x2 * y0 *
                                                                                                    y1 * pow(z1, 2) -
                                                                                                    8 * x0 * x2 * y0 *
                                                                                                    y1 * z1 * z2 +
                                                                                                    4 * x0 * x2 * y0 *
                                                                                                    y1 * pow(z2, 2) -
                                                                                                    4 * x0 * x2 * y0 *
                                                                                                    pow(y2, 3) -
                                                                                                    4 * x0 * x2 * y0 *
                                                                                                    y2 * pow(z1, 2) +
                                                                                                    8 * x0 * x2 * y0 *
                                                                                                    y2 * z1 * z2 -
                                                                                                    4 * x0 * x2 * y0 *
                                                                                                    y2 * pow(z2, 2) +
                                                                                                    4 * x0 * x2 *
                                                                                                    pow(y1, 2) * z0 *
                                                                                                    z1 - 4 * x0 * x2 *
                                                                                                         pow(y1, 2) *
                                                                                                         z0 * z2 -
                                                                                                    8 * x0 * x2 * y1 *
                                                                                                    y2 * z0 * z1 +
                                                                                                    8 * x0 * x2 * y1 *
                                                                                                    y2 * z0 * z2 +
                                                                                                    4 * x0 * x2 *
                                                                                                    pow(y2, 2) * z0 *
                                                                                                    z1 - 4 * x0 * x2 *
                                                                                                         pow(y2, 2) *
                                                                                                         z0 * z2 +
                                                                                                    4 * x0 * x2 * z0 *
                                                                                                    pow(z1, 3) -
                                                                                                    12 * x0 * x2 * z0 *
                                                                                                    pow(z1, 2) * z2 +
                                                                                                    12 * x0 * x2 * z0 *
                                                                                                    z1 * pow(z2, 2) -
                                                                                                    4 * x0 * x2 * z0 *
                                                                                                    pow(z2, 3) +
                                                                                                    pow(x1, 4) *
                                                                                                    pow(y0, 2) +
                                                                                                    pow(x1, 4) *
                                                                                                    pow(z0, 2) -
                                                                                                    4 * pow(x1, 3) *
                                                                                                    x2 * pow(y0, 2) -
                                                                                                    4 * pow(x1, 3) *
                                                                                                    x2 * pow(z0, 2) +
                                                                                                    6 * pow(x1, 2) *
                                                                                                    pow(x2, 2) *
                                                                                                    pow(y0, 2) +
                                                                                                    6 * pow(x1, 2) *
                                                                                                    pow(x2, 2) *
                                                                                                    pow(z0, 2) +
                                                                                                    2 * pow(x1, 2) *
                                                                                                    pow(y0, 2) *
                                                                                                    pow(z1, 2) -
                                                                                                    4 * pow(x1, 2) *
                                                                                                    pow(y0, 2) * z1 *
                                                                                                    z2 +
                                                                                                    2 * pow(x1, 2) *
                                                                                                    pow(y0, 2) *
                                                                                                    pow(z2, 2) -
                                                                                                    4 * pow(x1, 2) *
                                                                                                    y0 * y1 * z0 * z1 +
                                                                                                    4 * pow(x1, 2) *
                                                                                                    y0 * y1 * z0 * z2 +
                                                                                                    4 * pow(x1, 2) *
                                                                                                    y0 * y2 * z0 * z1 -
                                                                                                    4 * pow(x1, 2) *
                                                                                                    y0 * y2 * z0 * z2 +
                                                                                                    2 * pow(x1, 2) *
                                                                                                    pow(y1, 2) *
                                                                                                    pow(z0, 2) -
                                                                                                    4 * pow(x1, 2) *
                                                                                                    y1 * y2 *
                                                                                                    pow(z0, 2) +
                                                                                                    2 * pow(x1, 2) *
                                                                                                    pow(y2, 2) *
                                                                                                    pow(z0, 2) -
                                                                                                    4 * x1 *
                                                                                                    pow(x2, 3) *
                                                                                                    pow(y0, 2) -
                                                                                                    4 * x1 *
                                                                                                    pow(x2, 3) *
                                                                                                    pow(z0, 2) -
                                                                                                    4 * x1 * x2 *
                                                                                                    pow(y0, 2) *
                                                                                                    pow(z1, 2) +
                                                                                                    8 * x1 * x2 *
                                                                                                    pow(y0, 2) * z1 *
                                                                                                    z2 - 4 * x1 * x2 *
                                                                                                         pow(y0, 2) *
                                                                                                         pow(z2, 2) +
                                                                                                    8 * x1 * x2 * y0 *
                                                                                                    y1 * z0 * z1 -
                                                                                                    8 * x1 * x2 * y0 *
                                                                                                    y1 * z0 * z2 -
                                                                                                    8 * x1 * x2 * y0 *
                                                                                                    y2 * z0 * z1 +
                                                                                                    8 * x1 * x2 * y0 *
                                                                                                    y2 * z0 * z2 -
                                                                                                    4 * x1 * x2 *
                                                                                                    pow(y1, 2) *
                                                                                                    pow(z0, 2) +
                                                                                                    8 * x1 * x2 * y1 *
                                                                                                    y2 * pow(z0, 2) -
                                                                                                    4 * x1 * x2 *
                                                                                                    pow(y2, 2) *
                                                                                                    pow(z0, 2) +
                                                                                                    pow(x2, 4) *
                                                                                                    pow(y0, 2) +
                                                                                                    pow(x2, 4) *
                                                                                                    pow(z0, 2) +
                                                                                                    2 * pow(x2, 2) *
                                                                                                    pow(y0, 2) *
                                                                                                    pow(z1, 2) -
                                                                                                    4 * pow(x2, 2) *
                                                                                                    pow(y0, 2) * z1 *
                                                                                                    z2 +
                                                                                                    2 * pow(x2, 2) *
                                                                                                    pow(y0, 2) *
                                                                                                    pow(z2, 2) -
                                                                                                    4 * pow(x2, 2) *
                                                                                                    y0 * y1 * z0 * z1 +
                                                                                                    4 * pow(x2, 2) *
                                                                                                    y0 * y1 * z0 * z2 +
                                                                                                    4 * pow(x2, 2) *
                                                                                                    y0 * y2 * z0 * z1 -
                                                                                                    4 * pow(x2, 2) *
                                                                                                    y0 * y2 * z0 * z2 +
                                                                                                    2 * pow(x2, 2) *
                                                                                                    pow(y1, 2) *
                                                                                                    pow(z0, 2) -
                                                                                                    4 * pow(x2, 2) *
                                                                                                    y1 * y2 *
                                                                                                    pow(z0, 2) +
                                                                                                    2 * pow(x2, 2) *
                                                                                                    pow(y2, 2) *
                                                                                                    pow(z0, 2) -
                                                                                                    pow(y0, 2) *
                                                                                                    pow(y1, 4) +
                                                                                                    4 * pow(y0, 2) *
                                                                                                    pow(y1, 3) * y2 -
                                                                                                    6 * pow(y0, 2) *
                                                                                                    pow(y1, 2) *
                                                                                                    pow(y2, 2) +
                                                                                                    4 * pow(y0, 2) *
                                                                                                    y1 * pow(y2, 3) -
                                                                                                    pow(y0, 2) *
                                                                                                    pow(y2, 4) +
                                                                                                    pow(y0, 2) *
                                                                                                    pow(z1, 4) -
                                                                                                    4 * pow(y0, 2) *
                                                                                                    pow(z1, 3) * z2 +
                                                                                                    6 * pow(y0, 2) *
                                                                                                    pow(z1, 2) *
                                                                                                    pow(z2, 2) -
                                                                                                    4 * pow(y0, 2) *
                                                                                                    z1 * pow(z2, 3) +
                                                                                                    pow(y0, 2) *
                                                                                                    pow(z2, 4) -
                                                                                                    4 * y0 *
                                                                                                    pow(y1, 3) * z0 *
                                                                                                    z1 + 4 * y0 *
                                                                                                         pow(y1, 3) *
                                                                                                         z0 * z2 +
                                                                                                    12 * y0 *
                                                                                                    pow(y1, 2) * y2 *
                                                                                                    z0 * z1 - 12 * y0 *
                                                                                                              pow(y1,
                                                                                                                  2) *
                                                                                                              y2 * z0 *
                                                                                                              z2 -
                                                                                                    12 * y0 * y1 *
                                                                                                    pow(y2, 2) * z0 *
                                                                                                    z1 + 12 * y0 * y1 *
                                                                                                         pow(y2, 2) *
                                                                                                         z0 * z2 -
                                                                                                    4 * y0 * y1 * z0 *
                                                                                                    pow(z1, 3) +
                                                                                                    12 * y0 * y1 * z0 *
                                                                                                    pow(z1, 2) * z2 -
                                                                                                    12 * y0 * y1 * z0 *
                                                                                                    z1 * pow(z2, 2) +
                                                                                                    4 * y0 * y1 * z0 *
                                                                                                    pow(z2, 3) +
                                                                                                    4 * y0 *
                                                                                                    pow(y2, 3) * z0 *
                                                                                                    z1 - 4 * y0 *
                                                                                                         pow(y2, 3) *
                                                                                                         z0 * z2 +
                                                                                                    4 * y0 * y2 * z0 *
                                                                                                    pow(z1, 3) -
                                                                                                    12 * y0 * y2 * z0 *
                                                                                                    pow(z1, 2) * z2 +
                                                                                                    12 * y0 * y2 * z0 *
                                                                                                    z1 * pow(z2, 2) -
                                                                                                    4 * y0 * y2 * z0 *
                                                                                                    pow(z2, 3) +
                                                                                                    pow(y1, 4) *
                                                                                                    pow(z0, 2) -
                                                                                                    4 * pow(y1, 3) *
                                                                                                    y2 * pow(z0, 2) +
                                                                                                    6 * pow(y1, 2) *
                                                                                                    pow(y2, 2) *
                                                                                                    pow(z0, 2) -
                                                                                                    4 * y1 *
                                                                                                    pow(y2, 3) *
                                                                                                    pow(z0, 2) +
                                                                                                    pow(y2, 4) *
                                                                                                    pow(z0, 2) -
                                                                                                    pow(z0, 2) *
                                                                                                    pow(z1, 4) +
                                                                                                    4 * pow(z0, 2) *
                                                                                                    pow(z1, 3) * z2 -
                                                                                                    6 * pow(z0, 2) *
                                                                                                    pow(z1, 2) *
                                                                                                    pow(z2, 2) +
                                                                                                    4 * pow(z0, 2) *
                                                                                                    z1 * pow(z2, 3) -
                                                                                                    pow(z0, 2) *
                                                                                                    pow(z2, 4)) -
                  y2 * z0 * sqrt(-pow(x0, 2) * pow(x1, 4) + 4 * pow(x0, 2) * pow(x1, 3) * x2 -
                                 6 * pow(x0, 2) * pow(x1, 2) * pow(x2, 2) + 4 * pow(x0, 2) * x1 * pow(x2, 3) -
                                 pow(x0, 2) * pow(x2, 4) + pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 4 * x0 * pow(x1, 3) * y0 * y1 +
                                 4 * x0 * pow(x1, 3) * y0 * y2 - 4 * x0 * pow(x1, 3) * z0 * z1 +
                                 4 * x0 * pow(x1, 3) * z0 * z2 + 12 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 12 * x0 * pow(x1, 2) * x2 * y0 * y2 + 12 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 12 * x0 * pow(x1, 2) * x2 * z0 * z2 - 12 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 12 * x0 * x1 * pow(x2, 2) * y0 * y2 - 12 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 12 * x0 * x1 * pow(x2, 2) * z0 * z2 - 4 * x0 * x1 * y0 * pow(y1, 3) +
                                 12 * x0 * x1 * y0 * pow(y1, 2) * y2 - 12 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 4 * x0 * x1 * y0 * y1 * pow(z1, 2) + 8 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 4 * x0 * x1 * y0 * y1 * pow(z2, 2) + 4 * x0 * x1 * y0 * pow(y2, 3) +
                                 4 * x0 * x1 * y0 * y2 * pow(z1, 2) - 8 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 4 * x0 * x1 * y0 * y2 * pow(z2, 2) - 4 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 4 * x0 * x1 * pow(y1, 2) * z0 * z2 + 8 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 8 * x0 * x1 * y1 * y2 * z0 * z2 - 4 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 4 * x0 * x1 * pow(y2, 2) * z0 * z2 - 4 * x0 * x1 * z0 * pow(z1, 3) +
                                 12 * x0 * x1 * z0 * pow(z1, 2) * z2 - 12 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 4 * x0 * x1 * z0 * pow(z2, 3) + 4 * x0 * pow(x2, 3) * y0 * y1 -
                                 4 * x0 * pow(x2, 3) * y0 * y2 + 4 * x0 * pow(x2, 3) * z0 * z1 -
                                 4 * x0 * pow(x2, 3) * z0 * z2 + 4 * x0 * x2 * y0 * pow(y1, 3) -
                                 12 * x0 * x2 * y0 * pow(y1, 2) * y2 + 12 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 4 * x0 * x2 * y0 * y1 * pow(z1, 2) - 8 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 4 * x0 * x2 * y0 * y1 * pow(z2, 2) - 4 * x0 * x2 * y0 * pow(y2, 3) -
                                 4 * x0 * x2 * y0 * y2 * pow(z1, 2) + 8 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 4 * x0 * x2 * y0 * y2 * pow(z2, 2) + 4 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 4 * x0 * x2 * pow(y1, 2) * z0 * z2 - 8 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 8 * x0 * x2 * y1 * y2 * z0 * z2 + 4 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 4 * x0 * x2 * pow(y2, 2) * z0 * z2 + 4 * x0 * x2 * z0 * pow(z1, 3) -
                                 12 * x0 * x2 * z0 * pow(z1, 2) * z2 + 12 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 4 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) - 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 +
                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) - 4 * pow(x1, 2) * y0 * y1 * z0 * z1 +
                                 4 * pow(x1, 2) * y0 * y1 * z0 * z2 + 4 * pow(x1, 2) * y0 * y2 * z0 * z1 -
                                 4 * pow(x1, 2) * y0 * y2 * z0 * z2 + 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) -
                                 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) + 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) + 8 * x1 * x2 * pow(y0, 2) * z1 * z2 -
                                 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) + 8 * x1 * x2 * y0 * y1 * z0 * z1 -
                                 8 * x1 * x2 * y0 * y1 * z0 * z2 - 8 * x1 * x2 * y0 * y2 * z0 * z1 +
                                 8 * x1 * x2 * y0 * y2 * z0 * z2 - 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) +
                                 8 * x1 * x2 * y1 * y2 * pow(z0, 2) - 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) - 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 +
                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) - 4 * pow(x2, 2) * y0 * y1 * z0 * z1 +
                                 4 * pow(x2, 2) * y0 * y1 * z0 * z2 + 4 * pow(x2, 2) * y0 * y2 * z0 * z1 -
                                 4 * pow(x2, 2) * y0 * y2 * z0 * z2 + 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) -
                                 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) + 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) -
                                 pow(y0, 2) * pow(y1, 4) + 4 * pow(y0, 2) * pow(y1, 3) * y2 -
                                 6 * pow(y0, 2) * pow(y1, 2) * pow(y2, 2) + 4 * pow(y0, 2) * y1 * pow(y2, 3) -
                                 pow(y0, 2) * pow(y2, 4) + pow(y0, 2) * pow(z1, 4) - 4 * pow(y0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(y0, 2) * z1 * pow(z2, 3) +
                                 pow(y0, 2) * pow(z2, 4) - 4 * y0 * pow(y1, 3) * z0 * z1 +
                                 4 * y0 * pow(y1, 3) * z0 * z2 + 12 * y0 * pow(y1, 2) * y2 * z0 * z1 -
                                 12 * y0 * pow(y1, 2) * y2 * z0 * z2 - 12 * y0 * y1 * pow(y2, 2) * z0 * z1 +
                                 12 * y0 * y1 * pow(y2, 2) * z0 * z2 - 4 * y0 * y1 * z0 * pow(z1, 3) +
                                 12 * y0 * y1 * z0 * pow(z1, 2) * z2 - 12 * y0 * y1 * z0 * z1 * pow(z2, 2) +
                                 4 * y0 * y1 * z0 * pow(z2, 3) + 4 * y0 * pow(y2, 3) * z0 * z1 -
                                 4 * y0 * pow(y2, 3) * z0 * z2 + 4 * y0 * y2 * z0 * pow(z1, 3) -
                                 12 * y0 * y2 * z0 * pow(z1, 2) * z2 + 12 * y0 * y2 * z0 * z1 * pow(z2, 2) -
                                 4 * y0 * y2 * z0 * pow(z2, 3) + pow(y1, 4) * pow(z0, 2) -
                                 4 * pow(y1, 3) * y2 * pow(z0, 2) + 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) + pow(y2, 4) * pow(z0, 2) - pow(z0, 2) * pow(z1, 4) +
                                 4 * pow(z0, 2) * pow(z1, 3) * z2 - 6 * pow(z0, 2) * pow(z1, 2) * pow(z2, 2) +
                                 4 * pow(z0, 2) * z1 * pow(z2, 3) - pow(z0, 2) * pow(z2, 4))) /
                 (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                  pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) - 2 * x0 * x1 * y0 * y1 +
                  2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 + 2 * x0 * x1 * z0 * z2 + 2 * x0 * x2 * y0 * y1 -
                  2 * x0 * x2 * y0 * y2 + 2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 + pow(x1, 2) * pow(y0, 2) +
                  pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) - 2 * x1 * x2 * pow(z0, 2) +
                  pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) + pow(y0, 2) * pow(z1, 2) -
                  2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) - 2 * y0 * y1 * z0 * z1 + 2 * y0 * y1 * z0 * z2 +
                  2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 + pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) +
                  pow(y2, 2) * pow(z0, 2));
            y4 = (-pow(x0, 2) * pow(x1, 2) * y1 + pow(x0, 2) * pow(x1, 2) * y2 + 2 * pow(x0, 2) * x1 * x2 * y1 -
                  2 * pow(x0, 2) * x1 * x2 * y2 - pow(x0, 2) * pow(x2, 2) * y1 + pow(x0, 2) * pow(x2, 2) * y2 +
                  pow(x0, 2) * pow(y1, 2) * y2 - 2 * pow(x0, 2) * y1 * pow(y2, 2) + pow(x0, 2) * pow(y2, 3) +
                  pow(x0, 2) * y2 * pow(z1, 2) - 2 * pow(x0, 2) * y2 * z1 * z2 + pow(x0, 2) * y2 * pow(z2, 2) +
                  x0 * pow(x1, 3) * y0 - 3 * x0 * pow(x1, 2) * x2 * y0 + 3 * x0 * x1 * pow(x2, 2) * y0 -
                  x0 * x1 * y0 * pow(y1, 2) + x0 * x1 * y0 * pow(y2, 2) + x0 * x1 * y0 * pow(z1, 2) -
                  2 * x0 * x1 * y0 * z1 * z2 + x0 * x1 * y0 * pow(z2, 2) - 2 * x0 * x1 * y1 * z0 * z1 +
                  2 * x0 * x1 * y1 * z0 * z2 - x0 * pow(x2, 3) * y0 + x0 * x2 * y0 * pow(y1, 2) -
                  x0 * x2 * y0 * pow(y2, 2) - x0 * x2 * y0 * pow(z1, 2) + 2 * x0 * x2 * y0 * z1 * z2 -
                  x0 * x2 * y0 * pow(z2, 2) + 2 * x0 * x2 * y1 * z0 * z1 - 2 * x0 * x2 * y1 * z0 * z2 + x0 * z1 *
                                                                                                        sqrt(-pow(x0,
                                                                                                                  2) *
                                                                                                             pow(x1,
                                                                                                                 4) +
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(x1,
                                                                                                                 3) *
                                                                                                             x2 - 6 *
                                                                                                                  pow(x0,
                                                                                                                      2) *
                                                                                                                  pow(x1,
                                                                                                                      2) *
                                                                                                                  pow(x2,
                                                                                                                      2) +
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             x1 *
                                                                                                             pow(x2,
                                                                                                                 3) -
                                                                                                             pow(x0,
                                                                                                                 2) *
                                                                                                             pow(x2,
                                                                                                                 4) +
                                                                                                             pow(x0,
                                                                                                                 2) *
                                                                                                             pow(y1,
                                                                                                                 4) -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(y1,
                                                                                                                 3) *
                                                                                                             y2 + 6 *
                                                                                                                  pow(x0,
                                                                                                                      2) *
                                                                                                                  pow(y1,
                                                                                                                      2) *
                                                                                                                  pow(y2,
                                                                                                                      2) +
                                                                                                             2 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 2) -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             z1 * z2 +
                                                                                                             2 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 2) -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             y1 *
                                                                                                             pow(y2,
                                                                                                                 3) -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             y1 * y2 *
                                                                                                             pow(z1,
                                                                                                                 2) +
                                                                                                             8 * pow(x0,
                                                                                                                     2) *
                                                                                                             y1 * y2 *
                                                                                                             z1 * z2 -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             y1 * y2 *
                                                                                                             pow(z2,
                                                                                                                 2) +
                                                                                                             pow(x0,
                                                                                                                 2) *
                                                                                                             pow(y2,
                                                                                                                 4) +
                                                                                                             2 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 2) -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             z1 * z2 +
                                                                                                             2 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 2) +
                                                                                                             pow(x0,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 4) -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             pow(z1,
                                                                                                                 3) *
                                                                                                             z2 + 6 *
                                                                                                                  pow(x0,
                                                                                                                      2) *
                                                                                                                  pow(z1,
                                                                                                                      2) *
                                                                                                                  pow(z2,
                                                                                                                      2) -
                                                                                                             4 * pow(x0,
                                                                                                                     2) *
                                                                                                             z1 *
                                                                                                             pow(z2,
                                                                                                                 3) +
                                                                                                             pow(x0,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 4) -
                                                                                                             4 * x0 *
                                                                                                             pow(x1,
                                                                                                                 3) *
                                                                                                             y0 * y1 +
                                                                                                             4 * x0 *
                                                                                                             pow(x1,
                                                                                                                 3) *
                                                                                                             y0 * y2 -
                                                                                                             4 * x0 *
                                                                                                             pow(x1,
                                                                                                                 3) *
                                                                                                             z0 * z1 +
                                                                                                             4 * x0 *
                                                                                                             pow(x1,
                                                                                                                 3) *
                                                                                                             z0 * z2 +
                                                                                                             12 * x0 *
                                                                                                             pow(x1,
                                                                                                                 2) *
                                                                                                             x2 * y0 *
                                                                                                             y1 -
                                                                                                             12 * x0 *
                                                                                                             pow(x1,
                                                                                                                 2) *
                                                                                                             x2 * y0 *
                                                                                                             y2 +
                                                                                                             12 * x0 *
                                                                                                             pow(x1,
                                                                                                                 2) *
                                                                                                             x2 * z0 *
                                                                                                             z1 -
                                                                                                             12 * x0 *
                                                                                                             pow(x1,
                                                                                                                 2) *
                                                                                                             x2 * z0 *
                                                                                                             z2 -
                                                                                                             12 * x0 *
                                                                                                             x1 *
                                                                                                             pow(x2,
                                                                                                                 2) *
                                                                                                             y0 * y1 +
                                                                                                             12 * x0 *
                                                                                                             x1 *
                                                                                                             pow(x2,
                                                                                                                 2) *
                                                                                                             y0 * y2 -
                                                                                                             12 * x0 *
                                                                                                             x1 *
                                                                                                             pow(x2,
                                                                                                                 2) *
                                                                                                             z0 * z1 +
                                                                                                             12 * x0 *
                                                                                                             x1 *
                                                                                                             pow(x2,
                                                                                                                 2) *
                                                                                                             z0 * z2 -
                                                                                                             4 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             pow(y1,
                                                                                                                 3) +
                                                                                                             12 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             y2 -
                                                                                                             12 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             y1 *
                                                                                                             pow(y2,
                                                                                                                 2) -
                                                                                                             4 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             y1 *
                                                                                                             pow(z1,
                                                                                                                 2) +
                                                                                                             8 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             y1 * z1 *
                                                                                                             z2 -
                                                                                                             4 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             y1 *
                                                                                                             pow(z2,
                                                                                                                 2) +
                                                                                                             4 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             pow(y2,
                                                                                                                 3) +
                                                                                                             4 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             y2 *
                                                                                                             pow(z1,
                                                                                                                 2) -
                                                                                                             8 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             y2 * z1 *
                                                                                                             z2 +
                                                                                                             4 * x0 *
                                                                                                             x1 * y0 *
                                                                                                             y2 *
                                                                                                             pow(z2,
                                                                                                                 2) -
                                                                                                             4 * x0 *
                                                                                                             x1 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             z0 * z1 +
                                                                                                             4 * x0 *
                                                                                                             x1 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             z0 * z2 +
                                                                                                             8 * x0 *
                                                                                                             x1 * y1 *
                                                                                                             y2 * z0 *
                                                                                                             z1 -
                                                                                                             8 * x0 *
                                                                                                             x1 * y1 *
                                                                                                             y2 * z0 *
                                                                                                             z2 -
                                                                                                             4 * x0 *
                                                                                                             x1 *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             z0 * z1 +
                                                                                                             4 * x0 *
                                                                                                             x1 *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             z0 * z2 -
                                                                                                             4 * x0 *
                                                                                                             x1 * z0 *
                                                                                                             pow(z1,
                                                                                                                 3) +
                                                                                                             12 * x0 *
                                                                                                             x1 * z0 *
                                                                                                             pow(z1,
                                                                                                                 2) *
                                                                                                             z2 -
                                                                                                             12 * x0 *
                                                                                                             x1 * z0 *
                                                                                                             z1 *
                                                                                                             pow(z2,
                                                                                                                 2) +
                                                                                                             4 * x0 *
                                                                                                             x1 * z0 *
                                                                                                             pow(z2,
                                                                                                                 3) +
                                                                                                             4 * x0 *
                                                                                                             pow(x2,
                                                                                                                 3) *
                                                                                                             y0 * y1 -
                                                                                                             4 * x0 *
                                                                                                             pow(x2,
                                                                                                                 3) *
                                                                                                             y0 * y2 +
                                                                                                             4 * x0 *
                                                                                                             pow(x2,
                                                                                                                 3) *
                                                                                                             z0 * z1 -
                                                                                                             4 * x0 *
                                                                                                             pow(x2,
                                                                                                                 3) *
                                                                                                             z0 * z2 +
                                                                                                             4 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             pow(y1,
                                                                                                                 3) -
                                                                                                             12 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             y2 +
                                                                                                             12 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             y1 *
                                                                                                             pow(y2,
                                                                                                                 2) +
                                                                                                             4 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             y1 *
                                                                                                             pow(z1,
                                                                                                                 2) -
                                                                                                             8 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             y1 * z1 *
                                                                                                             z2 +
                                                                                                             4 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             y1 *
                                                                                                             pow(z2,
                                                                                                                 2) -
                                                                                                             4 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             pow(y2,
                                                                                                                 3) -
                                                                                                             4 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             y2 *
                                                                                                             pow(z1,
                                                                                                                 2) +
                                                                                                             8 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             y2 * z1 *
                                                                                                             z2 -
                                                                                                             4 * x0 *
                                                                                                             x2 * y0 *
                                                                                                             y2 *
                                                                                                             pow(z2,
                                                                                                                 2) +
                                                                                                             4 * x0 *
                                                                                                             x2 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             z0 * z1 -
                                                                                                             4 * x0 *
                                                                                                             x2 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             z0 * z2 -
                                                                                                             8 * x0 *
                                                                                                             x2 * y1 *
                                                                                                             y2 * z0 *
                                                                                                             z1 +
                                                                                                             8 * x0 *
                                                                                                             x2 * y1 *
                                                                                                             y2 * z0 *
                                                                                                             z2 +
                                                                                                             4 * x0 *
                                                                                                             x2 *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             z0 * z1 -
                                                                                                             4 * x0 *
                                                                                                             x2 *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             z0 * z2 +
                                                                                                             4 * x0 *
                                                                                                             x2 * z0 *
                                                                                                             pow(z1,
                                                                                                                 3) -
                                                                                                             12 * x0 *
                                                                                                             x2 * z0 *
                                                                                                             pow(z1,
                                                                                                                 2) *
                                                                                                             z2 +
                                                                                                             12 * x0 *
                                                                                                             x2 * z0 *
                                                                                                             z1 *
                                                                                                             pow(z2,
                                                                                                                 2) -
                                                                                                             4 * x0 *
                                                                                                             x2 * z0 *
                                                                                                             pow(z2,
                                                                                                                 3) +
                                                                                                             pow(x1,
                                                                                                                 4) *
                                                                                                             pow(y0,
                                                                                                                 2) +
                                                                                                             pow(x1,
                                                                                                                 4) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * pow(x1,
                                                                                                                     3) *
                                                                                                             x2 *
                                                                                                             pow(y0,
                                                                                                                 2) -
                                                                                                             4 * pow(x1,
                                                                                                                     3) *
                                                                                                             x2 *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             6 * pow(x1,
                                                                                                                     2) *
                                                                                                             pow(x2,
                                                                                                                 2) *
                                                                                                             pow(y0,
                                                                                                                 2) +
                                                                                                             6 * pow(x1,
                                                                                                                     2) *
                                                                                                             pow(x2,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             2 * pow(x1,
                                                                                                                     2) *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 2) -
                                                                                                             4 * pow(x1,
                                                                                                                     2) *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             z1 * z2 +
                                                                                                             2 * pow(x1,
                                                                                                                     2) *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 2) -
                                                                                                             4 * pow(x1,
                                                                                                                     2) *
                                                                                                             y0 * y1 *
                                                                                                             z0 * z1 +
                                                                                                             4 * pow(x1,
                                                                                                                     2) *
                                                                                                             y0 * y1 *
                                                                                                             z0 * z2 +
                                                                                                             4 * pow(x1,
                                                                                                                     2) *
                                                                                                             y0 * y2 *
                                                                                                             z0 * z1 -
                                                                                                             4 * pow(x1,
                                                                                                                     2) *
                                                                                                             y0 * y2 *
                                                                                                             z0 * z2 +
                                                                                                             2 * pow(x1,
                                                                                                                     2) *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * pow(x1,
                                                                                                                     2) *
                                                                                                             y1 * y2 *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             2 * pow(x1,
                                                                                                                     2) *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * x1 *
                                                                                                             pow(x2,
                                                                                                                 3) *
                                                                                                             pow(y0,
                                                                                                                 2) -
                                                                                                             4 * x1 *
                                                                                                             pow(x2,
                                                                                                                 3) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * x1 *
                                                                                                             x2 *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 2) +
                                                                                                             8 * x1 *
                                                                                                             x2 *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             z1 * z2 -
                                                                                                             4 * x1 *
                                                                                                             x2 *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 2) +
                                                                                                             8 * x1 *
                                                                                                             x2 * y0 *
                                                                                                             y1 * z0 *
                                                                                                             z1 -
                                                                                                             8 * x1 *
                                                                                                             x2 * y0 *
                                                                                                             y1 * z0 *
                                                                                                             z2 -
                                                                                                             8 * x1 *
                                                                                                             x2 * y0 *
                                                                                                             y2 * z0 *
                                                                                                             z1 +
                                                                                                             8 * x1 *
                                                                                                             x2 * y0 *
                                                                                                             y2 * z0 *
                                                                                                             z2 -
                                                                                                             4 * x1 *
                                                                                                             x2 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             8 * x1 *
                                                                                                             x2 * y1 *
                                                                                                             y2 *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * x1 *
                                                                                                             x2 *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             pow(x2,
                                                                                                                 4) *
                                                                                                             pow(y0,
                                                                                                                 2) +
                                                                                                             pow(x2,
                                                                                                                 4) *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             2 * pow(x2,
                                                                                                                     2) *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 2) -
                                                                                                             4 * pow(x2,
                                                                                                                     2) *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             z1 * z2 +
                                                                                                             2 * pow(x2,
                                                                                                                     2) *
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 2) -
                                                                                                             4 * pow(x2,
                                                                                                                     2) *
                                                                                                             y0 * y1 *
                                                                                                             z0 * z1 +
                                                                                                             4 * pow(x2,
                                                                                                                     2) *
                                                                                                             y0 * y1 *
                                                                                                             z0 * z2 +
                                                                                                             4 * pow(x2,
                                                                                                                     2) *
                                                                                                             y0 * y2 *
                                                                                                             z0 * z1 -
                                                                                                             4 * pow(x2,
                                                                                                                     2) *
                                                                                                             y0 * y2 *
                                                                                                             z0 * z2 +
                                                                                                             2 * pow(x2,
                                                                                                                     2) *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * pow(x2,
                                                                                                                     2) *
                                                                                                             y1 * y2 *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             2 * pow(x2,
                                                                                                                     2) *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(y1,
                                                                                                                 4) +
                                                                                                             4 * pow(y0,
                                                                                                                     2) *
                                                                                                             pow(y1,
                                                                                                                 3) *
                                                                                                             y2 - 6 *
                                                                                                                  pow(y0,
                                                                                                                      2) *
                                                                                                                  pow(y1,
                                                                                                                      2) *
                                                                                                                  pow(y2,
                                                                                                                      2) +
                                                                                                             4 * pow(y0,
                                                                                                                     2) *
                                                                                                             y1 *
                                                                                                             pow(y2,
                                                                                                                 3) -
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(y2,
                                                                                                                 4) +
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 4) -
                                                                                                             4 * pow(y0,
                                                                                                                     2) *
                                                                                                             pow(z1,
                                                                                                                 3) *
                                                                                                             z2 + 6 *
                                                                                                                  pow(y0,
                                                                                                                      2) *
                                                                                                                  pow(z1,
                                                                                                                      2) *
                                                                                                                  pow(z2,
                                                                                                                      2) -
                                                                                                             4 * pow(y0,
                                                                                                                     2) *
                                                                                                             z1 *
                                                                                                             pow(z2,
                                                                                                                 3) +
                                                                                                             pow(y0,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 4) -
                                                                                                             4 * y0 *
                                                                                                             pow(y1,
                                                                                                                 3) *
                                                                                                             z0 * z1 +
                                                                                                             4 * y0 *
                                                                                                             pow(y1,
                                                                                                                 3) *
                                                                                                             z0 * z2 +
                                                                                                             12 * y0 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             y2 * z0 *
                                                                                                             z1 -
                                                                                                             12 * y0 *
                                                                                                             pow(y1,
                                                                                                                 2) *
                                                                                                             y2 * z0 *
                                                                                                             z2 -
                                                                                                             12 * y0 *
                                                                                                             y1 *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             z0 * z1 +
                                                                                                             12 * y0 *
                                                                                                             y1 *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             z0 * z2 -
                                                                                                             4 * y0 *
                                                                                                             y1 * z0 *
                                                                                                             pow(z1,
                                                                                                                 3) +
                                                                                                             12 * y0 *
                                                                                                             y1 * z0 *
                                                                                                             pow(z1,
                                                                                                                 2) *
                                                                                                             z2 -
                                                                                                             12 * y0 *
                                                                                                             y1 * z0 *
                                                                                                             z1 *
                                                                                                             pow(z2,
                                                                                                                 2) +
                                                                                                             4 * y0 *
                                                                                                             y1 * z0 *
                                                                                                             pow(z2,
                                                                                                                 3) +
                                                                                                             4 * y0 *
                                                                                                             pow(y2,
                                                                                                                 3) *
                                                                                                             z0 * z1 -
                                                                                                             4 * y0 *
                                                                                                             pow(y2,
                                                                                                                 3) *
                                                                                                             z0 * z2 +
                                                                                                             4 * y0 *
                                                                                                             y2 * z0 *
                                                                                                             pow(z1,
                                                                                                                 3) -
                                                                                                             12 * y0 *
                                                                                                             y2 * z0 *
                                                                                                             pow(z1,
                                                                                                                 2) *
                                                                                                             z2 +
                                                                                                             12 * y0 *
                                                                                                             y2 * z0 *
                                                                                                             z1 *
                                                                                                             pow(z2,
                                                                                                                 2) -
                                                                                                             4 * y0 *
                                                                                                             y2 * z0 *
                                                                                                             pow(z2,
                                                                                                                 3) +
                                                                                                             pow(y1,
                                                                                                                 4) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * pow(y1,
                                                                                                                     3) *
                                                                                                             y2 *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             6 * pow(y1,
                                                                                                                     2) *
                                                                                                             pow(y2,
                                                                                                                 2) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             4 * y1 *
                                                                                                             pow(y2,
                                                                                                                 3) *
                                                                                                             pow(z0,
                                                                                                                 2) +
                                                                                                             pow(y2,
                                                                                                                 4) *
                                                                                                             pow(z0,
                                                                                                                 2) -
                                                                                                             pow(z0,
                                                                                                                 2) *
                                                                                                             pow(z1,
                                                                                                                 4) +
                                                                                                             4 * pow(z0,
                                                                                                                     2) *
                                                                                                             pow(z1,
                                                                                                                 3) *
                                                                                                             z2 - 6 *
                                                                                                                  pow(z0,
                                                                                                                      2) *
                                                                                                                  pow(z1,
                                                                                                                      2) *
                                                                                                                  pow(z2,
                                                                                                                      2) +
                                                                                                             4 * pow(z0,
                                                                                                                     2) *
                                                                                                             z1 *
                                                                                                             pow(z2,
                                                                                                                 3) -
                                                                                                             pow(z0,
                                                                                                                 2) *
                                                                                                             pow(z2,
                                                                                                                 4)) -
                  x0 * z2 * sqrt(-pow(x0, 2) * pow(x1, 4) + 4 * pow(x0, 2) * pow(x1, 3) * x2 -
                                 6 * pow(x0, 2) * pow(x1, 2) * pow(x2, 2) + 4 * pow(x0, 2) * x1 * pow(x2, 3) -
                                 pow(x0, 2) * pow(x2, 4) + pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                 6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) + 2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                 4 * pow(x0, 2) * y1 * pow(y2, 3) - 4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                 8 * pow(x0, 2) * y1 * y2 * z1 * z2 - 4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) +
                                 pow(x0, 2) * pow(y2, 4) + 2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                 4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 + 2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                 pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(x0, 2) * z1 * pow(z2, 3) +
                                 pow(x0, 2) * pow(z2, 4) - 4 * x0 * pow(x1, 3) * y0 * y1 +
                                 4 * x0 * pow(x1, 3) * y0 * y2 - 4 * x0 * pow(x1, 3) * z0 * z1 +
                                 4 * x0 * pow(x1, 3) * z0 * z2 + 12 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                 12 * x0 * pow(x1, 2) * x2 * y0 * y2 + 12 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                 12 * x0 * pow(x1, 2) * x2 * z0 * z2 - 12 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                 12 * x0 * x1 * pow(x2, 2) * y0 * y2 - 12 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                 12 * x0 * x1 * pow(x2, 2) * z0 * z2 - 4 * x0 * x1 * y0 * pow(y1, 3) +
                                 12 * x0 * x1 * y0 * pow(y1, 2) * y2 - 12 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                 4 * x0 * x1 * y0 * y1 * pow(z1, 2) + 8 * x0 * x1 * y0 * y1 * z1 * z2 -
                                 4 * x0 * x1 * y0 * y1 * pow(z2, 2) + 4 * x0 * x1 * y0 * pow(y2, 3) +
                                 4 * x0 * x1 * y0 * y2 * pow(z1, 2) - 8 * x0 * x1 * y0 * y2 * z1 * z2 +
                                 4 * x0 * x1 * y0 * y2 * pow(z2, 2) - 4 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                 4 * x0 * x1 * pow(y1, 2) * z0 * z2 + 8 * x0 * x1 * y1 * y2 * z0 * z1 -
                                 8 * x0 * x1 * y1 * y2 * z0 * z2 - 4 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                 4 * x0 * x1 * pow(y2, 2) * z0 * z2 - 4 * x0 * x1 * z0 * pow(z1, 3) +
                                 12 * x0 * x1 * z0 * pow(z1, 2) * z2 - 12 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                 4 * x0 * x1 * z0 * pow(z2, 3) + 4 * x0 * pow(x2, 3) * y0 * y1 -
                                 4 * x0 * pow(x2, 3) * y0 * y2 + 4 * x0 * pow(x2, 3) * z0 * z1 -
                                 4 * x0 * pow(x2, 3) * z0 * z2 + 4 * x0 * x2 * y0 * pow(y1, 3) -
                                 12 * x0 * x2 * y0 * pow(y1, 2) * y2 + 12 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                 4 * x0 * x2 * y0 * y1 * pow(z1, 2) - 8 * x0 * x2 * y0 * y1 * z1 * z2 +
                                 4 * x0 * x2 * y0 * y1 * pow(z2, 2) - 4 * x0 * x2 * y0 * pow(y2, 3) -
                                 4 * x0 * x2 * y0 * y2 * pow(z1, 2) + 8 * x0 * x2 * y0 * y2 * z1 * z2 -
                                 4 * x0 * x2 * y0 * y2 * pow(z2, 2) + 4 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                 4 * x0 * x2 * pow(y1, 2) * z0 * z2 - 8 * x0 * x2 * y1 * y2 * z0 * z1 +
                                 8 * x0 * x2 * y1 * y2 * z0 * z2 + 4 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                 4 * x0 * x2 * pow(y2, 2) * z0 * z2 + 4 * x0 * x2 * z0 * pow(z1, 3) -
                                 12 * x0 * x2 * z0 * pow(z1, 2) * z2 + 12 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                 4 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) + pow(x1, 4) * pow(z0, 2) -
                                 4 * pow(x1, 3) * x2 * pow(y0, 2) - 4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                 6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) + 6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) - 4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 +
                                 2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) - 4 * pow(x1, 2) * y0 * y1 * z0 * z1 +
                                 4 * pow(x1, 2) * y0 * y1 * z0 * z2 + 4 * pow(x1, 2) * y0 * y2 * z0 * z1 -
                                 4 * pow(x1, 2) * y0 * y2 * z0 * z2 + 2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) -
                                 4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) + 2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) -
                                 4 * x1 * pow(x2, 3) * pow(y0, 2) - 4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                 4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) + 8 * x1 * x2 * pow(y0, 2) * z1 * z2 -
                                 4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) + 8 * x1 * x2 * y0 * y1 * z0 * z1 -
                                 8 * x1 * x2 * y0 * y1 * z0 * z2 - 8 * x1 * x2 * y0 * y2 * z0 * z1 +
                                 8 * x1 * x2 * y0 * y2 * z0 * z2 - 4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) +
                                 8 * x1 * x2 * y1 * y2 * pow(z0, 2) - 4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) +
                                 pow(x2, 4) * pow(y0, 2) + pow(x2, 4) * pow(z0, 2) +
                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) - 4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 +
                                 2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) - 4 * pow(x2, 2) * y0 * y1 * z0 * z1 +
                                 4 * pow(x2, 2) * y0 * y1 * z0 * z2 + 4 * pow(x2, 2) * y0 * y2 * z0 * z1 -
                                 4 * pow(x2, 2) * y0 * y2 * z0 * z2 + 2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) -
                                 4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) + 2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) -
                                 pow(y0, 2) * pow(y1, 4) + 4 * pow(y0, 2) * pow(y1, 3) * y2 -
                                 6 * pow(y0, 2) * pow(y1, 2) * pow(y2, 2) + 4 * pow(y0, 2) * y1 * pow(y2, 3) -
                                 pow(y0, 2) * pow(y2, 4) + pow(y0, 2) * pow(z1, 4) - 4 * pow(y0, 2) * pow(z1, 3) * z2 +
                                 6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) - 4 * pow(y0, 2) * z1 * pow(z2, 3) +
                                 pow(y0, 2) * pow(z2, 4) - 4 * y0 * pow(y1, 3) * z0 * z1 +
                                 4 * y0 * pow(y1, 3) * z0 * z2 + 12 * y0 * pow(y1, 2) * y2 * z0 * z1 -
                                 12 * y0 * pow(y1, 2) * y2 * z0 * z2 - 12 * y0 * y1 * pow(y2, 2) * z0 * z1 +
                                 12 * y0 * y1 * pow(y2, 2) * z0 * z2 - 4 * y0 * y1 * z0 * pow(z1, 3) +
                                 12 * y0 * y1 * z0 * pow(z1, 2) * z2 - 12 * y0 * y1 * z0 * z1 * pow(z2, 2) +
                                 4 * y0 * y1 * z0 * pow(z2, 3) + 4 * y0 * pow(y2, 3) * z0 * z1 -
                                 4 * y0 * pow(y2, 3) * z0 * z2 + 4 * y0 * y2 * z0 * pow(z1, 3) -
                                 12 * y0 * y2 * z0 * pow(z1, 2) * z2 + 12 * y0 * y2 * z0 * z1 * pow(z2, 2) -
                                 4 * y0 * y2 * z0 * pow(z2, 3) + pow(y1, 4) * pow(z0, 2) -
                                 4 * pow(y1, 3) * y2 * pow(z0, 2) + 6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) -
                                 4 * y1 * pow(y2, 3) * pow(z0, 2) + pow(y2, 4) * pow(z0, 2) - pow(z0, 2) * pow(z1, 4) +
                                 4 * pow(z0, 2) * pow(z1, 3) * z2 - 6 * pow(z0, 2) * pow(z1, 2) * pow(z2, 2) +
                                 4 * pow(z0, 2) * z1 * pow(z2, 3) - pow(z0, 2) * pow(z2, 4)) +
                  pow(x1, 2) * pow(y0, 2) * y1 + pow(x1, 2) * y0 * z0 * z1 - pow(x1, 2) * y0 * z0 * z2 +
                  pow(x1, 2) * y2 * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) * y1 - 2 * x1 * x2 * y0 * z0 * z1 +
                  2 * x1 * x2 * y0 * z0 * z2 - 2 * x1 * x2 * y2 * pow(z0, 2) - x1 * z0 * sqrt(-pow(x0, 2) * pow(x1, 4) +
                                                                                              4 * pow(x0, 2) *
                                                                                              pow(x1, 3) * x2 -
                                                                                              6 * pow(x0, 2) *
                                                                                              pow(x1, 2) * pow(x2, 2) +
                                                                                              4 * pow(x0, 2) * x1 *
                                                                                              pow(x2, 3) -
                                                                                              pow(x0, 2) * pow(x2, 4) +
                                                                                              pow(x0, 2) * pow(y1, 4) -
                                                                                              4 * pow(x0, 2) *
                                                                                              pow(y1, 3) * y2 +
                                                                                              6 * pow(x0, 2) *
                                                                                              pow(y1, 2) * pow(y2, 2) +
                                                                                              2 * pow(x0, 2) *
                                                                                              pow(y1, 2) * pow(z1, 2) -
                                                                                              4 * pow(x0, 2) *
                                                                                              pow(y1, 2) * z1 * z2 +
                                                                                              2 * pow(x0, 2) *
                                                                                              pow(y1, 2) * pow(z2, 2) -
                                                                                              4 * pow(x0, 2) * y1 *
                                                                                              pow(y2, 3) -
                                                                                              4 * pow(x0, 2) * y1 * y2 *
                                                                                              pow(z1, 2) +
                                                                                              8 * pow(x0, 2) * y1 * y2 *
                                                                                              z1 * z2 -
                                                                                              4 * pow(x0, 2) * y1 * y2 *
                                                                                              pow(z2, 2) +
                                                                                              pow(x0, 2) * pow(y2, 4) +
                                                                                              2 * pow(x0, 2) *
                                                                                              pow(y2, 2) * pow(z1, 2) -
                                                                                              4 * pow(x0, 2) *
                                                                                              pow(y2, 2) * z1 * z2 +
                                                                                              2 * pow(x0, 2) *
                                                                                              pow(y2, 2) * pow(z2, 2) +
                                                                                              pow(x0, 2) * pow(z1, 4) -
                                                                                              4 * pow(x0, 2) *
                                                                                              pow(z1, 3) * z2 +
                                                                                              6 * pow(x0, 2) *
                                                                                              pow(z1, 2) * pow(z2, 2) -
                                                                                              4 * pow(x0, 2) * z1 *
                                                                                              pow(z2, 3) +
                                                                                              pow(x0, 2) * pow(z2, 4) -
                                                                                              4 * x0 * pow(x1, 3) * y0 *
                                                                                              y1 +
                                                                                              4 * x0 * pow(x1, 3) * y0 *
                                                                                              y2 -
                                                                                              4 * x0 * pow(x1, 3) * z0 *
                                                                                              z1 +
                                                                                              4 * x0 * pow(x1, 3) * z0 *
                                                                                              z2 +
                                                                                              12 * x0 * pow(x1, 2) *
                                                                                              x2 * y0 * y1 -
                                                                                              12 * x0 * pow(x1, 2) *
                                                                                              x2 * y0 * y2 +
                                                                                              12 * x0 * pow(x1, 2) *
                                                                                              x2 * z0 * z1 -
                                                                                              12 * x0 * pow(x1, 2) *
                                                                                              x2 * z0 * z2 -
                                                                                              12 * x0 * x1 *
                                                                                              pow(x2, 2) * y0 * y1 +
                                                                                              12 * x0 * x1 *
                                                                                              pow(x2, 2) * y0 * y2 -
                                                                                              12 * x0 * x1 *
                                                                                              pow(x2, 2) * z0 * z1 +
                                                                                              12 * x0 * x1 *
                                                                                              pow(x2, 2) * z0 * z2 -
                                                                                              4 * x0 * x1 * y0 *
                                                                                              pow(y1, 3) +
                                                                                              12 * x0 * x1 * y0 *
                                                                                              pow(y1, 2) * y2 -
                                                                                              12 * x0 * x1 * y0 * y1 *
                                                                                              pow(y2, 2) -
                                                                                              4 * x0 * x1 * y0 * y1 *
                                                                                              pow(z1, 2) +
                                                                                              8 * x0 * x1 * y0 * y1 *
                                                                                              z1 * z2 -
                                                                                              4 * x0 * x1 * y0 * y1 *
                                                                                              pow(z2, 2) +
                                                                                              4 * x0 * x1 * y0 *
                                                                                              pow(y2, 3) +
                                                                                              4 * x0 * x1 * y0 * y2 *
                                                                                              pow(z1, 2) -
                                                                                              8 * x0 * x1 * y0 * y2 *
                                                                                              z1 * z2 +
                                                                                              4 * x0 * x1 * y0 * y2 *
                                                                                              pow(z2, 2) -
                                                                                              4 * x0 * x1 * pow(y1, 2) *
                                                                                              z0 * z1 +
                                                                                              4 * x0 * x1 * pow(y1, 2) *
                                                                                              z0 * z2 +
                                                                                              8 * x0 * x1 * y1 * y2 *
                                                                                              z0 * z1 -
                                                                                              8 * x0 * x1 * y1 * y2 *
                                                                                              z0 * z2 -
                                                                                              4 * x0 * x1 * pow(y2, 2) *
                                                                                              z0 * z1 +
                                                                                              4 * x0 * x1 * pow(y2, 2) *
                                                                                              z0 * z2 -
                                                                                              4 * x0 * x1 * z0 *
                                                                                              pow(z1, 3) +
                                                                                              12 * x0 * x1 * z0 *
                                                                                              pow(z1, 2) * z2 -
                                                                                              12 * x0 * x1 * z0 * z1 *
                                                                                              pow(z2, 2) +
                                                                                              4 * x0 * x1 * z0 *
                                                                                              pow(z2, 3) +
                                                                                              4 * x0 * pow(x2, 3) * y0 *
                                                                                              y1 -
                                                                                              4 * x0 * pow(x2, 3) * y0 *
                                                                                              y2 +
                                                                                              4 * x0 * pow(x2, 3) * z0 *
                                                                                              z1 -
                                                                                              4 * x0 * pow(x2, 3) * z0 *
                                                                                              z2 + 4 * x0 * x2 * y0 *
                                                                                                   pow(y1, 3) -
                                                                                              12 * x0 * x2 * y0 *
                                                                                              pow(y1, 2) * y2 +
                                                                                              12 * x0 * x2 * y0 * y1 *
                                                                                              pow(y2, 2) +
                                                                                              4 * x0 * x2 * y0 * y1 *
                                                                                              pow(z1, 2) -
                                                                                              8 * x0 * x2 * y0 * y1 *
                                                                                              z1 * z2 +
                                                                                              4 * x0 * x2 * y0 * y1 *
                                                                                              pow(z2, 2) -
                                                                                              4 * x0 * x2 * y0 *
                                                                                              pow(y2, 3) -
                                                                                              4 * x0 * x2 * y0 * y2 *
                                                                                              pow(z1, 2) +
                                                                                              8 * x0 * x2 * y0 * y2 *
                                                                                              z1 * z2 -
                                                                                              4 * x0 * x2 * y0 * y2 *
                                                                                              pow(z2, 2) +
                                                                                              4 * x0 * x2 * pow(y1, 2) *
                                                                                              z0 * z1 -
                                                                                              4 * x0 * x2 * pow(y1, 2) *
                                                                                              z0 * z2 -
                                                                                              8 * x0 * x2 * y1 * y2 *
                                                                                              z0 * z1 +
                                                                                              8 * x0 * x2 * y1 * y2 *
                                                                                              z0 * z2 +
                                                                                              4 * x0 * x2 * pow(y2, 2) *
                                                                                              z0 * z1 -
                                                                                              4 * x0 * x2 * pow(y2, 2) *
                                                                                              z0 * z2 +
                                                                                              4 * x0 * x2 * z0 *
                                                                                              pow(z1, 3) -
                                                                                              12 * x0 * x2 * z0 *
                                                                                              pow(z1, 2) * z2 +
                                                                                              12 * x0 * x2 * z0 * z1 *
                                                                                              pow(z2, 2) -
                                                                                              4 * x0 * x2 * z0 *
                                                                                              pow(z2, 3) +
                                                                                              pow(x1, 4) * pow(y0, 2) +
                                                                                              pow(x1, 4) * pow(z0, 2) -
                                                                                              4 * pow(x1, 3) * x2 *
                                                                                              pow(y0, 2) -
                                                                                              4 * pow(x1, 3) * x2 *
                                                                                              pow(z0, 2) +
                                                                                              6 * pow(x1, 2) *
                                                                                              pow(x2, 2) * pow(y0, 2) +
                                                                                              6 * pow(x1, 2) *
                                                                                              pow(x2, 2) * pow(z0, 2) +
                                                                                              2 * pow(x1, 2) *
                                                                                              pow(y0, 2) * pow(z1, 2) -
                                                                                              4 * pow(x1, 2) *
                                                                                              pow(y0, 2) * z1 * z2 +
                                                                                              2 * pow(x1, 2) *
                                                                                              pow(y0, 2) * pow(z2, 2) -
                                                                                              4 * pow(x1, 2) * y0 * y1 *
                                                                                              z0 * z1 +
                                                                                              4 * pow(x1, 2) * y0 * y1 *
                                                                                              z0 * z2 +
                                                                                              4 * pow(x1, 2) * y0 * y2 *
                                                                                              z0 * z1 -
                                                                                              4 * pow(x1, 2) * y0 * y2 *
                                                                                              z0 * z2 + 2 * pow(x1, 2) *
                                                                                                        pow(y1, 2) *
                                                                                                        pow(z0, 2) -
                                                                                              4 * pow(x1, 2) * y1 * y2 *
                                                                                              pow(z0, 2) +
                                                                                              2 * pow(x1, 2) *
                                                                                              pow(y2, 2) * pow(z0, 2) -
                                                                                              4 * x1 * pow(x2, 3) *
                                                                                              pow(y0, 2) -
                                                                                              4 * x1 * pow(x2, 3) *
                                                                                              pow(z0, 2) -
                                                                                              4 * x1 * x2 * pow(y0, 2) *
                                                                                              pow(z1, 2) +
                                                                                              8 * x1 * x2 * pow(y0, 2) *
                                                                                              z1 * z2 -
                                                                                              4 * x1 * x2 * pow(y0, 2) *
                                                                                              pow(z2, 2) +
                                                                                              8 * x1 * x2 * y0 * y1 *
                                                                                              z0 * z1 -
                                                                                              8 * x1 * x2 * y0 * y1 *
                                                                                              z0 * z2 -
                                                                                              8 * x1 * x2 * y0 * y2 *
                                                                                              z0 * z1 +
                                                                                              8 * x1 * x2 * y0 * y2 *
                                                                                              z0 * z2 -
                                                                                              4 * x1 * x2 * pow(y1, 2) *
                                                                                              pow(z0, 2) +
                                                                                              8 * x1 * x2 * y1 * y2 *
                                                                                              pow(z0, 2) -
                                                                                              4 * x1 * x2 * pow(y2, 2) *
                                                                                              pow(z0, 2) +
                                                                                              pow(x2, 4) * pow(y0, 2) +
                                                                                              pow(x2, 4) * pow(z0, 2) +
                                                                                              2 * pow(x2, 2) *
                                                                                              pow(y0, 2) * pow(z1, 2) -
                                                                                              4 * pow(x2, 2) *
                                                                                              pow(y0, 2) * z1 * z2 +
                                                                                              2 * pow(x2, 2) *
                                                                                              pow(y0, 2) * pow(z2, 2) -
                                                                                              4 * pow(x2, 2) * y0 * y1 *
                                                                                              z0 * z1 +
                                                                                              4 * pow(x2, 2) * y0 * y1 *
                                                                                              z0 * z2 +
                                                                                              4 * pow(x2, 2) * y0 * y2 *
                                                                                              z0 * z1 -
                                                                                              4 * pow(x2, 2) * y0 * y2 *
                                                                                              z0 * z2 + 2 * pow(x2, 2) *
                                                                                                        pow(y1, 2) *
                                                                                                        pow(z0, 2) -
                                                                                              4 * pow(x2, 2) * y1 * y2 *
                                                                                              pow(z0, 2) +
                                                                                              2 * pow(x2, 2) *
                                                                                              pow(y2, 2) * pow(z0, 2) -
                                                                                              pow(y0, 2) * pow(y1, 4) +
                                                                                              4 * pow(y0, 2) *
                                                                                              pow(y1, 3) * y2 -
                                                                                              6 * pow(y0, 2) *
                                                                                              pow(y1, 2) * pow(y2, 2) +
                                                                                              4 * pow(y0, 2) * y1 *
                                                                                              pow(y2, 3) -
                                                                                              pow(y0, 2) * pow(y2, 4) +
                                                                                              pow(y0, 2) * pow(z1, 4) -
                                                                                              4 * pow(y0, 2) *
                                                                                              pow(z1, 3) * z2 +
                                                                                              6 * pow(y0, 2) *
                                                                                              pow(z1, 2) * pow(z2, 2) -
                                                                                              4 * pow(y0, 2) * z1 *
                                                                                              pow(z2, 3) +
                                                                                              pow(y0, 2) * pow(z2, 4) -
                                                                                              4 * y0 * pow(y1, 3) * z0 *
                                                                                              z1 +
                                                                                              4 * y0 * pow(y1, 3) * z0 *
                                                                                              z2 +
                                                                                              12 * y0 * pow(y1, 2) *
                                                                                              y2 * z0 * z1 -
                                                                                              12 * y0 * pow(y1, 2) *
                                                                                              y2 * z0 * z2 -
                                                                                              12 * y0 * y1 *
                                                                                              pow(y2, 2) * z0 * z1 +
                                                                                              12 * y0 * y1 *
                                                                                              pow(y2, 2) * z0 * z2 -
                                                                                              4 * y0 * y1 * z0 *
                                                                                              pow(z1, 3) +
                                                                                              12 * y0 * y1 * z0 *
                                                                                              pow(z1, 2) * z2 -
                                                                                              12 * y0 * y1 * z0 * z1 *
                                                                                              pow(z2, 2) +
                                                                                              4 * y0 * y1 * z0 *
                                                                                              pow(z2, 3) +
                                                                                              4 * y0 * pow(y2, 3) * z0 *
                                                                                              z1 -
                                                                                              4 * y0 * pow(y2, 3) * z0 *
                                                                                              z2 + 4 * y0 * y2 * z0 *
                                                                                                   pow(z1, 3) -
                                                                                              12 * y0 * y2 * z0 *
                                                                                              pow(z1, 2) * z2 +
                                                                                              12 * y0 * y2 * z0 * z1 *
                                                                                              pow(z2, 2) -
                                                                                              4 * y0 * y2 * z0 *
                                                                                              pow(z2, 3) +
                                                                                              pow(y1, 4) * pow(z0, 2) -
                                                                                              4 * pow(y1, 3) * y2 *
                                                                                              pow(z0, 2) +
                                                                                              6 * pow(y1, 2) *
                                                                                              pow(y2, 2) * pow(z0, 2) -
                                                                                              4 * y1 * pow(y2, 3) *
                                                                                              pow(z0, 2) +
                                                                                              pow(y2, 4) * pow(z0, 2) -
                                                                                              pow(z0, 2) * pow(z1, 4) +
                                                                                              4 * pow(z0, 2) *
                                                                                              pow(z1, 3) * z2 -
                                                                                              6 * pow(z0, 2) *
                                                                                              pow(z1, 2) * pow(z2, 2) +
                                                                                              4 * pow(z0, 2) * z1 *
                                                                                              pow(z2, 3) -
                                                                                              pow(z0, 2) * pow(z2, 4)) +
                  pow(x2, 2) * pow(y0, 2) * y1 + pow(x2, 2) * y0 * z0 * z1 - pow(x2, 2) * y0 * z0 * z2 +
                  pow(x2, 2) * y2 * pow(z0, 2) + x2 * z0 *
                                                 sqrt(-pow(x0, 2) * pow(x1, 4) + 4 * pow(x0, 2) * pow(x1, 3) * x2 -
                                                      6 * pow(x0, 2) * pow(x1, 2) * pow(x2, 2) +
                                                      4 * pow(x0, 2) * x1 * pow(x2, 3) - pow(x0, 2) * pow(x2, 4) +
                                                      pow(x0, 2) * pow(y1, 4) - 4 * pow(x0, 2) * pow(y1, 3) * y2 +
                                                      6 * pow(x0, 2) * pow(y1, 2) * pow(y2, 2) +
                                                      2 * pow(x0, 2) * pow(y1, 2) * pow(z1, 2) -
                                                      4 * pow(x0, 2) * pow(y1, 2) * z1 * z2 +
                                                      2 * pow(x0, 2) * pow(y1, 2) * pow(z2, 2) -
                                                      4 * pow(x0, 2) * y1 * pow(y2, 3) -
                                                      4 * pow(x0, 2) * y1 * y2 * pow(z1, 2) +
                                                      8 * pow(x0, 2) * y1 * y2 * z1 * z2 -
                                                      4 * pow(x0, 2) * y1 * y2 * pow(z2, 2) + pow(x0, 2) * pow(y2, 4) +
                                                      2 * pow(x0, 2) * pow(y2, 2) * pow(z1, 2) -
                                                      4 * pow(x0, 2) * pow(y2, 2) * z1 * z2 +
                                                      2 * pow(x0, 2) * pow(y2, 2) * pow(z2, 2) +
                                                      pow(x0, 2) * pow(z1, 4) - 4 * pow(x0, 2) * pow(z1, 3) * z2 +
                                                      6 * pow(x0, 2) * pow(z1, 2) * pow(z2, 2) -
                                                      4 * pow(x0, 2) * z1 * pow(z2, 3) + pow(x0, 2) * pow(z2, 4) -
                                                      4 * x0 * pow(x1, 3) * y0 * y1 + 4 * x0 * pow(x1, 3) * y0 * y2 -
                                                      4 * x0 * pow(x1, 3) * z0 * z1 + 4 * x0 * pow(x1, 3) * z0 * z2 +
                                                      12 * x0 * pow(x1, 2) * x2 * y0 * y1 -
                                                      12 * x0 * pow(x1, 2) * x2 * y0 * y2 +
                                                      12 * x0 * pow(x1, 2) * x2 * z0 * z1 -
                                                      12 * x0 * pow(x1, 2) * x2 * z0 * z2 -
                                                      12 * x0 * x1 * pow(x2, 2) * y0 * y1 +
                                                      12 * x0 * x1 * pow(x2, 2) * y0 * y2 -
                                                      12 * x0 * x1 * pow(x2, 2) * z0 * z1 +
                                                      12 * x0 * x1 * pow(x2, 2) * z0 * z2 -
                                                      4 * x0 * x1 * y0 * pow(y1, 3) +
                                                      12 * x0 * x1 * y0 * pow(y1, 2) * y2 -
                                                      12 * x0 * x1 * y0 * y1 * pow(y2, 2) -
                                                      4 * x0 * x1 * y0 * y1 * pow(z1, 2) +
                                                      8 * x0 * x1 * y0 * y1 * z1 * z2 -
                                                      4 * x0 * x1 * y0 * y1 * pow(z2, 2) +
                                                      4 * x0 * x1 * y0 * pow(y2, 3) +
                                                      4 * x0 * x1 * y0 * y2 * pow(z1, 2) -
                                                      8 * x0 * x1 * y0 * y2 * z1 * z2 +
                                                      4 * x0 * x1 * y0 * y2 * pow(z2, 2) -
                                                      4 * x0 * x1 * pow(y1, 2) * z0 * z1 +
                                                      4 * x0 * x1 * pow(y1, 2) * z0 * z2 +
                                                      8 * x0 * x1 * y1 * y2 * z0 * z1 -
                                                      8 * x0 * x1 * y1 * y2 * z0 * z2 -
                                                      4 * x0 * x1 * pow(y2, 2) * z0 * z1 +
                                                      4 * x0 * x1 * pow(y2, 2) * z0 * z2 -
                                                      4 * x0 * x1 * z0 * pow(z1, 3) +
                                                      12 * x0 * x1 * z0 * pow(z1, 2) * z2 -
                                                      12 * x0 * x1 * z0 * z1 * pow(z2, 2) +
                                                      4 * x0 * x1 * z0 * pow(z2, 3) + 4 * x0 * pow(x2, 3) * y0 * y1 -
                                                      4 * x0 * pow(x2, 3) * y0 * y2 + 4 * x0 * pow(x2, 3) * z0 * z1 -
                                                      4 * x0 * pow(x2, 3) * z0 * z2 + 4 * x0 * x2 * y0 * pow(y1, 3) -
                                                      12 * x0 * x2 * y0 * pow(y1, 2) * y2 +
                                                      12 * x0 * x2 * y0 * y1 * pow(y2, 2) +
                                                      4 * x0 * x2 * y0 * y1 * pow(z1, 2) -
                                                      8 * x0 * x2 * y0 * y1 * z1 * z2 +
                                                      4 * x0 * x2 * y0 * y1 * pow(z2, 2) -
                                                      4 * x0 * x2 * y0 * pow(y2, 3) -
                                                      4 * x0 * x2 * y0 * y2 * pow(z1, 2) +
                                                      8 * x0 * x2 * y0 * y2 * z1 * z2 -
                                                      4 * x0 * x2 * y0 * y2 * pow(z2, 2) +
                                                      4 * x0 * x2 * pow(y1, 2) * z0 * z1 -
                                                      4 * x0 * x2 * pow(y1, 2) * z0 * z2 -
                                                      8 * x0 * x2 * y1 * y2 * z0 * z1 +
                                                      8 * x0 * x2 * y1 * y2 * z0 * z2 +
                                                      4 * x0 * x2 * pow(y2, 2) * z0 * z1 -
                                                      4 * x0 * x2 * pow(y2, 2) * z0 * z2 +
                                                      4 * x0 * x2 * z0 * pow(z1, 3) -
                                                      12 * x0 * x2 * z0 * pow(z1, 2) * z2 +
                                                      12 * x0 * x2 * z0 * z1 * pow(z2, 2) -
                                                      4 * x0 * x2 * z0 * pow(z2, 3) + pow(x1, 4) * pow(y0, 2) +
                                                      pow(x1, 4) * pow(z0, 2) - 4 * pow(x1, 3) * x2 * pow(y0, 2) -
                                                      4 * pow(x1, 3) * x2 * pow(z0, 2) +
                                                      6 * pow(x1, 2) * pow(x2, 2) * pow(y0, 2) +
                                                      6 * pow(x1, 2) * pow(x2, 2) * pow(z0, 2) +
                                                      2 * pow(x1, 2) * pow(y0, 2) * pow(z1, 2) -
                                                      4 * pow(x1, 2) * pow(y0, 2) * z1 * z2 +
                                                      2 * pow(x1, 2) * pow(y0, 2) * pow(z2, 2) -
                                                      4 * pow(x1, 2) * y0 * y1 * z0 * z1 +
                                                      4 * pow(x1, 2) * y0 * y1 * z0 * z2 +
                                                      4 * pow(x1, 2) * y0 * y2 * z0 * z1 -
                                                      4 * pow(x1, 2) * y0 * y2 * z0 * z2 +
                                                      2 * pow(x1, 2) * pow(y1, 2) * pow(z0, 2) -
                                                      4 * pow(x1, 2) * y1 * y2 * pow(z0, 2) +
                                                      2 * pow(x1, 2) * pow(y2, 2) * pow(z0, 2) -
                                                      4 * x1 * pow(x2, 3) * pow(y0, 2) -
                                                      4 * x1 * pow(x2, 3) * pow(z0, 2) -
                                                      4 * x1 * x2 * pow(y0, 2) * pow(z1, 2) +
                                                      8 * x1 * x2 * pow(y0, 2) * z1 * z2 -
                                                      4 * x1 * x2 * pow(y0, 2) * pow(z2, 2) +
                                                      8 * x1 * x2 * y0 * y1 * z0 * z1 -
                                                      8 * x1 * x2 * y0 * y1 * z0 * z2 -
                                                      8 * x1 * x2 * y0 * y2 * z0 * z1 +
                                                      8 * x1 * x2 * y0 * y2 * z0 * z2 -
                                                      4 * x1 * x2 * pow(y1, 2) * pow(z0, 2) +
                                                      8 * x1 * x2 * y1 * y2 * pow(z0, 2) -
                                                      4 * x1 * x2 * pow(y2, 2) * pow(z0, 2) + pow(x2, 4) * pow(y0, 2) +
                                                      pow(x2, 4) * pow(z0, 2) +
                                                      2 * pow(x2, 2) * pow(y0, 2) * pow(z1, 2) -
                                                      4 * pow(x2, 2) * pow(y0, 2) * z1 * z2 +
                                                      2 * pow(x2, 2) * pow(y0, 2) * pow(z2, 2) -
                                                      4 * pow(x2, 2) * y0 * y1 * z0 * z1 +
                                                      4 * pow(x2, 2) * y0 * y1 * z0 * z2 +
                                                      4 * pow(x2, 2) * y0 * y2 * z0 * z1 -
                                                      4 * pow(x2, 2) * y0 * y2 * z0 * z2 +
                                                      2 * pow(x2, 2) * pow(y1, 2) * pow(z0, 2) -
                                                      4 * pow(x2, 2) * y1 * y2 * pow(z0, 2) +
                                                      2 * pow(x2, 2) * pow(y2, 2) * pow(z0, 2) -
                                                      pow(y0, 2) * pow(y1, 4) + 4 * pow(y0, 2) * pow(y1, 3) * y2 -
                                                      6 * pow(y0, 2) * pow(y1, 2) * pow(y2, 2) +
                                                      4 * pow(y0, 2) * y1 * pow(y2, 3) - pow(y0, 2) * pow(y2, 4) +
                                                      pow(y0, 2) * pow(z1, 4) - 4 * pow(y0, 2) * pow(z1, 3) * z2 +
                                                      6 * pow(y0, 2) * pow(z1, 2) * pow(z2, 2) -
                                                      4 * pow(y0, 2) * z1 * pow(z2, 3) + pow(y0, 2) * pow(z2, 4) -
                                                      4 * y0 * pow(y1, 3) * z0 * z1 + 4 * y0 * pow(y1, 3) * z0 * z2 +
                                                      12 * y0 * pow(y1, 2) * y2 * z0 * z1 -
                                                      12 * y0 * pow(y1, 2) * y2 * z0 * z2 -
                                                      12 * y0 * y1 * pow(y2, 2) * z0 * z1 +
                                                      12 * y0 * y1 * pow(y2, 2) * z0 * z2 -
                                                      4 * y0 * y1 * z0 * pow(z1, 3) +
                                                      12 * y0 * y1 * z0 * pow(z1, 2) * z2 -
                                                      12 * y0 * y1 * z0 * z1 * pow(z2, 2) +
                                                      4 * y0 * y1 * z0 * pow(z2, 3) + 4 * y0 * pow(y2, 3) * z0 * z1 -
                                                      4 * y0 * pow(y2, 3) * z0 * z2 + 4 * y0 * y2 * z0 * pow(z1, 3) -
                                                      12 * y0 * y2 * z0 * pow(z1, 2) * z2 +
                                                      12 * y0 * y2 * z0 * z1 * pow(z2, 2) -
                                                      4 * y0 * y2 * z0 * pow(z2, 3) + pow(y1, 4) * pow(z0, 2) -
                                                      4 * pow(y1, 3) * y2 * pow(z0, 2) +
                                                      6 * pow(y1, 2) * pow(y2, 2) * pow(z0, 2) -
                                                      4 * y1 * pow(y2, 3) * pow(z0, 2) + pow(y2, 4) * pow(z0, 2) -
                                                      pow(z0, 2) * pow(z1, 4) + 4 * pow(z0, 2) * pow(z1, 3) * z2 -
                                                      6 * pow(z0, 2) * pow(z1, 2) * pow(z2, 2) +
                                                      4 * pow(z0, 2) * z1 * pow(z2, 3) - pow(z0, 2) * pow(z2, 4)) +
                  pow(y0, 2) * y1 * pow(z1, 2) - 2 * pow(y0, 2) * y1 * z1 * z2 + pow(y0, 2) * y1 * pow(z2, 2) -
                  y0 * pow(y1, 2) * z0 * z1 + y0 * pow(y1, 2) * z0 * z2 + y0 * pow(y2, 2) * z0 * z1 -
                  y0 * pow(y2, 2) * z0 * z2 + y0 * z0 * pow(z1, 3) - 3 * y0 * z0 * pow(z1, 2) * z2 +
                  3 * y0 * z0 * z1 * pow(z2, 2) - y0 * z0 * pow(z2, 3) + pow(y1, 2) * y2 * pow(z0, 2) -
                  2 * y1 * pow(y2, 2) * pow(z0, 2) - y1 * pow(z0, 2) * pow(z1, 2) + 2 * y1 * pow(z0, 2) * z1 * z2 -
                  y1 * pow(z0, 2) * pow(z2, 2) + pow(y2, 3) * pow(z0, 2) + y2 * pow(z0, 2) * pow(z1, 2) -
                  2 * y2 * pow(z0, 2) * z1 * z2 + y2 * pow(z0, 2) * pow(z2, 2)) /
                 (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                  pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) - 2 * x0 * x1 * y0 * y1 +
                  2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 + 2 * x0 * x1 * z0 * z2 + 2 * x0 * x2 * y0 * y1 -
                  2 * x0 * x2 * y0 * y2 + 2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 + pow(x1, 2) * pow(y0, 2) +
                  pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) - 2 * x1 * x2 * pow(z0, 2) +
                  pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) + pow(y0, 2) * pow(z1, 2) -
                  2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) - 2 * y0 * y1 * z0 * z1 + 2 * y0 * y1 * z0 * z2 +
                  2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 + pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) +
                  pow(y2, 2) * pow(z0, 2));
            z4 = -sqrt(-(pow(x1, 2) - 2 * x1 * x2 + pow(x2, 2) + pow(y1, 2) - 2 * y1 * y2 + pow(y2, 2) + pow(z1, 2) -
                         2 * z1 * z2 + pow(z2, 2)) *
                       (pow(x0, 2) * pow(x1, 2) - 2 * pow(x0, 2) * x1 * x2 + pow(x0, 2) * pow(x2, 2) -
                        pow(x0, 2) * pow(y1, 2) + 2 * pow(x0, 2) * y1 * y2 - pow(x0, 2) * pow(y2, 2) -
                        pow(x0, 2) * pow(z1, 2) + 2 * pow(x0, 2) * z1 * z2 - pow(x0, 2) * pow(z2, 2) +
                        4 * x0 * x1 * y0 * y1 - 4 * x0 * x1 * y0 * y2 + 4 * x0 * x1 * z0 * z1 - 4 * x0 * x1 * z0 * z2 -
                        4 * x0 * x2 * y0 * y1 + 4 * x0 * x2 * y0 * y2 - 4 * x0 * x2 * z0 * z1 + 4 * x0 * x2 * z0 * z2 -
                        pow(x1, 2) * pow(y0, 2) - pow(x1, 2) * pow(z0, 2) + 2 * x1 * x2 * pow(y0, 2) +
                        2 * x1 * x2 * pow(z0, 2) - pow(x2, 2) * pow(y0, 2) - pow(x2, 2) * pow(z0, 2) +
                        pow(y0, 2) * pow(y1, 2) - 2 * pow(y0, 2) * y1 * y2 + pow(y0, 2) * pow(y2, 2) -
                        pow(y0, 2) * pow(z1, 2) + 2 * pow(y0, 2) * z1 * z2 - pow(y0, 2) * pow(z2, 2) +
                        4 * y0 * y1 * z0 * z1 - 4 * y0 * y1 * z0 * z2 - 4 * y0 * y2 * z0 * z1 + 4 * y0 * y2 * z0 * z2 -
                        pow(y1, 2) * pow(z0, 2) + 2 * y1 * y2 * pow(z0, 2) - pow(y2, 2) * pow(z0, 2) +
                        pow(z0, 2) * pow(z1, 2) - 2 * pow(z0, 2) * z1 * z2 + pow(z0, 2) * pow(z2, 2))) *
                 (x0 * y1 - x0 * y2 - x1 * y0 + x2 * y0) /
                 (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                  pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) - 2 * x0 * x1 * y0 * y1 +
                  2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 + 2 * x0 * x1 * z0 * z2 + 2 * x0 * x2 * y0 * y1 -
                  2 * x0 * x2 * y0 * y2 + 2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 + pow(x1, 2) * pow(y0, 2) +
                  pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) - 2 * x1 * x2 * pow(z0, 2) +
                  pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) + pow(y0, 2) * pow(z1, 2) -
                  2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) - 2 * y0 * y1 * z0 * z1 + 2 * y0 * y1 * z0 * z2 +
                  2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 + pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) +
                  pow(y2, 2) * pow(z0, 2)) -
                 (pow(x0, 2) * pow(x1, 2) * z1 - pow(x0, 2) * pow(x1, 2) * z2 - 2 * pow(x0, 2) * x1 * x2 * z1 +
                  2 * pow(x0, 2) * x1 * x2 * z2 + pow(x0, 2) * pow(x2, 2) * z1 - pow(x0, 2) * pow(x2, 2) * z2 -
                  pow(x0, 2) * pow(y1, 2) * z2 + 2 * pow(x0, 2) * y1 * y2 * z2 - pow(x0, 2) * pow(y2, 2) * z2 -
                  pow(x0, 2) * pow(z1, 2) * z2 + 2 * pow(x0, 2) * z1 * pow(z2, 2) - pow(x0, 2) * pow(z2, 3) -
                  x0 * pow(x1, 3) * z0 + 3 * x0 * pow(x1, 2) * x2 * z0 - 3 * x0 * x1 * pow(x2, 2) * z0 +
                  2 * x0 * x1 * y0 * y1 * z1 - 2 * x0 * x1 * y0 * y2 * z1 - x0 * x1 * pow(y1, 2) * z0 +
                  2 * x0 * x1 * y1 * y2 * z0 - x0 * x1 * pow(y2, 2) * z0 + x0 * x1 * z0 * pow(z1, 2) -
                  x0 * x1 * z0 * pow(z2, 2) + x0 * pow(x2, 3) * z0 - 2 * x0 * x2 * y0 * y1 * z1 +
                  2 * x0 * x2 * y0 * y2 * z1 + x0 * x2 * pow(y1, 2) * z0 - 2 * x0 * x2 * y1 * y2 * z0 +
                  x0 * x2 * pow(y2, 2) * z0 - x0 * x2 * z0 * pow(z1, 2) + x0 * x2 * z0 * pow(z2, 2) -
                  pow(x1, 2) * pow(y0, 2) * z2 - pow(x1, 2) * y0 * y1 * z0 + pow(x1, 2) * y0 * y2 * z0 -
                  pow(x1, 2) * pow(z0, 2) * z1 + 2 * x1 * x2 * pow(y0, 2) * z2 + 2 * x1 * x2 * y0 * y1 * z0 -
                  2 * x1 * x2 * y0 * y2 * z0 + 2 * x1 * x2 * pow(z0, 2) * z1 - pow(x2, 2) * pow(y0, 2) * z2 -
                  pow(x2, 2) * y0 * y1 * z0 + pow(x2, 2) * y0 * y2 * z0 - pow(x2, 2) * pow(z0, 2) * z1 +
                  pow(y0, 2) * pow(y1, 2) * z1 - pow(y0, 2) * pow(y1, 2) * z2 - 2 * pow(y0, 2) * y1 * y2 * z1 +
                  2 * pow(y0, 2) * y1 * y2 * z2 + pow(y0, 2) * pow(y2, 2) * z1 - pow(y0, 2) * pow(y2, 2) * z2 -
                  pow(y0, 2) * pow(z1, 2) * z2 + 2 * pow(y0, 2) * z1 * pow(z2, 2) - pow(y0, 2) * pow(z2, 3) -
                  y0 * pow(y1, 3) * z0 + 3 * y0 * pow(y1, 2) * y2 * z0 - 3 * y0 * y1 * pow(y2, 2) * z0 +
                  y0 * y1 * z0 * pow(z1, 2) - y0 * y1 * z0 * pow(z2, 2) + y0 * pow(y2, 3) * z0 -
                  y0 * y2 * z0 * pow(z1, 2) + y0 * y2 * z0 * pow(z2, 2) - pow(y1, 2) * pow(z0, 2) * z1 +
                  2 * y1 * y2 * pow(z0, 2) * z1 - pow(y2, 2) * pow(z0, 2) * z1) /
                 (pow(x0, 2) * pow(y1, 2) - 2 * pow(x0, 2) * y1 * y2 + pow(x0, 2) * pow(y2, 2) +
                  pow(x0, 2) * pow(z1, 2) - 2 * pow(x0, 2) * z1 * z2 + pow(x0, 2) * pow(z2, 2) - 2 * x0 * x1 * y0 * y1 +
                  2 * x0 * x1 * y0 * y2 - 2 * x0 * x1 * z0 * z1 + 2 * x0 * x1 * z0 * z2 + 2 * x0 * x2 * y0 * y1 -
                  2 * x0 * x2 * y0 * y2 + 2 * x0 * x2 * z0 * z1 - 2 * x0 * x2 * z0 * z2 + pow(x1, 2) * pow(y0, 2) +
                  pow(x1, 2) * pow(z0, 2) - 2 * x1 * x2 * pow(y0, 2) - 2 * x1 * x2 * pow(z0, 2) +
                  pow(x2, 2) * pow(y0, 2) + pow(x2, 2) * pow(z0, 2) + pow(y0, 2) * pow(z1, 2) -
                  2 * pow(y0, 2) * z1 * z2 + pow(y0, 2) * pow(z2, 2) - 2 * y0 * y1 * z0 * z1 + 2 * y0 * y1 * z0 * z2 +
                  2 * y0 * y2 * z0 * z1 - 2 * y0 * y2 * z0 * z2 + pow(y1, 2) * pow(z0, 2) - 2 * y1 * y2 * pow(z0, 2) +
                  pow(y2, 2) * pow(z0, 2));

            float point1[3],point2[3],point3[3],point4[3],point11[2],point22[2],point33[2],point44[2];
            point1[0]=x1;point1[1]=y1;point1[2]=z1;point2[0]=x2;point2[1]=y2;point2[2]=z2;point3[0]=x3;point3[1]=y3;point3[2]=z3;point4[0]=x4;point4[1]=y4;point4[2]=z4;
            //std::cerr <<"x3:"<<x33<<" y3:"<<y33<<" x4:"<<x44<<" y4:"<<y44<<endl;
            rs2_project_point_to_pixel(point11, &intrin, point1);
            rs2_project_point_to_pixel(point22, &intrin, point2);
            rs2_project_point_to_pixel(point33, &intrin, point3);
            rs2_project_point_to_pixel(point44, &intrin, point4);
            vector<rc_msgs::point> res0(4);
            res0[0].x = point11[0];
            res0[0].y = point11[1];
            res0[1].x = point22[0];
            res0[1].y = point22[1];
            res0[2].x = point33[0];
            res0[2].y = point33[1];
            res0[3].x = point44[0];
            res0[3].y = point44[1];
            res.data = res0;
            cout<<res0[0].x<<" "<<res0[0].y<<""<<res0[1].x<<" "<<res0[1].y<<" "<<res0[2].x<<" "<<res0[2].y<<" "<<res0[3].x<<" "<<res0[3].y<<endl;
            res_pub.publish(res);
        }
    }
}


int main(int argc, char **argv) {

    ros::init(argc, argv, "plane_extract"); // 节点名称
    ros::NodeHandle n;
    rc_msgs::stepConfig config;
    dynamic_reconfigure::Client<rc_msgs::stepConfig> client("/scheduler");

    client.setConfigurationCallback(&callback);
    ros::Subscriber img_sub = n.subscribe("/cloud", 10, cloudCallback);
    ros::Subscriber Identify_sub = n.subscribe("/isIdentify", 10, identifyCallback);
    res_pub = n.advertise<rc_msgs::calibrateResult>("/calibrateResult", 10);
    ros::Rate loop_rate(1);

    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
