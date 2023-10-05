#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Eigen/Dense"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include <omp.h>

cv::Mat rgb_cam, rgb_distor, depth_cam, depth_distor, rgb_R_depth, rgb_t_depth;
Eigen::Affine3d rgb_T_depth;
std::shared_ptr<ros::Publisher> registered_pc_publisher; 

struct ColoredPoint
{
  Eigen::Vector3d p; 
  uint8_t b,g,r; 
};

void callback(const sensor_msgs::ImageConstPtr& rgb_im_msg, const sensor_msgs::ImageConstPtr& depth_im_msg)
{
  // get the image through the cv:bridge
  cv_bridge::CvImageConstPtr rgb_cv_ptr, depth_cv_ptr;
  try
  {
    rgb_cv_ptr = cv_bridge::toCvShare(rgb_im_msg,"bgr8");
    depth_cv_ptr = cv_bridge::toCvShare(depth_im_msg);
  }
  catch(cv_bridge::Exception& e)
  {
    return;     
  }

  cv::Mat rgb_rect, depth_rect;
  rgb_cv_ptr->image.copyTo(rgb_rect);
  depth_cv_ptr->image.copyTo(depth_rect);
  double& fx_depth = depth_cam.at<double>(0,0);
  double& fy_depth = depth_cam.at<double>(1,1);
  double& cx_depth = depth_cam.at<double>(0,2);
  double& cy_depth = depth_cam.at<double>(1,2);
  double& fx_rgb = rgb_cam.at<double>(0,0);
  double& fy_rgb = rgb_cam.at<double>(1,1);
  double& cx_rgb = rgb_cam.at<double>(0,2);
  double& cy_rgb = rgb_cam.at<double>(1,2);  
  std::vector<ColoredPoint> points(depth_rect.rows*depth_rect.cols); 
  points.resize(depth_rect.rows*depth_rect.cols);
  #pragma omp parallel for collapse(2)
  for (int y=0; y<depth_rect.rows; y++)
  {
    for (int x=0; x<depth_rect.cols; x++)
    {
      ColoredPoint& cp = points[y*depth_rect.cols+x];
      //get the depth
      unsigned short depth = depth_rect.at<unsigned short>(y, x);
      Eigen::Vector3d depth_point;
      depth_point << (double)depth*((double)x-cx_depth)/fx_depth, (double)depth*((double)y-cy_depth)/fy_depth, (double)depth;
      depth_point = depth_point*0.001; // mm to m!
      cp.p = rgb_T_depth*depth_point;

      // project rgb-point to rgb-image
      double rgb_px_d = fx_rgb*cp.p(0)/cp.p(2)+cx_rgb;
      double rgb_py_d = fy_rgb*cp.p(1)/cp.p(2)+cy_rgb;
      int rgb_px = (int)std::round(rgb_px_d);
      int rgb_py = (int)std::round(rgb_py_d);

      // get the pixel values from rgb-image
      cv::Vec3b color(255,255,255);
      if(rgb_py > 0 && rgb_py < rgb_rect.rows; rgb_px > 0 && rgb_px < rgb_rect.cols)
      {
        color = rgb_rect.at<cv::Vec3b>(rgb_py,rgb_px);
      }
      cp.r = (color[2]);
      cp.g = (color[1]);
      cp.b = (color[0]);
    }
  }

  // register color to pointcloud: https://answers.ros.org/question/219876/using-sensor_msgspointcloud2-natively/
  //declare message and sizes
  sensor_msgs::PointCloud2 cloud;
  cloud.header.frame_id = "camera_rgb_frame";
  cloud.header.stamp = ros::Time::now();
  cloud.width  = points.size();
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = false; // there may be invalid points
  //for fields setup
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(2,"xyz","rgb");
  modifier.resize(points.size());
  //iterators
  sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_r(cloud, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_g(cloud, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_b(cloud, "b");
  // store to cloud
  for (int i=0; i<points.size(); i++)
  {
    //store xyz in point cloud, transforming from image coordinates, (Z Forward to X Forward)
    *out_x = (float)points[i].p(2);
    *out_y = -(float)points[i].p(0);
    *out_z = -(float)points[i].p(1);

    // store colors
    *out_r = points[i].r;
    *out_g = points[i].g;
    *out_b = points[i].b;

    //increment
    ++out_x;
    ++out_y;
    ++out_z;
    ++out_r;
    ++out_g;
    ++out_b;
  }
  registered_pc_publisher->publish(cloud);
}

void cv2eigen(cv::Mat& R, cv::Mat& t, Eigen::Affine3d& T)
{
  T.matrix()(0,0) = R.at<double>(0,0); 
  T.matrix()(1,0) = R.at<double>(1,0); 
  T.matrix()(2,0) = R.at<double>(2,0);
  T.matrix()(0,1) = R.at<double>(0,1); 
  T.matrix()(1,1) = R.at<double>(1,1); 
  T.matrix()(2,1) = R.at<double>(2,1);
  T.matrix()(0,2) = R.at<double>(0,2); 
  T.matrix()(1,2) = R.at<double>(1,2); 
  T.matrix()(2,2) = R.at<double>(2,2);

  T.matrix()(0,3) = t.at<double>(0); 
  T.matrix()(1,3) = t.at<double>(1); 
  T.matrix()(2,3) = t.at<double>(2);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rgb_depth_registration_node");
  ros::NodeHandle nh;

  std::string rgb_topic, depth_topic, calibration_file;  
  nh.getParam("rgb_depth_registration_node/rgb_topic", rgb_topic);
  nh.getParam("rgb_depth_registration_node/depth_topic", depth_topic);
  nh.getParam("rgb_depth_registration_node/calibration_file", calibration_file);

  // Read calibration data
  cv::FileStorage calibration_data(calibration_file, cv::FileStorage::READ);
  calibration_data["rgb_camera_matrix"] >> rgb_cam;
  calibration_data["rgb_dist_coeff"] >> rgb_distor;
  calibration_data["ir_camera_matrix"] >> depth_cam;
  calibration_data["ir_dist_coeff"] >> depth_distor;
  calibration_data["rgb_R_ir"] >> rgb_R_depth;
  calibration_data["rgb_t_ir"] >> rgb_t_depth;
  cv2eigen(rgb_R_depth,rgb_t_depth,rgb_T_depth);
  calibration_data.release();
  std::cout << "rgb_T_depth: " << std::endl << rgb_T_depth.matrix() << std::endl;

  message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, rgb_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, depth_topic, 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_sub, depth_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  registered_pc_publisher.reset(new ros::Publisher(nh.advertise<sensor_msgs::PointCloud2>("registered_pc", 1)));

  ros::spin();
  return 0; 
}