#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.h"
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Eigen/Dense"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <omp.h>
#include <chrono>
#include <functional>
#include <string>
#include <memory>

struct ColoredPoint
{
  Eigen::Vector3d p; 
  uint8_t b,g,r; 
};

class PoinCloudGenerator : public rclcpp::Node
{
  public: 
    PoinCloudGenerator() : Node("rgb_depth_registration_node")
    {
      this->declare_parameter<std::string>("calibration_file","");
      std::string calibration_file = this->get_parameter("calibration_file").as_string();
      this->declare_parameter<std::string>("rgb_topic","");
      std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
      this->declare_parameter<std::string>("depth_topic","");
      std::string depth_topic = this->get_parameter("depth_topic").as_string();

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

      // subscribers
      rgb_sub.subscribe(this, rgb_topic);
      depth_sub.subscribe(this, depth_topic);
      sync.reset(new Sync(MySyncPolicy(10),rgb_sub,depth_sub));
      sync->registerCallback(std::bind(&PoinCloudGenerator::callback,this, std::placeholders::_1, std::placeholders::_2));

      // publisher
      publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("registered_pc",10);
      cloud_msg.reset(new sensor_msgs::msg::PointCloud2);
    }

  private:
    // camera data from calibration
    cv::Mat rgb_cam, rgb_distor, depth_cam, depth_distor, rgb_R_depth, rgb_t_depth;
    Eigen::Affine3d rgb_T_depth;

    // time synchronization
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    std::shared_ptr<Sync> sync;

    // output message
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    // functions
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

    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_im_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth_im_msg)
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
      pcl::PointCloud<pcl::PointXYZRGB> cloud(depth_rect.cols, depth_rect.rows);
      #pragma omp parallel for collapse(2)
      for (int y=0; y<depth_rect.rows; y++)
      {
        for (int x=0; x<depth_rect.cols; x++)
        {
          pcl::PointXYZRGB& cp = cloud.at(x, y);
          //get the depth
          unsigned short depth = depth_rect.at<unsigned short>(y, x);
          Eigen::Vector3d depth_point;
          depth_point << (double)depth*((double)x-cx_depth)/fx_depth, (double)depth*((double)y-cy_depth)/fy_depth, (double)depth;
          depth_point = depth_point*0.001; // mm to m!
          Eigen::Vector3d rgb_point = rgb_T_depth*depth_point; // transform to rgb-frame
          cp.x = rgb_point.x(); cp.y = rgb_point.y(); cp.z = rgb_point.z();

          // project rgb-point to rgb-image
          double rgb_px_d = fx_rgb*cp.x/cp.z+cx_rgb;
          double rgb_py_d = fy_rgb*cp.y/cp.z+cy_rgb;
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
      pcl::toROSMsg(cloud, *cloud_msg);
      cloud_msg->header.frame_id = "camera_rgb_optical_frame";
      cloud_msg->header.stamp = now();
      publisher_->publish(*cloud_msg);

    }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoinCloudGenerator>());
  rclcpp::shutdown();
  return 0;
}