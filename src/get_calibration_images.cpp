#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>
#include <memory>

class ImageSaver : public rclcpp::Node
{
  public:
    ImageSaver() : Node("get_calibration_images_node")
    {
      this->declare_parameter<std::string>("image_topic","");
      std::string image_topic = this->get_parameter("image_topic").as_string();
      this->declare_parameter<int>("checkerboard_rows",1);
      checkerboard_rows = this->get_parameter("checkerboard_rows").as_int();
      this->declare_parameter<int>("checkerboard_cols",1);
      checkerboard_cols = this->get_parameter("checkerboard_cols").as_int();
      this->declare_parameter<std::string>("folder_path","");
      folder_path = this->get_parameter("folder_path").as_string();
      this->declare_parameter<double>("capture_frequency",1.0);
      capture_frequency = this->get_parameter("capture_frequency").as_double();

      last_capture_time =  std::chrono::system_clock::now();
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(image_topic, 10, std::bind(&ImageSaver::image_cb, this, std::placeholders::_1));
    }

    private:
      int checkerboard_rows, checkerboard_cols;
      int image_counter = 0; 
      std::string folder_path; 
      std::chrono::system_clock::time_point last_capture_time;
      double capture_frequency; 
      rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
      void image_cb(const sensor_msgs::msg::Image::ConstSharedPtr& im_msg)
      {
        // get the image through the cv:bridge
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
          cv_ptr = cv_bridge::toCvShare(im_msg);
        }
        catch(cv_bridge::Exception& e)
        {
          return;     
        }

        // copy the images into gray (and convert to CV_8UC1)
        cv::Mat gray, visu; 
        if(im_msg->encoding == "rgb8")
        {
          cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);
        } 
        else if(im_msg->encoding == "mono16")
        {
          cv_ptr->image.convertTo(gray,CV_8UC1);
        }
        gray.copyTo(visu); 

        std::chrono::duration<double> time_since_last_capture = std::chrono::system_clock::now() - last_capture_time;
        // find checkerboard
        // vector to store the pixel coordinates of detected checker board corners 
        std::vector<cv::Point2f> corner_pts;
        cv::Size checkerboard_size(checkerboard_rows, checkerboard_cols);
        bool success = cv::findChessboardCorners(gray, checkerboard_size, corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(success)
        {
          if(time_since_last_capture.count() > 1/capture_frequency)
          {
            // save the image to file
            std::stringstream ss;
            ss << folder_path;
            ss << "/";
            ss << image_counter;
            ss << ".png";
            std::cout << "save image: " << ss.str() << std::endl; 
            last_capture_time =  std::chrono::system_clock::now();
            cv::imwrite(ss.str(), gray);
            image_counter++;
          }
          // Displaying the detected corner points on the checker board
          cv::drawChessboardCorners(visu, checkerboard_size, corner_pts, success);
        }

        // show the image
        cv::imshow("image", visu); 
        cv::waitKey(1);
      }
};


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageSaver>());
  rclcpp::shutdown();
  return 0;
}