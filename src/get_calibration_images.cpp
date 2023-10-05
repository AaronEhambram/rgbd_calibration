#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

int checkerboard_rows, checkerboard_cols;
int image_counter = 0; 
std::string folder_path; 
std::chrono::system_clock::time_point last_capture_time;
double capture_frequency; 


void image_cb(const sensor_msgs::ImageConstPtr& im_msg)
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

  // calibration procedure: https://learnopencv.com/camera-calibration-using-opencv/
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "get_calibration_images_node");
  ros::NodeHandle nh;

  std::string image_topic;  
  nh.getParam("get_calibration_images_node/image_topic", image_topic);
  nh.getParam("get_calibration_images_node/checkerboard_rows", checkerboard_rows);
  nh.getParam("get_calibration_images_node/checkerboard_cols", checkerboard_cols);
  nh.getParam("get_calibration_images_node/folder_path", folder_path);
  nh.getParam("get_calibration_images_node/capture_frequency", capture_frequency);

  last_capture_time =  std::chrono::system_clock::now();

  ros::Subscriber im_sub = nh.subscribe(image_topic, 1, image_cb);
  
  ros::spin();
  return 0; 
}