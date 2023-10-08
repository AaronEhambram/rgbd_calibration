#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dirent.h>
#include "Eigen/Dense"

class Calibrator : public rclcpp::Node
{
  public: 
    Calibrator() : Node("calibrate_node")
    {
      int checkerboard_rows, checkerboard_cols;
      std::string folder_path;
      this->declare_parameter<int>("checkerboard_rows",1);
      checkerboard_rows = this->get_parameter("checkerboard_rows").as_int();
      this->declare_parameter<int>("checkerboard_cols",1);
      checkerboard_cols = this->get_parameter("checkerboard_cols").as_int();
      this->declare_parameter<std::string>("folder_path","");
      folder_path = this->get_parameter("folder_path").as_string();
      this->declare_parameter<double>("check_side_length",1.0);
      check_side_length = this->get_parameter("check_side_length").as_double();
      checkerboard_size = cv::Size(checkerboard_rows, checkerboard_cols);

      // read the rgb calibration images
      std::stringstream rgb_only_image_folder;
      rgb_only_image_folder << folder_path << "/rgb";
      std::vector<cv::Mat> rgb_image;
      read_images(rgb_only_image_folder, rgb_image);
      // calibrate the rgb camera
      std::stringstream rgb_calibration_results_file; 
      rgb_calibration_results_file << folder_path << "/test_rgb_camera.txt"; 
      cv::Mat rgb_camera_matrix, rgb_dist_coeff; 
      calibrate_camera(rgb_image, rgb_calibration_results_file, rgb_camera_matrix, rgb_dist_coeff);

      // read the ir calibration images
      std::stringstream ir_only_image_folder;
      ir_only_image_folder << folder_path << "/ir";
      std::vector<cv::Mat> ir_image;
      read_images(ir_only_image_folder, ir_image);
      // calibrate the rgb camera
      std::stringstream ir_calibration_results_file; 
      ir_calibration_results_file << folder_path << "/test_ir_camera.txt"; 
      cv::Mat ir_camera_matrix, ir_dist_coeff; 
      calibrate_camera(ir_image, ir_calibration_results_file, ir_camera_matrix, ir_dist_coeff);
      
      // read the calibration images where ir and rgb captured the same scene
      std::stringstream rgb_both_image_folder;
      rgb_both_image_folder << folder_path << "/both/rgb";
      std::stringstream ir_both_image_folder;
      ir_both_image_folder << folder_path << "/both/ir";
      std::vector<std::pair<cv::Mat,cv::Mat>> image_pairs;
      read_image_pairs(rgb_both_image_folder, ir_both_image_folder, image_pairs);
      // compte the relative transform
      cv::Mat rgb_R_ir, rgb_t_ir; 
      compute_relative_transform(rgb_camera_matrix,rgb_dist_coeff,ir_camera_matrix,ir_dist_coeff,image_pairs, rgb_R_ir, rgb_t_ir);

      // Save the calibration results
      std::stringstream calibration_results_file; 
      calibration_results_file << folder_path << "/test_rgbd_calibration.txt"; 
      cv::FileStorage file(calibration_results_file.str(), cv::FileStorage::WRITE);
      file << "rgb_camera_matrix" << rgb_camera_matrix;
      file << "rgb_dist_coeff" << rgb_dist_coeff;
      file << "ir_camera_matrix" << ir_camera_matrix;
      file << "ir_dist_coeff" << ir_dist_coeff;
      file << "rgb_R_ir" << rgb_R_ir;
      file << "rgb_t_ir" << rgb_t_ir;
    }

  private:
    cv::Size checkerboard_size;
    double check_side_length; 

    void read_images(std::stringstream& image_folder, std::vector<cv::Mat>& images)
    {
      DIR *dir;
      std::vector<std::string> rgb_files; 
      struct dirent *diread;
      if ((dir = opendir(image_folder.str().c_str())) != NULL) 
      {
        while ((diread = readdir(dir)) != NULL) 
        {
          std::stringstream file_path; 
          std::string file_name(diread->d_name);
          if(file_name != "." && file_name != "..")
          {
            file_path << image_folder.str() << "/" << file_name; 
            rgb_files.push_back(file_path.str());
          }
        }
      }
      images.resize(rgb_files.size());
      for(int i = 0; i < images.size(); i++)
      {
        images[i] = cv::imread(rgb_files[i]);
        cv::cvtColor(images[i], images[i], CV_BGR2GRAY);
      }
    }

    void read_image_pairs(std::stringstream& image_folder_rgb, std::stringstream& image_folder_ir, std::vector<std::pair<cv::Mat,cv::Mat>>& image_pairs)
    {
      // get rgb file names and paths
      std::vector<std::string> rgb_file_paths; 
      std::vector<std::string> rgb_file_names;
      {
        DIR *dir; 
        struct dirent *diread;
        if ((dir = opendir(image_folder_rgb.str().c_str())) != NULL) 
        {
          while ((diread = readdir(dir)) != NULL) 
          {
            std::stringstream file_path; 
            std::string file_name(diread->d_name);
            if(file_name != "." && file_name != "..")
            {
              file_path << image_folder_rgb.str() << "/" << file_name; 
              rgb_file_names.push_back(file_name);
              rgb_file_paths.push_back(file_path.str());
            }
          }
        }
      }

      // get ir file names and paths
      std::vector<std::string> ir_file_paths; 
      std::vector<std::string> ir_file_names;
      {
        DIR *dir; 
        struct dirent *diread;
        if ((dir = opendir(image_folder_ir.str().c_str())) != NULL) 
        {
          while ((diread = readdir(dir)) != NULL) 
          {
            std::stringstream file_path; 
            std::string file_name(diread->d_name);
            if(file_name != "." && file_name != "..")
            {
              file_path << image_folder_ir.str() << "/" << file_name; 
              ir_file_names.push_back(file_name);
              ir_file_paths.push_back(file_path.str());
            }
          }
        }
      }

      // Iterate through the rgb filenames and if match to ir -> get images and save to vector
      for(int rgb_counter = 0; rgb_counter < rgb_file_names.size(); rgb_counter++)
      {
        for(int ir_counter = 0; ir_counter < ir_file_names.size(); ir_counter++)
        {
          if(rgb_file_names[rgb_counter] == ir_file_names[ir_counter])
          {
            std::pair<cv::Mat,cv::Mat> new_pair; 
            image_pairs.push_back(new_pair);
            std::pair<cv::Mat,cv::Mat>& pair = image_pairs.back(); // first rgb, secon ir

            // get the rgb image
            pair.first = cv::imread(rgb_file_paths[rgb_counter]);
            cv::cvtColor(pair.first, pair.first, CV_BGR2GRAY);

            // get the IR image
            pair.second = cv::imread(ir_file_paths[ir_counter]);
            cv::cvtColor(pair.second, pair.second, CV_BGR2GRAY);
          }
        }
      }
    }

    void calibrate_camera(std::vector<cv::Mat>& images, std::stringstream& calibration_results_file, cv::Mat& cameraMatrix, cv::Mat& distCoeffs)
    {
      // create object points
      std::vector<cv::Point3f> objp;
      for(int i = 0; i<checkerboard_size.height; i++)
      {
        for(int j = 0; j<checkerboard_size.width; j++)
        {
          objp.push_back(cv::Point3f((float)j*(float)check_side_length,(float)i*(float)check_side_length,0));
        }
      }

      // Creating vector to store vectors of 3D points for each checkerboard image
      std::vector<std::vector<cv::Point3f> > objpoints;
      // Creating vector to store vectors of 2D points for each checkerboard image
      std::vector<std::vector<cv::Point2f> > imgpoints;

      // corner points detected in each image
      std::vector<cv::Point2f> corner_pts;
      bool success;
      double image_counter = 0; 
      for(cv::Mat& im : images)
      {
        corner_pts.clear(); 
        bool success = cv::findChessboardCorners(im, checkerboard_size, corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(success)
        {
          cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
          // refining pixel coordinates for given 2d points.
          cv::cornerSubPix(im,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
          objpoints.push_back(objp);
          imgpoints.push_back(corner_pts);
        }
        image_counter++;
        double processing_status = image_counter/(double) images.size(); 
        std::cout << "Processing: " << processing_status*100.0 << " %" << std::endl; 
      }

      cv::Mat R,T;
      // calibrate camera
      std::cout << "start calibrating" << std::endl; 
      cv::calibrateCamera(objpoints, imgpoints, cv::Size(images[0].rows,images[0].cols), cameraMatrix, distCoeffs, R, T);
      std::cout << "end calibrating" << std::endl;

      std::cout << "Save calibration results to : " << calibration_results_file.str() << std::endl;

      cv::FileStorage file(calibration_results_file.str(), cv::FileStorage::WRITE);
      file << "cameraMatrix" << cameraMatrix;
      file << "distCoeffs" << distCoeffs;
      file << "R" << R;
      file << "t" << T;
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

    void eigen2cv(Eigen::Affine3d& T, cv::Mat& R, cv::Mat& t)
    {
      R = cv::Mat(3,3,CV_64FC1);
      t = cv::Mat(3,1,CV_64FC1);

      R.at<double>(0,0) = T.matrix()(0,0); 
      R.at<double>(1,0) = T.matrix()(1,0); 
      R.at<double>(2,0) = T.matrix()(2,0);
      R.at<double>(0,1) = T.matrix()(0,1); 
      R.at<double>(1,1) = T.matrix()(1,1); 
      R.at<double>(2,1) = T.matrix()(2,1);
      R.at<double>(0,2) = T.matrix()(0,2); 
      R.at<double>(1,2) = T.matrix()(1,2); 
      R.at<double>(2,2) = T.matrix()(2,2);

      t.at<double>(0) = T.matrix()(0,3); 
      t.at<double>(1) = T.matrix()(1,3); 
      t.at<double>(2) = T.matrix()(2,3);
    }

    void average_poses(std::vector<Eigen::Affine3d>& Ts, Eigen::Affine3d T_avrg)
    {
      // Average translation
      Eigen::Vector3d t_avrg;
      t_avrg.setZero();
      for(Eigen::Affine3d& T : Ts)
      {
        t_avrg = t_avrg + T.translation();
      }
      t_avrg = t_avrg/(double)Ts.size();
      T_avrg.translation() = t_avrg;

      // Average rotation 
      Eigen::Vector3d ea_avrg;
      ea_avrg.setZero();
      for(Eigen::Affine3d& T : Ts)
      {
        Eigen::Vector3d ea = T.linear().eulerAngles(0, 1, 2);
        ea_avrg = ea_avrg+ea;
      }
      ea_avrg = ea_avrg/(double)Ts.size();
      Eigen::Quaterniond q_avrg = Eigen::AngleAxisd(ea_avrg[0], Eigen::Vector3d::UnitX())
                            * Eigen::AngleAxisd(ea_avrg[1], Eigen::Vector3d::UnitY())
                            * Eigen::AngleAxisd(ea_avrg[2], Eigen::Vector3d::UnitZ());
      T_avrg.linear() = q_avrg.normalized().toRotationMatrix(); 
    }

    void compute_relative_transform(cv::Mat& rgb_cam, cv::Mat& rgb_distort, cv::Mat& ir_cam, cv::Mat& ir_distort, std::vector<std::pair<cv::Mat,cv::Mat>>& image_pairs, cv::Mat& R, cv::Mat& t)
    {
      // create object points
      std::vector<cv::Point3f> objp;
      for(int i = 0; i<checkerboard_size.height; i++)
      {
        for(int j = 0; j<checkerboard_size.width; j++)
        {
          objp.push_back(cv::Point3f((float)j*(float)check_side_length,(float)i*(float)check_side_length,0));
        }
      }

      // detect the checkerboard corners
      // corner points detected in each image
      std::vector<cv::Point2f> rgb_corner_pts;
      bool rgb_success;
      std::vector<cv::Point2f> ir_corner_pts;
      bool ir_success;
      std::vector<Eigen::Affine3d> rgb_T_ir_estimates; 
      for(std::pair<cv::Mat,cv::Mat>& im_pair : image_pairs)
      {
        rgb_corner_pts.clear(); 
        rgb_success = cv::findChessboardCorners(im_pair.first, checkerboard_size, rgb_corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        ir_corner_pts.clear(); 
        ir_success = cv::findChessboardCorners(im_pair.second, checkerboard_size, ir_corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(rgb_success && ir_success)
        {
          cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
          // refining pixel coordinates for given 2d points in rgb.
          cv::cornerSubPix(im_pair.first,rgb_corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
          // refine ir
          cv::cornerSubPix(im_pair.second,ir_corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

          // get rgb-pose with respect to board
          cv::Mat rgb_r_board, rgb_R_board, rgb_t_board; 
          cv::solvePnP(objp,rgb_corner_pts,rgb_cam,rgb_distort,rgb_r_board,rgb_t_board); 
          cv::Rodrigues(rgb_r_board,rgb_R_board);
          Eigen::Affine3d rgb_T_board; 
          cv2eigen(rgb_R_board, rgb_t_board, rgb_T_board);

          // get ir-pose with respect to board
          cv::Mat ir_r_board, ir_R_board, ir_t_board; 
          cv::solvePnP(objp,ir_corner_pts,ir_cam,ir_distort,ir_r_board,ir_t_board); 
          cv::Rodrigues(ir_r_board,ir_R_board);
          Eigen::Affine3d ir_T_board; 
          cv2eigen(ir_R_board, ir_t_board, ir_T_board);

          Eigen::Affine3d rgb_T_ir = rgb_T_board*ir_T_board.inverse();
          rgb_T_ir_estimates.push_back(rgb_T_ir);
        }
      }
      Eigen::Affine3d rgb_T_ir; 
      average_poses(rgb_T_ir_estimates, rgb_T_ir);
      eigen2cv(rgb_T_ir,R,t); 
    }
};


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Calibrator>());
  rclcpp::shutdown();
  return 0; 
}