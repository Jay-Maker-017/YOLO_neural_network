#include "tasks/pnp_solver.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  const auto loaded_config = auto_aim::PnPConfig::load("configs/pnp.yaml");
  assert(loaded_config.calibration_size == cv::Size(960, 720));
  assert(loaded_config.calibration_resolution_verified);
  assert(!loaded_config.camera_model_verified);
  assert(std::abs(loaded_config.camera_matrix.at<double>(0, 0) - 1667.491479) < 1e-5);
  assert(std::abs(loaded_config.camera_matrix.at<double>(1, 1) - 1557.461069) < 1e-5);
  assert(std::abs(loaded_config.camera_matrix.at<double>(0, 2) - 479.417745) < 1e-5);
  assert(std::abs(loaded_config.camera_matrix.at<double>(1, 2) - 352.712393) < 1e-5);

  auto_aim::PnPConfig config;
  config.camera_matrix = (cv::Mat_<double>(3, 3) <<
    900.0, 0.0, 640.0,
    0.0, 910.0, 360.0,
    0.0, 0.0, 1.0);
  config.distortion_coefficients = (cv::Mat_<double>(1, 5) << 0.01, -0.005, 0.0, 0.0, 0.0);
  config.calibration_size = {1280, 720};
  config.calibration_roi = {0, 0, 1280, 720};
  config.axis_length_mm = 55.0;
  auto_aim::PnPSolver solver(config);

  const cv::Vec3d truth_rvec(0.12, -0.18, 0.05);
  const cv::Vec3d truth_tvec(35.0, -20.0, 950.0);
  const auto object_points = auto_aim::PnPSolver::object_points(auto_aim::small);
  const auto big_object_points = auto_aim::PnPSolver::object_points(auto_aim::big);
  assert(std::abs(cv::norm(object_points[0] - object_points[3]) - 135.0) < 1e-4);
  assert(std::abs(cv::norm(big_object_points[0] - big_object_points[3]) - 225.0) < 1e-4);
  assert(std::abs(cv::norm(big_object_points[0] - big_object_points[1]) - 55.0) < 1e-4);
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(
    object_points, truth_rvec, truth_tvec, config.camera_matrix,
    config.distortion_coefficients, image_points);

  auto_aim::Armor armor(
    3, 0.9F, cv::Rect(0, 0, 100, 80), image_points);
  armor.type = auto_aim::small;
  const auto pose = solver.solve(armor, config.calibration_size);
  assert(pose.success && "synthetic planar PnP solve should succeed");
  assert(pose.reprojection_error_px < 1e-3);
  const auto metrics = auto_aim::PnPSolver::compare_with_ground_truth(pose, truth_rvec, truth_tvec);
  assert(metrics.valid);
  assert(metrics.translation_error_mm < 1e-2);
  assert(metrics.rotation_error_rad < 1e-4);

  auto cropped_config = config;
  cropped_config.calibration_roi = {100, 50, 1000, 600};
  auto cropped_camera_matrix = config.camera_matrix.clone();
  cropped_camera_matrix.at<double>(0, 0) *= 0.8;
  cropped_camera_matrix.at<double>(1, 1) *= 0.8;
  cropped_camera_matrix.at<double>(0, 2) = (640.0 - 100.0) * 0.8;
  cropped_camera_matrix.at<double>(1, 2) = (360.0 - 50.0) * 0.8;
  std::vector<cv::Point2f> cropped_image_points;
  cv::projectPoints(
    object_points, truth_rvec, truth_tvec, cropped_camera_matrix,
    config.distortion_coefficients, cropped_image_points);
  auto_aim::Armor cropped_armor(
    3, 0.9F, cv::Rect(0, 0, 100, 80), cropped_image_points);
  cropped_armor.type = auto_aim::small;
  auto_aim::PnPSolver cropped_solver(cropped_config);
  const auto cropped_pose = cropped_solver.solve(cropped_armor, {800, 480});
  assert(cropped_pose.success);
  assert(cv::norm(cropped_pose.tvec_mm - truth_tvec) < 1e-2);

  cv::Mat visualization(config.calibration_size, CV_8UC3, cv::Scalar::all(0));
  solver.draw_axes(visualization, pose);
  int red_pixels = 0;
  int green_pixels = 0;
  int blue_pixels = 0;
  for (int y = 0; y < visualization.rows; ++y) {
    for (int x = 0; x < visualization.cols; ++x) {
      const auto pixel = visualization.at<cv::Vec3b>(y, x);
      if (pixel[2] > pixel[0] + pixel[1]) ++red_pixels;
      if (pixel[1] > pixel[0] + pixel[2]) ++green_pixels;
      if (pixel[0] > pixel[1] + pixel[2]) ++blue_pixels;
    }
  }
  assert(red_pixels > 10 && green_pixels > 10 && blue_pixels > 10);

  auto_aim::Armor invalid_armor = armor;
  invalid_armor.points[2] = invalid_armor.points[1];
  const auto invalid_pose = solver.solve(invalid_armor, config.calibration_size);
  assert(!invalid_pose.success);

  std::cout << "PnP synthetic pose and RGB axis checks passed\n";
  return 0;
}
