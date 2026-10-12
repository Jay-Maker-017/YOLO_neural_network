#ifndef AUTO_AIM__PNP_SOLVER_HPP
#define AUTO_AIM__PNP_SOLVER_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "armor.hpp"

namespace auto_aim
{
struct PnPConfig
{
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  cv::Size calibration_size;
  bool calibration_resolution_verified = false;
  bool camera_model_verified = false;
  // ROI in calibration-image coordinates. The ROI is resized to the input photo.
  cv::Rect calibration_roi;
  double axis_length_mm = 55.0;
  double calibration_rms_px = 0.0;

  static PnPConfig load(const std::string & path);
};

struct PoseResult
{
  bool success = false;
  cv::Vec3d rvec{0.0, 0.0, 0.0};
  cv::Vec3d tvec_mm{0.0, 0.0, 0.0};
  double reprojection_error_px = 0.0;
  std::string error;
};

// Validation adapter: future ground-truth image records can call this without
// binding the solver to a particular annotation file format.
struct PoseValidationMetrics
{
  bool valid = false;
  double translation_error_mm = 0.0;
  double rotation_error_rad = 0.0;
  double reprojection_error_px = 0.0;
};

class PnPSolver
{
public:
  explicit PnPSolver(PnPConfig config);
  explicit PnPSolver(const std::string & config_path);

  PoseResult solve(const Armor & armor, const cv::Size & image_size) const;
  void draw_axes(cv::Mat & image, const PoseResult & pose) const;

  static std::vector<cv::Point3f> object_points(ArmorType type);
  static PoseValidationMetrics compare_with_ground_truth(
    const PoseResult & estimate, const cv::Vec3d & ground_truth_rvec,
    const cv::Vec3d & ground_truth_tvec_mm);

private:
  bool camera_matrix_for_image(const cv::Size & image_size, cv::Mat & camera_matrix,
                               std::string & error) const;
  PnPConfig config_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PNP_SOLVER_HPP
