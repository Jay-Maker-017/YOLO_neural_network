#include "pnp_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace auto_aim
{
namespace
{
bool finite_point(const cv::Point2f & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool finite_vector(const cv::Vec3d & vector)
{
  return std::isfinite(vector[0]) && std::isfinite(vector[1]) && std::isfinite(vector[2]);
}

bool finite_point(const cv::Point2d & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool projectable(const cv::Point2d & point)
{
  constexpr double max_coordinate = 1.0e7;
  return finite_point(point) && std::abs(point.x) < max_coordinate &&
         std::abs(point.y) < max_coordinate;
}
}  // namespace

PnPConfig PnPConfig::load(const std::string & path)
{
  cv::FileStorage storage(path, cv::FileStorage::READ);
  if (!storage.isOpened()) {
    throw std::runtime_error("无法打开 PnP 标定配置: " + path);
  }

  PnPConfig config;
  storage["camera_matrix"] >> config.camera_matrix;
  storage["distortion_coefficients"] >> config.distortion_coefficients;

  int width = 0;
  int height = 0;
  storage["calibration_width"] >> width;
  storage["calibration_height"] >> height;
  config.calibration_size = {width, height};
  int resolution_verified = 0;
  storage["calibration_resolution_verified"] >> resolution_verified;
  config.calibration_resolution_verified = resolution_verified != 0;

  int roi_x = 0;
  int roi_y = 0;
  int roi_width = width;
  int roi_height = height;
  const cv::FileNode roi_node = storage["calibration_roi"];
  if (!roi_node.empty()) {
    roi_node["x"] >> roi_x;
    roi_node["y"] >> roi_y;
    roi_node["width"] >> roi_width;
    roi_node["height"] >> roi_height;
  }
  config.calibration_roi = {roi_x, roi_y, roi_width, roi_height};

  const cv::FileNode axis_node = storage["axis_length_mm"];
  if (!axis_node.empty()) axis_node >> config.axis_length_mm;
  const cv::FileNode rms_node = storage["calibration_rms_px"];
  if (!rms_node.empty()) rms_node >> config.calibration_rms_px;

  if (config.camera_matrix.empty() || config.camera_matrix.rows != 3 ||
      config.camera_matrix.cols != 3) {
    throw std::runtime_error("camera_matrix 必须为 3x3 矩阵");
  }
  if (config.distortion_coefficients.empty()) {
    throw std::runtime_error("distortion_coefficients 不能为空");
  }
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("calibration_width/height 必须为正数");
  }
  if (roi_x < 0 || roi_y < 0 || roi_width <= 0 || roi_height <= 0 ||
      roi_x + roi_width > width || roi_y + roi_height > height) {
    throw std::runtime_error("calibration_roi 超出标定图像范围");
  }
  if (!(config.axis_length_mm > 0.0) || !std::isfinite(config.axis_length_mm)) {
    throw std::runtime_error("axis_length_mm 必须为正有限值");
  }

  config.camera_matrix.convertTo(config.camera_matrix, CV_64F);
  config.distortion_coefficients = config.distortion_coefficients.reshape(1, 1);
  config.distortion_coefficients.convertTo(config.distortion_coefficients, CV_64F);
  return config;
}

PnPSolver::PnPSolver(PnPConfig config) : config_(std::move(config))
{
  if (config_.camera_matrix.empty() || config_.camera_matrix.rows != 3 ||
      config_.camera_matrix.cols != 3 || config_.distortion_coefficients.empty() ||
      config_.calibration_size.width <= 0 || config_.calibration_size.height <= 0 ||
      config_.calibration_roi.width <= 0 || config_.calibration_roi.height <= 0 ||
      config_.calibration_roi.x < 0 || config_.calibration_roi.y < 0 ||
      config_.calibration_roi.x + config_.calibration_roi.width > config_.calibration_size.width ||
      config_.calibration_roi.y + config_.calibration_roi.height > config_.calibration_size.height ||
      !(config_.axis_length_mm > 0.0) || !std::isfinite(config_.axis_length_mm)) {
    throw std::invalid_argument("PnP 配置参数无效");
  }
  config_.camera_matrix.convertTo(config_.camera_matrix, CV_64F);
  config_.distortion_coefficients = config_.distortion_coefficients.reshape(1, 1);
  config_.distortion_coefficients.convertTo(config_.distortion_coefficients, CV_64F);
  if (!(config_.camera_matrix.at<double>(0, 0) > 0.0) ||
      !(config_.camera_matrix.at<double>(1, 1) > 0.0)) {
    throw std::invalid_argument("相机焦距必须为正数");
  }
  for (int row = 0; row < config_.camera_matrix.rows; ++row) {
    for (int column = 0; column < config_.camera_matrix.cols; ++column) {
      if (!std::isfinite(config_.camera_matrix.at<double>(row, column))) {
        throw std::invalid_argument("camera_matrix 包含非有限值");
      }
    }
  }
  for (int index = 0; index < config_.distortion_coefficients.cols; ++index) {
    if (!std::isfinite(config_.distortion_coefficients.at<double>(0, index))) {
      throw std::invalid_argument("distortion_coefficients 包含非有限值");
    }
  }
}

PnPSolver::PnPSolver(const std::string & config_path) : PnPSolver(PnPConfig::load(config_path)) {}

std::vector<cv::Point3f> PnPSolver::object_points(ArmorType type)
{
  double width = 0.0;
  double height = 55.0;
  switch (type) {
    case small:
      width = 135.0;
      break;
    case big:
      width = 225.0;
      break;
    default:
      throw std::invalid_argument("未知装甲板类型");
  }

  const float half_width = static_cast<float>(width / 2.0);
  const float half_height = static_cast<float>(height / 2.0);
  // Match Armor::points: upper-left, lower-left, lower-right, upper-right.
  // Local +X points right, +Y points up; +Z follows the right-handed normal.
  return {{-half_width, half_height, 0.0F},
          {-half_width, -half_height, 0.0F},
          {half_width, -half_height, 0.0F},
          {half_width, half_height, 0.0F}};
}

bool PnPSolver::camera_matrix_for_image(
  const cv::Size & image_size, cv::Mat & camera_matrix, std::string & error) const
{
  if (image_size.width <= 0 || image_size.height <= 0) {
    error = "输入图像尺寸无效";
    return false;
  }

  const double scale_x = static_cast<double>(image_size.width) / config_.calibration_roi.width;
  const double scale_y = static_cast<double>(image_size.height) / config_.calibration_roi.height;
  camera_matrix = config_.camera_matrix.clone();
  camera_matrix.at<double>(0, 0) *= scale_x;
  camera_matrix.at<double>(1, 1) *= scale_y;
  camera_matrix.at<double>(0, 2) =
    (camera_matrix.at<double>(0, 2) - config_.calibration_roi.x) * scale_x;
  camera_matrix.at<double>(1, 2) =
    (camera_matrix.at<double>(1, 2) - config_.calibration_roi.y) * scale_y;
  return true;
}

PoseResult PnPSolver::solve(const Armor & armor, const cv::Size & image_size) const
{
  PoseResult result;
  if (armor.points.size() != 4) {
    result.error = "PnP 需要恰好 4 个角点";
    return result;
  }
  for (const auto & point : armor.points) {
    if (!finite_point(point)) {
      result.error = "角点包含非有限坐标";
      return result;
    }
  }
  std::vector<cv::Point2f> image_points = armor.points;
  if (!cv::isContourConvex(image_points) || std::abs(cv::contourArea(image_points)) < 1.0) {
    result.error = "角点构成退化或非凸四边形";
    return result;
  }

  std::vector<cv::Point3f> model_points;
  try {
    model_points = object_points(armor.type);
  } catch (const std::exception & exception) {
    result.error = exception.what();
    return result;
  }

  cv::Mat camera_matrix;
  if (!camera_matrix_for_image(image_size, camera_matrix, result.error)) return result;

  try {
    if (!cv::solvePnP(
          model_points, image_points, camera_matrix, config_.distortion_coefficients,
          result.rvec, result.tvec_mm, false, cv::SOLVEPNP_ITERATIVE)) {
      result.error = "cv::solvePnP 未找到有效解";
      return result;
    }

    if (!finite_vector(result.rvec) || !finite_vector(result.tvec_mm)) {
      result.error = "PnP 返回非有限姿态";
      return result;
    }

    cv::Mat rotation_matrix;
    cv::Rodrigues(result.rvec, rotation_matrix);
    cv::Matx33d rotation;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        rotation(row, column) = rotation_matrix.at<double>(row, column);
      }
    }
    for (const auto & point : model_points) {
      const cv::Vec3d camera_point =
        rotation * cv::Vec3d(point.x, point.y, point.z) + result.tvec_mm;
      if (!(camera_point[2] > 0.0) || !finite_vector(camera_point)) {
        result.error = "求解结果位于相机后方或深度无效";
        return result;
      }
    }

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(
      model_points, result.rvec, result.tvec_mm, camera_matrix,
      config_.distortion_coefficients, projected_points);
    double squared_error = 0.0;
    for (std::size_t index = 0; index < projected_points.size(); ++index) {
      const cv::Point2f difference = projected_points[index] - image_points[index];
      squared_error += difference.dot(difference);
    }
    result.reprojection_error_px = std::sqrt(squared_error / projected_points.size());
    result.success = std::isfinite(result.reprojection_error_px);
    if (!result.success) result.error = "重投影误差无效";
  } catch (const cv::Exception & exception) {
    result.error = std::string("OpenCV PnP 异常: ") + exception.what();
  }
  return result;
}

void PnPSolver::draw_axes(cv::Mat & image, const PoseResult & pose) const
{
  if (image.empty() || !pose.success || !finite_vector(pose.rvec) || !finite_vector(pose.tvec_mm)) return;

  cv::Mat camera_matrix;
  std::string error;
  if (!camera_matrix_for_image(image.size(), camera_matrix, error)) return;

  const std::array<cv::Point3f, 4> axes = {
    cv::Point3f(0.0F, 0.0F, 0.0F),
    cv::Point3f(static_cast<float>(config_.axis_length_mm), 0.0F, 0.0F),
    cv::Point3f(0.0F, static_cast<float>(config_.axis_length_mm), 0.0F),
    cv::Point3f(0.0F, 0.0F, static_cast<float>(config_.axis_length_mm))};

  cv::Mat rotation_matrix;
  cv::Rodrigues(pose.rvec, rotation_matrix);
  cv::Matx33d rotation;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      rotation(row, column) = rotation_matrix.at<double>(row, column);
    }
  }
  const cv::Vec3d origin_camera = pose.tvec_mm;
  if (!(origin_camera[2] > 0.0) || !finite_vector(origin_camera)) return;

  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    std::vector<cv::Point3f>(axes.begin(), axes.end()), pose.rvec, pose.tvec_mm,
    camera_matrix, config_.distortion_coefficients, projected);
  if (projected.size() != axes.size()) return;

  const cv::Rect canvas(0, 0, image.cols, image.rows);
  const std::array<cv::Scalar, 3> colors = {
    cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
  const cv::Point2d origin(projected[0].x, projected[0].y);
  if (!projectable(origin)) return;
  for (std::size_t axis = 0; axis < colors.size(); ++axis) {
    const cv::Point3f & endpoint = axes[axis + 1];
    const cv::Vec3d camera_endpoint =
      rotation * cv::Vec3d(endpoint.x, endpoint.y, endpoint.z) + pose.tvec_mm;
    const cv::Point2d projected_endpoint(projected[axis + 1].x, projected[axis + 1].y);
    if (!(camera_endpoint[2] > 0.0) || !finite_vector(camera_endpoint) ||
        !projectable(projected_endpoint)) {
      continue;
    }

    cv::Point start(cvRound(origin.x), cvRound(origin.y));
    cv::Point finish(cvRound(projected_endpoint.x), cvRound(projected_endpoint.y));
    if (cv::clipLine(canvas, start, finish)) {
      cv::line(image, start, finish, colors[axis], 3, cv::LINE_AA);
    }
  }
}

PoseValidationMetrics PnPSolver::compare_with_ground_truth(
  const PoseResult & estimate, const cv::Vec3d & ground_truth_rvec,
  const cv::Vec3d & ground_truth_tvec_mm)
{
  PoseValidationMetrics metrics;
  if (!estimate.success || !finite_vector(estimate.rvec) || !finite_vector(estimate.tvec_mm) ||
      !finite_vector(ground_truth_rvec) || !finite_vector(ground_truth_tvec_mm)) {
    return metrics;
  }

  cv::Mat estimated_rotation;
  cv::Mat ground_truth_rotation;
  cv::Rodrigues(estimate.rvec, estimated_rotation);
  cv::Rodrigues(ground_truth_rvec, ground_truth_rotation);
  const cv::Mat relative_rotation = estimated_rotation * ground_truth_rotation.t();
  const double cosine = std::clamp((cv::trace(relative_rotation)[0] - 1.0) / 2.0, -1.0, 1.0);
  metrics.translation_error_mm = cv::norm(estimate.tvec_mm - ground_truth_tvec_mm);
  metrics.rotation_error_rad = std::acos(cosine);
  metrics.reprojection_error_px = estimate.reprojection_error_px;
  metrics.valid = std::isfinite(metrics.translation_error_mm) &&
                  std::isfinite(metrics.rotation_error_rad) &&
                  std::isfinite(metrics.reprojection_error_px);
  return metrics;
}

}  // namespace auto_aim
