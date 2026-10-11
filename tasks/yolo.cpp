#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

#include "yolos/infantry_armor.hpp"
#include "yolos/yolov5.hpp"

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, const std::string & requested_model_name, bool debug)
{
  auto yaml = YAML::LoadFile(config_path);
  model_name_ = requested_model_name.empty()
                  ? yaml["yolo_name"].as<std::string>()
                  : requested_model_name;

  if (model_name_ == "yolov5") {
    yolo_ = std::make_unique<YOLOV5>(config_path, debug, "", "yolov5");
  } else if (model_name_ == "infantry_v5") {
    yolo_ = std::make_unique<InfantryArmor>(config_path, "armor_v5", "infantry_v5");
  } else if (model_name_ == "infantry_v8") {
    yolo_ = std::make_unique<InfantryArmor>(config_path, "armor_v8", "infantry_v8");
  } else if (model_name_ == "infantry_v5_legacy") {
    const auto model_path = yaml["yolov5_legacy_infantry_v5_model_path"].as<std::string>();
    yolo_ = std::make_unique<YOLOV5>(config_path, debug, model_path, "infantry_v5_legacy");
  } else {
    throw std::runtime_error(
      "Unknown model name: " + model_name_ +
      ". Available models: yolov5, infantry_v5, infantry_v8, infantry_v5_legacy");
  }
}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
  //虚派发到 YOLOV5::detect（tasks/yolos/yolov5.cpp:57）
}

}  // namespace auto_aim