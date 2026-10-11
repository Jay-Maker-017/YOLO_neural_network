#include "infantry_armor.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <utility>

#include "tools/logger.hpp"

namespace auto_aim
{
struct InfantryOptions
{
  std::string json_path;
  std::string model_folder;
  std::string model_key;
  double min_confidence;
  int my_color;
};

namespace
{
InfantryOptions load_options(
  const std::string & config_path, const std::string & model_key,
  const std::string & threshold_key)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto thresholds = yaml["model_thresholds"][threshold_key];
  if (!thresholds) {
    throw std::runtime_error("Missing model_thresholds entry: " + threshold_key);
  }

  return {
    yaml["nndeployment_json"].as<std::string>(),
    yaml["nndeployment_model_dir"].as<std::string>(),
    model_key,
    thresholds["final_confidence_threshold"].as<double>(),
    yaml["my_color"].as<int>()};
}

ArmorName armor_name_from_id(int armor_id)
{
  switch (armor_id) {
    case 0: return ArmorName::sentry;
    case 1: return ArmorName::one;
    case 2: return ArmorName::two;
    case 3: return ArmorName::three;
    case 4: return ArmorName::four;
    case 5: return ArmorName::five;
    case 6: return ArmorName::outpost;
    case 7:
    case 8: return ArmorName::base;
    default: return ArmorName::not_armor;
  }
}
}  // namespace

InfantryArmor::InfantryArmor(
  const std::string & config_path, const std::string & model_key,
  const std::string & threshold_key)
: InfantryArmor(load_options(config_path, model_key, threshold_key))
{
}

InfantryArmor::InfantryArmor(InfantryOptions options)
: armor_model_(JsonConfig{
    std::move(options.json_path), std::move(options.model_key), std::move(options.model_folder)}),
  min_confidence_(options.min_confidence), my_color_(options.my_color)
{
}

std::list<Armor> InfantryArmor::detect(const cv::Mat & img, int frame_count)
{
  (void)frame_count;
  if (img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  const auto detections = armor_model_.netProcess(img, my_color_);
  std::list<Armor> armors;
  for (const auto & detection : detections) {
    if (detection.points.size() != 4 || detection.armor_id < 0 || detection.armor_id > 8 ||
        detection.color_id < 0 || detection.color_id > 2) {
      continue;
    }

    std::vector<cv::Point2f> points;
    points.reserve(detection.points.size());
    for (const auto & point : detection.points) {
      points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }
    const cv::Rect box = cv::boundingRect(points);
    Armor armor(
      detection.color_id, detection.armor_id, static_cast<float>(detection.score), box,
      std::move(points));

    // 对齐模型库语义：ID 7/8 均为基地，尺寸由模型输出的 size 字段决定。
    armor.name = armor_name_from_id(detection.armor_id);
    armor.type = detection.size == 1 ? ArmorType::big : ArmorType::small;
    armor.class_id = detection.armor_id;
    armor.duplicated = false;
    armor.center_norm = {
      armor.center.x / static_cast<float>(img.cols),
      armor.center.y / static_cast<float>(img.rows)};

    if (!check_name(armor) || !check_type(armor)) {
      continue;
    }
    armors.emplace_back(std::move(armor));
  }
  return armors;
}

bool InfantryArmor::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}

bool InfantryArmor::check_type(const Armor & armor) const
{
  return (armor.type == ArmorType::small)
           ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
           : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
              armor.name != ArmorName::outpost);
}
}  // namespace auto_aim