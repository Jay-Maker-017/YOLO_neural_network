#ifndef AUTO_AIM__INFANTRY_ARMOR_HPP
#define AUTO_AIM__INFANTRY_ARMOR_HPP

#include <list>
#include <string>

#include "network_deployment_interface.hpp"
#include "tasks/armor.hpp"
#include "tasks/yolo.hpp"

namespace auto_aim
{
struct InfantryOptions;

class InfantryArmor : public YOLOBase
{
public:
  InfantryArmor(
    const std::string & config_path, const std::string & model_key,
    const std::string & threshold_key);

  std::list<Armor> detect(const cv::Mat & img, int frame_count) override;

private:
  explicit InfantryArmor(InfantryOptions options);

  ArmorModel armor_model_;
  double min_confidence_;
  int my_color_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__INFANTRY_ARMOR_HPP