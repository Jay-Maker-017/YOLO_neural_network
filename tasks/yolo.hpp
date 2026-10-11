#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

#include "armor.hpp"

namespace auto_aim
{
class YOLOBase
{
public:
  virtual ~YOLOBase() = default;
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

};

class YOLO
{
public:
  YOLO(const std::string & config_path, const std::string & model_name = "", bool debug = true);

  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);
  const std::string & model_name() const { return model_name_; }

private:
  std::unique_ptr<YOLOBase> yolo_;
  std::string model_name_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP