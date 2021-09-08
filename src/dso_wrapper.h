#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class DsoImpl;
class DsoWrapper {
 public:
  /** @brief constructor
   * @param[in]img_size: input image size, image will be resize to this before process.
   * @param[in]K: camera intrinsic. DSO does NOT support dynamic camera intrinsic.
   * @param[in]D: camera distortion. not used, to be implemented.
   */
  DsoWrapper(const cv::Size& img_size, const cv::Mat& K, const cv::Mat& D = cv::Mat());

  /** @brief DSO set the first camera pose as world frame, which will not be gravity rectified.
   * One can input init Rcw, which can be got from IMU, to make the world frame to be gravity rectified.
   * If this set, the first camera pose willl be: Rcw + Position(0,0,0).
   * @param[in]Rcw: Rotation matrix from camera to world. Default: Identity.
   */
  void setCurRot(const Eigen::Matrix3d& Rcw);

  /** @brief call DSO process a frame
   * @param[in]img: image
   * @param[out]pose: camera pose on world frame
   * @return whether pose ok. if return false, output pose will be invalid.
   */
  bool process(const cv::Mat& img, Eigen::Isometry3d& pose);

  /** @brief get depth */
  bool getDepth(cv::Mat& depth, Eigen::Isometry3d& pose);

  cv::Mat getDepthDraw();

 private:
  std::shared_ptr<DsoImpl> impl_;
};
