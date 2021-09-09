#include "dso_wrapper.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"
#include "IOWrapper/Output3DWrapper.h"

using namespace dso;

class MyOutputWrapper : public IOWrap::Output3DWrapper {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MyOutputWrapper() : has_pose(false) {}

  virtual ~MyOutputWrapper() {}

  virtual void publishKeyframes(std::vector<FrameHessian*>& frames, bool final, CalibHessian* HCalib) override {
    // printf("KF_size:%d : ", frames.size());
    // for(const auto& f: frames) printf("%d ", f->flaggedForMarginalization);
    // printf("\n");
  }

  virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override {
    // printf("publishCamPose\n");
    cur_pose.matrix() = frame->camToWorld.matrix();
    has_pose = true;
  }

  virtual void pushDepthImage(MinimalImageB3* image) override {
    depth_draw = cv::Mat(image->h, image->w, CV_8UC3);
    memcpy(depth_draw.data, image->data, image->h * image->w * 3);
  }

  virtual bool needPushDepthImage() override { return false; }

  virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF) override {
    kf_depth = cv::Mat(image->h, image->w, CV_32FC1);
    float* ptr_src = image->data;
    float* ptr_tar = reinterpret_cast<float*>(kf_depth.data);
    for (int i = 0; i < image->h * image->w; i++) {
      float idepth = *ptr_src++;
      *ptr_tar++ = idepth > 1e-5 ? 1.0f / idepth : 0;
    }
    kf_pose = KF->shell->camToWorld.matrix();
  }

 public:
  bool has_pose;               ///< whether has camera pose
  Eigen::Isometry3d cur_pose;  ///< current camera pose
  cv::Mat kf_depth;            ///< float depth for keyframe
  Eigen::Isometry3d kf_pose;   ///< keyframe camera pose
  cv::Mat depth_draw;          ///< draw kf depth

  std::vector<cv::Point3f> cur_pts_immature;
  std::vector<cv::Point3f> cur_pts_active;
  std::vector<cv::Point3f> cur_pts_marginalized;
  std::vector<cv::Point3f> cur_pts_outlier;

  std::vector<cv::Point3f> world_pts_immature;
  std::vector<cv::Point3f> world_pts_active;
  std::vector<cv::Point3f> world_pts_marginalized;
  std::vector<cv::Point3f> world_pts_outlier;
};

class DsoImpl {
 public:
  DsoImpl(const cv::Size& img_size, const cv::Mat& K, const cv::Mat& D = cv::Mat());

  ~DsoImpl();

  void reset();

  bool process(const cv::Mat& img, Eigen::Isometry3d& pose);

  bool getDepth(cv::Mat& depth, Eigen::Isometry3d& pose);

  cv::Mat getDepthDraw();

  void setCurRot(const Eigen::Matrix3d& Rcw) { cur_Rcw_ = Rcw; }

 private:
  int idx_;
  cv::Size img_size_;
  cv::Mat K_, D_;
  bool init_rot_set_;
  Eigen::Matrix3d init_Rcw_, cur_Rcw_;
  std::shared_ptr<FullSystem> full_system_;
  std::shared_ptr<Undistort> undistort_;
  std::shared_ptr<MyOutputWrapper> output_wrapper_;

  void defaultSetting();
};

DsoImpl::DsoImpl(const cv::Size& img_size, const cv::Mat& K, const cv::Mat& D)
    : idx_(0), img_size_(img_size), K_(K), D_(D) {
  init_Rcw_.setIdentity();
  cur_Rcw_.setIdentity();
  reset();
}

DsoImpl::~DsoImpl() {}

void DsoImpl::defaultSetting() {
  setting_desiredImmatureDensity = 1500;
  setting_desiredPointDensity = 2000;
  setting_minFrames = 5;
  setting_maxFrames = 7;
  setting_maxOptIterations = 6;
  setting_minOptIterations = 1;

  setting_logStuff = false;
  setting_photometricCalibration = 0;
  setting_affineOptModeA = 0;  //-1: fix. >=0: optimize (with prior, if > 0).
  setting_affineOptModeB = 0;  //-1: fix. >=0: optimize (with prior, if > 0).

  disableAllDisplay = true;
  setting_debugout_runquiet = true;
}

void DsoImpl::reset() {
  defaultSetting();

  cv::Mat Kf;
  K_.convertTo(Kf, CV_32FC1);
  Eigen::Matrix3f Ke;
  Ke << Kf.at<float>(0, 0), Kf.at<float>(0, 1), Kf.at<float>(0, 2), Kf.at<float>(1, 0), Kf.at<float>(1, 1),
      Kf.at<float>(1, 2), Kf.at<float>(2, 0), Kf.at<float>(2, 1), Kf.at<float>(2, 2);
  setGlobalCalib(img_size_.width, img_size_.height, Ke);

  full_system_.reset(new FullSystem());
  full_system_->setGammaFunction(NULL);
  full_system_->linearizeOperation = 1;
  output_wrapper_.reset(new MyOutputWrapper());
  full_system_->outputWrapper.push_back(output_wrapper_.get());

  setting_fullResetRequested = false;

  init_Rcw_.setIdentity();
  init_rot_set_ = false;
}

bool DsoImpl::process(const cv::Mat& img, Eigen::Isometry3d& pose) {
  cv::Mat gray;
  if (img.channels() == 1)
    gray = img;
  else
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  if (gray.size() != img_size_) {
    cv::Mat timg;
    cv::resize(gray, timg, img_size_);
    gray = timg;
  }
  ImageAndExposure input_img(gray.cols, gray.rows);
  cv::Mat input_cv_mat(input_img.h, input_img.w, CV_32FC1, input_img.image);
  gray.convertTo(input_cv_mat, CV_32F);

  full_system_->addActiveFrame(&input_img, idx_);

  if (full_system_->initFailed || setting_fullResetRequested) reset();

  if (full_system_->isLost) return false;

  if (output_wrapper_->has_pose) {
    if (!init_rot_set_) {
      init_rot_set_ = true;
      init_Rcw_ = cur_Rcw_;
    }
    pose = init_Rcw_ * output_wrapper_->cur_pose;
    return true;
  } else {
    return false;
  }
}

bool DsoImpl::getDepth(cv::Mat& depth, Eigen::Isometry3d& pose) {
  if (output_wrapper_->kf_depth.empty()) return false;
  depth = output_wrapper_->kf_depth;
  pose = init_Rcw_ * output_wrapper_->kf_pose;
  return true;
}

cv::Mat DsoImpl::getDepthDraw() { return output_wrapper_->depth_draw; }

DsoWrapper::DsoWrapper(const cv::Size& img_size, const cv::Mat& K, const cv::Mat& D) {
  impl_ = std::make_shared<DsoImpl>(img_size, K, D);
}

void DsoWrapper::setCurRot(const Eigen::Matrix3d& Rcw) { impl_->setCurRot(Rcw); }

bool DsoWrapper::process(const cv::Mat& img, Eigen::Isometry3d& pose) { return impl_->process(img, pose); }

bool DsoWrapper::getDepth(cv::Mat& depth, Eigen::Isometry3d& pose) { return impl_->getDepth(depth, pose); }

cv::Mat DsoWrapper::getDepthDraw() { return impl_->getDepthDraw(); }
