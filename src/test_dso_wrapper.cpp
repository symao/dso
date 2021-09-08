#include <vs_common/vs_common.h>
#include "dso_wrapper.h"

#define ENABLE_VIZ 0
#define ENABLE_IMSHOW 1

std::vector<cv::Point3f> triangulate(const cv::Mat& depth, const Eigen::Isometry3d& pose, const cv::Mat& K) {
  std::vector<cv::Point3f> pts;
  cv::Mat Kf;
  K.convertTo(Kf, CV_32F);
  float fx = Kf.at<float>(0, 0);
  float fy = Kf.at<float>(1, 1);
  float cx = Kf.at<float>(0, 2);
  float cy = Kf.at<float>(1, 2);
  for (int i = 0; i < depth.rows; i++) {
    const float* ptr = depth.ptr<float>(i);
    for (int j = 0; j < depth.cols; j++) {
      float z = *ptr++;
      if (z > 1e-4 && z < 5) {
        float x = (j - cx) / fx * z;
        float y = (i - cy) / fy * z;
        auto pw = pose * Eigen::Vector3d(x, y, z);
        pts.push_back(cv::Point3f(pw.x(), pw.y(), pw.z()));
      }
    }
  }
  return pts;
}

int main(int argc, char** argv) {
  cv::Size img_size(640, 480);
  cv::Mat K = (cv::Mat_<float>(3, 3) << 494.1246, 0, 313.105, 0, 494.1246, 240.6499, 0, 0, 1);
  DsoWrapper dso(img_size, K);

  std::string flog("/home/symao/data/arkit_data/1606136518.log");
  std::string img_dir = flog.substr(0, flog.size() - 4) + "-img";
  std::ifstream fin_log(flog.c_str());
  for (int idx = 0;; idx++) {
    int tstamp = 0;
    fin_log >> tstamp;
    if (tstamp <= 0) break;
    fin_log.get();  //删除换行符

    std::string lineT, lineK;
    std::getline(fin_log, lineT);
    std::getline(fin_log, lineK);
    auto Tvec = vs::str2vec(lineT, ',');
    auto Kvec = vs::str2vec(lineK, ',');
    Eigen::Isometry3d T;
    T.matrix() << Tvec[0], Tvec[4], Tvec[8], Tvec[12], Tvec[1], Tvec[5], Tvec[9], Tvec[13], Tvec[2], Tvec[6], Tvec[10],
        Tvec[14], Tvec[3], Tvec[7], Tvec[11], Tvec[15];
    Eigen::Matrix3d R;
    R << 0, -1, 0, -1, 0, 0, 0, 0, -1;
    T.linear() = T.linear() * R;
    cv::Vec4f intrin(Kvec[0], Kvec[4], Kvec[6], Kvec[7]);

    if (idx % 5 == 0) continue;

    char fimg[128] = {0};
    snprintf(fimg, 128, "%s/%d.jpg", img_dir.c_str(), tstamp);
    cv::Mat img = cv::imread(fimg);
    if (img.empty()) continue;
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    cv::resize(gray, gray, img_size);

    Eigen::Isometry3d pose;
    dso.setCurRot(T.linear());
    cv::Mat depth;
    Eigen::Isometry3d depth_pose;
    bool ok = dso.process(gray, pose) && dso.getDepth(depth, depth_pose);

#if ENABLE_IMSHOW
    cv::Mat depth_draw = dso.getDepthDraw();
    if (depth.empty()) cv::cvtColor(gray, depth_draw, cv::COLOR_GRAY2BGR);
    cv::Mat frame_draw = vs::toRgb(gray);
    if (ok) {
#if ENABLE_VIZ
      static vs::Viz3D viz;
      static std::vector<cv::Affine3f> traj;
      cv::Matx44f m;
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) m(i, j) = pose.matrix()(i, j);
      cv::Affine3f aff(m);
      traj.push_back(aff);
      viz.updateWidget("cood", cv::viz::WCoordinateSystem());
      viz.updateWidget("traj", cv::viz::WTrajectory(traj, 2, 1.0, cv::viz::Color::blue()));
      std::vector<cv::Affine3f> temp = {aff};
      viz.updateWidget("cur_pose", cv::viz::WTrajectoryFrustums(temp, cv::Vec2d(60 * 0.017453, 50 * 0.017453), 0.5,
                                                                cv::viz::Color::red()));
      if (!depth.empty()) {
        auto pts = triangulate(depth, depth_pose, K);
        if (!pts.empty()) viz.updateWidget("cur_pts", cv::viz::WCloud(pts, cv::viz::Color::green()));
      }
#endif
      // draw coord
      static Eigen::Isometry3d init_model_pose = Eigen::Isometry3d::Identity();
      static bool set_init_model_pose = false;
      if (!set_init_model_pose) {
        init_model_pose.translation() = pose.linear().col(2) + pose.translation();
        set_init_model_pose = true;
      }
      auto pw = init_model_pose.translation();
      cv::Mat rvec, tvec;
      vs::isom2rt(pose.inverse(), rvec, tvec);
      std::vector<cv::Point3f> cube_centers = {cv::Point3f(pw.x(), pw.y(), pw.z()),
                                               cv::Point3f(pw.x() - 0.8, pw.y(), pw.z()),
                                               cv::Point3f(pw.x() - 1.5, pw.y(), pw.z())};
      vs::drawCubes(frame_draw, rvec, tvec, K, cv::Mat(), cube_centers, 0.1);
    }
    cv::imshow("img", vs::hstack({frame_draw, depth_draw}));

    auto key = cv::waitKey(20);
    if (key == 27) break;
  }
#endif
}
