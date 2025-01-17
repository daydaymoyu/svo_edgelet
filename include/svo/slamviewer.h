#ifndef SLAMVIEWER_H
#define SLAMVIEWER_H

#include "opencv2/core/core.hpp"
#include <sophus/se3.h>

#include "pangolin/pangolin.h"
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>

#include <mutex>

namespace SLAM_VIEWER {

class Viewer {
  public:
    Viewer(svo::FrameHandlerMono* vo);
    void init();
    void run();
    void draw();
    bool CheckFinish();
    void DrawKeyFrames(const bool bDrawKF);
    void DrawMapRegionPoints();
    void DrawMapSeeds();

    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M);
    void DrawCurrentCamera(pangolin::OpenGlMatrix& Twc);

  private:
    svo::FrameHandlerMono* _vo;

    std::mutex mMutexCurrentPose;
    std::vector<Sophus::SE3> _pos;
    Sophus::SE3 _CurrentPoseTwc;
    int _drawedframeID = 0;

    void SetFinish();
    bool mbFinished;
    std::mutex mMutexFinish;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
};

} // namespace SLAM_VIEWER
#endif // SLAMVIEWER_H
