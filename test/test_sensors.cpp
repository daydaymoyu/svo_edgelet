// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <svo/math_lib.h>
#include <svo/camera_model.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>

#include <svo/slamviewer.h>
#include <thread>

static double time_now() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

template <int Rows = Eigen::Dynamic, int Cols = Rows, bool UseRowMajor = false, typename T = double>
using matrix = typename std::conditional<
    Rows != 1 && Cols != 1,
    Eigen::Matrix<T, Rows, Cols, UseRowMajor ? Eigen::RowMajor : Eigen::ColMajor>,
    Eigen::Matrix<T, Rows, Cols>>::type;

template <int Dimension = Eigen::Dynamic, bool RowVector = false, typename T = double>
using vector = typename std::conditional<
    RowVector,
    matrix<1, Dimension, false, T>,
    matrix<Dimension, 1, false, T>>::type;

using quaternion = Eigen::Quaternion<double>;

struct OutputPose {
    quaternion q;
    vector<3> p;
};

class OutputWriter {
  public:
    virtual ~OutputWriter() = default;
    virtual void write_pose(const double& t, const OutputPose& pose) = 0;
    virtual void close_file() = 0;
};

class TumOutputWriter : public OutputWriter {
    std::ofstream file;

  public:
    TumOutputWriter(const std::string& filename) {
        file.open(filename.c_str());
        if (!file.is_open()) {
            std::cerr << "Cannot open file " << std::endl;
        }
        file.precision(15);
    }

    ~TumOutputWriter() = default;

    void write_pose(const double& t, const OutputPose& pose) override {
        file << t << " " << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " "
             << pose.q.x() << " " << pose.q.y() << " " << pose.q.z() << " " << pose.q.w() << "\n";
        file.flush();
    }

    void close_file() override {
        file.close();
    }
};

OutputWriter* output_writer;

namespace svo {

struct RawDataReader {
    FILE* fp;

    RawDataReader(const std::string file_name) {
        std::string temp = (file_name).c_str();
        fp = fopen((file_name).c_str(), "rb");
        if (fp == NULL) {
            fprintf(stderr, "%s fopen error!\n", file_name.c_str());
        }
        std::cout << "test file opened success." << std::endl;
    }

    ~RawDataReader() {
        if (fp) {
            fclose(fp);
            fp = NULL;
        }
    }

    template <typename T>
    void Read(T* data, int size, const int N = 1) {
        fread(data, size, N, fp);
    }
};

class BenchmarkNode {
    svo::AbstractCamera* cam_;
    svo::PinholeCamera* cam_pinhole_;
    svo::FrameHandlerMono* vo_;

    SLAM_VIEWER::Viewer* viewer_;
    std::thread* viewer_thread_;

  public:
    BenchmarkNode();
    ~BenchmarkNode();
    void runFromFolder();
};

BenchmarkNode::BenchmarkNode() {
    //tum rgbd dataset fr2
    // cam_ = new svo::PinholeCamera(640, 480, 520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172);
    //tum rgbd dataset fr1
    //cam_ = new svo::PinholeCamera(640, 480, 517.3,516.5,318.6,	255.3,	0.2624,	-0.9531,-0.0054,0.0026,	1.1633);

    // ICL dataset
    //cam_ = new svo::PinholeCamera(640, 480, 481.20, 480.00, 319.50, 239.50);

    // kitti
    //cam_ = new svo::PinholeCamera(1226, 370, 707.0912, 707.0912, 601.8873, 183.1104);
    // tum mono dataset
    //cam_ = new svo::PinholeCamera(1226, 370, 707.0912, 707.0912, 601.8873, 183.1104);
    // xiaomi2s
    // cam_ = new svo::PinholeCamera(640, 480, 490, 490, 318.5, 237.5);

    // xiaomi8
    cam_ = new svo::PinholeCamera(640, 480, 493.017, 491.55953, 317.97856, 242.392);

    vo_ = new svo::FrameHandlerMono(cam_);
    vo_->start();

    // viewer_ = new SLAM_VIEWER::Viewer(vo_);
    // viewer_thread_ = new std::thread(&SLAM_VIEWER::Viewer::run, viewer_);
    // viewer_thread_->detach();
    // viewer_thread_->join();
}

BenchmarkNode::~BenchmarkNode() {
    delete vo_;
    delete cam_;
    delete cam_pinhole_;

    delete viewer_;
    delete viewer_thread_;
}

void BenchmarkNode::runFromFolder() {
    unsigned char type;
    double img_time, gravity_time;
    int width, height;
    double gravity[3];
    size_t id = 0;
    size_t img_id = 0;

    output_writer = new TumOutputWriter("trajectory.tum");

    // std::string filepath = std::string("/Users/shuxiangqian/DataSet/evaluate/lvo/slow.sensors"); // xiaomi2s
    std::string filepath = std::string("/Users/shuxiangqian/DataSet/evaluate/lvo/09_0.sensors");    // xiaomi8
    RawDataReader reader(filepath);
    while (true) {
        reader.Read<unsigned char>(&type, sizeof(unsigned char));
        if (type == 0) {
            reader.Read<double>(&img_time, sizeof(double));
            reader.Read<int>(&width, sizeof(int));
            reader.Read<int>(&height, sizeof(int));
            unsigned char* data = new unsigned char[width * height];
            reader.Read<unsigned char>(data, sizeof(unsigned char), width * height);
            cv::Mat image(height, width, CV_8UC1, data);

            vo_->addImage(image, 0.01 * img_id);

            if (vo_->lastFrame() != NULL) {
                std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                          << "#Features: " << vo_->lastNumObservations() << " \n";
            }

            OutputPose pose;
            pose.q = vo_->lastFrame()->T_f_w_.inverse().unit_quaternion();
            pose.p = vo_->lastFrame()->T_f_w_.inverse().translation();
            output_writer->write_pose(time_now(), pose);

            img_id++;

        } else if (type == 18) {
            reader.Read<double>(&gravity_time, sizeof(double));
            reader.Read<double>(gravity, sizeof(double), 3);
        } else {
            break;
        }
    }
}

} // namespace svo

int main(int argc, char** argv) {
    svo::BenchmarkNode benchmark;
    benchmark.runFromFolder();

    return 0;
}
