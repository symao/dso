/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vs_common/vs_common.h>

#include <boost/thread.hpp>
#include <thread>

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/DatasetReader.h"
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"

std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start = 0;
int end = 100000;
bool prefetch = false;
float playbackSpeed = 0;  // 0 for linearize (play as fast as possible, while sequentializing
                          // tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
bool useSampleOutput = false;

int mode = 0;

bool firstRosSpin = false;

using namespace dso;

void my_exit_handler(int s) {
    printf("Caught signal %d\n", s);
    exit(1);
}

void exitThread() {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    firstRosSpin = true;
    while (true) pause();
}

void settingsDefault(int preset) {
    printf("\n=============== PRESET Settings: ===============\n");
    if (preset == 0 || preset == 1) {
        printf(
            "DEFAULT settings:\n"
            "- %s real-time enforcing\n"
            "- 2000 active points\n"
            "- 5-7 active frames\n"
            "- 1-6 LM iteration each KF\n"
            "- original image resolution\n",
            preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0 : 1);
        preload = preset == 1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
    }

    if (preset == 2 || preset == 3) {
        printf(
            "FAST settings:\n"
            "- %s real-time enforcing\n"
            "- 800 active points\n"
            "- 4-6 active frames\n"
            "- 1-4 LM iteration each KF\n"
            "- 424 x 320 image resolution\n",
            preset == 0 ? "no " : "5x");

        playbackSpeed = (preset == 2 ? 0 : 5);
        preload = preset == 3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations = 4;
        setting_minOptIterations = 1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}

void parseArgument(char* arg) {
    int option;
    float foption;
    char buf[1000];

    if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
        if (option == 1) {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "quiet=%d", &option)) {
        if (option == 1) {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "preset=%d", &option)) {
        settingsDefault(option);
        return;
    }

    if (1 == sscanf(arg, "rec=%d", &option)) {
        if (option == 0) {
            disableReconfigure = true;
            printf("DISABLE RECONFIGURE!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "noros=%d", &option)) {
        if (option == 1) {
            disableROS = true;
            disableReconfigure = true;
            printf("DISABLE ROS (AND RECONFIGURE)!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "nolog=%d", &option)) {
        if (option == 1) {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "reverse=%d", &option)) {
        if (option == 1) {
            reverse = true;
            printf("REVERSE!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nogui=%d", &option)) {
        if (option == 1) {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nomt=%d", &option)) {
        if (option == 1) {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "prefetch=%d", &option)) {
        if (option == 1) {
            prefetch = true;
            printf("PREFETCH!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "start=%d", &option)) {
        start = option;
        printf("START AT %d!\n", start);
        return;
    }
    if (1 == sscanf(arg, "end=%d", &option)) {
        end = option;
        printf("END AT %d!\n", start);
        return;
    }

    if (1 == sscanf(arg, "files=%s", buf)) {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
        return;
    }

    if (1 == sscanf(arg, "calib=%s", buf)) {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }

    if (1 == sscanf(arg, "vignette=%s", buf)) {
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }

    if (1 == sscanf(arg, "gamma=%s", buf)) {
        gammaCalib = buf;
        printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
        return;
    }

    if (1 == sscanf(arg, "rescale=%f", &foption)) {
        rescale = foption;
        printf("RESCALE %f!\n", rescale);
        return;
    }

    if (1 == sscanf(arg, "speed=%f", &foption)) {
        playbackSpeed = foption;
        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
        return;
    }

    if (1 == sscanf(arg, "save=%d", &option)) {
        if (option == 1) {
            debugSaveImages = true;
            if (42 == system("rm -rf images_out"))
                printf(
                    "system call returned 42 - what are the odds?. This is only here to shut up "
                    "the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf(
                    "system call returned 42 - what are the odds?. This is only here to shut up "
                    "the compiler.\n");
            if (42 == system("rm -rf images_out"))
                printf(
                    "system call returned 42 - what are the odds?. This is only here to shut up "
                    "the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf(
                    "system call returned 42 - what are the odds?. This is only here to shut up "
                    "the compiler.\n");
            printf("SAVE IMAGES!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "mode=%d", &option)) {
        mode = option;
        if (option == 0) {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if (option == 1) {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0;  //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0;  //-1: fix. >=0: optimize (with prior, if > 0).
        }
        if (option == 2) {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1;  //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1;  //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd = 3;
        }
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}

int main(int argc, char** argv) {
    // setlocale(LC_ALL, "");
    for (int i = 1; i < argc; i++) parseArgument(argv[i]);

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);

    ImageFolderReader* reader = new ImageFolderReader(source, calib, gammaCalib, vignette);
    reader->setGlobalCalibration();

    if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0) {
        printf(
            "ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or "
            "mode=2 ");
        exit(1);
    }

    int lstart = start;
    int lend = end;
    int linc = 1;
    if (reverse) {
        printf("REVERSE!!!!");
        lstart = end - 1;
        if (lstart >= reader->getNumImages())
            lstart = reader->getNumImages() - 1;
        lend = start;
        linc = -1;
    }

    FullSystem* fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    IOWrap::PangolinDSOViewer* viewer = 0;
    if (!disableAllDisplay) {
        viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if (useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::string flog("/home/symao/data/arkit_data/1606136518.log");
        std::string img_dir = flog.substr(0, flog.size() - 4) + "-img";
        std::ifstream fin_log(flog.c_str());
        for (int idx = 0;; idx++) {
            int tstamp = 0;
            fin_log >> tstamp;
            if (tstamp <= 0)
                break;
            fin_log.get();  //删除换行符

            std::string lineT, lineK;
            std::getline(fin_log, lineT);
            std::getline(fin_log, lineK);
            auto Tvec = vs::str2vec(lineT, ',');
            auto Kvec = vs::str2vec(lineK, ',');
            Eigen::Isometry3d T;
            T.matrix() << Tvec[0], Tvec[4], Tvec[8], Tvec[12], Tvec[1], Tvec[5], Tvec[9], Tvec[13],
                Tvec[2], Tvec[6], Tvec[10], Tvec[14], Tvec[3], Tvec[7], Tvec[11], Tvec[15];
            Eigen::Matrix3d R;
            R << 0, -1, 0, -1, 0, 0, 0, 0, -1;
            T.linear() = T.linear() * R;
            cv::Vec4f intrin(Kvec[0], Kvec[4], Kvec[6], Kvec[7]);

            if(idx % 5 == 0) continue;

            char fimg[128] = {0};
            snprintf(fimg, 128, "%s/%d.jpg", img_dir.c_str(), tstamp);
            cv::Mat img = cv::imread(fimg);
            if (img.empty())
                continue;
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
            cv::resize(gray, gray, cv::Size(640, 480));

            SE3 pose(T.linear(), T.translation());
            ImageAndExposure input_img(gray.cols, gray.rows);
            cv::Mat input_cv_mat(input_img.h, input_img.w, CV_32FC1, input_img.image);
            gray.convertTo(input_cv_mat, CV_32F);

            fullSystem->addActiveFrame(&input_img, idx, pose);

            if (fullSystem->initFailed || setting_fullResetRequested) {
                if (idx < 250 || setting_fullResetRequested) {
                    printf("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for (IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                    fullSystem = new FullSystem();
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);

                    fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested = false;
                }
            }

            if (fullSystem->isLost) {
                printf("LOST!!\n");
                break;
            }
        }
        fullSystem->blockUntilMappingIsFinished();
    });

    if (viewer != 0)
        viewer->run();

    runthread.join();

    for (IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper) {
        ow->join();
        delete ow;
    }

    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    printf("DELETE READER!\n");
    delete reader;

    printf("EXIT NOW!\n");
    return 0;
}
