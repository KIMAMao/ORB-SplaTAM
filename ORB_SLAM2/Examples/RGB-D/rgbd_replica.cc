/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include<System.h>
#include<dirent.h>
#include<sys/stat.h>

using namespace std;

void LoadImages(const string &path, vector<string> &vstrImageFilenamesRGB, vector<string> &vstrImageFilenamesD);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./rgbd_replica path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    string path = string(argv[3]);
    LoadImages(path, vstrImageFilenamesRGB, vstrImageFilenamesD);

    // Check consistency in the number of images and depth maps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for RGB and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni = 0; ni < nImages; ni++)
    {
        // Read image and depth map from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        imD = cv::imread(vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = ni; // Simulate timestamps for replica dataset

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB, imD, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &path, vector<string> &vstrImageFilenamesRGB, vector<string> &vstrImageFilenamesD)
{
    string resultsPath = path + "/results/";
    DIR *dir;
    struct dirent *entry;

    vector<string> allRGBFiles;
    vector<string> allDepthFiles;

    dir = opendir(resultsPath.c_str());
    if (!dir)
    {
        cerr << "Unable to open results directory: " << resultsPath << endl;
        exit(1);
    }

    while ((entry = readdir(dir)) != NULL)
    {
        string filename = entry->d_name;

        // 加载 RGB 图像（以 "frame" 开头，后缀为 .jpg）
        if (filename.find("frame") == 0 && filename.find(".jpg") != string::npos)
        {
            allRGBFiles.push_back(resultsPath + filename);
        }

        // 加载深度图像（以 "depth" 开头，后缀为 .png）
        if (filename.find("depth") == 0 && filename.find(".png") != string::npos)
        {
            allDepthFiles.push_back(resultsPath + filename);
        }
    }

    closedir(dir);

    // 对文件名进行排序以确保顺序一致
    sort(allRGBFiles.begin(), allRGBFiles.end());
    sort(allDepthFiles.begin(), allDepthFiles.end());

    vstrImageFilenamesRGB = allRGBFiles;
    vstrImageFilenamesD = allDepthFiles;

    // 验证 RGB 和深度图像数量是否一致
    if (vstrImageFilenamesRGB.size() != vstrImageFilenamesD.size())
    {
        cerr << "Error: RGB and Depth image counts do not match!" << endl;
        exit(1);
    }

    cout << "Successfully loaded " << vstrImageFilenamesRGB.size() << " RGB and depth image pairs." << endl;
}

