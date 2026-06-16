#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// === 用户参数（根据实际情况修改） ===
const cv::Size boardSize(8, 11);     // 棋盘内角点个数 (列, 行)
const float squareSize = 50.0f;      // 每个格子的边长 (mm)
const string imageDir = "E:/code/opcv/实验课2/实验课2";  // 图片存放路径
const int imageCount = 7;            // 图片数量（根据实际图片总数修改）
const string outputFile = "E:/code/opcv/实验课2/calib_result.yml"; // 输出文件路径

int main() {
    vector<vector<Point3f>> objpoints;  // 3D 世界点
    vector<vector<Point2f>> imgpoints;  // 2D 图像点
    vector<string> validImages;

    // 生成世界坐标系下的棋盘角点坐标
    vector<Point3f> objp;
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++)
            objp.emplace_back(j * squareSize, i * squareSize, 0);

    // 读取图片（适配纯数字命名：1.bmp, 2.bmp, ..., n.bmp）
    vector<string> imageFiles;
    for (int i = 1; i <= imageCount; ++i) {
        // 直接使用数字作为文件名（例如：1.bmp, 2.bmp）
        string filename = imageDir + "/" + to_string(i) + ".bmp";
        imageFiles.push_back(filename);
    }

    cout << "共准备图片 " << imageFiles.size() << " 张。" << endl;

    Size imageSize;
    for (const auto& file : imageFiles) {
        Mat img = imread(file);
        if (img.empty()) {
            cout << "无法读取: " << file << endl;
            continue;
        }

        if (imageSize.empty())
            imageSize = img.size();

        vector<Point2f> corners;
        bool found = findChessboardCorners(img, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001));
            drawChessboardCorners(img, boardSize, corners, found);
            imshow("Corners", img);
            waitKey(300);

            imgpoints.push_back(corners);
            objpoints.push_back(objp);
            validImages.push_back(file);
            cout << "成功检测角点: " << file << endl;
        }
        else {
            cout << "未检测到角点: " << file << endl;
        }
    }

    destroyAllWindows();

    if (imgpoints.size() < 3) {
        cerr << "有效图片不足 3 张，无法标定！" << endl;
        return -1;
    }

    cout << "\n开始标定..." << endl;
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;

    double rms = calibrateCamera(objpoints, imgpoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs);

    cout << "\n标定完成" << endl;
    cout << "RMS 重投影误差: " << rms << " 像素\n" << endl;

    cout << "相机内参矩阵 K = \n" << cameraMatrix << endl;
    cout << "畸变系数 [k1, k2, p1, p2, k3] = \n" << distCoeffs << endl;

    // 计算每张图的重投影误差
    double totalErr = 0;
    vector<double> perViewError(objpoints.size());
    for (size_t i = 0; i < objpoints.size(); i++) {
        vector<Point2f> reprojected;
        projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, reprojected);
        double err = norm(imgpoints[i], reprojected, NORM_L2) / reprojected.size();
        perViewError[i] = err;
        totalErr += err;
    }
    double avgError = totalErr / perViewError.size();

    cout << "\n每张图像重投影误差：" << endl;
    for (size_t i = 0; i < perViewError.size(); ++i)
        cout << validImages[i] << " : " << perViewError[i] << " 像素" << endl;

    cout << "\n平均重投影误差: " << avgError << " 像素" << endl;

    // 保存结果到 YAML 文件（带中文注释）
    FileStorage fs(outputFile, FileStorage::WRITE);
    
    // 写入图像尺寸
    fs.writeComment("图像宽度（像素）", false);
    fs << "image_width" << imageSize.width;
    fs.writeComment("图像高度（像素）", false);
    fs << "image_height" << imageSize.height;
    
    // 写入相机内参矩阵
    fs.writeComment("相机内参矩阵 K (3x3): fx, fy 为焦距, cx, cy 为主点坐标", false);
    fs << "camera_matrix" << cameraMatrix;
    
    // 写入畸变系数
    fs.writeComment("畸变系数: [k1, k2, p1, p2, k3] (径向畸变k1-k3, 切向畸变p1-p2)", false);
    fs << "dist_coeffs" << distCoeffs;
    
    // 写入重投影误差
    fs.writeComment("平均重投影误差（像素）", false);
    fs << "avg_reproj_error" << avgError;
    
    // 写入标定板信息
    fs.writeComment("棋盘格内角点数量 (宽x高)", false);
    fs << "board_size" << boardSize;
    fs.writeComment("棋盘格每个格子的边长（毫米）", false);
    fs << "square_size" << squareSize;
    
    fs.release();

    cout << "\n结果已保存到: " << outputFile << endl;

    // 去畸变演示
    for (size_t i = 0; i < validImages.size(); ++i) {
        Mat img = imread(validImages[i]);
        if (img.empty()) continue;
        Mat undistorted;
        undistort(img, undistorted, cameraMatrix, distCoeffs);
        imshow("Original", img);
        imshow("Undistorted", undistorted);
        waitKey(0);
    }

    return 0;
}
