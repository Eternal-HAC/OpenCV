#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/**
 * @brief Detects the inner contour of an archway in an image.
 * @param inputPath The path to the input image.
 * @param outputPath The path where the result image will be saved.
 */
void detectArchContour(const string& inputPath, const string& outputPath) {
    Mat img = imread(inputPath);  // 修改：使用传入的路径参数
    if (img.empty()) {
        cout << "[ERROR] Could not load image: " << inputPath << endl;
        return;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Gaussian blur to reduce noise
    GaussianBlur(gray, gray, Size(5, 5), 1.55);

    // Canny edge detection
    Mat edges;
    Canny(gray, edges, 50, 155);

    // Morphological closing to connect broken edges
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel);

    // Find all external contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    int bestIdx = -1;

    for (int i = 0; i < (int)contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area < 20000) continue; // Filter out small areas

        // Calculate circularity (to determine if it's an arch)
        double perimeter = arcLength(contours[i], true);
        if (perimeter == 0) continue;
        double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        if (circularity < 0.05) continue; // Exclude non-circular/non-arch structures

        if (area > maxArea) {
            maxArea = area;
            bestIdx = i;
        }
    }

    Mat result = img.clone();

    if (bestIdx != -1) {
        // Create a mask and fill the original contour
        Mat mask = Mat::zeros(img.size(), CV_8UC1);
        drawContours(mask, contours, bestIdx, Scalar(255), FILLED);

        // Erode the mask to shrink it inwards (controls the inset)
        Mat erodedMask;
        Mat erodeKernel = getStructuringElement(MORPH_ELLIPSE, Size(21, 21)); // Adjustable: inset size
        erode(mask, erodedMask, erodeKernel);

        // Find new contours in the eroded mask (the inner contour)
        vector<vector<Point>> innerContours;
        findContours(erodedMask, innerContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!innerContours.empty()) {
            // Select the largest inner contour (usually there's only one)
            int innerBest = 0;
            double maxInnerArea = contourArea(innerContours[0]);
            for (size_t j = 1; j < innerContours.size(); ++j) {
                double a = contourArea(innerContours[j]);
                if (a > maxInnerArea) {
                    maxInnerArea = a;
                    innerBest = j;
                }
            }
            drawContours(result, innerContours, innerBest, Scalar(0, 255, 0), 3);
            cout << "[SUCCESS] Arch contour found and shrunk. Original area: " << maxArea
                << ", Shrunk area: " << maxInnerArea << endl;
        }
        else {
            // Fallback to the original contour if shrinking fails
            drawContours(result, contours, bestIdx, Scalar(0, 255, 0), 3);
            cout << "[WARNING] No valid contour after shrinking. Drawing the original contour. Area: " << maxArea << endl;
        }
    }
    else {
        cout << "[ERROR] No suitable arch contour found for " << inputPath << "!" << endl;
    }

    // 保存结果图像
    imwrite(outputPath, result);
    
    // ========== 新增：直接展示结果 ==========
    // 缩放图像以适应屏幕显示
    float scale = 0.6;
    Mat showOriginal, showEdges, showResult;
    
    resize(img, showOriginal, Size(), scale, scale);
    resize(edges, showEdges, Size(), scale, scale);
    resize(result, showResult, Size(), scale, scale);
    
    // 显示处理过程和结果
    imshow("1. Original Image - " + inputPath, showOriginal);
    imshow("2. Edge Detection - " + inputPath, showEdges);
    imshow("3. Final Result - " + inputPath, showResult);
    
    cout << "[INFO] Press any key to continue to next image, or ESC to exit..." << endl;
    
    // 等待用户按键
    int key = waitKey(0);
    if (key == 27) { // ESC键
        cout << "[INFO] User pressed ESC, exiting..." << endl;
        destroyAllWindows();
        exit(0);
    }
    
    // 关闭当前图像的窗口
    destroyWindow("1. Original Image - " + inputPath);
    destroyWindow("2. Edge Detection - " + inputPath);
    destroyWindow("3. Final Result - " + inputPath);
}

int main() {
    // 修改：使用你项目中实际存在的图像文件
    vector<string> filenames = {
        "1.jpg",
        "2.jpg", 
        "3.jpg"
    };

    // 修改：使用指定的保存路径
    string projectRootPath = "";

    // 输出文件夹 - 使用你指定的路径
    string outputFolder = "E:/01_code/opcv/实验课5/实验5/实验5/";

    // 输出路径已指定为绝对路径，无需创建文件夹

    cout << "=== 拱形门内轮廓检测与展示程序 ===" << endl;
    cout << "处理图像文件：";
    for (const string& name : filenames) {
        cout << name << " ";
    }
    cout << endl << endl;

    // Loop through and process each image
    for (const string& filename : filenames) {
        // Construct the correct input path relative to the executable's location
        string inputPath = projectRootPath + filename;

        // Construct the correct output path
        string outputPath = outputFolder + filename;

        cout << "Processing: " << inputPath << endl;
        detectArchContour(inputPath, outputPath);
    }

    cout << "\n=== 处理完成 ===" << endl;
    cout << "All images processed! Results saved to '" << outputFolder << "' folder." << endl;
    cout << "Press any key to exit..." << endl;
    waitKey(0);
    destroyAllWindows();
    
    return 0;
}
