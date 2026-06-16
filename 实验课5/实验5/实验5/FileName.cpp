#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>


using namespace cv;
using namespace std;

struct Params {
    string tag;          // 该组参数的名字
    int blurK = 5;       // 高斯核（奇数）
    double blurSigma = 1.5;

    // Canny
    int c1 = 50;
    int c2 = 150;

    // Sobel 后二值化阈值：0 表示用 Otsu
    int sobelThresh = 0;

    // 闭运算核大小（奇数更推荐）
    int closeK = 7;

    // 回退方案：腐蚀核大小（奇数）
    int erodeK = 21;
};

// 保存工具
static void saveImg(const string& path, const Mat& img) {
    imwrite(path, img);
}

// 画点/写字
static void putLabel(Mat& img, const string& text, Point org, Scalar color) {
    putText(img, text, org, FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
}

// 从轮廓层级中找“最大外轮廓”的“最大子轮廓”（更接近真实内轮廓）
static bool findOuterAndInnerByTree(const vector<vector<Point>>& contours,
    const vector<Vec4i>& hierarchy,
    int& outerIdx, int& innerIdx)
{
    outerIdx = -1;
    innerIdx = -1;

    double maxOuterArea = 0.0;

    // 找 parent==-1 的最大外轮廓
    for (int i = 0; i < (int)contours.size(); ++i) {
        if (hierarchy[i][3] != -1) continue; // 不是顶层外轮廓
        double area = contourArea(contours[i]);
        if (area > maxOuterArea) {
            maxOuterArea = area;
            outerIdx = i;
        }
    }
    if (outerIdx == -1) return false;

    // 找 outerIdx 的最大 child（内轮廓）
    int child = hierarchy[outerIdx][2]; // first child
    if (child == -1) return false;

    double maxChildArea = 0.0;
    int bestChild = -1;

    while (child != -1) {
        double a = contourArea(contours[child]);
        if (a > maxChildArea) {
            maxChildArea = a;
            bestChild = child;
        }
        child = hierarchy[child][0]; // next sibling
    }
    if (bestChild == -1) return false;

    innerIdx = bestChild;
    return true;
}

// 回退：把外轮廓填充成白色后腐蚀，得到“内圈”
static bool findInnerByErodeFallback(const Size& imgSize,
    const vector<vector<Point>>& contours,
    int outerIdx, int erodeK,
    vector<vector<Point>>& innerContours,
    int& innerBestIdx)
{
    Mat mask = Mat::zeros(imgSize, CV_8UC1);
    drawContours(mask, contours, outerIdx, Scalar(255), FILLED);

    Mat erodedMask;
    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(erodeK, erodeK));
    erode(mask, erodedMask, k);

    vector<Vec4i> hi;
    findContours(erodedMask, innerContours, hi, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (innerContours.empty()) return false;

    innerBestIdx = 0;
    double maxA = contourArea(innerContours[0]);
    for (int i = 1; i < (int)innerContours.size(); ++i) {
        double a = contourArea(innerContours[i]);
        if (a > maxA) {
            maxA = a;
            innerBestIdx = i;
        }
    }
    return true;
}

// 生成边缘：Canny
static Mat edgesByCanny(const Mat& grayBlur, const Params& p) {
    Mat edges;
    Canny(grayBlur, edges, p.c1, p.c2);
    return edges;
}

// 生成边缘：Sobel（梯度幅值→阈值）
static Mat edgesBySobel(const Mat& grayBlur, const Params& p) {
    Mat gx, gy;
    Sobel(grayBlur, gx, CV_32F, 1, 0, 3);
    Sobel(grayBlur, gy, CV_32F, 0, 1, 3);

    Mat mag;
    magnitude(gx, gy, mag);

    Mat mag8u;
    mag.convertTo(mag8u, CV_8U);

    Mat bin;
    if (p.sobelThresh == 0) {
        threshold(mag8u, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }
    else {
        threshold(mag8u, bin, p.sobelThresh, 255, THRESH_BINARY);
    }
    return bin;
}

// 一次实验：指定方法（CANNY/SOBEL）+ 参数，输出中间结果与最终叠加图
static void runOne(const Mat& srcBgr, const string& outDir,
    const string& methodName, const Params& p)
{
    // 1) 灰度
    Mat gray;
    cvtColor(srcBgr, gray, COLOR_BGR2GRAY);

    // 2) 高斯滤波
    int bk = (p.blurK % 2 == 1) ? p.blurK : (p.blurK + 1);
    Mat blurImg;
    GaussianBlur(gray, blurImg, Size(bk, bk), p.blurSigma);

    // 3) 边缘
    Mat edges = (methodName == "CANNY") ? edgesByCanny(blurImg, p)
        : edgesBySobel(blurImg, p);

    // 4) 闭运算（连接断裂边缘）
    int ck = (p.closeK % 2 == 1) ? p.closeK : (p.closeK + 1);
    Mat closed = edges.clone();
    Mat closeKernel = getStructuringElement(MORPH_ELLIPSE, Size(ck, ck));
    morphologyEx(closed, closed, MORPH_CLOSE, closeKernel);

    // 5) 轮廓（用 TREE 获取内外层级）
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Mat result = srcBgr.clone();
    int outerIdx = -1, innerIdx = -1;

    bool okTree = false;
    if (!contours.empty()) {
        okTree = findOuterAndInnerByTree(contours, hierarchy, outerIdx, innerIdx);
    }

    // 画轮廓：外轮廓蓝色、内轮廓绿色
    if (okTree) {
        drawContours(result, contours, outerIdx, Scalar(255, 0, 0), 3); // 外：蓝
        drawContours(result, contours, innerIdx, Scalar(0, 255, 0), 3); // 内：绿
        putLabel(result, methodName + "-" + p.tag, Point(20, 40), Scalar(0, 255, 255));
    }
    else if (!contours.empty()) {
        // 回退：用最大外轮廓 + 腐蚀得到内轮廓
        // 先找最大外轮廓（不看 parent）
        double maxA = 0; outerIdx = -1;
        for (int i = 0; i < (int)contours.size(); ++i) {
            double a = contourArea(contours[i]);
            if (a > maxA) { maxA = a; outerIdx = i; }
        }

        vector<vector<Point>> innerContours;
        int innerBest = -1;
        bool okErode = findInnerByErodeFallback(srcBgr.size(), contours, outerIdx,
            p.erodeK, innerContours, innerBest);

        drawContours(result, contours, outerIdx, Scalar(255, 0, 0), 3);
        if (okErode) drawContours(result, innerContours, innerBest, Scalar(0, 255, 0), 3);
        putLabel(result, methodName + "-" + p.tag + "(fallback)", Point(20, 40), Scalar(0, 255, 255));
    }
    else {
        putLabel(result, methodName + "-" + p.tag + "(no contour)", Point(20, 40), Scalar(0, 0, 255));
    }

    // 6) 保存中间结果（便于写实验报告“过程图”）
    string prefix = outDir + "/" + methodName + "_" + p.tag;
    saveImg(prefix + "_1_gray.png", gray);
    saveImg(prefix + "_2_blur.png", blurImg);
    saveImg(prefix + "_3_edges.png", edges);
    saveImg(prefix + "_4_closed.png", closed);
    saveImg(prefix + "_5_result.png", result);

    cout << "[OK] 输出: " << prefix << "_*.png" << endl;
}

int main()
{
    // ========= 你的路径 =========
    string baseFolder = "E:/01_code/opcv/实验课5/实验5/实验5/";
    string inputName = "3.jpg";
    string inputPath = baseFolder + inputName;

    // 输出目录
    string outDir = baseFolder + "out";
    std::filesystem::create_directories(outDir);

    Mat img = imread(inputPath);
    if (img.empty()) {
        cout << "无法读取图像: " << inputPath << endl;
        return -1;
    }
    cout << "图像尺寸: " << img.cols << " x " << img.rows << endl;
    cout << "输出目录: " << outDir << endl;

    // ========= 3组 Canny 参数（满足“参数对比”要求）=========
    vector<Params> cannySets = {
        {"P1_low",  5, 1.5, 30,  90, 0, 5, 15},
        {"P2_mid",  5, 1.5, 50, 155, 0, 7, 21},   // 类似你原来那套
        {"P3_high", 5, 1.5, 80, 240, 0, 9, 31}
    };

    // ========= Sobel 一套（满足“算法对比”要求）=========
    Params sobelP;
    sobelP.tag = "Sobel_Otsu";
    sobelP.blurK = 5;
    sobelP.blurSigma = 1.5;
    sobelP.sobelThresh = 0; // Otsu
    sobelP.closeK = 7;
    sobelP.erodeK = 21;

    // 运行：Canny 3组
    for (auto& p : cannySets) {
        runOne(img, outDir, "CANNY", p);
    }

    // 运行：Sobel 1组
    runOne(img, outDir, "SOBEL", sobelP);

    cout << "\n实验完成：已输出 Canny(3组) + Sobel(1组) 的过程图与结果图。" << endl;
    cout << "你写报告时，直接对比 out 文件夹里的 *_result.png 即可。" << endl;

    return 0;
}
