#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // 读取原图
    Mat img = imread("xixi.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cout << "无法读取图片！请检查路径和文件名。" << endl;
        return -1;
    }

    //-------------------------
    // 1. 原图
    //-------------------------
    namedWindow("嘻嘻-原图", WINDOW_AUTOSIZE);
    imshow("嘻嘻-原图", img);

    //-------------------------
    // 2. 图像处理功能（灰度 + 模糊 + 边缘）
    //-------------------------
    Mat gray, blurImg, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);              // 灰度
    GaussianBlur(img, blurImg, Size(15, 15), 0);      // 模糊
    Canny(img, edges, 100, 200);                      // 边缘检测

    // 灰度 & 边缘 转换成 3 通道，保证拼接时一致
    Mat gray3, edges3;
    cvtColor(gray, gray3, COLOR_GRAY2BGR);
    cvtColor(edges, edges3, COLOR_GRAY2BGR);

    // 拼接展示
    Mat procTop, procAll;
    hconcat(img, gray3, procTop);          // 原图 + 灰度
    hconcat(blurImg, edges3, procAll);     // 模糊 + 边缘
    vconcat(procTop, procAll, procAll);    // 上下拼接

    imshow("嘻嘻-图像处理功能", procAll);

    //-------------------------
    // 3. 图像操作功能（缩放 + 翻转）
    //-------------------------
    Mat small, big, flipImg;
    resize(img, small, Size(), 0.5, 0.5); // 缩小
    resize(img, big, Size(), 1.5, 1.5);   // 放大
    flip(img, flipImg, 1);                // 水平翻转

    // 为了拼接，先调整尺寸一致（缩放到和原图同大小）
    resize(small, small, img.size());
    resize(big, big, img.size());
    resize(flipImg, flipImg, img.size());

    Mat opTop, opAll;
    hconcat(small, big, opTop);           // 缩小 + 放大
    hconcat(img, flipImg, opAll);         // 原图 + 镜像
    vconcat(opTop, opAll, opAll);

    imshow("嘻嘻-图像操作功能", opAll);

    //-------------------------
    // 4. 绘图功能（画线 + 画圆 + 加文字）
    //-------------------------
    Mat drawImg = img.clone();
    line(drawImg, Point(50, 50), Point(200, 200), Scalar(0, 0, 255), 3);
    circle(drawImg, Point(250, 150), 80, Scalar(0, 255, 0), 3);
    putText(drawImg, "Hello xixi!", Point(50, 300),
        FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0), 3);

    imshow("嘻嘻-绘图功能", drawImg);

    //-------------------------
    // 等待键盘输入
    //-------------------------
    waitKey(0);
    return 0;
}
