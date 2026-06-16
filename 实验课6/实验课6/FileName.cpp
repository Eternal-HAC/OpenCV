#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

// 计算放大后某一点对应到原图上的连续坐标
Point2d dstToSrcCoord(int dstX, int dstY, double scale)
{
    double sx = dstX / scale;   // x方向缩放回原图
    double sy = dstY / scale;   // y方向缩放回原图
    return Point2d(sx, sy);
}

// 手写最近邻插值：在放大倍数 scale 下，求放大图 (dstX, dstY) 的灰度值
uchar myNearestNeighbor(const Mat& gray, int dstX, int dstY, double scale)
{
    Point2d src = dstToSrcCoord(dstX, dstY, scale);
    double sx = src.x;
    double sy = src.y;

    int nx = (int)std::round(sx);
    int ny = (int)std::round(sy);

    // 边界检查
    nx = std::max(0, std::min(nx, gray.cols - 1));
    ny = std::max(0, std::min(ny, gray.rows - 1));

    return gray.at<uchar>(ny, nx);
}

// 手写双线性插值：在放大倍数 scale 下，求放大图 (dstX, dstY) 的灰度值
uchar myBilinear(const Mat& gray, int dstX, int dstY, double scale)
{
    Point2d src = dstToSrcCoord(dstX, dstY, scale);
    double sx = src.x;
    double sy = src.y;

    // 左上角整数坐标
    int x0 = (int)floor(sx);
    int y0 = (int)floor(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 边界处理
    x0 = std::max(0, std::min(x0, gray.cols - 1));
    y0 = std::max(0, std::min(y0, gray.rows - 1));
    x1 = std::max(0, std::min(x1, gray.cols - 1));
    y1 = std::max(0, std::min(y1, gray.rows - 1));

    double fx = sx - x0;   // x方向小数部分
    double fy = sy - y0;   // y方向小数部分

    double f00 = gray.at<uchar>(y0, x0); // (x0, y0)
    double f10 = gray.at<uchar>(y0, x1); // (x1, y0)
    double f01 = gray.at<uchar>(y1, x0); // (x0, y1)
    double f11 = gray.at<uchar>(y1, x1); // (x1, y1)

    // 双线性插值公式
    double val =
        (1 - fx) * (1 - fy) * f00 +
        fx * (1 - fy) * f10 +
        (1 - fx) * fy * f01 +
        fx * fy * f11;

    val = std::max(0.0, std::min(255.0, val));
    return (uchar)(val + 0.5); // 四舍五入
}

int main()
{
    // ======================== 1. 读取原图并转灰度 ========================
    string imgPath = "E:/01_code/opcv/实验课6/实验课6/1.jpg";
    Mat img = imread(imgPath);
    if (img.empty())
    {
        cout << "❌ 无法读取图像，请检查路径: " << imgPath << endl;
        return -1;
    }

    cout << "原图尺寸: " << img.cols << " x " << img.rows << endl;

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 原图指定点坐标
    int x_src = 326;
    int y_src = 177;

    if (x_src < 0 || x_src >= gray.cols || y_src < 0 || y_src >= gray.rows)
    {
        cout << "❌ 指定点 (" << x_src << "," << y_src << ") 超出原图范围！" << endl;
        return -1;
    }

    uchar gray_original = gray.at<uchar>(y_src, x_src);
    cout << "原图中点 (" << x_src << "," << y_src << ") 的灰度值 = "
        << (int)gray_original << endl;

    // ======================== 2. 放大 2 倍并计算灰度 ========================
    double scale = 2.0;                     // 放大 2 倍
    // 老师要求：放大 2 倍后取 (2x+1, 2y+1)
    Point dstPt(2 * x_src + 1, 2 * y_src + 1);  // (653, 355)
    cout << "放大 2 倍后指定的新坐标 = (" << dstPt.x << ", " << dstPt.y << ")\n";

    // 手写插值
    uchar val_nearest_manual = myNearestNeighbor(gray, dstPt.x, dstPt.y, scale);
    uchar val_bilinear_manual = myBilinear(gray, dstPt.x, dstPt.y, scale);

    // OpenCV resize 对比
    Mat dst_near, dst_linear;
    resize(gray, dst_near, Size(), scale, scale, INTER_NEAREST);
    resize(gray, dst_linear, Size(), scale, scale, INTER_LINEAR);

    if (dstPt.x >= dst_near.cols || dstPt.y >= dst_near.rows)
    {
        cout << "❌ 放大后点 (" << dstPt.x << "," << dstPt.y
            << ") 超出新图像范围！" << endl;
        cout << "放大后尺寸 (nearest): " << dst_near.cols
            << " x " << dst_near.rows << endl;
        return -1;
    }

    uchar val_nearest_cv = dst_near.at<uchar>(dstPt.y, dstPt.x);
    uchar val_bilinear_cv = dst_linear.at<uchar>(dstPt.y, dstPt.x);

    // ======================== 3. 打印所有数值结果 ========================
    cout << "\n===== 放大倍数: " << scale << "，目标点 (" << dstPt.x << "," << dstPt.y << ") =====" << endl;
    cout << "[手写最近邻插值] 灰度值 = " << (int)val_nearest_manual << endl;
    cout << "[OpenCV INTER_NEAREST] 灰度值 = " << (int)val_nearest_cv << endl;

    cout << "\n[手写双线性插值] 灰度值 = " << (int)val_bilinear_manual << endl;
    cout << "[OpenCV INTER_LINEAR] 灰度值 = " << (int)val_bilinear_cv << endl;

    // ======================== 4. 在灰度图上显示变换前后点 ========================
    // 原图灰度图上画出原点 (326,177)
    Mat gray_show_src;
    cvtColor(gray, gray_show_src, COLOR_GRAY2BGR);  // 变成3通道方便画彩色点
    circle(gray_show_src, Point(x_src, y_src), 5, Scalar(0, 0, 255), -1); // 红色实心圆
    putText(gray_show_src, "(326,177)", Point(x_src + 10, y_src - 10),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

    // 放大后的灰度图（这里用双线性结果）上画出新点 (653,355)
    Mat gray_scaled = dst_linear.clone();          // 双线性放大图
    Mat gray_show_dst;
    cvtColor(gray_scaled, gray_show_dst, COLOR_GRAY2BGR);
    circle(gray_show_dst, dstPt, 5, Scalar(0, 0, 255), -1);
    putText(gray_show_dst, "(653,355)", Point(dstPt.x + 10, dstPt.y - 10),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

    // ======================== 5. 显示窗口 ========================
    imshow("原图", img);
    imshow("原图灰度 + 原点(326,177)", gray_show_src);
    imshow("放大2倍灰度(双线性) + 新点(653,355)", gray_show_dst);
    imshow("放大 2 倍 (Nearest)", dst_near);
    imshow("放大 2 倍 (Bilinear)", dst_linear);

    cout << "\n✅ 实验完成：原图和放大后的灰度图上已经标出变换前后的点，按任意键退出。" << endl;
    waitKey(0);
    return 0;
}
   