#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // ================= 0. 设置基础信息 =================
    string save_dir = "./"; // 图片保存路径
    string img_path = "E:/01_code/opcv/实验课4/实验课4/实验课3二值法/1.jpg";

    // ================= 1. 读取图像 =================
    Mat img_src = imread(img_path);
    if (img_src.empty()) {
        cout << "❌ 无法读取图像！请检查路径: " << img_path << endl;
        return -1;
    }
    cout << "图像读取成功，尺寸: " << img_src.cols << "x" << img_src.rows << endl;

    // ================= 2. 预处理 (模糊 + 降亮) =================
    Mat img_blurred;
    GaussianBlur(img_src, img_blurred, Size(9, 9), 0);

    Mat img_darkened;
    img_blurred.convertTo(img_darkened, -1, 1, -20); // 降低亮度压暗背景

    // ================= 3. HSV 颜色分割 =================
    Mat img_hsv;
    cvtColor(img_darkened, img_hsv, COLOR_BGR2HSV);

    Mat img_mask;
    // 提取偏白/亮色区域
    inRange(img_hsv, Scalar(0, 0, 180), Scalar(180, 100, 255), img_mask);

    // V通道二次约束
    vector<Mat> hsv_channels;
    split(img_hsv, hsv_channels);
    Mat val_mask;
    threshold(hsv_channels[2], val_mask, 120, 255, THRESH_BINARY);
    bitwise_and(img_mask, val_mask, img_mask);

    // ================= 4. 底部位置过滤 (去反光) =================
    Mat img_filtered_binary = img_mask.clone();
    vector<vector<Point>> contours;
    findContours(img_filtered_binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int height_threshold = int(img_src.rows * 0.75);
    for (auto& contour : contours) {
        Rect bound = boundingRect(contour);
        // 如果轮廓底部超过了阈值线，视为反光，填充黑色
        if (bound.y + bound.height >= height_threshold) {
            drawContours(img_filtered_binary, vector<vector<Point>>{contour}, -1, Scalar(0), FILLED);
        }
    }

    // ================= 5. 形态学操作 =================
    int kernel_size = 6;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));

    Mat img_eroded, img_dilated, img_opened, img_closed;
    erode(img_filtered_binary, img_eroded, kernel);
    dilate(img_filtered_binary, img_dilated, kernel);
    morphologyEx(img_filtered_binary, img_opened, MORPH_OPEN, kernel);
    morphologyEx(img_filtered_binary, img_closed, MORPH_CLOSE, kernel);

    // ================= 6. 外圈检测 (最终结果) =================
    Mat img_result = img_src.clone();
    vector<vector<Point>> final_contours;

    // 只检测外轮廓 (RETR_EXTERNAL)
    findContours(img_closed, final_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < final_contours.size(); i++) {
        double area = contourArea(final_contours[i]);
        if (area > 500) { // 简单面积过滤
            // 绘制绿色外圈，线宽2
            drawContours(img_result, final_contours, i, Scalar(0, 255, 0), 2);
        }
    }

    // ================= 7. 单独展示与保存 (8个独立窗口) =================
    cout << "正在弹出独立窗口并保存图片..." << endl;

    // 定义一个辅助宏，方便写重复代码
    // 参数: 窗口名, 图像变量, 文件名
    auto show_and_save = [&](string name, Mat& m, string filename) {
        namedWindow(name, WINDOW_NORMAL); // 允许调整窗口大小
        imshow(name, m);
        imwrite(save_dir + filename, m);
        };

    show_and_save("1. Original", img_src, "01_Original.jpg");
    show_and_save("2. HSV Mask", img_mask, "02_HSV_Mask.jpg");
    show_and_save("3. Filtered Binary", img_filtered_binary, "03_Filtered_Binary.jpg");
    show_and_save("4. Eroded", img_eroded, "04_Eroded.jpg");
    show_and_save("5. Dilated", img_dilated, "05_Dilated.jpg");
    show_and_save("6. Opened", img_opened, "06_Opened.jpg");
    show_and_save("7. Closed", img_closed, "07_Closed.jpg");
    show_and_save("8. Final Result", img_result, "08_Final_Result.jpg");

    // ================= 8. 生成无变形拼接大图 =================
    // 为了拼接，必须先将所有单通道图转为3通道(BGR)
    Mat d_mask, d_filtered, d_eroded, d_dilated, d_opened, d_closed;
    cvtColor(img_mask, d_mask, COLOR_GRAY2BGR);
    cvtColor(img_filtered_binary, d_filtered, COLOR_GRAY2BGR);
    cvtColor(img_eroded, d_eroded, COLOR_GRAY2BGR);
    cvtColor(img_dilated, d_dilated, COLOR_GRAY2BGR);
    cvtColor(img_opened, d_opened, COLOR_GRAY2BGR);
    cvtColor(img_closed, d_closed, COLOR_GRAY2BGR);

    // 拼接布局：2行 x 4列
    Mat row1, row2, combined;
    // 第一行: 原图 | HSV掩码 | 过滤后二值 | 腐蚀
    hconcat(vector<Mat>{img_src, d_mask, d_filtered, d_eroded}, row1);
    // 第二行: 膨胀 | 开运算 | 闭运算 | 最终结果
    hconcat(vector<Mat>{d_dilated, d_opened, d_closed, img_result}, row2);
    // 垂直合并
    vconcat(row1, row2, combined);

    // --- 关键修改：按比例缩放，杜绝变形 ---
    // 之前可能用了固定尺寸(如300x200)，那会导致拉伸变形。
    // 现在我们使用缩放因子 (fx, fy)。
    // 0.4 表示长宽都变为原来的 40%，保持原始宽高比。
    Mat combined_show;
    resize(combined, combined_show, Size(), 0.4, 0.4);

    // 显示拼接图
    namedWindow("Process Flow (All in One)", WINDOW_NORMAL);
    imshow("Process Flow (All in One)", combined_show);

    // 保存拼接图 (保存全尺寸高清版)
    imwrite(save_dir + "00_All_In_One_Result.jpg", combined);

    cout << "✅ 所有操作完成！按任意键退出..." << endl;
    waitKey(0);
    destroyAllWindows();
    return 0;
}