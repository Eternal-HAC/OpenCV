#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// 定义线段结构体：新增平均x坐标（用于左右排序）
struct LineSegment {
    Vec4i coords;   // 线段坐标 (x1,y1,x2,y2)
    double length;  // 线段长度
    double slope;   // 线段斜率
    double a, b, c; // 直线一般式 ax + by + c = 0（归一化后）
    int x_mean;     // 线段的平均x坐标（用于左右排序）
};

// 计算线段长度
double calculateLineLength(const Vec4i& line) {
    int x1 = line[0], y1 = line[1];
    int x2 = line[2], y2 = line[3];
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// 按可调节间隔采样计算斜率 + 计算一般式参数+平均x坐标
double calculateSlopeWithAdjustableGap(const Vec4i& line, int sample_gap, double& a, double& b, double& c, int& x_mean) {
    int x1 = line[0], y1 = line[1];
    int x2 = line[2], y2 = line[3];
    double line_length = calculateLineLength(line);
    x_mean = (x1 + x2) / 2; // 计算线段平均x坐标（核心：区分左右）

    int sx1, sy1, sx2, sy2;
    if (line_length <= sample_gap || sample_gap <= 0) {
        sx1 = x1; sy1 = y1;
        sx2 = x2; sy2 = y2;
    }
    else {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double unit_dx = dx / line_length;
        double unit_dy = dy / line_length;
        sx1 = cvRound(x1 + unit_dx * sample_gap);
        sy1 = cvRound(y1 + unit_dy * sample_gap);
        sx2 = cvRound(x2 - unit_dx * sample_gap);
        sy2 = cvRound(y2 - unit_dy * sample_gap);
    }

    // 计算直线一般式
    a = (sy2 - sy1);
    b = (sx1 - sx2);
    c = (sx2 * sy1 - sx1 * sy2);
    double norm = sqrt(a * a + b * b);
    if (norm > 1e-6) {
        a /= norm;
        b /= norm;
        c /= norm;
    }

    if (sx2 == sx1) return 1e9; // 垂直直线
    return (double)(sy2 - sy1) / (sx2 - sx1);
}

// 计算两条平行直线的垂直距离
double calculateLineDistance(const LineSegment& line1, const LineSegment& line2) {
    return abs(line1.c - line2.c);
}

// 筛选间距大于阈值的直线（去重）
vector<LineSegment> filterLinesByDistance(const vector<LineSegment>& sorted_lines, double distance_threshold, int target_num) {
    vector<LineSegment> filtered_lines;
    if (sorted_lines.empty()) return filtered_lines;

    filtered_lines.push_back(sorted_lines[0]);
    for (size_t i = 1; i < sorted_lines.size(); i++) {
        bool is_valid = true;
        for (const auto& selected : filtered_lines) {
            double dist = calculateLineDistance(sorted_lines[i], selected);
            if (dist < distance_threshold) {
                is_valid = false;
                break;
            }
        }
        if (is_valid) {
            filtered_lines.push_back(sorted_lines[i]);
            if (filtered_lines.size() >= target_num) break;
        }
    }
    return filtered_lines;
}

// 核心修改：过滤从右往左数第二条和第七条直线
vector<LineSegment> filterSecondAndSeventhRightLine(vector<LineSegment>& lines) {
    // 步骤1：按平均x坐标降序排序（从右到左）
    sort(lines.begin(), lines.end(), [](const LineSegment& a, const LineSegment& b) {
        return a.x_mean > b.x_mean; // x越大越靠右，降序=从右到左
        });

    // 步骤2：剔除从右数第二条（索引1）和第七条（索引6）
    vector<LineSegment> result;
    for (size_t i = 0; i < lines.size(); i++) {
        if (i != 1 && i != 6) { // 跳过索引1（右2）和索引6（右7）
            result.push_back(lines[i]);
        }
    }

    // 步骤3：恢复按长度降序排序（保证保留最长的有效线条）
    sort(result.begin(), result.end(), [](const LineSegment& a, const LineSegment& b) {
        return a.length > b.length;
        });

    return result;
}

// 线段双向延长函数
Vec4i extendLine(const Vec4i& line, int extend_len) {
    int x1 = line[0], y1 = line[1];
    int x2 = line[2], y2 = line[3];

    double dx = x2 - x1;
    double dy = y2 - y1;
    double line_len = calculateLineLength(line);
    if (line_len < 1e-6) return line;

    double unit_dx = dx / line_len;
    double unit_dy = dy / line_len;
    double unit_dx_rev = -unit_dx;
    double unit_dy_rev = -unit_dy;

    int new_x1 = cvRound(x1 + unit_dx_rev * extend_len);
    int new_y1 = cvRound(y1 + unit_dy_rev * extend_len);
    int new_x2 = cvRound(x2 + unit_dx * extend_len);
    int new_y2 = cvRound(y2 + unit_dy * extend_len);

    return Vec4i(new_x1, new_y1, new_x2, new_y2);
}

// 输出直线信息
void printLineEquation(const LineSegment& line) {
    int x1 = line.coords[0], y1 = line.coords[1];
    int x2 = line.coords[2], y2 = line.coords[3];

    if (x2 == x1) {
        cout << "  方程：x = " << x1 << endl;
    }
    else {
        double k = line.slope;
        double b = y1 - k * x1;
        cout << "  方程：y = " << fixed << setprecision(2) << k << "x + " << b << endl;
    }
    cout << "  斜率：" << fixed << setprecision(4) << line.slope << endl;
    cout << "  长度：" << fixed << setprecision(2) << line.length << " 像素" << endl;
    cout << "  平均x坐标：" << line.x_mean << endl;
}

int main() {
    // ===================== 1. 配置参数（核心修改：目标数量+预留数量） =====================
    string img_path = "E:/01_code/opcv/实验7/1.jpg";        // 你的图像路径
    double slope_min = -2;              // 斜率下限
    double slope_max = 2;               // 斜率上限
    int target_line_num = 5;            // 最终保留5条线（过滤2条后）
    int prepare_line_num = 7;           // 先提取7条线（用于过滤右2+右7）
    int slope_sample_gap = 15;          // 斜率采样间隔un u8  
    double distance_threshold = 10.0;   // 间距阈值
    int line_extend_length = 220;       // 延长长度

    // ===================== 2. 图像预处理 =====================
    Mat img = imread(img_path);
    if (img.empty()) {
        cout << "无法读取图像！" << endl;
        return -1;
    }
    Mat img_gray, img_blur, img_thresh, img_edge, img_result = img.clone();

    GaussianBlur(img, img_blur, Size(7, 7), 0);
    cvtColor(img_blur, img_gray, COLOR_BGR2GRAY);
    adaptiveThreshold(img_gray, img_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 5);
    Canny(img_thresh, img_edge, 30, 90);

    // ===================== 3. 提取候选线段 =====================
    vector<Vec4i> raw_lines;
    HoughLinesP(
        img_edge,
        raw_lines,
        1,
        CV_PI / 180,
        35,
        45,
        5
    );

    // ===================== 4. 计算参数+筛选斜率 =====================
    vector<LineSegment> valid_lines;
    for (const auto& line : raw_lines) {
        double a, b, c;
        int x_mean;
        double length = calculateLineLength(line);
        double slope = calculateSlopeWithAdjustableGap(line, slope_sample_gap, a, b, c, x_mean);
        if (slope >= slope_min && slope <= slope_max) {
            valid_lines.push_back({ line, length, slope, a, b, c, x_mean });
        }
    }

    // ===================== 5. 排序+去重（先提取7条） =====================
    sort(valid_lines.begin(), valid_lines.end(), [](const LineSegment& a, const LineSegment& b) {
        return a.length > b.length;
        });
    // 先提取7条（预留2条用于过滤）
    vector<LineSegment> filtered_by_distance = filterLinesByDistance(valid_lines, distance_threshold, prepare_line_num);

    // ===================== 6. 核心：过滤从右数第二条和第七条直线 =====================
    vector<LineSegment> final_lines = filterSecondAndSeventhRightLine(filtered_by_distance);
    // 确保最终数量为目标值（5条）
    if (final_lines.size() > target_line_num) {
        final_lines.resize(target_line_num);
    }

    // ===================== 7. 绘制+延长 =====================
    cout << "========== 检测结果（过滤从右数第二条+第七条线+延长" << line_extend_length << "像素）==========" << endl;
    cout << "斜率采样间隔：" << slope_sample_gap << " 像素" << endl;
    cout << "间距阈值：" << distance_threshold << " 像素" << endl;
    cout << "------------------------------------------------" << endl;

    for (size_t i = 0; i < final_lines.size(); i++) {
        LineSegment& curr_line = final_lines[i];
        Vec4i extended_line = extendLine(curr_line.coords, line_extend_length);
        // 绘制延长后的红线
        line(img_result, Point(extended_line[0], extended_line[1]),
            Point(extended_line[2], extended_line[3]), Scalar(0, 0, 255), 3);

        // 输出信息
        cout << "第" << i + 1 << "条直线：" << endl;
        printLineEquation(curr_line);
        for (size_t j = 0; j < final_lines.size(); j++) {
            if (i != j) {
                double dist = calculateLineDistance(curr_line, final_lines[j]);
                cout << "  与第" << j + 1 << "条直线的垂直距离：" << fixed << setprecision(2) << dist << " 像素" << endl;
            }
        }
        cout << "------------------------------------------------" << endl;
    }

    // 提示信息
    if (final_lines.size() < target_line_num) {
        cout << "警告：仅检测到 " << final_lines.size() << " 条目标直线（需" << target_line_num << "条）" << endl;
    }

    // ===================== 8. 显示+保存 =====================
    namedWindow("Result", cv::WINDOW_KEEPRATIO);
    imshow("1. 原始图像", img);
    imshow("2. 预处理边缘图", img_edge);
    imshow("Result", img_result);
    waitKey(0);
    destroyAllWindows();

    imwrite("filtered_5_lines_result.jpg", img_result);

    return 0;
}