#ifndef SOBELKERNEL_H
#define SOBELKERNEL_H
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <fstream>
#include <streambuf>

#define SOURCE_DIR "${CMAKE_SOURCE_DIR}"

#define filterSepCol_Src_Path OPENCL_RESOURCE_DIR"/filterSepCol.cl"
#define filterSepRow_Src_Path OPENCL_RESOURCE_DIR"/filterSepRow.cl"

inline int DIVUP(int total, int grain)
{
    return (total + grain - 1) / grain;
}

namespace myocl {

class Sobel
{
public:
    Sobel(){}
    bool create(cv::Size _src_size, int _stype, int _ddepth, int _dx, int _dy, int _ksize,
                double _scale, double _delta, int _border_type);
    void apply(cv::InputArray _src, cv::OutputArray _dst, bool abs_thresh = false, uchar thresh = 0, uchar max_val = 255);

private:
    std::shared_ptr<cv::ocl::Kernel> kernel_row_filter_fast8uc1, kernel_row_filter;
    std::shared_ptr<cv::ocl::Kernel> kernel_col_filter;
    cv::UMat buf;
    cv::Mat kernelX, kernelY;
    cv::Size src_size;
    int stype;
    int bdepth;
    int ddepth;
    int dtype;
    int dx;
    int dy;
    int ksize;
    double scale;
    double delta;
    int borderType;

private:
    const int optimizedSepFilterLocalWidth  = 16;
    const int optimizedSepFilterLocalHeight = 8;
    static bool createBitExactKernel_32S(const cv::Mat& kernel, cv::Mat& kernel_dst, int bits);

    static bool ocl_sepFilter2D(cv::InputArray _src, cv::OutputArray _dst, int ddepth,
                                cv::InputArray _kernelX, cv::InputArray _kernelY, cv::Point anchor,
                                double delta, int borderType);

    std::shared_ptr<cv::ocl::Kernel> ocl_sepRowFilter2D(int anchor, bool fast8uc1,
                                                        bool int_arithm, int shift_bits);

    std::shared_ptr<cv::ocl::Kernel> ocl_sepColFilter2D(int anchor,
                                                        bool int_arithm, int shift_bits);
};

}
#endif // SOBELKERNEL_H
