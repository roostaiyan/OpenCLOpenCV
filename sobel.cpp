#include "sobel.h"

using namespace cv;

namespace myocl{

bool Sobel::create(cv::Size _src_size, int _stype, int _ddepth,
                   int _dx, int _dy, int _ksize,
                   double _scale, double _delta, int _border_type)
{
    src_size = _src_size;
    stype = _stype;
    ddepth = _ddepth;
    dx = _dx;
    dy = _dy;
    ksize = _ksize;
    scale = _scale;
    delta = _delta;
    borderType = _border_type;
    //! Parse Kernel Properties
    int type = stype, sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if (cn > 4)
        return false;

    if (ddepth < 0)
        ddepth = sdepth;

    dtype = CV_MAKE_TYPE(ddepth, cn);

    int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

    Mat _kernelX, _kernelY;
    getDerivKernels( _kernelX, _kernelY, dx, dy, ksize, false, ktype );
    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differentiating part
        if( dx == 0 )
            _kernelX *= scale;
        else
            _kernelY *= scale;
    }

    kernelX = _kernelX.reshape(1, 1);
    if (kernelX.cols % 2 != 1)
        return false;
    kernelY = _kernelY.reshape(1, 1);
    if (kernelY.cols % 2 != 1)
        return false;

    Point anchor(-1, -1);
    if (anchor.x < 0)
        anchor.x = kernelX.cols >> 1;
    if (anchor.y < 0)
        anchor.y = kernelY.cols >> 1;

    bdepth = CV_32F;
    bool int_arithm = false;
    int shift_bits = 0;

    kernel_row_filter = ocl_sepRowFilter2D(anchor.x, false, int_arithm, shift_bits);
    kernel_row_filter_fast8uc1 = ocl_sepRowFilter2D(anchor.x, true, int_arithm, shift_bits);
    kernel_col_filter = ocl_sepColFilter2D(anchor.y, int_arithm, shift_bits);

    if (!kernel_row_filter || !kernel_row_filter_fast8uc1 || !kernel_col_filter)
        return false;
}

void Sobel::apply(InputArray _src, OutputArray _dst, bool abs_thresh, uchar thresh, uchar max_val)
{
    assert(!_src.empty());

    if(cv::ocl::useOpenCL() && _src.isUMat() && _dst.isUMat()
            && !kernel_row_filter->empty() && !kernel_row_filter_fast8uc1->empty() && !kernel_col_filter->empty()){
        UMat src = _src.getUMat();
        int cn = CV_MAT_CN(stype);
        _dst.create(src_size, CV_MAKETYPE(ddepth, cn));
        UMat dst = _dst.getUMat();
        //! Running of ocl_sepRowFilter2D
        int radiusY = (buf.rows - src_size.height) >> 1;
        bool fast8uc1 = false;
        if (stype == CV_8UC1)
        {
            Size srcWholeSize;
            Point srcOffset;
            src.locateROI(srcWholeSize, srcOffset);
            fast8uc1 = srcOffset.x % 4 == 0 && src.cols % 4 == 0 && src.step % 4 == 0;
        }
        {
            std::shared_ptr<cv::ocl::Kernel> kernel_rowf;
            Size srcWholeSize; Point srcOffset;
            src.locateROI(srcWholeSize, srcOffset);
            size_t localsize[2] = {16, 16};

            Size bufSize = buf.size();
            size_t globalsize[2] = {DIVUP(bufSize.width, localsize[0]) * localsize[0], DIVUP(bufSize.height, localsize[1]) * localsize[1]};
            if (fast8uc1)
                globalsize[0] = DIVUP((bufSize.width + 3) >> 2, localsize[0]) * localsize[0];

            if (fast8uc1){
                kernel_rowf = kernel_row_filter_fast8uc1;
                kernel_rowf->args(ocl::KernelArg::PtrReadOnly(src), (int)(src.step / src.elemSize()), srcOffset.x,
                                  srcOffset.y, src.cols, src.rows, srcWholeSize.width, srcWholeSize.height,
                                  ocl::KernelArg::PtrWriteOnly(buf), (int)(buf.step / buf.elemSize()),
                                  buf.cols, buf.rows, radiusY);
            }
            else {
                kernel_rowf = kernel_row_filter;
                kernel_rowf->args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, srcOffset.x,
                                  srcOffset.y, src.cols, src.rows, srcWholeSize.width, srcWholeSize.height,
                                  ocl::KernelArg::PtrWriteOnly(buf), (int)buf.step, buf.cols, buf.rows, radiusY);
            }
            kernel_rowf->run(2, globalsize, localsize, false);
        }
        //! Running of ocl_sepColFilter2D
        {
            size_t localsize[2] = { 16, 16 };
            size_t globalsize[2] = { 0, 0 };

            Size sz = dst.size();

            globalsize[1] = DIVUP(sz.height, localsize[1]) * localsize[1];
            globalsize[0] = DIVUP(sz.width, localsize[0]) * localsize[0];

            int shift_bits = 0;
            kernel_col_filter->args(ocl::KernelArg::ReadOnly(buf), ocl::KernelArg::WriteOnly(dst),
                                    static_cast<float>(delta * (1u << (2 * shift_bits))),
                                    uchar(abs_thresh), thresh, max_val);

            kernel_col_filter->run(2, globalsize, localsize, false);
        }
        cv::ocl::finish();
    }
    else {
        //! Mat Functions
        cv::Mat _dst_16S;
        cv::Sobel(_src, _dst_16S, ddepth, dx, dy, ksize, scale, delta, borderType);

        // Revealing both edges.
        if(abs_thresh) {
            convertScaleAbs(_dst_16S, _dst);
            // Thresholding to zero helps reducing output noise again:
            cv::threshold(_dst, _dst, thresh, max_val, cv::THRESH_TOZERO);
        }
    }
}

std::shared_ptr<cv::ocl::Kernel> Sobel::ocl_sepRowFilter2D(int anchor, bool fast8uc1,
                                                           bool int_arithm, int shift_bits)
{
    CV_Assert(shift_bits == 0 || int_arithm);

    int cn = CV_MAT_CN(stype), sdepth = CV_MAT_DEPTH(stype);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    int buf_type = CV_MAKETYPE(bdepth, cn);
    Size bufSize(src_size.width, src_size.height + kernelY.cols - 1);
    buf.create(bufSize, buf_type);

    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return nullptr;

    size_t localsize[2] = {16, 16};

    size_t globalsize[2] = {DIVUP(bufSize.width, localsize[0]) * localsize[0], DIVUP(bufSize.height, localsize[1]) * localsize[1]};
    if (fast8uc1)
        globalsize[0] = DIVUP((bufSize.width + 3) >> 2, localsize[0]) * localsize[0];

    int radiusX = anchor, radiusY = (buf.rows - src_size.height) >> 1;

    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT_101" },
            * const btype = borderMap[borderType & ~BORDER_ISOLATED];

    bool extra_extrapolation = src_size.height < (int)((-radiusY + globalsize[1]) >> 1) + 1;
    extra_extrapolation |= src_size.height < radiusY;
    extra_extrapolation |= src_size.width < (int)((-radiusX + globalsize[0] + 8 * localsize[0] + 3) >> 1) + 1;
    extra_extrapolation |= src_size.width < radiusX;

    char cvt[40];
    cv::String build_options = cv::format("-D RADIUSX=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d -D %s -D %s -D %s"
                                          " -D srcT=%s -D dstT=%s -D convertToDstT=%s -D srcT1=%s -D dstT1=%s%s%s",
                                          radiusX, (int)localsize[0], (int)localsize[1], cn, btype,
            extra_extrapolation ? "EXTRA_EXTRAPOLATION" : "NO_EXTRA_EXTRAPOLATION",
            isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
            ocl::typeToStr(stype), ocl::typeToStr(buf_type),
            ocl::convertTypeStr(sdepth, bdepth, cn, cvt),
            ocl::typeToStr(sdepth), ocl::typeToStr(bdepth),
            doubleSupport ? " -D DOUBLE_SUPPORT" : "",
            int_arithm ? " -D INTEGER_ARITHMETIC" : "");
    build_options += ocl::kernelToStr(kernelX, bdepth);

    String kernelName("row_filter");
    if (fast8uc1)
        kernelName += "_C1_D0";

    //! Try to Compile OpenCL Kernel & Keep it Into Gpu Memory
    std::ifstream t(filterSepRow_Src_Path);
    std::string source((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource filterSepRow_ProgramSource(source);
    std::shared_ptr<ocl::Kernel> kernel =
            std::make_shared<ocl::Kernel>(kernelName.c_str(), filterSepRow_ProgramSource, build_options);
    if (kernel->empty())
        return nullptr;
    return kernel;
}

std::shared_ptr<cv::ocl::Kernel> Sobel::ocl_sepColFilter2D(int anchor,
                                          bool int_arithm, int shift_bits)
{
    assert(shift_bits == 0 || int_arithm);
    int cn = CV_MAT_CN(stype), sdepth = CV_MAT_DEPTH(stype);

    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;
    if (ddepth == CV_64F && !doubleSupport)
        return nullptr;

    size_t localsize[2] = { 16, 16 };
    size_t globalsize[2] = { 0, 0 };

    char cvt[2][40];
    int floatT = std::max(CV_32F, bdepth);
    cv::String build_options = cv::format("-D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D CN=%d"
                                          " -D srcT=%s -D dstT=%s -D convertToFloatT=%s -D floatT=%s -D convertToDstT=%s"
                                          " -D srcT1=%s -D dstT1=%s -D SHIFT_BITS=%d%s%s",
                                          anchor, (int)localsize[0], (int)localsize[1], cn,
            ocl::typeToStr(buf.type()), ocl::typeToStr(dtype),
            ocl::convertTypeStr(bdepth, floatT, cn, cvt[0]),
            ocl::typeToStr(CV_MAKETYPE(floatT, cn)),
            ocl::convertTypeStr(shift_bits ? floatT : bdepth, ddepth, cn, cvt[1]),
            ocl::typeToStr(bdepth), ocl::typeToStr(ddepth),
            2*shift_bits, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
            int_arithm ? " -D INTEGER_ARITHMETIC" : "");
    build_options += ocl::kernelToStr(kernelY, bdepth);

    //! Try to Compile OpenCL Kernel & Keep it Into Gpu Memory
    std::ifstream t(filterSepCol_Src_Path);
    std::string source((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource filterSepCol_ProgramSource(source);

    std::shared_ptr<ocl::Kernel> kernel =
            std::make_shared<ocl::Kernel>("col_filter", filterSepCol_ProgramSource, build_options);
    if (kernel.get()->empty())
        return nullptr;
    return kernel;
}


}
