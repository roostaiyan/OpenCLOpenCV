#ifndef OCL_H
#define OCL_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace myocl {
class device {
public:
    static bool loadOpenCL(int dev_idx){
        if(dev_idx<0){
            cv::ocl::setUseOpenCL(false);
            return true;
        }
        cv::ocl::Context context;
        if (!context.create(cv::ocl::Device::TYPE_GPU))
        {
            std::cout << "Failed to create the OpenCL Context..." << std::endl;
            std::cout << "Plase Install The Graphic Driver including OpenCL Vendor " << std::endl;
        }

        std::cout << context.ndevices() << " GPU devices are detected." << std::endl; //This bit provides an overview of the OpenCL devices you have in your computer
        for (int i = 0; i < context.ndevices(); i++)
        {
            cv::ocl::Device device = context.device(i);
            std::cout << "name:              " << device.name() << std::endl;
            std::cout << "available:         " << device.available() << std::endl;
            std::cout << "imageSupport:      " << device.imageSupport() << std::endl;
            std::cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << std::endl;
            std::cout << std::endl;
        }
        //! Here is where you change which GPU to use (e.g. 0 or 1),
        //! We use the first OpenCL Device which is available.
        if(dev_idx<context.ndevices())
            cv::ocl::Device(context.device(dev_idx));
        cv::ocl::setUseOpenCL(true);
        return cv::ocl::useOpenCL();
    }
};
}
#endif // OCL_H
