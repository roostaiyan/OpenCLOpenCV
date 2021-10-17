#include <iostream>
#include "ocl_loader.h"
#include "sobel.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if(!myocl::device::loadOpenCL(0)){
        std::cout << "Failed to Run on GPU. Running on Cpu ... " << std::endl;
    }

    //std::cout << cv::ocl::
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
     VideoCapture cap("test.mp4");

     // Check if camera opened successfully
     if(!cap.isOpened()){
       cout << "Error opening video stream or file" << endl;
       return -1;
     }

     while(1){

       Mat frame;
       // Capture frame-by-frame
       cap >> frame;

       // If the frame is empty, break immediately
       if (frame.empty())
         break;

       //! Plase call create just once for better time
       double t0 = static_cast<double>(cv::getTickCount());
       std::shared_ptr<myocl::Sobel> sobel_ocl = std::make_shared<myocl::Sobel>();
       sobel_ocl->create(frame.size(), CV_8UC1, CV_8U, 0, 1, 3, 1, 0, BORDER_DEFAULT);

       double t1 = static_cast<double>(cv::getTickCount());
       double secs = (t1-t0)/cv::getTickFrequency()*1000;
       std::cout <<" Create & Compile Kernel Time = " << secs << " , [It should be done just once]" << std::endl;
       t0 = static_cast<double>(cv::getTickCount());
       // Convert image to gray scale
       cv::UMat im_gray, grad_x, grad_scaled;
       if (frame.channels() == 3)
           cvtColor(frame, im_gray, COLOR_RGB2GRAY);
       else
           frame.copyTo(im_gray);
       if(sobel_ocl){
           sobel_ocl->apply(im_gray, grad_scaled, true, 17, 255);
       } else {
           // Vertical edges from the sobel filter is used to extract regions that are capable of having text.
           cv::Sobel(im_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);

           // Revealing both edges.
           convertScaleAbs(grad_x, grad_scaled);

           // Thresholding to zero helps reducing output noise again:
           threshold(grad_scaled, grad_scaled, 17, 255, THRESH_TOZERO);
       }
       t1 = static_cast<double>(cv::getTickCount());
       secs = (t1-t0)/cv::getTickFrequency()*1000;
       std::cout <<" OpenCL Time  = " << secs << std::endl;
       imshow( "Gradient of X", grad_scaled );

       // Press  ESC on keyboard to exit
       char c=(char)waitKey(25);
       if(c==27)
         break;
     }

     // When everything done, release the video capture object
     cap.release();

     // Closes all the frames
     destroyAllWindows();

    getchar();
    return 0;
}
