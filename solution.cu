#include "htk.h"
#include <nppi.h>

#include "opencv2/opencv.hpp"
using namespace cv;

#if defined(USE_DOUBLE)
#define EPSILON 0.00005
#define FABS fabs
typedef double real_t;

#else
#define EPSILON 0.00005f
#define FABS fabsf
typedef float real_t;
#endif

#define val(arry, i, j) arry[(i)*width + (j)]

int iterations;

#define RGB_TYPE float

static void hot_plate(real_t *out, real_t *in, real_t *mask, int width, int height, real_t epsilon) {
  real_t *u = in;
  real_t *w = out;

  // Variables for the convolution
  int conv_step = width * sizeof(RGB_TYPE);
  NppiSize oSrcSize = {width, height};
  NppiPoint oSrcOffset = {1, 1};
  NppiSize convROI = {width-2 , height-2};
  NppiSize oKernelSize = {9, 3};
  NppiPoint oAnchor = {1, 1};
  NppiBorderType eborderType = NPP_BORDER_REPLICATE;

  NppStatus convStatus = nppiFilterBorder_32f_C1R(
                              u + width + 1, // *pSrc
                              conv_step, // nSrcStep
                              oSrcSize, // oSrcSize
                              oSrcOffset, // oSrcOffset
                              w + width + 1, // *pDst
                              conv_step, // nDstStep
                              convROI, // oSizeROI
                              mask, // *pKernel
                              oKernelSize, // oKernelSize
                              oAnchor, // oAnchor
                              eborderType // eBorderType
                              );

printf("STATUS %d\n", convStatus);

}

int main(int argc, char *argv[]) {
    int width;
  int height;

    VideoCapture capture;
#define VIDEO_FILE "test_video.mp4"
#define SIZEOFPIXEL 3
    
  capture.open(VIDEO_FILE);
    if (!capture.isOpened()) {
        printf("Error: Unable to open the video file.");
        return -1;
    }  Mat frame;
  capture >> frame;
    if (frame.empty()) {
        printf("Error: Unable to open the frame file.");
        return -1;
    }

  width = frame.cols * 3;
  height = frame.rows;
  //Get a Mat (this is the matrix that will hold a single frame)
  int matElements = frame.cols * frame.rows * SIZEOFPIXEL;
// unsigned char* oneDFrame = new unsigned char[matElements];
// unsigned char* oneDFrameOutput = new unsigned char[matElements];
  float* oneDFrame = new float[matElements];
  float* oneDFrameOutput = new float[matElements];
  printf("Frame Elements %d\n", matElements);
  MatIterator_<cv::Vec3b> it, end;
  int i = 0;

    for ( it = frame.begin<cv::Vec3b>(), end = frame.end<cv::Vec3b>(); it != end; ++it ) {

        //get current bgr pixels:
        uchar &r = (*it)[2];
        uchar &g = (*it)[1];
        uchar &b = (*it)[0];

        //Store them into array, as a cv::Scalar:
        oneDFrame[i] = b;
        oneDFrame[i + 1] = g;
        oneDFrame[i + 2] = r;
        i +=3;

    }
    unsigned char* tempFrame = new unsigned char [matElements];
    for(int i = 0; i < width * height; i++){
      unsigned char temp = (unsigned char)oneDFrame[i];
      tempFrame[i] = temp;
  }
    Mat reconstructed(frame.rows, frame.cols, CV_8UC3, tempFrame);

    // Save the image
    imwrite("no_conv_image.png", reconstructed);

    // std::cout << "Image saved as output_image.png" << std::endl;
    // return 0;

  htkArg_t args;
  // int channels;
  // char *inputFile;
  // htkImage_t input;
  // htkImage_t output;
  // float *hostInputData;
  float *hostOutputData = new float[matElements];
  // float hostMask[9] = {0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0};
  #define BOX_BLUR_SIZE 3 * 3 * 3
  #define BOX_BLUR_COEFF float(1.0/9.0)
  float boxBlur[BOX_BLUR_SIZE] = { BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, 
                                    BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, 
                                    BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0};

  // args = htkArg_read(argc, argv);
  // if (args.inputCount != 1) {htkLog(ERROR, "Missing input"); return 1;}

  // htkTime_start(IO, "Importing data and creating memory on host");
  // inputFile = htkArg_getInputFile(args, 0);
  // input = htkImport(inputFile);
  // width  = htkImage_getWidth(input);
  // height = htkImage_getHeight(input);
  // channels  = htkImage_getChannels(input);
  // if (channels != 1) {htkLog(ERROR, "Expecting gray scale image"); return 1;}
  // output = htkImage_new(width, height, channels);
  // hostInputData  = htkImage_getData(input);
  // hostOutputData = htkImage_getData(output);

  htkTime_stop(IO, "Importing data and creating memory on host");
  htkLog(TRACE, "Image dimensions WxH are ", width, " x ", height);

  // Allocate GPU memory
  real_t *devInputData;
  real_t *devOutputData;

  real_t *devMask;

  htkTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&devInputData, width * height * sizeof(real_t));
  cudaMalloc((void **)&devOutputData, width * height * sizeof(real_t));
  cudaMalloc((void **)&devMask, 9 * 3 * sizeof(real_t));
  htkTime_stop(GPU, "Allocating GPU memory.");
  
  

  // Copy memory to the GPU
  htkTime_start(IO, "Copying memory to the GPU.");
  cudaMemcpy(devInputData, oneDFrame, width * height * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devOutputData, oneDFrameOutput, width * height * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devMask, boxBlur, 9 * 3 * sizeof(real_t), cudaMemcpyHostToDevice);
  htkTime_stop(IO, "Copying memory to the GPU.");

  htkTime_start(Compute, "Doing the computation");


  hot_plate(devOutputData, devInputData, devMask, width, height, EPSILON);
  
  
  htkTime_stop(Compute, "Doing the computation");
  htkLog(TRACE, "Solution iterations: ", iterations);

  // Copy the GPU memory back to the CPU
  htkTime_start(IO, "Copying memory back to the CPU.");
  cudaMemcpy(hostOutputData, devOutputData, width * height * sizeof(real_t), cudaMemcpyDeviceToHost);
  htkTime_stop(IO, "Copying memory back to the CPU.");
  unsigned char* oneDFrameSafe = new unsigned char[matElements];
  // printf("SIZE %d\n", sizeof(hostOutputData));
  for(int i = 0; i < width * height; i++){
      unsigned char temp = (unsigned char)hostOutputData[i];
      oneDFrameSafe[i] = temp;
  }

  std::cout <<"TRYING SOMETHING" <<std::endl;
  Mat reconstructedA(frame.rows, frame.cols, CV_8UC3, oneDFrameSafe);

  // Save the image
  imwrite("output_image_blur.png", reconstructedA);

  std::cout << "Image saved as output_image.png" << std::endl;

  // Free the GPU memory
  htkTime_start(GPU, "Freeing GPU memory.");
  cudaFree(devInputData);
  cudaFree(devOutputData);
  cudaFree(devMask);
  htkTime_stop(GPU, "Freeing GPU memory.");

  return 0;
}








