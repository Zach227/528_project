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

#define VIDEO_FILE "test_video.mp4"
#define BOX_BLUR_SIZE 3 * 3 * 3
#define BOX_BLUR_COEFF float(1.0 / 9.0)

int iterations;

#define RGB_TYPE float

static void genericMultiChannelFilter(Npp8u *out, Npp8u *in,  Npp32s *mask, int width, int height, real_t epsilon, int maskWidth) {
  Npp8u *u = in;
  Npp8u *w = out;
  // width = width * 3;
  printf("Width: %d\n", width);
  // Variables for the convolution
  int conv_step = width * sizeof(Npp8u);
  NppiSize oSrcSize = {width, height};
  NppiPoint oSrcOffset = {1, 1};
  NppiSize convROI = {width , height};
  NppiSize oKernelSize = {maskWidth, maskWidth};
  NppiPoint oAnchor = {1, 1};
  NppiBorderType eborderType = NPP_BORDER_REPLICATE;

  NppStatus convStatus = nppiFilterBorder_8u_C3R(
                              u , // *pSrc
                              conv_step, // nSrcStep
                              oSrcSize, // oSrcSize
                              oSrcOffset, // oSrcOffset
                              w , // *pDst
                              conv_step, // nDstStep
                              convROI, // oSizeROI
                              mask, // *pKernel
                              oKernelSize, // oKernelSize
                              oAnchor, // oAnchor
                              1, // nDivisor
                              eborderType // eBorderType
                              );

  printf("STATUS %d\n", convStatus);


}
  
static void superBoxBlur(real_t *out, real_t *in, real_t *mask, int width, int height, real_t epsilon) {
  real_t *u = in;
  real_t *w = out;

  // Variables for the convolution
  int conv_step = width * sizeof(RGB_TYPE);
  NppiSize oSrcSize = {width, height};
  NppiPoint oSrcOffset = {0, 0};
  NppiSize convROI = {width , height};
  NppiSize oKernelSize = {18, 6};
  NppiPoint oAnchor = {0, 0};
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
static void genericKernelFilter(real_t *out, real_t *in, real_t *mask, int width, int height, real_t epsilon, int maskWidth) {
  real_t *u = in;
  real_t *w = out;

  // Variables for the convolution
  int conv_step = width * sizeof(RGB_TYPE);
  NppiSize oSrcSize = {width, height};
  NppiPoint oSrcOffset = {0, 0};
  NppiSize convROI = {width , height};
  NppiSize oKernelSize = {maskWidth * 3, maskWidth};
  NppiPoint oAnchor = {0, 0};
  NppiBorderType eborderType = NPP_BORDER_REPLICATE;

  NppStatus convStatus = nppiFilterBorder_32f_C1R(
                              u + width + 1, // *pSrc
                              conv_step, // nSrcStep
                              oSrcSize, // oSrcSize
                              oSrcOffset, // oSrcOffset
                              u + width + 1, // *pDst
                              conv_step, // nDstStep
                              convROI, // oSizeROI
                              mask, // *pKernel
                              oKernelSize, // oKernelSize
                              oAnchor, // oAnchor
                              eborderType // eBorderType
                              );

  printf("STATUS %d\n", convStatus);

}
static void boxBlur(real_t *out, real_t *in, real_t *mask, int width, int height, real_t epsilon) {
  real_t *u = in;
  real_t *w = out;

  // Variables for the convolution
  int conv_step = width * sizeof(RGB_TYPE);
  NppiSize oSrcSize = {width, height};
  NppiPoint oSrcOffset = {0, 0};
  NppiSize convROI = {width , height};
  NppiSize oKernelSize = {9, 3};
  NppiPoint oAnchor = {0, 0};
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

template<int N>
void createBoxBlurMask(float (&mask)[N * N * 3]) {
    for (int i = 0; i < N * N * 3; ++i) {
        mask[i] = 0.0f;  // Initialize all elements to 0
    }
    
    // Set BOX_BLUR_COEFF for every third element
    for (int i = 0; i < N * N * 3; i += 3) {
        mask[i] = 1.0/(N * N);
    }
}


int main(int argc, char *argv[]) {
  VideoCapture capture;
    
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

  // Set width and height
  int channels = frame.channels();
  int width = frame.cols;
  int height = frame.rows;
  printf("Loaded image with dimensions %d x %d x %d\n", width, height, channels);
  width = width * channels;
  #define BOX_BLUR_ULTRA_SIZE 16
  float boxBlurUltraMask[ BOX_BLUR_ULTRA_SIZE * BOX_BLUR_ULTRA_SIZE * 3];
  createBoxBlurMask<BOX_BLUR_ULTRA_SIZE>(boxBlurUltraMask);
  
  // Get a Mat (this is the matrix that will hold a single frame)
  int matElements = width * height;
  // unsigned char* oneDFrame = new unsigned char[matElements];
  float *oneDFrame = new float[matElements];
  printf("Total Elements: %d\n", matElements);
  MatIterator_<cv::Vec3b> it, end;
  int i = 0;

  // Flatten the frame into a 1D array
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

  // Save the image before processing
  imwrite("no_conv_image.png", reconstructed);

  // Define the kernel  
  float boxBlurMask[BOX_BLUR_SIZE] = { BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, 
                                    BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, 
                                    BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0};

  #define SUPER_BOX_BLUR_COEFF 1.0/36.0

  float superBoxBlurMask[36 * 3] = {    SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0,  SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, 
                                        SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0,  SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, 
                                        SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0,  SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, 
                                        SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0,  SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, 
                                        SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0,  SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, 
                                        SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0,  SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0, SUPER_BOX_BLUR_COEFF, 0, 0
                                    };

//A bunch of random kernels Using One channel
   Npp32s kernelRidge [9] = {
     0, -1, 0,
    -1, 4, -1, 
    0, -1, 0
  };

  // Allocate GPU memory
  real_t *devInputData;
  real_t *devOutputData;
  real_t *devMask;

  htkTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&devInputData, width * height * sizeof(real_t));
  cudaMalloc((void **)&devOutputData, width * height * sizeof(real_t));
  cudaMalloc((void **)&devMask,  3 * 3 * sizeof(real_t));
  htkTime_stop(GPU, "Allocating GPU memory.");
  
  // Copy memory to the GPU
  htkTime_start(IO, "Copying memory to the GPU.");
  cudaMemcpy(devInputData, oneDFrame, width * height * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devOutputData, oneDFrame, width * height * sizeof(real_t), cudaMemcpyHostToDevice);
  htkTime_stop(IO, "Copying memory to the GPU.");

  // Call convolution function
  htkTime_start(Compute, "Doing the computation");
  // boxBlur(devOutputData, devInputData, devMask, width, height, EPSILON);
  // superBoxBlur(devOutputData, devInputData, devMask, width, height, EPSILON);
  // genericKernelFilter(devOutputData, devInputData, devMask, width, height, EPSILON, 16);
  int width3Channel = width;
  Npp32s *devMask3Channel = new Npp32s[3 * 3];
  Npp8u *devInputData3Channel;
  Npp8u *devOutputData3Channel;
  Npp8u * tempInput = new Npp8u[width * height];
  // for ( it = frame.begin<cv::Vec3b>(), end = frame.end<cv::Vec3b>(); it != end; ++it ) {

  //     //get current bgr pixels:
  //     uchar &r = (*it)[2];
  //     uchar &g = (*it)[1];
  //     uchar &b = (*it)[0];

  //     //Store them into array, as a cv::Scalar:
  //     tempInput[i] = b;
  //     tempInput[i + 1] = g;
  //     tempInput[i + 2] = r;
  //     i +=3;

  // }

  // cudaMemcpy(D, kernelRidge, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&devInputData3Channel, width * height * sizeof(Npp8u));
  cudaMalloc((void **)&devOutputData3Channel, width * height * sizeof(Npp8u));

  cvtColor(frame, frame, COLOR_BGR2RGB);
  cudaMemcpy(devMask3Channel, kernelRidge, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
  printf("IS continue%d\n", frame.isContinuous());
  cudaMemcpy(devInputData3Channel, frame.ptr<Npp8u>(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
  cudaMemcpy(devOutputData3Channel, frame.ptr<Npp8u>(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
  genericMultiChannelFilter(devOutputData3Channel, devInputData3Channel, devMask3Channel, width3Channel, height, EPSILON, 3);
  htkTime_stop(Compute, "Doing the computation");
  htkLog(TRACE, "Solution iterations: ", iterations);

  // Copy the GPU memory back to the CPU
  float *hostOutputData = new float[matElements];
  Npp8u *hostOutputData3Channel = new Npp8u[matElements];
  htkTime_start(IO, "Copying memory back to the CPU.");
  cudaMemcpy(hostOutputData, devOutputData, width * height * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutputData3Channel, devOutputData3Channel, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  
  htkTime_stop(IO, "Copying memory back to the CPU.");
  unsigned char* oneDFrameSafe = new unsigned char[matElements];

  // Convert back to unsigned char before reconstructing the image
  // for(int i = 0; i < width * height; i++){
  //     unsigned char temp = (unsigned char)hostOutputData[i];
  //     oneDFrameSafe[i] = temp;
  // }

  Mat reconstructedA(frame.rows, frame.cols, CV_8UC3, hostOutputData3Channel);

  // Save the image
  imwrite("output_image_ridge_detection_3_channel.png", reconstructedA);

  printf("Image saved as output_image.png");

  // Free the GPU memory
  htkTime_start(GPU, "Freeing GPU memory.");
  cudaFree(devInputData);
  cudaFree(devOutputData);
  cudaFree(devMask);
  htkTime_stop(GPU, "Freeing GPU memory.");

  return 0;
}








