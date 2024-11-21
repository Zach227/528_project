  // Variables for the convolution
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

static void genericOneChannelFilter(Npp8u *out, Npp8u *in,  Npp32s *mask, int width, int height, real_t epsilon, int maskWidth) {
    Npp8u *u = in;
    Npp8u *w = out;
  // width = width * 3;
    printf("Width: %d\n", width);
  // Variables for the convolution
    int conv_step = width * sizeof(Npp8u);
    NppiSize oSrcSize = {width, height};
  NppiPoint oSrcOffset = {0, 0};
  NppiSize convROI = {width , height};
  NppiSize oKernelSize = {maskWidth * 3, maskWidth};
  NppiPoint oAnchor = {0, 0};
  NppiBorderType eborderType = NPP_BORDER_REPLICATE;

  NppStatus convStatus = nppiFilterBorder_8u_C1R(
                              u , // *pSrc
                              conv_step, // nSrcStep
                              oSrcSize, // oSrcSize
                              oSrcOffset, // oSrcOffset
                              w  , // *pDst
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

static void test3ChannelFilter(Npp8u *out, Npp8u *in,  Npp32s *mask, int width, int height, real_t epsilon, int maskWidth, int nDivisor) {
    Npp8u *u = in;
    Npp8u *w = out;
  // width = width * 3;
    printf("Width: %d\n", width);
  // Variables for the convolution
    int conv_step = width * sizeof(Npp8u);
    NppiSize oSrcSize = {width / 3, height};
  NppiPoint oSrcOffset = {1, 1};
  NppiSize convROI = {width /3, height};
  NppiSize oKernelSize = {maskWidth , maskWidth};
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
                              nDivisor, // nDivisor
                              eborderType // eBorderType
                              );

  printf("STATUS %d\n", convStatus);

}


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
  

__global__ void filterGreyscale(int height, int width, Npp8u *input, Npp8u *output){
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < height && j < width){
    //If Red pixel
    if(j % 3 == 0){
      output[i * width + j] = .299 * input[i * width + j];
    }
    //If Green Pixel
    else if(j % 3 == 1){
      output[i * width + j] = .587 * input[i * width + j];
    }
    //If Blue Pixel
    else{
      output[i * width + j] = .114 * input[i * width + j];
    }
    // output[i * width + j] = input[i * width + j];
  }
};  
__global__ void filterSepia(int height, int width, Npp8u *input, Npp8u *output) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    
    if (i < height && j < width) {
        int idx = (i * width + j) * 3; // Index of the current pixel in the interleaved RGB array

        // Extract original color channels
        Npp8u red = input[idx];
        Npp8u green = input[idx + 1];
        Npp8u blue = input[idx + 2];

        // Apply sepia filter
        output[idx]     = Npp8u(min(255, int(393 * red + 769 * green + 189 * blue)/1000)); // Red channel
        output[idx + 1] = Npp8u(min(255, int(349 * red + 686 * green + 168 * blue)/1000)); // Green channel
        output[idx + 2] = Npp8u(min(255, int(272 * red + 534 * green + 131 * blue)/1000)); // Blue channel
        // output[idx] = red;
        // output[idx + 1] = green;
        // output[idx + 2] = blue;
    }
};
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
  width = width * 3;
  #define BOX_BLUR_ULTRA_SIZE 16
  float boxBlurUltraMask[ BOX_BLUR_ULTRA_SIZE * BOX_BLUR_ULTRA_SIZE * 3];
  createBoxBlurMask<BOX_BLUR_ULTRA_SIZE>(boxBlurUltraMask);
  
  // Get a Mat (this is the matrix that will hold a single frame)
  int matElements = width * height;
  // unsigned char* oneDFrame = new unsigned char[matElements];
  Npp8u *oneDFrame = new Npp8u[matElements];
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

// //A bunch of random kernels Using One channel
//    Npp32s kernelRidge  [9 * 3] = {
//     0, 0, 0, -1, 0, 0, 0, 0, 0,
//     -1, 0, 0, 4, 0, 0, -1, 0, 0, 
//     0, 0, 0, -1, 0, 0, 0, 0, 0
//   };
   Npp32s kernelRidge  [3 * 3] = {
    0, -1, 0,
    -1, 4, -1, 
    0, -1, 0  
    };
  Npp32s sepiaFilter [3*3] = {
    101, 198, 48,  // 0.393 * 256, 0.769 * 256, 0.189 * 256
    89, 176, 43,   // 0.349 * 256, 0.686 * 256, 0.168 * 256
    69, 137, 33    // 0.272 * 256, 0.534 * 256, 0.131 * 256
};

Npp32s sobelFilter [9] = {
    1, 0, -1,
    2, 0, -2,
    1, 0, -1
};

Npp32s sharpenFilter [9] = {
  0, -1, 0,
  -1, 5, -1,
  0, -1, 0
};

Npp32s embossFilter [9] = {
  -2, -1, 0,
  -1, 1, 1,
  0, 1, 2
};

Npp32s gaussFilter [25] = {
  1, 4, 7, 4, 1,
  4, 16, 26, 16, 4,
  7, 26, 41, 26, 7,
  4, 16, 26, 16, 4,
  1, 4, 7, 4, 1


};


  // Allocate GPU memory
  Npp8u *devInputData;
  Npp8u *devOutputData;
  Npp32s *devMask;
  Npp32s *sepiaMask;
  Npp32s *sobelMask;
  Npp32s *sharpenMask;
  Npp32s *embossMask;
  Npp32s *gaussMask;

  htkTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&devInputData, width * height * sizeof(Npp8u));
  cudaMalloc((void **)&devOutputData, width * height * sizeof(Npp8u));
  cudaMalloc((void **)&devMask,  3 * 3 * sizeof(Npp32s));
  cudaMalloc((void **)&sobelMask, 3 * 3 * sizeof(Npp32s));
  cudaMalloc((void **)&sepiaMask, 3 * 3 * sizeof(Npp32s));
  cudaMalloc((void **)&sharpenMask, 3 * 3 * sizeof(Npp32s));
  cudaMalloc((void **)&embossMask, 3 * 3 * sizeof(Npp32s));
  cudaMalloc((void **)&gaussMask, 5 * 5 * sizeof(Npp32s));
  htkTime_stop(GPU, "Allocating GPU memory.");
  
  // Copy memory to the GPU
  htkTime_start(IO, "Copying memory to the GPU.");
  cudaMemcpy(devInputData, oneDFrame, width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
  cudaMemcpy(devOutputData, oneDFrame, width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
  cudaMemcpy(devMask, kernelRidge, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(sepiaMask, sepiaFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(sobelMask, sobelFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(sharpenMask, sharpenFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(embossMask, embossFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(gaussMask, gaussFilter, 5 * 5 * sizeof(Npp32s), cudaMemcpyHostToDevice);
  

  // test3ChannelFilter(devOutputData, devInputData, devMask, width, height, EPSILON, 3, 1);
  // test3ChannelFilter(devOutputData, devInputData, sepiaMask, width, height, EPSILON, 3, 256);
  // test3ChannelFilter(devOutputData, devInputData, sobelMask, width, height, EPSILON, 3, 1);
  // test3ChannelFilter(devOutputData, devInputData, sharpenMask, width, height, EPSILON, 3, 1);
  // test3ChannelFilter(devOutputData, devInputData, embossMask, width, height, EPSILON, 3, 1);
  test3ChannelFilter(devOutputData, devInputData, gaussMask, width, height, EPSILON, 5, 273);
  
    #define DIMENSIONS 16
    
    dim3 dimGrid(ceil(width/(float)DIMENSIONS), ceil(height/3/(float)DIMENSIONS), 1);
    dim3 dimBlock(DIMENSIONS, DIMENSIONS, 1);

    // filterGreyscale<<<dimGrid, dimBlock>>>(height, width, devInputData, devOutputData);
    // filterSepia<<<dimGrid, dimBlock>>>(height, width, devInputData, devOutputData);
    
  htkTime_stop(IO, "Copying memory to the GPU.");

  // Call convolution function
  htkTime_start(Compute, "Doing the computation");
  htkTime_stop(Compute, "Doing the computation");
  htkLog(TRACE, "Solution iterations: ", iterations);

  // Copy the GPU memory back to the CPU
  Npp8u* hostOutputData = new Npp8u[matElements];
  // Npp8u *hostOutputData3Channel = new Npp8u[matElements];
  htkTime_start(IO, "Copying memory back to the CPU.");
  cudaMemcpy(hostOutputData, devOutputData, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
//   cudaMemcpy(hostOutputData3Channel, devOutputData3Channel, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  
  htkTime_stop(IO, "Copying memory back to the CPU.");
//   unsigned char* oneDFrameSafe = new unsigned char[matElements];

  // Convert back to unsigned char before reconstructing the image
  // for(int i = 0; i < width * height; i++){
  //     unsigned char temp = (unsigned char)hostOutputData[i];
  //     oneDFrameSafe[i] = temp;
  // }
  //Print out first 9 pixels of input and output
  for(int i = 0; i < 9; i++){
    printf("Input: %d, Output: %d\n", oneDFrame[i], hostOutputData[i]);
  }


  Mat reconstructedA(frame.rows, frame.cols, CV_8UC3, hostOutputData);

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








