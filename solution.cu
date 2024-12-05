  // Variables for the convolution
#include "htk.h"
#include <nppi.h>
#include <chrono>
#include "filter.h"

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

int iterations;

#define RGB_TYPE float

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FILTER FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static void generic3ChannelFilter(Npp8u *out, Npp8u *in,  Npp32s *mask, int width, int height, real_t epsilon, int maskWidth, int nDivisor) {
    Npp8u *u = in;
    Npp8u *w = out;
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

}


// __global__ void filterGreyscale(int height, int width, Npp8u *input, Npp8u *output){
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
//   if(i < height && j < width){
//     //If Red pixel
//     if(j % 3 == 0){
//       output[i * width + j] = .299 * input[i * width + j];
//     }
//     //If Green Pixel
//     else if(j % 3 == 1){
//       output[i * width + j] = .587 * input[i * width + j];
//     }
//     //If Blue Pixel
//     else{
//       output[i * width + j] = .114 * input[i * width + j];
//     }
//     // output[i * width + j] = input[i * width + j];
//   }
// };  
__global__ void filterGreyscale(int height, int width, Npp8u *input, Npp8u *output) {
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
    }
};

int main(int argc, char *argv[]) {

    std::fill(superblurFilter, superblurFilter + 81, 1); // Fill with 1

    String video_name;
    String filter_name;


    if (argc != 3) {
        printf("Usage: ./solution <video_name> <filter_name>\n");
        return -1;
    } else {
        video_name = argv[1];
        filter_name = argv[2];
    }

    VideoCapture capture;
        
    capture.open(video_name.c_str());
    if (!capture.isOpened()) {
        printf("Error: Unable to open the video file.");
        return -1;
    }  

    Mat frame;
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
    
    double fps = capture.get(cv::CAP_PROP_FPS);    
    printf("Frames per second %f\n", fps);

    //Calculate period
    int period_ms =  1000 /fps;
     

    width = width * 3;
    
    int matElements = width * height;

    // Get a Mat (this is the matrix that will hold a single frame)
    // unsigned char* oneDFrame = new unsigned char[matElements];
    Npp8u *oneDFrame = new Npp8u[matElements];
    // unsigned char *oneDFrame = new unsigned char[matElements];
    printf("Total Elements: %d\n", matElements);

    if (filter_name == "greyscale") {
        printf("Applying greyscale filter\n");
    } else if (filter_name == "sobel") {
        printf("Applying sobel filter\n");
    } else if (filter_name == "ridge") {
        printf("Applying ridge filter\n");
    } else if (filter_name == "ridge2") {
        printf("Applying ridge2 filter\n");
    } else if (filter_name == "sharpen") {
        printf("Applying sharpen filter\n");
    } else if (filter_name == "emboss") {
        printf("Applying emboss filter\n");
    } else if (filter_name == "gauss") {
        printf("Applying gauss filter\n");
    } else if (filter_name == "blur") {
        printf("Applying blur filter\n");
    } else if (filter_name == "super_blur") {
        printf("Applying super blur filter\n");
    } else {
        printf("Error: Invalid filter name.\n");
        return -1;
    }

    // Allocate GPU memory
    Npp8u *devInputData;
    Npp8u *devOutputData;
    Npp32s *devMask;
    Npp32s *blurMask;
    Npp32s *greyscaleMask;
    Npp32s *sobelMask;
    Npp32s *sharpenMask;
    Npp32s *embossMask;
    Npp32s *gaussMask;
    Npp32s *superblurMask;
    Npp32s *ridgeMask;
    Npp32s *ridgeMask2;

    htkTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **)&devInputData, width * height * sizeof(Npp8u));
    cudaMalloc((void **)&devOutputData, width * height * sizeof(Npp8u));
    cudaMalloc((void **)&devMask,  3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&ridgeMask,  3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&ridgeMask2,  3 * 3 * sizeof(Npp32s));

    cudaMalloc((void **)&sobelMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&greyscaleMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&blurMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&sharpenMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&embossMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&gaussMask, 5 * 5 * sizeof(Npp32s));
    cudaMalloc((void **)&superblurMask, 9 * 9 * sizeof(Npp32s));

    htkTime_stop(GPU, "Allocating GPU memory.");


    // cudaMemcpy(devMask, kernelRidge, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(greyscaleMask, sepiaFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(sobelMask, sobelFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(ridgeMask, ridgeFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(ridgeMask2, ridgeFilter2, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);

    cudaMemcpy(sharpenMask, sharpenFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(embossMask, embossFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(gaussMask, gaussFilter, 5 * 5 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(blurMask, blurFilter, 3 * 3* sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(superblurMask, superblurFilter, 9 * 9 * sizeof(Npp32s), cudaMemcpyHostToDevice);

    Npp8u *hostOutputData = new Npp8u[matElements];

    int max_frame_time = 0;
    int min_frame_time = 1000000;
    int total_frame_time = 0;

    int max_compute_time = 0;
    int min_compute_time = 1000000;
    int total_compute_time = 0;

    int max_flatten_time = 0;
    int min_flatten_time = 1000000;
    int total_flatten_time = 0;

    int frame_num = 0;
    while (1) {
        // htkTime_start(IO, "Processing Total Frame");
        auto start_time = std::chrono::high_resolution_clock::now();
        // htkTime_start(IO, "Flattening Frame\n");
        MatIterator_<cv::Vec3b> it, end;

        auto flatten_start = std::chrono::high_resolution_clock::now();
        // Flatten the frame into a 1D array
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                int idx = (i * frame.cols + j) * 3; // Compute index based on row and column
                Vec3b pixel = frame.at<Vec3b>(i, j);
                oneDFrame[idx] = pixel[0];      // Blue
                oneDFrame[idx + 1] = pixel[1];  // Green
                oneDFrame[idx + 2] = pixel[2];  // Red
            }
        }
        auto flatten_end = std::chrono::high_resolution_clock::now();
        int flatten_duration = std::chrono::duration_cast<std::chrono::microseconds>(flatten_end - flatten_start).count();
        if (flatten_duration > max_flatten_time) {
            max_flatten_time = flatten_duration;
        }
        if (flatten_duration < min_flatten_time) {
            min_flatten_time = flatten_duration;
        }
        total_flatten_time += flatten_duration;

        auto copy1_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(devInputData, oneDFrame, width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
        auto copy1_end = std::chrono::high_resolution_clock::now();
        int copy1_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy1_end - copy1_start).count();

        #define GREYSCALE_KERNEL_DIM 16

        auto comput_start = std::chrono::high_resolution_clock::now();
        if (filter_name == "greyscale") {
            dim3 dimGrid(ceil(width/(float)GREYSCALE_KERNEL_DIM), ceil(height/3/(float)GREYSCALE_KERNEL_DIM), 1);
            dim3 dimBlock(GREYSCALE_KERNEL_DIM, GREYSCALE_KERNEL_DIM, 1);
            filterGreyscale<<<dimGrid, dimBlock>>>(height, width, devInputData, devOutputData);
        } else if (filter_name == "sobel") {
            generic3ChannelFilter(devOutputData, devInputData, sobelMask, width, height, EPSILON, 3, 1);
        } else if (filter_name == "ridge") {
            generic3ChannelFilter(devOutputData, devInputData, ridgeMask, width, height, EPSILON, 3, 1);
        } else if (filter_name == "ridge2") {
            generic3ChannelFilter(devOutputData, devInputData, ridgeMask2, width, height, EPSILON, 3, 1);
        } else if (filter_name == "sharpen") {
            generic3ChannelFilter(devOutputData, devInputData, sharpenMask, width, height, EPSILON, 3, 1);
        } else if (filter_name == "emboss") {
            generic3ChannelFilter(devOutputData, devInputData, embossMask, width, height, EPSILON, 3, 1);
        } else if (filter_name == "gauss") {
            generic3ChannelFilter(devOutputData, devInputData, gaussMask, width, height, EPSILON, 5, 273);
        } else if (filter_name == "blur") {
            generic3ChannelFilter(devOutputData, devInputData, blurMask, width, height, EPSILON, 3, 9);
        } else if (filter_name == "super_blur") {
            generic3ChannelFilter(devOutputData, devInputData, superblurMask, width, height, EPSILON, 9, 81);
        } else {
            printf("Error: Invalid filter name.");
            return -1;
        }
        auto compute_end = std::chrono::high_resolution_clock::now();
        int compute_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - comput_start).count();

        if (compute_duration > max_compute_time) {
            max_compute_time = compute_duration;
        }
        if (compute_duration < min_compute_time) {
            min_compute_time = compute_duration;
        }
        total_compute_time += compute_duration;
        
        // Copy the GPU memory back to the CPU
        auto copy2_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(hostOutputData, devOutputData, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        auto copy2_end = std::chrono::high_resolution_clock::now();
        int copy2_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy2_end - copy2_start).count();

        
        auto reconstruct_start = std::chrono::high_resolution_clock::now();
        Mat reconstructedA(frame.rows, frame.cols, CV_8UC3, hostOutputData);
        auto reconstruct_end = std::chrono::high_resolution_clock::now();
        int reconstruct_duration = std::chrono::duration_cast<std::chrono::microseconds>(reconstruct_end - reconstruct_start).count();

        auto capture_start = std::chrono::high_resolution_clock::now();
        capture >> frame;
        frame_num++;
        auto capture_end = std::chrono::high_resolution_clock::now();
        int capture_duration = std::chrono::duration_cast<std::chrono::microseconds>(capture_end - capture_start).count();

        imshow("Frame", reconstructedA);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        int duration_int = int(duration.count()/1000);
        if(duration_int > period_ms - 1)
            duration_int = period_ms - 1;    

        char c = (char)waitKey(period_ms  - duration_int);
        if (c == 27)
            break;


        if (duration.count() > max_frame_time)
        {
            max_frame_time = duration.count();
        }
        if (duration.count() < min_frame_time)
        {
            min_frame_time = duration.count();
        }
        total_frame_time += duration.count();

        if (frame.empty())
        {
            printf("Final Frame\n");
            break;
        }

    }

    printf("----------------------- Stats -----------------------");
    printf("Max Frame Time: %d us\n", max_frame_time);
    printf("Min Frame Time: %d us\n", min_frame_time);
    printf("Average Frame Time: %f us\n", total_frame_time / (float)frame_num);
    printf("Max Compute Time: %d us\n", max_compute_time);
    printf("Min Compute Time: %d us\n", min_compute_time);
    printf("Average Compute Time: %f us\n", total_compute_time / (float)frame_num);
    printf("Max Flatten Time: %d us\n", max_flatten_time);
    printf("Min Flatten Time: %d us\n", min_flatten_time);
    printf("Average Flatten Time: %f us\n", total_flatten_time / (float)frame_num);

    // Free the GPU memory
    htkTime_start(GPU, "Freeing GPU memory.");
    cudaFree(devInputData);
    cudaFree(devOutputData);
    cudaFree(devMask);
    cudaFree(sobelMask);
    cudaFree(greyscaleMask);
    cudaFree(blurMask);
    cudaFree(sharpenMask);
    cudaFree(embossMask);
    cudaFree(gaussMask);
    cudaFree(superblurMask);
    htkTime_stop(GPU, "Freeing GPU memory.");

    return 0;
}