  // Variables for the convolution
#include "htk.h"
#include <nppi.h>
#include <chrono>

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

#define VIDEO_FILE "video_2.mp4"
#define BOX_BLUR_SIZE 3 * 3 * 3
#define BOX_BLUR_COEFF float(1.0 / 9.0)

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
        // output[idx] = red;
        // output[idx + 1] = green;
        // output[idx + 2] = blue;
    }
};

   Npp32s ridgeFilter  [3 * 3] = {
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

Npp32s blurFilter [ 9 ] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
};

Npp32s ridgeFilter2 [ 9 ] = {
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1
};
Npp32s superblurFilter [ 81] = {};

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
    Npp32s *sepiaMask;
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
    cudaMalloc((void **)&sepiaMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&blurMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&sharpenMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&embossMask, 3 * 3 * sizeof(Npp32s));
    cudaMalloc((void **)&gaussMask, 5 * 5 * sizeof(Npp32s));
    cudaMalloc((void **)&superblurMask, 9 * 9 * sizeof(Npp32s));

    htkTime_stop(GPU, "Allocating GPU memory.");


    // cudaMemcpy(devMask, kernelRidge, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(sepiaMask, sepiaFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(sobelMask, sobelFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(ridgeMask, ridgeFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(ridgeMask2, ridgeFilter2, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);

    cudaMemcpy(sharpenMask, sharpenFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(embossMask, embossFilter, 3 * 3 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(gaussMask, gaussFilter, 5 * 5 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(blurMask, blurFilter, 3 * 3* sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(superblurMask, superblurFilter, 9 * 9 * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy(devOutputData, oneDFrame, width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);



    int frame_num = 0;
    while (1) {
        // htkTime_start(IO, "Processing Total Frame");
        auto start_time = std::chrono::high_resolution_clock::now();
        htkTime_start(IO, "Flattening Frame\n");
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

        htkTime_stop(IO, "Flattening Frame\n");

        htkTime_start(Copy, "Copying input memory to the GPU.");
        cudaMemcpy(devInputData, oneDFrame, width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
        htkTime_stop(Copy, "Copying input memory to the GPU.");

        htkTime_start(GPU, "Computing New Frame\n");
        // Copy memory to the GPU

        #define DIMENSIONS 16

        
        if (filter_name == "greyscale") {
            dim3 dimGrid(ceil(width/(float)DIMENSIONS), ceil(height/3/(float)DIMENSIONS), 1);
            dim3 dimBlock(DIMENSIONS, DIMENSIONS, 1);
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
        htkTime_stop(GPU, "Computing New Frame\n");
        
        // Copy the GPU memory back to the CPU
        Npp8u* hostOutputData = new Npp8u[matElements];
        // htkTime_start(IO, "Copying memory back to the CPU.");
        htkTime_start(Copy, "Copying image back to CPU.");
        cudaMemcpy(hostOutputData, devOutputData, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        htkTime_stop(Copy, "Copying image back to CPU.");

        
        // htkTime_stop(IO, "Copying memory back to the CPU.");

        Mat reconstructedA(frame.rows, frame.cols, CV_8UC3, hostOutputData);

        capture >> frame;
        frame_num++;
        // htkTime_stop(IO, "Show and Load new frame\n");


        // Save the image
        // htkTime_stop(IO, "Processing Total Frame");
        // htkTime_start(IO, "Show and Load new frame\n");
        imshow("Frame", reconstructedA);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        printf("duration: %d\n", int(duration.count()));
        int duration_int = int(duration.count());
        if(duration_int > period_ms - 1)
            duration_int = period_ms - 1;    

        char c = (char)waitKey(period_ms  - duration_int);
        if (c == 27)
            break;

        printf("Processed frame %d\n", frame_num);
        
        

        if (frame.empty())
        {
            printf("Error: Unable to open the frame file.");
            break;
        }

    }

    // Free the GPU memory
    htkTime_start(GPU, "Freeing GPU memory.");
    cudaFree(devInputData);
    cudaFree(devOutputData);
    cudaFree(devMask);
    cudaFree(sobelMask);
    cudaFree(sepiaMask);
    cudaFree(blurMask);
    cudaFree(sharpenMask);
    cudaFree(embossMask);
    cudaFree(gaussMask);
    cudaFree(superblurMask);
    htkTime_stop(GPU, "Freeing GPU memory.");

    return 0;
}








