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

static void hot_plate(real_t *out, real_t *in, real_t *mask, int width, int height, real_t epsilon)
{
    real_t *u = in;
    real_t *w = out;

    // Variables for the convolution
    int conv_step = width * sizeof(RGB_TYPE);
    NppiSize oSrcSize = {width, height};
    NppiPoint oSrcOffset = {1, 1};
    NppiSize convROI = {width - 2, height - 2};
    NppiSize oKernelSize = {9, 3};
    NppiPoint oAnchor = {1, 1};
    NppiBorderType eborderType = NPP_BORDER_REPLICATE;

    NppStatus convStatus = nppiFilterBorder_32f_C1R(
        u + width + 1, // *pSrc
        conv_step,     // nSrcStep
        oSrcSize,      // oSrcSize
        oSrcOffset,    // oSrcOffset
        w + width + 1, // *pDst
        conv_step,     // nDstStep
        convROI,       // oSizeROI
        mask,          // *pKernel
        oKernelSize,   // oKernelSize
        oAnchor,       // oAnchor
        eborderType    // eBorderType
    );

    printf("STATUS %d\n", convStatus);
}

int main(int argc, char *argv[])
{
    VideoCapture capture;

    capture.open(VIDEO_FILE);
    if (!capture.isOpened())
    {
        printf("Error: Unable to open the video file.");
        return -1;
    }

    Mat frame;
    capture >> frame;
    if (frame.empty())
    {
        printf("Error: Unable to open the frame file.");
        return -1;
    }

    // Set width and height
    int channels = frame.channels();
    int width = frame.cols;
    int height = frame.rows;
    printf("Loaded image with dimensions %d x %d x %d\n", width, height, channels);
    width = width * channels;

    // Define the kernel
    float boxBlur[BOX_BLUR_SIZE] = {BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0,
                                    BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0,
                                    BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0, BOX_BLUR_COEFF, 0, 0};

    // Allocate GPU memory
    real_t *devInputData;
    real_t *devOutputData;
    real_t *devMask;

    htkTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **)&devInputData, width * height * sizeof(real_t));
    cudaMalloc((void **)&devOutputData, width * height * sizeof(real_t));
    cudaMalloc((void **)&devMask, 9 * 3 * sizeof(real_t));
    htkTime_stop(GPU, "Allocating GPU memory.");

    // Copy the mask to the GPU
    cudaMemcpy(devMask, boxBlur, 9 * 3 * sizeof(real_t), cudaMemcpyHostToDevice);
    int frame_num = 1;
    while (1)
    {
        // Get a Mat (this is the matrix that will hold a single frame)
        int matElements = width * height;
        float *oneDFrame = new float[matElements];
        printf("Total Elements: %d\n", matElements);
        MatIterator_<cv::Vec3b> it, end;
        int i = 0;

        // Flatten the frame into a 1D array
        for (it = frame.begin<cv::Vec3b>(), end = frame.end<cv::Vec3b>(); it != end; ++it)
        {
            // get current bgr pixels:
            uchar &r = (*it)[2];
            uchar &g = (*it)[1];
            uchar &b = (*it)[0];

            // Store them into array, as a cv::Scalar:
            oneDFrame[i] = b;
            oneDFrame[i + 1] = g;
            oneDFrame[i + 2] = r;
            i += 3;
        }

        // Copy memory to the GPU
        htkTime_start(IO, "Copying memory to the GPU.");
        cudaMemcpy(devInputData, oneDFrame, width * height * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(devOutputData, oneDFrame, width * height * sizeof(real_t), cudaMemcpyHostToDevice);
        htkTime_stop(IO, "Copying memory to the GPU.");

        // Call convolution function
        htkTime_start(Compute, "Doing the computation");
        hot_plate(devOutputData, devInputData, devMask, width, height, EPSILON);
        htkTime_stop(Compute, "Doing the computation");
        htkLog(TRACE, "Solution iterations: ", iterations);

        // Copy the GPU memory back to the CPU
        float *hostOutputData = new float[matElements];
        htkTime_start(IO, "Copying memory back to the CPU.");
        cudaMemcpy(hostOutputData, devOutputData, width * height * sizeof(real_t), cudaMemcpyDeviceToHost);
        htkTime_stop(IO, "Copying memory back to the CPU.");
        unsigned char *oneDFrameSafe = new unsigned char[matElements];

        // Convert back to unsigned char before reconstructing the image
        for (int i = 0; i < width * height; i++)
        {
            unsigned char temp = (unsigned char)hostOutputData[i];
            oneDFrameSafe[i] = temp;
        }

        Mat reconstructedA(frame.rows, frame.cols, CV_8UC3, oneDFrameSafe);

        // imshow("Frame", reconstructedA);

        // char c = (char)waitKey(25);
        // if (c == 27)
        //     break;

        printf("Processed frame %d\n", frame_num);

        capture >> frame;
        frame_num++;
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
    htkTime_stop(GPU, "Freeing GPU memory.");

    // When everything done, release the video capture object
    capture.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}