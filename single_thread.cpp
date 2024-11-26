#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// Function to apply greyscale filter (CPU equivalent of the CUDA kernel)
void filterGreyscaleCPU(int height, int width, unsigned char *input, unsigned char *output) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;

            // Extract original color channels
            unsigned char red = input[idx];
            unsigned char green = input[idx + 1];
            unsigned char blue = input[idx + 2];

            // Apply sepia filter
            output[idx]     = min(255, (393 * red + 769 * green + 189 * blue) / 1000); // Red channel
            output[idx + 1] = min(255, (349 * red + 686 * green + 168 * blue) / 1000); // Green channel
            output[idx + 2] = min(255, (272 * red + 534 * green + 131 * blue) / 1000); // Blue channel
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./solution <video_name> <filter_name>\n");
        return -1;
    }

    String video_name = argv[1];
    String filter_name = argv[2];

    VideoCapture capture(video_name);
    if (!capture.isOpened()) {
        cerr << "Error: Unable to open the video file." << endl;
        return -1;
    }

    Mat frame;
    capture >> frame;
    if (frame.empty()) {
        cerr << "Error: Unable to read the first frame." << endl;
        return -1;
    }

    int width = frame.cols;
    int height = frame.rows;
    int channels = frame.channels();
    cout << "Loaded video with dimensions " << width << " x " << height << " x " << channels << endl;

    double fps = capture.get(CAP_PROP_FPS);
    cout << "Frames per second: " << fps << endl;

    int period_ms = 1000 / fps;

    // Buffer for the flattened image and the processed output
    int matElements = width * height * channels;
    unsigned char *inputFrame = new unsigned char[matElements];
    unsigned char *outputFrame = new unsigned char[matElements];

    if (filter_name != "greyscale") {
        cerr << "Error: Invalid filter name. Only 'greyscale' is supported." << endl;
        return -1;
    }

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
    while (true) {
        auto start_time = chrono::high_resolution_clock::now();

        // Flatten the frame into a 1D array
        auto flatten_start = chrono::high_resolution_clock::now();
        int idx = 0;
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                Vec3b pixel = frame.at<Vec3b>(i, j);
                inputFrame[idx] = pixel[0];      // Blue
                inputFrame[idx + 1] = pixel[1];  // Green
                inputFrame[idx + 2] = pixel[2];  // Red
                idx += 3;
            }
        }
        auto flatten_end = chrono::high_resolution_clock::now();
        int flatten_duration = chrono::duration_cast<chrono::microseconds>(flatten_end - flatten_start).count();
        if (flatten_duration > max_flatten_time) {
            max_flatten_time = flatten_duration;
        }
        if (flatten_duration < min_flatten_time) {
            min_flatten_time = flatten_duration;
        }
        total_flatten_time += flatten_duration;
        // printf("flatten_duration: %d us\n", flatten_duration);

        // Apply the greyscale filter
        auto compute_start = chrono::high_resolution_clock::now();
        filterGreyscaleCPU(height, width, inputFrame, outputFrame);
        auto compute_end = chrono::high_resolution_clock::now();
        int compute_duration = chrono::duration_cast<chrono::microseconds>(compute_end - compute_start).count();
        if (compute_duration > max_compute_time) {
            max_compute_time = compute_duration;
        }
        if (compute_duration < min_compute_time) {
            min_compute_time = compute_duration;
        }
        total_compute_time += compute_duration;
        // printf("compute_duration: %d us\n", compute_duration);

        // Reconstruct the processed frame
        Mat reconstructedA(frame.rows, frame.cols, CV_8UC3, outputFrame);

        // Display the frame
        // imshow("Processed Frame", reconstructedA);

        // Measure frame processing time
        auto end_time = chrono::high_resolution_clock::now();
        int duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
        if (duration > max_frame_time)
        {
            max_frame_time = duration;
        }
        if (duration < min_frame_time)
        {
            min_frame_time = duration;
        }
        total_frame_time += duration;
        // printf("Frame duration: %d us\n", duration);
        duration = duration/1000;
        int delay = max(1, period_ms - duration);

        char c = (char)waitKey(delay);
        if (c == 27) // Exit on 'ESC' key
            break;

        // cout << "Processed frame " << ++frame_num << endl;
        frame_num++;
        // printf("------------------------------------\n");

        // Read the next frame
        capture >> frame;
        if (frame.empty()) {
            cout << "End of video." << endl;
            break;
        }
    }

    printf("\nMax Frame Time: %d us\n", max_frame_time);
    printf("Min Frame Time: %d us\n", min_frame_time);
    printf("Average Frame Time: %f us\n", (float)total_frame_time / (float)frame_num);
    printf("Max Compute Time: %d us\n", max_compute_time);
    printf("Min Compute Time: %d us\n", min_compute_time);
    printf("Average Compute Time: %f us\n", (float)total_compute_time / (float)frame_num);
    printf("Max Flatten Time: %d us\n", max_flatten_time);
    printf("Min Flatten Time: %d us\n", min_flatten_time);
    printf("Average Flatten Time: %f us\n", (float)total_flatten_time / (float)frame_num);

    // Cleanup
    delete[] inputFrame;
    delete[] outputFrame;

    return 0;
}
