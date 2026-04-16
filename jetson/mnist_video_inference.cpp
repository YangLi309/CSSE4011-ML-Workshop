#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <chrono>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Skip info messages
        if (severity == Severity::kINFO) return;
        
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            case Severity::kVERBOSE: std::cerr << "VERBOSE: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

// Destroy TensorRT objects
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
            obj->destroy();
    }
};

// Signal handler for clean shutdown
bool signal_received = false;

void sig_handler(int signo)
{
    if (signo == SIGINT)
    {
        std::cout << "Received SIGINT" << std::endl;
        signal_received = true;
    }
}

// MNIST Inference class
class MNISTInference
{
private:
    Logger logger;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context;
    cudaStream_t stream;
    
    // Input and output buffer pointers
    void* buffers[2]; // Assuming one input, one output
    int inputIndex;
    int outputIndex;
    size_t inputSize;
    size_t outputSize;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;
    
    // FPS calculation
    float networkFPS;
    std::chrono::steady_clock::time_point lastTime;

public:
    MNISTInference(const std::string& engineFile) : stream(nullptr), networkFPS(0)
    {
        // Load the engine
        std::cout << "Loading TensorRT engine: " << engineFile << std::endl;
        std::ifstream file(engineFile, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Failed to open engine file: " + engineFile);
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        
        if (!file) {
            throw std::runtime_error("Failed to read engine file");
        }
        
        // Create runtime and deserialize engine
        std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        
        if (!engine) {
            throw std::runtime_error("Failed to deserialize engine");
        }
        
        // Create execution context
        context.reset(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }
        
        // Create CUDA stream
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
        
        // Find input and output binding indices and allocate memory
        inputIndex = -1;
        outputIndex = -1;
        
        for (int i = 0; i < engine->getNbBindings(); i++) {
            if (engine->bindingIsInput(i)) {
                inputIndex = i;
                inputDims = engine->getBindingDimensions(i);
            } else {
                outputIndex = i;
                outputDims = engine->getBindingDimensions(i);
            }
        }
        
        if (inputIndex == -1 || outputIndex == -1) {
            throw std::runtime_error("Could not find input or output binding");
        }
        
        // Calculate sizes and allocate memory
        inputSize = 1;
        for (int i = 0; i < inputDims.nbDims; i++) {
            inputSize *= inputDims.d[i];
        }
        inputSize *= sizeof(float);
        
        outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; i++) {
            outputSize *= outputDims.d[i];
        }
        outputSize *= sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&buffers[inputIndex], inputSize);
        cudaMalloc(&buffers[outputIndex], outputSize);
        
        std::cout << "TensorRT engine loaded successfully" << std::endl;
        std::cout << "Input shape: ";
        for (int i = 0; i < inputDims.nbDims; i++) {
            std::cout << inputDims.d[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Output shape: ";
        for (int i = 0; i < outputDims.nbDims; i++) {
            std::cout << outputDims.d[i] << " ";
        }
        std::cout << std::endl;
        
        // Initialize FPS timer
        lastTime = std::chrono::steady_clock::now();
    }
    
    ~MNISTInference()
    {
        // Free allocated resources
        if (buffers[inputIndex]) cudaFree(buffers[inputIndex]);
        if (buffers[outputIndex]) cudaFree(buffers[outputIndex]);
        if (stream) cudaStreamDestroy(stream);
    }
    
    // Process a single image
    int processImage(const cv::Mat& image, float* confidence = nullptr)
    {
        // Start timing for FPS calculation
        auto startTime = std::chrono::steady_clock::now();
        
        // Get dimensions
        const int batchSize = inputDims.d[0];
        const int channels = inputDims.d[1];
        const int modelHeight = inputDims.d[2];
        const int modelWidth = inputDims.d[3];
        
        // Resize and preprocess image
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(modelWidth, modelHeight));
        
        // Convert to grayscale if needed
        cv::Mat gray;
        if (channels == 1 && resized.channels() == 3) {
            cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        } else if (channels == 3 && resized.channels() == 1) {
            cv::cvtColor(resized, gray, cv::COLOR_GRAY2BGR);
        } else {
            gray = resized;
        }
        
        // Normalize to [0,1]
        cv::Mat normalized;
        gray.convertTo(normalized, CV_32F, 1.0f/255.0f);
        
        // Allocate host memory for input
        std::vector<float> inputData(inputSize / sizeof(float));
        
        // Copy data to input buffer (assuming NCHW format)
        if (channels == 1) {
            // For grayscale, just copy the data
            for (int h = 0; h < modelHeight; h++) {
                for (int w = 0; w < modelWidth; w++) {
                    inputData[h * modelWidth + w] = normalized.at<float>(h, w);
                }
            }
        } else {
            // For color, copy each channel
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < modelHeight; h++) {
                    for (int w = 0; w < modelWidth; w++) {
                        inputData[(c * modelHeight * modelWidth) + (h * modelWidth) + w] = 
                            normalized.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
        }
        
        // Copy input data to GPU
        cudaMemcpy(buffers[inputIndex], inputData.data(), inputSize, cudaMemcpyHostToDevice);
        
        // Execute inference
        if (!context->enqueueV2(buffers, stream, nullptr)) {
            throw std::runtime_error("Failed to execute inference");
        }
        
        // Allocate host memory for output
        std::vector<float> outputData(outputSize / sizeof(float));
        
        // Copy output back to host
        cudaMemcpy(outputData.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);
        
        // Synchronize stream
        cudaStreamSynchronize(stream);
        
        // Find the class with highest confidence (MNIST has 10 classes, 0-9)
        int numClasses = outputDims.d[1];  // Assuming output shape is [batch_size, num_classes]
        auto maxElement = std::max_element(outputData.begin(), outputData.begin() + numClasses);
        int classId = std::distance(outputData.begin(), maxElement);
        
        // Set confidence value if requested
        if (confidence != nullptr) {
            *confidence = *maxElement;
        }
        
        // Update FPS calculation
        auto endTime = std::chrono::steady_clock::now();
        
        // Calculate time in ms
        float ms = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        
        // Update running average of network time
        networkFPS = 1000.0f / ms;
        
        return classId;
    }
    
    float GetNetworkFPS() const { return networkFPS; }
};

void printUsage()
{
    std::cout << "Usage: mnist_video_inference <engine_file> <camera_id or video_file>" << std::endl;
    std::cout << "  engine_file: Path to TensorRT engine file" << std::endl;
    std::cout << "  camera_id: Camera device ID (e.g., 0 for default camera)" << std::endl;
    std::cout << "  video_file: Path to video file (if using a file instead of camera)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  mnist_video_inference alexnet_mnist.engine 0" << std::endl;
    std::cout << "  mnist_video_inference alexnet_mnist.engine /path/to/video.mp4" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printUsage();
        return 1;
    }
    
    // Parse command line
    std::string engineFile = argv[1];
    std::string videoSource = argv[2];
    
    // Attach signal handler
    if (signal(SIGINT, sig_handler) == SIG_ERR) {
        std::cerr << "Can't catch SIGINT" << std::endl;
        return 1;
    }
    
    // Create video capture
    cv::VideoCapture cap;
    
    // Try to open the video source as a number (camera index)
    try {
        int cameraIndex = std::stoi(videoSource);
        if (!cap.open(cameraIndex)) {
            std::cerr << "Failed to open camera device " << cameraIndex << std::endl;
            return 1;
        }
        std::cout << "Opened camera device " << cameraIndex << std::endl;
    } catch (const std::invalid_argument&) {
        // If not a number, treat as a file path
        if (!cap.open(videoSource)) {
            std::cerr << "Failed to open video file: " << videoSource << std::endl;
            return 1;
        }
        std::cout << "Opened video file: " << videoSource << std::endl;
    }
    
    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Video dimensions: " << width << "x" << height << std::endl;
    
    try {
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Create MNIST inference object
        MNISTInference inference(engineFile);
        
        // Create window for display
        cv::namedWindow("MNIST Inference", cv::WINDOW_NORMAL);
        
        // Processing loop
        cv::Mat frame;
        
        while (!signal_received) {
            // Capture next frame
            if (!cap.read(frame)) {
                std::cout << "End of video stream" << std::endl;
                break;
            }
            
            if (frame.empty()) {
                std::cerr << "Empty frame received" << std::endl;
                continue;
            }
            
            // Run inference on the frame
            float confidence = 0.0f;
            int digit = inference.processImage(frame, &confidence);
            
            // Convert confidence to percentage
            confidence *= 100.0f;
            
            // Draw the recognized digit and confidence
            char str[100];
            sprintf(str, "Digit: %d (%.2f%%)", digit, confidence);
            cv::putText(frame, str, cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            
            // Display FPS
            sprintf(str, "FPS: %.1f", inference.GetNetworkFPS());
            cv::putText(frame, str, cv::Point(10, 60), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Display the frame
            cv::imshow("MNIST Inference", frame);
            
            // Check for keyboard input (press 'q' or ESC to quit)
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) { // 'q' or ESC
                break;
            }
        }
        
        // Cleanup
        cap.release();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Shutdown complete" << std::endl;
    return 0;
} 