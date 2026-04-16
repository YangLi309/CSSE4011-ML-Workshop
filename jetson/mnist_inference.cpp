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
#include <dirent.h>

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

public:
    MNISTInference(const std::string& engineFile) : stream(nullptr)
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
    }
    
    ~MNISTInference()
    {
        // Free allocated resources
        if (buffers[inputIndex]) cudaFree(buffers[inputIndex]);
        if (buffers[outputIndex]) cudaFree(buffers[outputIndex]);
        if (stream) cudaStreamDestroy(stream);
    }
    
    // Process a single image
    int processImage(const cv::Mat& image)
    {
        // Get dimensions
        const int batchSize = inputDims.d[0];
        const int channels = inputDims.d[1];
        const int height = inputDims.d[2];
        const int width = inputDims.d[3];
        
        // Resize and preprocess image
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height));
        
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
        gray.convertTo(normalized, CV_32F, 1.0/255.0);
        
        // Allocate host memory for input
        std::vector<float> inputData(inputSize / sizeof(float));
        
        // Copy data to input buffer (assuming NCHW format)
        if (channels == 1) {
            // For grayscale, just copy the data
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    inputData[h * width + w] = normalized.at<float>(h, w);
                }
            }
        } else {
            // For color, copy each channel
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        inputData[(c * height * width) + (h * width) + w] = 
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
        int classId = std::distance(outputData.begin(), 
                                   std::max_element(outputData.begin(), outputData.begin() + numClasses));
        
        return classId;
    }
};

// Utility function to list image files in a directory
std::vector<std::string> listImageFiles(const std::string& directory)
{
    std::vector<std::string> files;
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        std::cerr << "Failed to open directory: " << directory << std::endl;
        return files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        // Check if the file is an image (based on extension)
        if (filename.size() > 4) {
            std::string ext = filename.substr(filename.size() - 4);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".png" || ext == ".pgm" || ext == "jpeg" || 
                filename.substr(filename.size() - 5) == ".jpeg") {
                files.push_back(directory + "/" + filename);
            }
        }
    }
    closedir(dir);
    
    // Sort files to ensure consistent order
    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_directory>" << std::endl;
        return 1;
    }
    
    std::string engineFile = argv[1];
    std::string imageDir = argv[2];
    
    try {
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Create inference object
        MNISTInference inference(engineFile);
        
        // Get list of image files
        std::vector<std::string> imageFiles = listImageFiles(imageDir);
        
        if (imageFiles.empty()) {
            std::cout << "No image files found in directory: " << imageDir << std::endl;
            return 0;
        }
        
        std::cout << "Found " << imageFiles.size() << " images. Processing..." << std::endl;
        
        // Process each image
        for (const auto& file : imageFiles) {
            cv::Mat image = cv::imread(file, cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                std::cerr << "Failed to read image: " << file << std::endl;
                continue;
            }
            
            int digit = inference.processImage(image);
            std::cout << "Image: " << file << " -> Digit: " << digit << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 