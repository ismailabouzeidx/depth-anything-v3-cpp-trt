#include "tensorrt_engine.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

TensorRTEngine::TensorRTEngine()
    : runtime_(nullptr), engine_(nullptr), context_(nullptr),
      input_buffer_(nullptr), output_buffer_(nullptr),
      input_size_(0), output_size_(0) {
    cudaStreamCreate(&stream_);
}

TensorRTEngine::~TensorRTEngine() {
    free_buffers();
    // TensorRT 10.x: objects are automatically managed, no need to call destroy()
    // Just set pointers to nullptr
    if (context_) {
        context_ = nullptr;
    }
    if (engine_) {
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_ = nullptr;
    }
    cudaStreamDestroy(stream_);
}

bool TensorRTEngine::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Cannot open engine file: " << engine_path << std::endl;
        return false;
    }

    // Read engine file
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create runtime and deserialize engine
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return false;
    }

    // Get input/output information (TensorRT 10.x API)
    int num_io_tensors = engine_->getNbIOTensors();
    for (int i = 0; i < num_io_tensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            auto dims = engine_->getTensorShape(name);
            input_shape_.clear();
            for (int j = 0; j < dims.nbDims; ++j) {
                input_shape_.push_back(dims.d[j]);
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name_ = name;
            auto dims = engine_->getTensorShape(name);
            output_shape_.clear();
            for (int j = 0; j < dims.nbDims; ++j) {
                output_shape_.push_back(dims.d[j]);
            }
        }
    }

    std::cout << "Engine loaded successfully" << std::endl;
    std::cout << "  Input: " << input_name_ << " shape: [";
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        std::cout << input_shape_[i];
        if (i < input_shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Output: " << output_name_ << " shape: [";
    for (size_t i = 0; i < output_shape_.size(); ++i) {
        std::cout << output_shape_[i];
        if (i < output_shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    return allocate_buffers();
}

bool TensorRTEngine::allocate_buffers() {
    // Calculate buffer sizes
    input_size_ = 1;
    for (int dim : input_shape_) {
        input_size_ *= dim;
    }
    input_size_ *= sizeof(float);

    output_size_ = 1;
    for (int dim : output_shape_) {
        output_size_ *= dim;
    }
    output_size_ *= sizeof(float);

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&input_buffer_, input_size_);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to allocate input buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&output_buffer_, output_size_);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to allocate output buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(input_buffer_);
        return false;
    }

    return true;
}

void TensorRTEngine::free_buffers() {
    if (input_buffer_) {
        cudaFree(input_buffer_);
        input_buffer_ = nullptr;
    }
    if (output_buffer_) {
        cudaFree(output_buffer_);
        output_buffer_ = nullptr;
    }
}

bool TensorRTEngine::infer(const float* input_data, float* output_data) {
    if (!engine_ || !context_) {
        std::cerr << "Error: Engine not loaded" << std::endl;
        return false;
    }

    // Copy input to GPU
    cudaError_t err = cudaMemcpyAsync(input_buffer_, input_data, input_size_, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to copy input to GPU: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // TensorRT 10.x API: Use setTensorAddress instead of bindings
    if (!context_->setTensorAddress(input_name_.c_str(), input_buffer_)) {
        std::cerr << "Error: Failed to set input tensor address" << std::endl;
        return false;
    }
    if (!context_->setTensorAddress(output_name_.c_str(), output_buffer_)) {
        std::cerr << "Error: Failed to set output tensor address" << std::endl;
        return false;
    }
    
    // Execute inference (TensorRT 10.x uses enqueueV3)
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        std::cerr << "Error: Inference execution failed" << std::endl;
        return false;
    }

    // Copy output from GPU
    err = cudaMemcpyAsync(output_data, output_buffer_, output_size_, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to copy output from GPU: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream_);

    return true;
}

