#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    // Load engine from file
    bool load_engine(const std::string& engine_path);
    
    // Run inference with preprocessed input tensor
    // input_data: Preprocessed input tensor (B, N, C, H, W) as float array
    // output_data: Output buffer for depth maps (B, N, H, W) as float array
    bool infer(const float* input_data, float* output_data);
    
    // Get input/output info
    std::vector<int> get_input_shape() const { return input_shape_; }
    std::vector<int> get_output_shape() const { return output_shape_; }
    std::string get_input_name() const { return input_name_; }
    std::string get_output_name() const { return output_name_; }
    
    // Check if engine is loaded
    bool is_loaded() const { return engine_ != nullptr; }

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    std::string input_name_;
    std::string output_name_;
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    
    void* input_buffer_;
    void* output_buffer_;
    size_t input_size_;
    size_t output_size_;
    
    cudaStream_t stream_;
    
    bool allocate_buffers();
    void free_buffers();
};

