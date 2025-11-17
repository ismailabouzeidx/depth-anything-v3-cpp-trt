#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorrt_engine.h"
#include "preprocessing.h"
#include "postprocessing.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <engine_path> <image1> [image2] ... [output_dir]" << std::endl;
        std::cout << "Example: " << argv[0] << " model.engine img1.jpg img2.jpg output/" << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    std::vector<std::string> image_paths;
    std::string output_dir = "output";
    
    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_dir = argv[++i];
            }
        } else {
            image_paths.push_back(arg);
        }
    }
    
    if (image_paths.empty()) {
        std::cerr << "Error: No input images provided" << std::endl;
        return 1;
    }

    // Create output directory
    #ifdef _WIN32
        (void)system(("mkdir " + output_dir).c_str());
    #else
        (void)system(("mkdir -p " + output_dir).c_str());
    #endif

    std::cout << "Loading TensorRT engine: " << engine_path << std::endl;
    TensorRTEngine engine;
    if (!engine.load_engine(engine_path)) {
        std::cerr << "Failed to load engine" << std::endl;
        return 1;
    }

    // Preprocessing parameters
    preprocessing::PreprocessParams preprocess_params;
    preprocess_params.target_height = 280;
    preprocess_params.target_width = 504;
    preprocess_params.patch_size = 14;
    preprocess_params.method = "upper_bound_resize";

    // Get expected number of images from engine input shape
    std::vector<int> engine_input_shape = engine.get_input_shape();
    size_t expected_num_images = 1;
    if (engine_input_shape.size() >= 2) {
        expected_num_images = static_cast<size_t>(engine_input_shape[1]);  // Second dimension is num_images
    }
    
    // Validate image count matches model requirements
    if (image_paths.size() != expected_num_images) {
        std::cerr << "Error: Model expects exactly " << expected_num_images 
                  << " image(s), but " << image_paths.size() << " provided." << std::endl;
        std::cerr << "Please provide exactly " << expected_num_images << " image(s)." << std::endl;
        return 1;
    }

    std::cout << "\nPreprocessing " << image_paths.size() << " images..." << std::endl;
    std::vector<cv::Size> original_sizes;
    std::vector<float> input_tensor = preprocessing::preprocessImages(
        image_paths, preprocess_params, original_sizes);

    if (input_tensor.empty()) {
        std::cerr << "Error: Preprocessing failed" << std::endl;
        return 1;
    }

    std::cout << "Input tensor shape: [1, " << image_paths.size() << ", 3, " 
              << preprocess_params.target_height << ", " << preprocess_params.target_width << "]" << std::endl;

    // Get output shape for allocation
    std::vector<int> output_shape = engine.get_output_shape();
    size_t output_size = 1;
    for (int dim : output_shape) {
        output_size *= dim;
    }
    
    // Allocate output buffer
    std::vector<float> output_data(output_size);
    
    // Run inference
    std::cout << "\nRunning inference..." << std::endl;
    if (!engine.infer(input_tensor.data(), output_data.data())) {
        std::cerr << "Error: Inference failed" << std::endl;
        return 1;
    }
    std::cout << "✓ Inference completed successfully" << std::endl;
    
    // Postprocessing parameters
    postprocessing::PostprocessParams postprocess_params;
    postprocess_params.normalize_depth = true;
    postprocess_params.apply_colormap = true;
    postprocess_params.colormap_type = cv::COLORMAP_INFERNO;

    // Process output from engine
    std::cout << "\nPostprocessing..." << std::endl;
    int batch_size = 1;
    int num_images = static_cast<int>(image_paths.size());
    int height = output_shape.size() >= 3 ? output_shape[output_shape.size() - 2] : preprocess_params.target_height;
    int width = output_shape.size() >= 2 ? output_shape[output_shape.size() - 1] : preprocess_params.target_width;
    
    std::vector<cv::Mat> depth_maps = postprocessing::processDepthMaps(
        output_data.data(), batch_size, num_images, height, width, original_sizes, postprocess_params);
    
    if (depth_maps.empty()) {
        std::cerr << "Error: Postprocessing failed" << std::endl;
        return 1;
    }
    std::cout << "Postprocessing completed (" << depth_maps.size() << " depth maps)" << std::endl;

    // Save depth maps
    std::cout << "\nSaving results..." << std::endl;
    for (size_t i = 0; i < image_paths.size() && i < depth_maps.size(); ++i) {
        std::string base_name = image_paths[i].substr(image_paths[i].find_last_of("/\\") + 1);
        size_t dot_pos = base_name.find_last_of(".");
        if (dot_pos != std::string::npos) {
            base_name = base_name.substr(0, dot_pos);
        }
        
        std::string depth_path = output_dir + "/" + base_name + "_depth.png";
        std::string colored_path = output_dir + "/" + base_name + "_depth_colored.png";
        
        if (postprocessing::save_depth_map(depth_maps[i], depth_path)) {
            std::cout << "  Saved: " << depth_path << std::endl;
        } else {
            std::cerr << "  Warning: Failed to save " << depth_path << std::endl;
        }
        
        if (postprocessing::save_depth_map_colored(depth_maps[i], colored_path, cv::COLORMAP_INFERNO)) {
            std::cout << "  Saved: " << colored_path << std::endl;
        } else {
            std::cerr << "  Warning: Failed to save " << colored_path << std::endl;
        }
    }

    std::cout << "Done! Results saved to: " << output_dir << std::endl;
    return 0;
}

