#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace postprocessing {

// Postprocessing parameters
struct PostprocessParams {
    bool normalize_depth = true;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    bool apply_colormap = true;
    int colormap_type = cv::COLORMAP_INFERNO;  // OpenCV colormap
    bool resize_to_original = false;  // If false, keep model output size (matches Python)
};

// Process depth maps from model output
// Input: Raw model output tensor (B, N, H, W)
// Output: Processed depth maps
std::vector<cv::Mat> processDepthMaps(
    const float* output_data,
    int batch_size,
    int num_images,
    int height,
    int width,
    const std::vector<cv::Size>& original_sizes,
    const PostprocessParams& params = PostprocessParams()
);

// Normalize depth map to [0, 1] or specified range
cv::Mat normalize_depth(const cv::Mat& depth, float min_val = 0.0f, float max_val = 1.0f);

// Apply colormap to depth for visualization
cv::Mat apply_colormap(const cv::Mat& depth, int colormap_type = cv::COLORMAP_INFERNO);

// Resize depth map back to original image size
cv::Mat resize_depth_to_original(const cv::Mat& depth, const cv::Size& original_size);

// Save depth map
bool save_depth_map(const cv::Mat& depth, const std::string& output_path, bool normalized = true);

// Save depth map with colormap
bool save_depth_map_colored(const cv::Mat& depth, const std::string& output_path, int colormap_type = cv::COLORMAP_INFERNO);

} // namespace postprocessing

