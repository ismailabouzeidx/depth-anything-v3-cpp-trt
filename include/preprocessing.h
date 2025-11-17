#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace preprocessing {

// Image preprocessing parameters
struct PreprocessParams {
    int target_height = 280;
    int target_width = 504;
    int patch_size = 14;
    std::string method = "upper_bound_resize";  // "upper_bound_resize" or "upper_bound_crop"
    
    // ImageNet normalization
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};

// Load and preprocess images
// Input: List of image paths or cv::Mat images
// Output: Preprocessed tensor ready for model input (B, N, 3, H, W)
std::vector<float> preprocessImages(
    const std::vector<std::string>& image_paths,
    const PreprocessParams& params,
    std::vector<cv::Size>& original_sizes
);

std::vector<float> preprocessImages(
    const std::vector<cv::Mat>& images,
    const PreprocessParams& params,
    std::vector<cv::Size>& original_sizes
);

// Resize image with upper bound constraint (preserves aspect ratio)
cv::Mat resize_upper_bound(const cv::Mat& img, int max_size);

// Make dimensions divisible by patch_size
cv::Mat make_divisible_by_patch_size(const cv::Mat& img, int patch_size, bool use_crop = false);

// Normalize image (ImageNet normalization)
cv::Mat normalize_image(const cv::Mat& img, const float mean[3], const float std[3]);

// Convert BGR to RGB
cv::Mat bgr_to_rgb(const cv::Mat& img);

} // namespace preprocessing

