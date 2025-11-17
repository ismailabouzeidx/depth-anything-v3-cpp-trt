#include "preprocessing.h"
#include <algorithm>
#include <cmath>

namespace preprocessing {

cv::Mat bgr_to_rgb(const cv::Mat& img) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

cv::Mat resize_upper_bound(const cv::Mat& img, int max_size) {
    int h = img.rows;
    int w = img.cols;
    
    if (h <= max_size && w <= max_size) {
        return img.clone();
    }
    
    float scale = std::min(static_cast<float>(max_size) / h, static_cast<float>(max_size) / w);
    int new_h = static_cast<int>(std::round(h * scale));
    int new_w = static_cast<int>(std::round(w * scale));
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    return resized;
}

cv::Mat make_divisible_by_patch_size(const cv::Mat& img, int patch_size, bool use_crop) {
    int h = img.rows;
    int w = img.cols;
    
    int new_h = (h / patch_size) * patch_size;
    int new_w = (w / patch_size) * patch_size;
    
    if (use_crop) {
        // Center crop
        int crop_h = (h - new_h) / 2;
        int crop_w = (w - new_w) / 2;
        return img(cv::Rect(crop_w, crop_h, new_w, new_h)).clone();
    } else {
        // Resize to nearest multiple
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        return resized;
    }
}

cv::Mat normalize_image(const cv::Mat& img, const float mean[3], const float std[3]) {
    cv::Mat normalized;
    img.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    
    cv::merge(channels, normalized);
    return normalized;
}

std::vector<float> preprocessImages(
    const std::vector<std::string>& image_paths,
    const PreprocessParams& params,
    std::vector<cv::Size>& original_sizes) {
    
    std::vector<cv::Mat> images;
    int failed_count = 0;
    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error: Failed to load image: " << path << std::endl;
            std::cerr << "  Please check that the file exists and is a valid image format." << std::endl;
            failed_count++;
            continue;
        }
        images.push_back(img);
    }
    
    if (images.empty()) {
        std::cerr << "Error: All images failed to load (" << failed_count << " failed)" << std::endl;
        return {};
    }
    
    if (failed_count > 0) {
        std::cerr << "Warning: " << failed_count << " image(s) failed to load, processing " 
                  << images.size() << " image(s)" << std::endl;
    }
    
    return preprocessImages(images, params, original_sizes);
}

std::vector<float> preprocessImages(
    const std::vector<cv::Mat>& images,
    const PreprocessParams& params,
    std::vector<cv::Size>& original_sizes) {
    
    if (images.empty()) {
        return {};
    }
    
    original_sizes.clear();
    std::vector<cv::Mat> processed_images;
    
    for (const auto& img : images) {
        original_sizes.push_back(cv::Size(img.cols, img.rows));
        
        // Convert BGR to RGB
        cv::Mat rgb = bgr_to_rgb(img);
        
        // Resize with upper bound
        cv::Mat resized = resize_upper_bound(rgb, std::max(params.target_height, params.target_width));
        
        // Make divisible by patch size
        bool use_crop = params.method.find("crop") != std::string::npos;
        cv::Mat divisible = make_divisible_by_patch_size(resized, params.patch_size, use_crop);
        
        // Normalize
        cv::Mat normalized = normalize_image(divisible, params.mean, params.std);
        
        processed_images.push_back(normalized);
    }
    
    // Get final dimensions (should be same for all after processing)
    int B = 1;  // batch size
    int N = processed_images.size();  // number of images
    int C = 3;
    int H = processed_images[0].rows;
    int W = processed_images[0].cols;
    
    // Stack images into tensor format (B, N, C, H, W)
    std::vector<float> tensor(B * N * C * H * W);
    
    for (int n = 0; n < N; ++n) {
        const cv::Mat& img = processed_images[n];
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w_idx = 0; w_idx < W; ++w_idx) {
                    int idx = ((0 * N + n) * C + c) * H * W + h * W + w_idx;
                    tensor[idx] = img.at<cv::Vec3f>(h, w_idx)[c];
                }
            }
        }
    }
    
    return tensor;
}

} // namespace preprocessing

