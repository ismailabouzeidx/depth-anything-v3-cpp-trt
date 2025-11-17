#include "postprocessing.h"
#include <algorithm>
#include <cmath>

namespace postprocessing {

cv::Mat normalize_depth(const cv::Mat& depth, float min_val, float max_val) {
    double min, max;
    cv::minMaxLoc(depth, &min, &max);
    
    cv::Mat normalized;
    if (max > min) {
        depth.convertTo(normalized, CV_32F, (max_val - min_val) / (max - min), 
                       min_val - min * (max_val - min_val) / (max - min));
    } else {
        normalized = cv::Mat::zeros(depth.size(), CV_32F);
    }
    
    return normalized;
}

cv::Mat apply_colormap(const cv::Mat& depth, int colormap_type) {
    cv::Mat normalized = normalize_depth(depth, 0.0f, 255.0f);
    normalized.convertTo(normalized, CV_8U);
    
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, colormap_type);
    return colored;
}

cv::Mat resize_depth_to_original(const cv::Mat& depth, const cv::Size& original_size) {
    cv::Mat resized;
    cv::resize(depth, resized, original_size, 0, 0, cv::INTER_LINEAR);
    return resized;
}

std::vector<cv::Mat> processDepthMaps(
    const float* output_data,
    int batch_size,
    int num_images,
    int height,
    int width,
    const std::vector<cv::Size>& original_sizes,
    const PostprocessParams& params) {
    
    std::vector<cv::Mat> depth_maps;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int n = 0; n < num_images; ++n) {
            // Extract depth map for this image
            cv::Mat depth(height, width, CV_32F);
            int offset = ((b * num_images + n) * height * width);
            
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    depth.at<float>(h, w) = output_data[offset + h * width + w];
                }
            }
            
            // Normalize if requested
            if (params.normalize_depth) {
                depth = normalize_depth(depth, params.depth_min, params.depth_max);
            }
            
            // Resize to original size only if requested (default: false to match Python)
            if (params.resize_to_original && n < static_cast<int>(original_sizes.size())) {
                depth = resize_depth_to_original(depth, original_sizes[n]);
            }
            
            depth_maps.push_back(depth);
        }
    }
    
    return depth_maps;
}

bool save_depth_map(const cv::Mat& depth, const std::string& output_path, bool normalized) {
    cv::Mat to_save;
    if (normalized) {
        cv::Mat normalized = normalize_depth(depth, 0.0f, 65535.0f);
        normalized.convertTo(to_save, CV_16U);
    } else {
        depth.convertTo(to_save, CV_16U);
    }
    
    return cv::imwrite(output_path, to_save);
}

bool save_depth_map_colored(const cv::Mat& depth, const std::string& output_path, int colormap_type) {
    cv::Mat colored = apply_colormap(depth, colormap_type);
    return cv::imwrite(output_path, colored);
}

} // namespace postprocessing

