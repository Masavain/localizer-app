#include "orb_extractor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>

int main() {
    auto extractor = openvslam::feature::orb_extractor(10000, 1.2, 16, 20, 7);
    auto image = cv::imread("./img.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    extractor.extract(image, cv::Mat(), keypoints, descriptors);
    std::cout << descriptors << "\n";
    return 0;
}
