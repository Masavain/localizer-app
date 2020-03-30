#include "orb_extractor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>
#include <fstream>

int main() {
    auto extractor = openvslam::feature::orb_extractor(2000, 1.2, 16, 20, 7);
    auto image = cv::imread("./img.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    extractor.extract(image, cv::Mat(), keypoints, descriptors);
    
    std::ofstream myfile ("out.txt", std::ofstream::out);
    if (myfile.is_open())
    {
        myfile << descriptors << "\n";
        myfile.close();
    }
    else std::cout << "Unable lol";
    //std::cout << descriptors << "\n";
}
