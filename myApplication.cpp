#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>

void MyApplication()
{
    const std::filesystem::path imagePath =
        std::filesystem::path("data") / "images" / "Table1.jpeg";

    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    std::cout << "Working directory: "
              << std::filesystem::current_path() << '\n';
    std::cout << "Loading: " << imagePath << '\n';

    const cv::Mat image =
        cv::imread(imagePath.string(), cv::IMREAD_COLOR);

    if (image.empty())
    {
        throw std::runtime_error(
            "Could not load Table1.jpeg. Check the image location "
            "and run the program from the project root.");
    }

    std::cout << "Image dimensions: "
              << image.cols << " columns x "
              << image.rows << " rows\n";

    cv::namedWindow("OpenCV setup test", cv::WINDOW_NORMAL);
    cv::imshow("OpenCV setup test", image);

    std::cout << "Press a key in the image window to exit.\n";

    cv::waitKey(0);
    cv::destroyAllWindows();
}