#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Ground Truth
struct GroundTruthBall
{
    std::string label;
    cv::Point2f centre;
};

std::vector<GroundTruthBall> LoadGroundTruth(
    const std::string &csv_path,
    const std::string &image_name)
{
    std::ifstream file(csv_path);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Could not open ground-truth file: " + csv_path);
    }

    std::vector<GroundTruthBall> balls;
    std::string line;

    // Skip the CSV header.
    std::getline(file, line);

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream stream(line);

        std::string image;
        std::string label;
        std::string row_text;
        std::string column_text;

        std::getline(stream, image, ',');
        std::getline(stream, label, ',');
        std::getline(stream, row_text, ',');
        std::getline(stream, column_text, ',');

        if (image != image_name)
        {
            continue;
        }

        // Missing coordinates do not represent a ball at (0, 0).
        if (row_text.empty() && column_text.empty())
        {
            continue;
        }

        if (row_text.empty() || column_text.empty())
        {
            throw std::runtime_error(
                "Incomplete ground-truth coordinate for " +
                image + ", ball " + label);
        }

        const float row = std::stof(row_text);
        const float column = std::stof(column_text);

        GroundTruthBall ball;
        ball.label = label;

        // Spreadsheet: (row, column).
        // OpenCV: (x, y) = (column, row).
        ball.centre = cv::Point2f(column, row);

        balls.push_back(ball);
    }

    return balls;
}

cv::Mat DrawGroundTruth(
    const cv::Mat &rectified_image,
    const std::vector<GroundTruthBall> &balls,
    const std::string &image_name)
{
    cv::Mat display = rectified_image.clone();

    int drawn_count = 0;
    int invalid_count = 0;

    for (const GroundTruthBall &ball : balls)
    {
        const float x = ball.centre.x;
        const float y = ball.centre.y;

        if (x < 0 || x >= rectified_image.cols ||
            y < 0 || y >= rectified_image.rows)
        {
            std::cerr
                << "Ground-truth warning: " << image_name
                << ", ball " << ball.label
                << ", row=" << y
                << ", column=" << x
                << " is outside the rectified image.\n";

            ++invalid_count;
            continue;
        }

        const cv::Point centre(
            cvRound(x),
            cvRound(y));

        cv::drawMarker(
            display,
            centre,
            cv::Scalar(0, 0, 255),
            cv::MARKER_CROSS,
            12,
            2);

        cv::putText(
            display,
            ball.label,
            centre + cv::Point(8, -8),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            1);

        ++drawn_count;
    }

    std::cout
        << image_name
        << ": " << balls.size() << " nonblank annotations, "
        << drawn_count << " drawn, "
        << invalid_count << " outside image bounds.\n";

    return display;
}

// Ball Dectection
struct DetectedBall
{
    cv::Point2f centre;
    float radius;
};

std::vector<DetectedBall> DetectBalls(
    const cv::Mat &rectified_image,
    cv::Mat &smoothed_gray)
{
    if (rectified_image.empty())
    {
        throw std::runtime_error(
            "DetectBalls received an empty image.");
    }

    cv::Mat gray;

    cv::cvtColor(
        rectified_image,
        gray,
        cv::COLOR_BGR2GRAY);

    cv::medianBlur(gray, smoothed_gray, 5);

    // Initial baseline settings, not validated final parameters.
    const double dp = 1.0;
    const double minimum_centre_distance = 30.0;
    const double canny_high_threshold = 100.0;
    const double centre_vote_threshold = 15.0;

    const int minimum_radius = 15;
    const int maximum_radius = 30;

    std::vector<cv::Vec3f> circles;

    cv::HoughCircles(
        smoothed_gray,
        circles,
        cv::HOUGH_GRADIENT,
        dp,
        minimum_centre_distance,
        canny_high_threshold,
        centre_vote_threshold,
        minimum_radius,
        maximum_radius);

    std::vector<DetectedBall> detections;
    detections.reserve(circles.size());

    for (const cv::Vec3f &circle : circles)
    {
        detections.push_back({cv::Point2f(circle[0], circle[1]),
                              circle[2]});
    }

    return detections;
}

// Contour-based ellipse-fitting diagnostic
cv::Mat DrawEllipseCandidates(
    const cv::Mat &rectified_image,
    const cv::Mat &edge_image)
{
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(
        edge_image.clone(),
        contours,
        cv::RETR_LIST,
        cv::CHAIN_APPROX_NONE);

    cv::Mat display = rectified_image.clone();

    int candidate_count = 0;

    for (const std::vector<cv::Point> &contour : contours)
    {
        // A diagnostic support filter, not a final detection rule.
        if (contour.size() < 20)
        {
            continue;
        }

        const cv::RotatedRect fitted =
            cv::fitEllipse(contour);

        const float minor_axis =
            std::min(fitted.size.width, fitted.size.height);

        const float major_axis =
            std::max(fitted.size.width, fitted.size.height);

        // Broad diagnostic limits, not validated ball parameters.
        if (!std::isfinite(minor_axis) ||
            !std::isfinite(major_axis) ||
            !std::isfinite(fitted.center.x) ||
            !std::isfinite(fitted.center.y))
        {
            continue;
        }

        if (minor_axis < 10.0f || major_axis > 150.0f)
        {
            continue;
        }

        if (fitted.center.x < 0 ||
            fitted.center.x >= rectified_image.cols ||
            fitted.center.y < 0 ||
            fitted.center.y >= rectified_image.rows)
        {
            continue;
        }

        ++candidate_count;

        cv::ellipse(
            display,
            fitted,
            cv::Scalar(0, 255, 255),
            1);

        const cv::Point centre(
            cvRound(fitted.center.x),
            cvRound(fitted.center.y));

        cv::drawMarker(
            display,
            centre,
            cv::Scalar(255, 0, 255),
            cv::MARKER_CROSS,
            8,
            1);

        std::cout
            << "Ellipse candidate " << candidate_count
            << ": x=" << fitted.center.x
            << ", y=" << fitted.center.y
            << ", minor axis=" << minor_axis
            << ", major axis=" << major_axis
            << '\n';
    }

    return display;
}

// Foreground Mask
cv::Mat BuildForegroundMask(
    const cv::Mat &rectified_image,
    cv::Mat &cloth_mask,
    int maximum_cloth_saturation)
{
    if (maximum_cloth_saturation < 25 ||
        maximum_cloth_saturation > 255)
    {
        throw std::runtime_error(
            "Maximum cloth saturation must be between 25 and 255.");
    }

    cv::Mat hsv;

    cv::cvtColor(
        rectified_image,
        hsv,
        cv::COLOR_BGR2HSV);

    const cv::Scalar cloth_lower(35, 25, 25);

    const cv::Scalar cloth_upper(
        95,
        maximum_cloth_saturation,
        255);

    cv::inRange(
        hsv,
        cloth_lower,
        cloth_upper,
        cloth_mask);

    cv::Mat foreground_mask;
    cv::bitwise_not(cloth_mask, foreground_mask);

    return foreground_mask;
}

cv::Mat DrawMaskCandidates(
    const cv::Mat &rectified_image,
    const cv::Mat &foreground_mask)
{
    CV_Assert(foreground_mask.type() == CV_8UC1);
    CV_Assert(foreground_mask.size() == rectified_image.size());

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    const int number_of_labels =
        cv::connectedComponentsWithStats(
            foreground_mask,
            labels,
            stats,
            centroids,
            8,
            CV_32S);

    cv::Mat display = rectified_image.clone();

    int shown_count = 0;

    // Label 0 is background.
    for (int label = 1; label < number_of_labels; ++label)
    {
        const int area =
            stats.at<int>(label, cv::CC_STAT_AREA);

        // Diagnostic noise filter, not a ball-size rule.
        if (area < 100)
        {
            continue;
        }

        const int x =
            stats.at<int>(label, cv::CC_STAT_LEFT);

        const int y =
            stats.at<int>(label, cv::CC_STAT_TOP);

        const int width =
            stats.at<int>(label, cv::CC_STAT_WIDTH);

        const int height =
            stats.at<int>(label, cv::CC_STAT_HEIGHT);

        const bool touches_border =
            x == 0 ||
            y == 0 ||
            x + width == foreground_mask.cols ||
            y + height == foreground_mask.rows;

        const cv::Scalar colour =
            touches_border
                ? cv::Scalar(0, 0, 255)
                : cv::Scalar(0, 255, 255);

        cv::rectangle(
            display,
            cv::Rect(x, y, width, height),
            colour,
            2);

        cv::putText(
            display,
            "C" + std::to_string(label),
            cv::Point(x, std::max(15, y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1);

        std::cout
            << "Component C" << label
            << ": area=" << area
            << ", width=" << width
            << ", height=" << height
            << ", touches border="
            << (touches_border ? "yes" : "no")
            << '\n';

        ++shown_count;
    }

    std::cout
        << shown_count
        << " foreground components above the diagnostic area limit.\n";

    return display;
}

void MyApplication()
{
    const char *file_location = "data/images/";

    const char *pool_files[] = {
        "AllBalls.jpeg",
        "Table1.jpeg",
        "Table2.jpeg",
        "Table3.jpeg",
        "Table4.jpeg",
        "Table5.jpeg",
        "Table6.jpeg"};

    const int number_of_images =
        sizeof(pool_files) / sizeof(pool_files[0]);

    std::vector<cv::Mat> pool_image(number_of_images);

    // Table dimensions: width = columns, height = rows.
    const int table_width = 1434;
    const int table_height = 683;

    // Order: top left, top right, bottom right, bottom left.
    // Coordinates: (column, row).
    const cv::Point2f source_points[4] = {
        cv::Point2f(337.0f, 35.0f),
        cv::Point2f(1570.0f, 21.0f),
        cv::Point2f(1888.0f, 451.0f),
        cv::Point2f(38.0f, 472.0f)};

    const cv::Point2f destination_points[4] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(
            static_cast<float>(table_width - 1),
            0.0f),
        cv::Point2f(
            static_cast<float>(table_width - 1),
            static_cast<float>(table_height - 1)),
        cv::Point2f(
            0.0f,
            static_cast<float>(table_height - 1))};

    // Use the same perspective transformation for all table images.
    const cv::Mat perspective_matrix =
        cv::getPerspectiveTransform(
            source_points,
            destination_points);

    // std::cout
    //     << "Working directory: "
    //     << std::filesystem::current_path()
    //     << '\n';

    for (int file_no = 0; file_no < number_of_images; ++file_no)
    {
        const std::string image_filename = pool_files[file_no];

        std::cout
            << "\nProcessing: "
            << image_filename
            << '\n';

        std::string filename(file_location);
        filename.append(image_filename);

        pool_image[file_no] =
            cv::imread(filename, cv::IMREAD_COLOR);

        if (pool_image[file_no].empty())
        {
            throw std::runtime_error(
                "Could not open or find the image: " + filename);
        }

        cv::namedWindow(
            "Original Image",
            cv::WINDOW_NORMAL);

        cv::imshow(
            "Original Image",
            pool_image[file_no]);

        cv::Mat rectified_image;

        cv::warpPerspective(
            pool_image[file_no],
            rectified_image,
            perspective_matrix,
            cv::Size(table_width, table_height),
            cv::INTER_LINEAR);

        const std::string image_name =
            std::filesystem::path(image_filename)
                .stem()
                .string();

        const std::filesystem::path image_output =
            std::filesystem::path("output") / image_name;

        const std::filesystem::path debug_dir =
            image_output / "debug";

        const std::filesystem::path ground_truth_dir =
            image_output / "ground_truth";

        const std::filesystem::path detection_dir =
            image_output / "detections";

        const std::filesystem::path ellipse_dir =
            image_output / "ellipse";

        const std::filesystem::path mask_dir =
            image_output / "mask";

        const std::filesystem::path hsv_dir =
            image_output / "hsv";

        std::filesystem::create_directories(debug_dir);
        std::filesystem::create_directories(ground_truth_dir);
        std::filesystem::create_directories(detection_dir);
        std::filesystem::create_directories(ellipse_dir);
        std::filesystem::create_directories(mask_dir);
        std::filesystem::create_directories(hsv_dir);

        const std::string output_path =
            (debug_dir /
             (image_name + "_rectified.png"))
                .string();

        const std::vector<GroundTruthBall> ground_truth =
            LoadGroundTruth(
                "data/ground_truth/ground_truth.csv",
                image_filename);

        if (ground_truth.empty())
        {
            throw std::runtime_error(
                "No ground-truth annotations loaded for: " +
                image_filename);
        }

        const cv::Mat ground_truth_display =
            DrawGroundTruth(
                rectified_image,
                ground_truth,
                image_filename);

        cv::Mat smoothed_gray;

        const std::vector<DetectedBall> detected_balls =
            DetectBalls(rectified_image, smoothed_gray);

        cv::Mat detection_display = rectified_image.clone();

        for (std::size_t i = 0; i < detected_balls.size(); ++i)
        {
            const DetectedBall &ball = detected_balls[i];

            const cv::Point centre(
                cvRound(ball.centre.x),
                cvRound(ball.centre.y));

            circle(
                detection_display,
                centre,
                cvRound(ball.radius),
                cv::Scalar(0, 255, 0),
                2);

            drawMarker(
                detection_display,
                centre,
                cv::Scalar(255, 0, 255),
                cv::MARKER_CROSS,
                8,
                1);

            // Candidate index only: not a predicted ball identity.
            putText(
                detection_display,
                "D" + std::to_string(i + 1),
                centre + cv::Point(8, -8),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(0, 255, 0),
                1);

            std::cout
                << image_filename
                << ", D" << (i + 1)
                << ": x=" << ball.centre.x
                << ", y=" << ball.centre.y
                << ", radius=" << ball.radius
                << '\n';
        }

        cv::Mat diagnostic_edges;

        // Mirrors the threshold pair associated with param1 = 100.
        // This is a diagnostic image, not an input to HoughCircles.
        cv::Canny(
            smoothed_gray,
            diagnostic_edges,
            50.0,
            100.0);

        const cv::Mat ellipse_display =
            DrawEllipseCandidates(
                rectified_image,
                diagnostic_edges);

        cv::Mat cloth_mask;
        int maximum_cloth_saturation = 95;

        const cv::Mat foreground_mask =
            BuildForegroundMask(
                rectified_image,
                cloth_mask,
                maximum_cloth_saturation);

        const cv::Mat opening_kernel =
            cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(3, 3));

        cv::Mat opened_mask;

        cv::morphologyEx(
            foreground_mask,
            opened_mask,
            cv::MORPH_OPEN,
            opening_kernel,
            cv::Point(-1, -1),
            1);

        cv::Mat foreground_preview;

        rectified_image.copyTo(
            foreground_preview,
            foreground_mask);

        const cv::Mat mask_candidates =
            DrawMaskCandidates(
                rectified_image,
                foreground_mask);

        const cv::Mat opened_regions =
            DrawMaskCandidates(
                rectified_image,
                opened_mask);

        cv::Mat opened_preview;

        rectified_image.copyTo(
            opened_preview,
            opened_mask);

        cv::Mat hsv;
        cv::cvtColor(rectified_image, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);

        // Stretch hue's 0–179 range for visualisation only.
        // Do not use this stretched image for HSV thresholding.
        cv::Mat hue_display;
        hsv_channels[0].convertTo(
            hue_display,
            CV_8U,
            255.0 / 179.0);

        const std::string mask_tag =
            "_S" + std::to_string(maximum_cloth_saturation) + "_V25";

        const std::string opening_tag = "_open3";

        const std::string ground_truth_output =
            (ground_truth_dir /
             (image_name + "_ground_truth.png"))
                .string();

        const std::string gray_path =
            (debug_dir /
             (image_name + "_smoothed_gray.png"))
                .string();

        const std::string detection_path =
            (detection_dir /
             (image_name + "_hough_detections.png"))
                .string();

        const std::string edges_path =
            (debug_dir /
             (image_name + "_edges.png"))
                .string();

        const std::string ellipse_path =
            (ellipse_dir /
             (image_name + "_ellipse_candidates.png"))
                .string();

        const std::string cloth_path =
            (mask_dir /
             (image_name + "_cloth" + mask_tag + ".png"))
                .string();

        const std::string foreground_path =
            (mask_dir /
             (image_name + "_foreground" + mask_tag + ".png"))
                .string();

        const std::string preview_path =
            (mask_dir /
             (image_name + "_preview" + mask_tag + ".png"))
                .string();

        const std::string hue_path =
            (hsv_dir /
             (image_name + "_hue.png"))
                .string();

        const std::string saturation_path =
            (hsv_dir /
             (image_name + "_saturation.png"))
                .string();

        const std::string value_path =
            (hsv_dir /
             (image_name + "_value.png"))
                .string();

        const std::string candidates_path =
            (mask_dir /
             (image_name + "_regions" + mask_tag + ".png"))
                .string();

        const std::string opened_mask_path =
            (mask_dir /
             (image_name + "_foreground" +
              mask_tag + opening_tag + ".png"))
                .string();

        const std::string opened_preview_path =
            (mask_dir /
             (image_name + "_preview" +
              mask_tag + opening_tag + ".png"))
                .string();

        const std::string opened_regions_path =
            (mask_dir /
             (image_name + "_regions" +
              mask_tag + opening_tag + ".png"))
                .string();

        if (!cv::imwrite(opened_mask_path, opened_mask) ||
            !cv::imwrite(opened_preview_path, opened_preview) ||
            !cv::imwrite(opened_regions_path, opened_regions))
        {
            throw std::runtime_error(
                "Could not save opening diagnostics for: " +
                image_name);
        }

        if (!cv::imwrite(output_path, rectified_image))
        {
            throw std::runtime_error(
                "Failed to write the rectified image to: " +
                output_path);
        }

        if (!cv::imwrite(
                ground_truth_output,
                ground_truth_display))
        {
            throw std::runtime_error(
                "Could not save image: " +
                ground_truth_output);
        }

        if (!imwrite(gray_path, smoothed_gray))
        {
            throw std::runtime_error("Could not save: " + gray_path);
        }

        if (!imwrite(detection_path, detection_display))
        {
            throw std::runtime_error(
                "Could not save: " + detection_path);
        }

        if (!cv::imwrite(edges_path, diagnostic_edges))
        {
            throw std::runtime_error(
                "Could not save: " + edges_path);
        }

        if (!cv::imwrite(ellipse_path, ellipse_display))
        {
            throw std::runtime_error(
                "Could not save: " + ellipse_path);
        }

        if (!cv::imwrite(cloth_path, cloth_mask) ||
            !cv::imwrite(foreground_path, foreground_mask) ||
            !cv::imwrite(preview_path, foreground_preview))
        {
            throw std::runtime_error(
                "Could not save segmentation diagnostics for: " +
                image_name);
        }

        if (!cv::imwrite(candidates_path, mask_candidates))
        {
            throw std::runtime_error(
                "Could not save: " + candidates_path);
        }

        if (!cv::imwrite(hue_path, hue_display) ||
            !cv::imwrite(saturation_path, hsv_channels[1]) ||
            !cv::imwrite(value_path, hsv_channels[2]))
        {
            throw std::runtime_error(
                "Could not save HSV channels for: " + image_name);
        }

        std::cout << image_filename
                  << ": " << detected_balls.size()
                  << " Hough circle candidates.\n";

        // cv::namedWindow(
        //     "Rectified Table",
        //     cv::WINDOW_NORMAL);

        // cv::imshow(
        //     "Rectified Table",
        //     rectified_image);

        // cv::namedWindow(
        //     "Ground Truth",
        //     cv::WINDOW_NORMAL);

        // cv::imshow(
        //     "Ground Truth",
        //     ground_truth_display);

        // cv::namedWindow(
        //     "Detected balls",
        //     cv::WINDOW_NORMAL);

        // cv::imshow(
        //     "Detected balls",
        //     detection_display);

        // cv::namedWindow(
        //     "Ellipse candidates",
        //     cv::WINDOW_NORMAL);

        // cv::imshow(
        //     "Ellipse candidates",
        //     ellipse_display);

        // cv::namedWindow(
        //     "Foreground mask",
        //     cv::WINDOW_NORMAL);

        // cv::imshow(
        //     "Foreground mask",
        //     foreground_mask);

        // cv::namedWindow(
        //     "Foreground preview",
        //     cv::WINDOW_NORMAL);

        // cv::imshow(
        //     "Foreground preview",
        //     foreground_preview);

        // std::cout
        //     << "Saved: " << output_path << '\n'
        //     << "Saved: " << ground_truth_output << '\n'
        //     << "Rectified dimensions: "
        //     << rectified_image.cols << " columns x "
        //     << rectified_image.rows << " rows\n";

        // std::cout
        //     << "Press a key in an image window "
        //     << "to continue to the next image...\n";

        // cv::waitKey(0);
        cv::destroyAllWindows();
    }
}
