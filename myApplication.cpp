#include "filesystem"
#include "iostream"
#include "stdexcept"
#include "fstream"
#include "sstream"
#include "vector"
#include "Utilities.h"

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

        // A blank coordinate pair is not a ball at (0, 0).
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
        ball.centre = cv::Point2f(column, row);

        balls.push_back(ball);
    }
    return balls;
}

cv::Mat DrawGroundTruth(
    const cv::Mat& rectified_image,
    const std::vector<GroundTruthBall>& balls,
    const std::string& image_name)
{
    cv::Mat display = rectified_image.clone();

    int drawn_count = 0;
    int invalid_count = 0;

    for (const GroundTruthBall& ball : balls)
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

        const cv::Point centre(cvRound(x), cvRound(y));

        cv::drawMarker(
            display,
            centre,
            cv::Scalar(0, 0, 255),
            cv::MARKER_CROSS,
            12,
            2
        );

        cv::putText(
            display,
            ball.label,
            centre + cv::Point(8, -8),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            1
        );

        ++drawn_count;
    }

    std::cout
        << image_name
        << ": " << balls.size() << " nonblank annotations, "
        << drawn_count << " drawn, "
        << invalid_count << " outside image bounds.\n";

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

    // Load images
    int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);

    // Fixed-sized array for pool images
    Mat pool_image[number_of_images];

    // Table Dimensions:
    // Width = Column
    // Height = Row
    const int table_width = 1434;
    const int table_height = 683;

    // Order: Top Left, Top Right, Bottom Right, Bottom Left
    const Point2f source_points[4] = {
        Point2f(337.0f, 35.0f),   // Top Left
        Point2f(1570.0f, 21.0f),  // Top Right
        Point2f(1880.0f, 451.0f), // Bottom Right
        Point2f(38.0f, 472.0f)    // Bottom Left
    };

    const Point2f destination_points[4] = {
        Point2f(0.0f, 0.0f),
        Point2f(1433.0f, 0.0f),
        Point2f(1433.0f, 682.0f),
        Point2f(0.0f, 682.0f)};

    // Caluate same mapping for all images
    // Compute the perspective transformation matrix
    const Mat perspective_matrix = getPerspectiveTransform(source_points, destination_points);

    std::filesystem::create_directories("output/debug");

    for (int file_no = 0; file_no < number_of_images; file_no++)
    {
        cout << "Processing: " << pool_files[file_no] << std::endl;
        string filename(file_location);
        filename.append(pool_files[file_no]);
        pool_image[file_no] = imread(filename, IMREAD_COLOR);

        if (pool_image[file_no].empty())
        {
            throw std::runtime_error("Could not open or find the image: " + filename);
        }

        namedWindow("Original Image", WINDOW_NORMAL);
        imshow("Original Image", pool_image[file_no]);

        const bool is_training_image = std::string(pool_files[file_no]) == "AllBalls.jpeg";

        if (!is_training_image)
        {
            Mat rectified_image;

            warpPerspective(
                pool_image[file_no],
                rectified_image,
                perspective_matrix,
                Size(table_width, table_height),
                INTER_LINEAR);

            const std::string image_name = std::filesystem::path(pool_files[file_no]).stem().string();
            const std::string output_path = "output/debug/" + image_name + "_rectified.png";

            const std::vector<GroundTruthBall> ground_truth = LoadGroundTruth("data/ground_truth/ground_truth.csv", pool_files[file_no]);
            const std::string ground_truth_output = "output/debug/" + image_name + "_ground_truth.png";

            const Mat ground_truth_display = DrawGroundTruth(rectified_image, ground_truth, pool_files[file_no]);

            if (!imwrite(output_path, rectified_image))
            {
                throw std::runtime_error("Failed to write the rectified image to: " + output_path);
            }

           if (!imwrite(ground_truth_output, ground_truth_display))
            {
                throw std::runtime_error( "Could not save image: " + ground_truth_output);
            }

            namedWindow("Rectified Table", WINDOW_NORMAL);
            imshow("Rectified Table", rectified_image);

            namedWindow("Ground truth", WINDOW_NORMAL);
            imshow("Ground truth", ground_truth_display);

            cout << "Saved " << output_path << std::endl;
            cout << "Dimensions of Rectified Image: "
                 << rectified_image.cols << " Columns x "
                 << rectified_image.rows << " Rows \n"
                 << std::endl;
        }
        else
        {
            cout << "Training Image: No table transformation applied.\n ";
        }
        cout << "Press a key in an image window to continue to the next image... \n";
        waitKey(0);
        destroyAllWindows();
    }
}
