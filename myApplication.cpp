#include "filesystem"
#include "iostream"
#include "stdexcept"
#include "Utilities.h"

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

            if (!imwrite(output_path, rectified_image))
            {
                throw std::runtime_error("Failed to write the rectified image to: " + output_path);
            }

            namedWindow("Rectified Table", WINDOW_NORMAL);
            imshow("Rectified Table", rectified_image);

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
