#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// All assignment-specific implementation stays in this file.
// Sections: diagnostics, candidate selection, evaluation, identity, preprocessing, entry point.
// Shared settings: pooldiag::Settings, ballvision::SelectorSettings, ballid::Settings.
// AllBalls supplies training only; Table1--Table6 annotations are evaluation only.

// -----------------------------------------------------------------------------
// Diagnostic measurements and output organisation
// -----------------------------------------------------------------------------
namespace pooldiag
{
// Keep output routing in one place so every writer uses the same folder layout.
// Unknown filenames are left relative to dir (for example, a per-ball crop).
inline std::filesystem::path OutputPath(const std::filesystem::path &dir,
                                        const std::string &filename)
{
    std::filesystem::path folder = dir;
    const auto begins = [&filename](const std::string &prefix)
    {
        return filename.rfind(prefix, 0) == 0;
    };

    if (filename == "settings.yml" || filename == "selector_settings.yml" ||
        filename == "summary.txt")
        folder /= "00_settings";
    else if (filename == "hue.png" || filename == "saturation.png" || filename == "value.png")
        folder /= "01_preprocessing/hsv";
    else if (filename == "rectified.png" || filename == "smoothed_gray.png" ||
             filename == "edges.png")
        folder /= "01_preprocessing";
    else if (begins("components_") || begins("foreground_") || begins("regions_") ||
             filename == "cloth.png" || filename == "opened_preview.png")
        folder /= "02_segmentation";
    else if (begins("seeds") || begins("peak") || begins("border_peak") || begins("local_maxima") ||
             begins("distance") || filename == "markers.png" || filename == "border_markers.png")
        folder /= "03_seeds";
    else if (begins("ellipse_crops_"))
        folder /= "04_candidates/ellipse_crops";
    else if (begins("ellips") || begins("contours") || begins("overlapping_") || begins("hough") ||
             filename == "rejected_candidates.png")
        folder /= "04_candidates";
    else if (filename == "selection.csv" || filename == "detections.csv" ||
             filename == "validated_detections.png" || filename == "selector_summary.txt" ||
             filename == "pocket_change.png")
        folder /= "05_detections";
    else if (filename == "crop_quality.csv")
        folder /= "06_ball_crops";
    else if (begins("localisation_") || begins("ground_truth") ||
             filename == "evaluation_notes.txt")
        folder /= "07_evaluation";

    std::filesystem::create_directories(folder);
    return folder / filename;
}

// Small navigation files keep the detailed evidence discoverable without moving it.
inline void WriteImageIndex(const std::filesystem::path &dir)
{
    std::ofstream index(dir / "INDEX.md");
    if (!index)
        throw std::runtime_error("Cannot write image index: " + dir.string());

    index << "# " << dir.filename().string() << " diagnostics\n\n"
          << "Start with [accepted detections](05_detections/validated_detections.png) "
          << "and [localisation evaluation](07_evaluation/localisation_evaluation.png).\n\n"
          << "Detection labels are proposal IDs. See [recognised balls](08_recognition/recognised_balls.png) for identities.\n\n"
          << "| Stage | Evidence |\n|---|---|\n"
          << "| Settings | [Preprocessing](00_settings/settings.yml), "
          << "[selector](00_settings/selector_settings.yml) |\n"
          << "| Preprocessing | [Rectified RGB](01_preprocessing/rectified.png), "
          << "[edges](01_preprocessing/edges.png), [HSV](01_preprocessing/hsv/) |\n"
          << "| Segmentation | [Raw mask](02_segmentation/foreground_raw.png), "
          << "[opened mask](02_segmentation/foreground_opened.png) |\n"
          << "| Seeds | [Interior](03_seeds/markers.png), "
          << "[border](03_seeds/border_markers.png), "
          << "[suppression audit](03_seeds/peak_selection.csv) |\n"
          << "| Candidates | [Numbered ellipses](04_candidates/ellipses_numbered.png), "
          << "[measurements](04_candidates/ellipses.csv), "
          << "[rejections](04_candidates/rejected_candidates.png) |\n"
          << "| Detections | [Positions](05_detections/detections.csv), "
          << "[selection audit](05_detections/selection.csv) |\n"
          << "| Ball crops | [Crop folders](06_ball_crops/), "
          << "[appearance quality](06_ball_crops/crop_quality.csv) |\n"
          << "| Evaluation | [Metrics](07_evaluation/localisation_metrics.csv), "
          << "[matches](07_evaluation/localisation_matches.csv), "
          << "[notes](07_evaluation/evaluation_notes.txt) |\n\n"
          << "Evaluation files may be absent when no annotations are available. "
          << "Supplied annotations are preserved, not corrected automatically.\n";
    index.close();
    if (!index)
        throw std::runtime_error("Cannot finish image index: " + dir.string());
}

inline void WriteRunOverview(const std::filesystem::path &root,
                             const std::vector<std::string> &images)
{
    std::ofstream index(root / "INDEX.md");
    if (!index)
        throw std::runtime_error("Cannot write run index: " + root.string());

    index << "# Pool vision run " << root.filename().string() << "\n\n"
          << "[Run summary](run_summary.txt). Open an image index below.\n\n"
          << "AllBalls is training data; Table1--Table6 are evaluation only. "
          << "Recognition uses audited AllBalls training appearances; supplied evaluation labels are unchanged.\n\n";
    for (const std::string &image : images)
    {
        WriteImageIndex(root / image);
        index << "- [" << image << "](" << image << "/INDEX.md)\n";
    }
    index.close();
    if (!index)
        throw std::runtime_error("Cannot finish run index: " + root.string());

    std::ofstream summary(root / "run_summary.txt");
    if (!summary)
        throw std::runtime_error("Cannot write run overview: " + root.string());

    summary << "Pool vision run: " << root.filename().string() << '\n'
            << "Images processed: " << images.size() << "\n\n"
            << "Start with <image>/05_detections/validated_detections.png\n"
            << "Then inspect <image>/07_evaluation/localisation_evaluation.png\n"
            << "Detailed candidates: 04_candidates; ball crops: 06_ball_crops/<id>.\n\n"
            << "Pocket rejection uses static AllBalls reference differences only near six pockets.\n"
            << "Identity results: <image>/08_recognition/recognised_balls.png\n"
            << "Training audit and descriptors: training/.\n"
            << "AllBalls is training data; Table1--Table6 are evaluation images.\n"
            << "Supplied annotations are unchanged and may contain errors.\n\n";

    for (const std::string &image : images)
    {
        summary << "[" << image << "]\n";
        const auto metrics_path = root / image / "07_evaluation/localisation_metrics.csv";
        std::ifstream metrics(metrics_path);
        if (metrics)
            summary << metrics.rdbuf();
        else
            summary << "Evaluation unavailable; inspect annotation availability.\n";
        std::ifstream identities(root / image / "07_evaluation/recognition_metrics.csv");
        if (identities)
            summary << "Recognition (supplied reference):\n" << identities.rdbuf();
        summary << '\n';
    }
    summary.close();
    if (!summary)
        throw std::runtime_error("Cannot finish run overview: " + root.string());

    std::ofstream latest("diagnostics/latest.txt");
    latest << root.generic_string() << '\n';
    latest.close();
    if (!latest)
        throw std::runtime_error("Cannot write diagnostics/latest.txt");
}

struct Settings
{
    // Preprocessing and the independent Hough diagnostic branch.
    int median_kernel = 5;
    double hough_dp = 1.0, hough_spacing = 30.0;
    double hough_canny = 100.0, hough_votes = 15.0;
    int hough_min_radius = 15, hough_max_radius = 30;
    double canny_low = 50.0, canny_high = 100.0;

    // Cloth segmentation and one conservative opening.
    int cloth_h_min = 35, cloth_h_max = 95;
    int cloth_s_min = 25, cloth_s_max = 95, cloth_v_min = 25;
    int opening_kernel = 3, opening_iterations = 1;

    // Distance-surface maxima; thresholds below are in rectified pixels.
    int peak_neighbourhood = 21, distance_smooth_kernel = 5;
    double distance_smooth_sigma = 1.0;
    float minimum_interior_distance = 12.0f;
    double minimum_peak_spacing = 24.0;

    // Broad proposals: final acceptance happens in ballvision::Select().
    int ellipse_min_points = 20;
    double ellipse_min_minor = 10.0, ellipse_max_major = 150.0;

    // Measurement definitions only: NOT acceptance thresholds.
    int perimeter_samples = 360, angular_sectors = 12;
    double sector_support_fraction = 0.5, duplicate_report_iou = 0.5;
};
inline const Settings settings{};
struct PointRecord
{
    std::string id;
    cv::Point2f centre;
};
struct Seed
{
    std::string id;
    cv::Point point;
    int component;
    bool border;
};

inline std::ofstream Csv(const std::filesystem::path &path)
{
    const std::filesystem::path folder = path.parent_path();
    if (!folder.empty())
    {
        std::filesystem::create_directories(folder);
    }

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Cannot write diagnostics: " + path.string());
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::fixed << std::setprecision(5);
    return out;
}
inline std::string Quote(const std::string &s)
{
    std::string q = "\"";
    for (char c : s)
    {
        if (c == '\"')
            q += '\"';
        q += c;
    }
    return q + "\"";
}
inline void Save(const std::filesystem::path &path, const cv::Mat &image)
{
    if (image.empty())
    {
        throw std::runtime_error("Cannot save an empty diagnostic image: " + path.string());
    }

    const std::filesystem::path folder = path.parent_path();
    if (!folder.empty())
    {
        std::filesystem::create_directories(folder);
    }

    if (!cv::imwrite(path.string(), image))
    {
        throw std::runtime_error("Cannot save diagnostics: " + path.string());
    }
}
inline void Text(cv::Mat &image,
                 const std::string &text,
                 cv::Point p,
                 cv::Scalar colour = cv::Scalar(255, 255, 255),
                 double scale = 0.4)
{
    p.x = std::clamp(p.x, 0, std::max(0, image.cols - 100));
    p.y = std::clamp(p.y, 13, std::max(13, image.rows - 3));
    cv::putText(image, text, p, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, text, p, cv::FONT_HERSHEY_SIMPLEX, scale, colour, 1);
}
inline std::string RunTag()
{
    const char *env = std::getenv("POOL_RUN");
    if (env && *env)
    {
        const std::string tag(env);
        if (tag.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
            std::string::npos)
            throw std::runtime_error("POOL_RUN must contain only letters, numbers, '_' or '-'.");
        return tag;
    }
    const std::time_t now = std::time(nullptr);
    std::ostringstream tag;
    tag << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    return tag.str();
}
inline bool Border(const cv::Mat &stats, int c, cv::Size size)
{
    return stats.at<int>(c, cv::CC_STAT_LEFT) == 0 || stats.at<int>(c, cv::CC_STAT_TOP) == 0 ||
           stats.at<int>(c, cv::CC_STAT_LEFT) + stats.at<int>(c, cv::CC_STAT_WIDTH) == size.width ||
           stats.at<int>(c, cv::CC_STAT_TOP) + stats.at<int>(c, cv::CC_STAT_HEIGHT) == size.height;
}
inline cv::Mat Distance(const cv::Mat &mask)
{
    cv::Mat pad, d;
    cv::copyMakeBorder(mask, pad, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::distanceTransform(pad, d, cv::DIST_L2, 5);
    return d(cv::Rect(1, 1, mask.cols, mask.rows)).clone();
}
struct Ellipse
{
    std::string id;
    int contour = -1, parent = -1, depth = 0, points = 0;
    cv::RotatedRect fit;
    double minor = 0, major = 0, area = 0, residual = 0;
    double support1 = 0, support2 = 0, support3 = 0, visible = 0;
    double foreground = 0, ring_foreground = 0;
    int sectors_visible = 0, sectors_supported = 0;
    int own_sectors = 0, interior_seeds = 0, border_seeds = 0;
    double nearest_seed = -1;
    bool clipped = false, broad = false;
    std::string reason;
};
inline cv::Point2f Unit(const cv::RotatedRect &r, cv::Point2f p)
{
    const double a = r.angle * CV_PI / 180.0;
    const double dx = p.x - r.center.x, dy = p.y - r.center.y;
    return cv::Point2f(
        static_cast<float>((std::cos(a) * dx + std::sin(a) * dy) / (r.size.width * 0.5)),
        static_cast<float>((-std::sin(a) * dx + std::cos(a) * dy) / (r.size.height * 0.5)));
}
inline std::vector<Ellipse> Ellipses(const cv::Mat &edges,
                                     const cv::Mat &opened,
                                     const std::vector<Seed> &seeds,
                                     const std::filesystem::path &dir)
{
    const Settings &s = settings;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges.clone(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    cv::Mat inverse, edge_distance;
    cv::bitwise_not(edges, inverse);
    const bool has_edges = cv::countNonZero(edges) != 0;
    if (has_edges)
        cv::distanceTransform(inverse, edge_distance, cv::DIST_L2, 5);
    auto audit = Csv(pooldiag::OutputPath(dir, "contours.csv"));
    audit << "contour_id,parent_id,depth,points,contour_area,arc_length,ellipse_id,fit_status\n";
    std::vector<Ellipse> result;
    for (std::size_t i = 0; i < contours.size(); ++i)
    {
        const auto &c = contours[i];
        int depth = 0;
        for (int p = hierarchy[i][3]; p >= 0; p = hierarchy[p][3])
            ++depth;
        audit << i << ',' << hierarchy[i][3] << ',' << depth << ',' << c.size() << ','
              << cv::contourArea(c) << ',' << cv::arcLength(c, true) << ',';
        if (c.size() < 5)
        {
            audit << ",too_few_to_fit\n";
            continue;
        }
        Ellipse e;
        e.id = "E" + std::to_string(result.size() + 1);
        e.contour = static_cast<int>(i);
        e.parent = hierarchy[i][3];
        e.depth = depth;
        e.points = static_cast<int>(c.size());
        try
        {
            e.fit = cv::fitEllipse(c);
        }
        catch (const cv::Exception &)
        {
            audit << ",fit_exception\n";
            continue;
        }
        e.minor = std::min(e.fit.size.width, e.fit.size.height);
        e.major = std::max(e.fit.size.width, e.fit.size.height);
        if (!std::isfinite(e.fit.center.x) || !std::isfinite(e.fit.center.y) ||
            !std::isfinite(e.fit.angle) || !std::isfinite(e.minor) || !std::isfinite(e.major) ||
            e.minor <= 0 || e.major <= 0)
        {
            audit << ",degenerate_fit\n";
            continue;
        }
        e.area = CV_PI * e.minor * e.major * 0.25;
        if (e.points < s.ellipse_min_points)
            e.reason += "points_lt20;";
        if (e.minor < s.ellipse_min_minor)
            e.reason += "minor_lt10;";
        if (e.major > s.ellipse_max_major)
            e.reason += "major_gt150;";
        if (e.fit.center.x < 0 || e.fit.center.x >= edges.cols || e.fit.center.y < 0 ||
            e.fit.center.y >= edges.rows)
            e.reason += "centre_outside;";
        e.broad = e.reason.empty();
        if (e.broad)
            e.reason = "broad_pass_NOT_a_detection";
        std::vector<int> counts(s.angular_sectors, 0), hits(s.angular_sectors, 0),
            own(s.angular_sectors, 0);
        int visible = 0, h1 = 0, h2 = 0, h3 = 0;
        const double a = e.fit.angle * CV_PI / 180.0;
        for (int k = 0; k < s.perimeter_samples; ++k)
        {
            const double t = 2 * CV_PI * k / s.perimeter_samples;
            const double u = e.fit.size.width * 0.5 * std::cos(t),
                         v = e.fit.size.height * 0.5 * std::sin(t);
            const double x = e.fit.center.x + std::cos(a) * u - std::sin(a) * v;
            const double y = e.fit.center.y + std::sin(a) * u + std::cos(a) * v;
            if (x < 0 || x > edges.cols - 1 || y < 0 || y > edges.rows - 1)
                continue;
            const int sector =
                std::min(s.angular_sectors - 1, k * s.angular_sectors / s.perimeter_samples);
            ++visible;
            ++counts[sector];
            const float d = has_edges ? edge_distance.at<float>(cvRound(y), cvRound(x)) : 1e9f;
            h1 += d <= 1;
            h2 += d <= 2;
            h3 += d <= 3;
            hits[sector] += d <= 2;
        }
        e.visible = static_cast<double>(visible) / s.perimeter_samples;
        e.clipped = visible < s.perimeter_samples;
        if (visible)
        {
            e.support1 = double(h1) / visible;
            e.support2 = double(h2) / visible;
            e.support3 = double(h3) / visible;
        }
        for (const cv::Point &p : c)
        {
            const cv::Point2f u = Unit(e.fit, p);
            e.residual += std::abs(std::hypot(u.x, u.y) - 1.0);
            double t = std::atan2(u.y, u.x);
            if (t < 0)
                t += 2 * CV_PI;
            ++own[std::min(s.angular_sectors - 1, int(t * s.angular_sectors / (2 * CV_PI)))];
        }
        e.residual /= c.size();
        for (int k = 0; k < s.angular_sectors; ++k)
        {
            if (counts[k])
            {
                ++e.sectors_visible;
                if (double(hits[k]) / counts[k] >= s.sector_support_fraction)
                    ++e.sectors_supported;
            }
            if (own[k])
                ++e.own_sectors;
        }
        // Bound computation to the image even for numerically extreme rejected fits.
        const double extent = e.major * 0.65;
        auto clamp_x = [&](double x) { return int(std::clamp(x, 0.0, double(opened.cols))); };
        auto clamp_y = [&](double y) { return int(std::clamp(y, 0.0, double(opened.rows))); };
        const int left = clamp_x(std::floor(e.fit.center.x - extent)),
                  right = clamp_x(std::ceil(e.fit.center.x + extent + 1));
        const int top = clamp_y(std::floor(e.fit.center.y - extent)),
                  bottom = clamp_y(std::ceil(e.fit.center.y + extent + 1));
        int inside = 0, fg = 0, ring = 0, ringfg = 0;
        for (int y = top; y < bottom; ++y)
            for (int x = left; x < right; ++x)
            {
                const cv::Point2f u = Unit(e.fit, cv::Point2f(float(x), float(y)));
                const double r = std::hypot(u.x, u.y);
                if (r <= 1)
                {
                    ++inside;
                    fg += opened.at<uchar>(y, x) != 0;
                }
                else if (r >= 1.05 && r <= 1.25)
                {
                    ++ring;
                    ringfg += opened.at<uchar>(y, x) != 0;
                }
            }
        e.foreground = inside ? double(fg) / inside : -1;
        e.ring_foreground = ring ? double(ringfg) / ring : -1;
        for (const Seed &seed : seeds)
        {
            const double d = cv::norm(e.fit.center - cv::Point2f(seed.point));
            if (e.nearest_seed < 0 || d < e.nearest_seed)
                e.nearest_seed = d;
            const cv::Point2f u = Unit(e.fit, cv::Point2f(seed.point));
            if (std::hypot(u.x, u.y) <= 1.0)
            {
                if (seed.border)
                    ++e.border_seeds;
                else
                    ++e.interior_seeds;
            }
        }
        audit << e.id << ",ok\n";
        result.push_back(e);
    }
    return result;
}
inline void Components(const cv::Mat &mask,
                       const cv::Mat &hsv,
                       const cv::Mat &image,
                       const std::vector<Seed> &seeds,
                       const std::filesystem::path &dir,
                       const std::string &tag)
{
    cv::Mat labels, stats, centres;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centres, 8, CV_32S);
    const cv::Mat distance = Distance(mask);
    std::vector<double> sh(n, 0), ss(n, 0), sv(n, 0), maxd(n, 0);
    std::vector<int> seedn(n, 0);
    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
        {
            const int c = labels.at<int>(y, x);
            if (!c)
                continue;
            const cv::Vec3b p = hsv.at<cv::Vec3b>(y, x);
            sh[c] += p[0];
            ss[c] += p[1];
            sv[c] += p[2];
            maxd[c] = std::max(maxd[c], double(distance.at<float>(y, x)));
        }
    for (const Seed &seed : seeds)
    {
        const int c = labels.at<int>(seed.point);
        if (c)
            ++seedn[c];
    }
    auto out = Csv(pooldiag::OutputPath(dir, ("components_" + tag + ".csv")));
    out << "component_id,x,y,width,height,area,centroid_x,centroid_y,bbox_fill,border_connected,"
           "max_distance_px,retained_seeds,mean_h_linear,mean_s,mean_v\n";
    cv::Mat overlay = image.clone();
    for (int c = 1; c < n; ++c)
    {
        const int x = stats.at<int>(c, cv::CC_STAT_LEFT), y = stats.at<int>(c, cv::CC_STAT_TOP);
        const int w = stats.at<int>(c, cv::CC_STAT_WIDTH), h = stats.at<int>(c, cv::CC_STAT_HEIGHT),
                  area = stats.at<int>(c, cv::CC_STAT_AREA);
        const bool border = Border(stats, c, mask.size());
        out << c << ',' << x << ',' << y << ',' << w << ',' << h << ',' << area << ','
            << centres.at<double>(c, 0) << ',' << centres.at<double>(c, 1) << ','
            << double(area) / (w * h) << ',' << border << ',' << maxd[c] << ',' << seedn[c] << ','
            << sh[c] / area << ',' << ss[c] / area << ',' << sv[c] / area << '\n';
        if (area >= 100)
        {
            const cv::Scalar colour = border ? cv::Scalar(0, 165, 255) : cv::Scalar(255, 255, 0);
            cv::rectangle(overlay, cv::Rect(x, y, w, h), colour, 1);
            Text(overlay, "C" + std::to_string(c), cv::Point(x, y - 4), colour);
        }
    }
    Save(pooldiag::OutputPath(dir, ("components_" + tag + ".png")), overlay);
}
inline std::vector<Ellipse> Write(const std::filesystem::path &dir,
                                  const std::string &image_name,
                                  const cv::Mat &image,
                                  const cv::Mat &gray,
                                  const cv::Mat &edges,
                                  const cv::Mat &raw,
                                  const cv::Mat &opened,
                                  const cv::Mat &peaks,
                                  const cv::Mat &border_peaks,
                                  const std::vector<PointRecord> &hough,
                                  const std::vector<PointRecord> &truth,
                                  const cv::Mat &transform,
                                  cv::Size source_size)
{
    CV_Assert(image.type() == CV_8UC3 && edges.type() == CV_8UC1);
    CV_Assert(raw.size() == image.size() && opened.size() == image.size() &&
              peaks.size() == image.size() && border_peaks.size() == image.size());
    std::filesystem::create_directories(dir);
    const Settings &s = settings;
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat labels, stats, centres;
    cv::connectedComponentsWithStats(opened, labels, stats, centres, 8, CV_32S);
    const cv::Mat distance = Distance(opened);
    std::vector<Seed> seeds;
    // Stable IDs are scan-order, distinct from strength-ordered P/B labels in legacy overlays.
    for (int group = 0; group < 2; ++group)
    {
        std::vector<cv::Point> points;
        cv::findNonZero(group ? border_peaks : peaks, points);
        for (std::size_t i = 0; i < points.size(); ++i)
            seeds.push_back({std::string(group ? "SB" : "SI") + std::to_string(i + 1),
                             points[i],
                             labels.at<int>(points[i]),
                             group != 0});
    }
    const std::vector<Ellipse> ellipses = Ellipses(edges, opened, seeds, dir);
    auto es = Csv(pooldiag::OutputPath(dir, "ellipses.csv"));
    es << "ellipse_id,contour_id,parent_id,depth,points,x,y,minor_px,major_px,major_minor_ratio,"
          "angle_degrees,ellipse_area_px,normalized_radial_residual,own_contour_sectors,visible_"
          "fraction,edge_support_1px,edge_support_2px,edge_support_3px,visible_sectors,supported_"
          "sectors_2px,foreground_inside,foreground_outer_ring,interior_seeds_inside,border_seeds_"
          "inside,nearest_seed_px,clipped,broad_pass,broad_reason\n";
    cv::Mat numbered = image.clone(), supported = image.clone(), seed_overlay = image.clone();
    int broad_count = 0;
    for (const Ellipse &e : ellipses)
    {
        es << e.id << ',' << e.contour << ',' << e.parent << ',' << e.depth << ',' << e.points
           << ',' << e.fit.center.x << ',' << e.fit.center.y << ',' << e.minor << ',' << e.major
           << ',' << e.major / e.minor << ',' << e.fit.angle << ',' << e.area << ',' << e.residual
           << ',' << e.own_sectors << ',' << e.visible << ',' << e.support1 << ',' << e.support2
           << ',' << e.support3 << ',' << e.sectors_visible << ',' << e.sectors_supported << ','
           << e.foreground << ',' << e.ring_foreground << ',' << e.interior_seeds << ','
           << e.border_seeds << ',' << e.nearest_seed << ',' << e.clipped << ',' << e.broad << ','
           << Quote(e.reason) << '\n';
        if (!e.broad)
            continue;
        ++broad_count;
        cv::ellipse(numbered, e.fit, cv::Scalar(0, 255, 255), 1);
        Text(numbered, e.id, e.fit.center + cv::Point2f(5, -5), cv::Scalar(0, 255, 255));
        // Continuous support visualization, not a keep/reject threshold.
        const cv::Scalar colour(0, 255 * e.support2, 255 * (1 - e.support2));
        cv::ellipse(supported, e.fit, colour, 2);
        Text(supported, e.id, e.fit.center + cv::Point2f(5, -5), colour);
    }
    auto seedcsv = Csv(pooldiag::OutputPath(dir, "seeds.csv"));
    seedcsv << "seed_id,x,y,component_id,border_connected,raw_distance_px,h,s,v,broad_ellipses_"
               "containing_seed\n";
    for (const Seed &seed : seeds)
    {
        const cv::Vec3b hsv_value = hsv.at<cv::Vec3b>(seed.point);
        int containing = 0;
        for (const Ellipse &e : ellipses)
            if (e.broad)
            {
                const cv::Point2f u = Unit(e.fit, cv::Point2f(seed.point));
                if (std::hypot(u.x, u.y) <= 1)
                    ++containing;
            }
        seedcsv << seed.id << ',' << seed.point.x << ',' << seed.point.y << ',' << seed.component
                << ',' << seed.border << ',' << distance.at<float>(seed.point) << ','
                << int(hsv_value[0]) << ',' << int(hsv_value[1]) << ',' << int(hsv_value[2]) << ','
                << containing << '\n';
        const cv::Scalar colour = seed.border ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255);
        cv::drawMarker(seed_overlay, seed.point, colour, cv::MARKER_CROSS, 10, 1);
        Text(seed_overlay, seed.id, seed.point + cv::Point(5, -5), colour, 0.32);
    }
    auto pairs = Csv(pooldiag::OutputPath(dir, "overlapping_ellipses.csv"));
    pairs << "ellipse_a,ellipse_b,centre_distance_px,minor_ratio,major_ratio,visible_raster_iou,"
             "action\n";
    cv::Mat ma(image.size(), CV_8UC1), mb;
    std::vector<cv::Mat> raster;
    std::vector<const Ellipse *> broad;
    for (const Ellipse &e : ellipses)
        if (e.broad)
        {
            ma.setTo(0);
            cv::ellipse(ma, e.fit, 255, cv::FILLED);
            raster.push_back(ma.clone());
            broad.push_back(&e);
        }
    for (std::size_t i = 0; i < broad.size(); ++i)
        for (std::size_t j = i + 1; j < broad.size(); ++j)
        {
            const Ellipse &a = *broad[i], &b = *broad[j];
            if (cv::norm(a.fit.center - b.fit.center) > (a.major + b.major) * 0.5)
                continue;
            cv::bitwise_and(raster[i], raster[j], mb);
            const int intersection = cv::countNonZero(mb);
            const int total =
                cv::countNonZero(raster[i]) + cv::countNonZero(raster[j]) - intersection;
            const double iou = total ? double(intersection) / total : 0;
            if (iou >= s.duplicate_report_iou)
                pairs << a.id << ',' << b.id << ',' << cv::norm(a.fit.center - b.fit.center) << ','
                      << std::min(a.minor, b.minor) / std::max(a.minor, b.minor) << ','
                      << std::min(a.major, b.major) / std::max(a.major, b.major) << ',' << iou
                      << ",review_only_no_suppression\n";
        }
    Components(raw, hsv, image, seeds, dir, "raw");
    Components(opened, hsv, image, seeds, dir, "opened");
    // Annotation use starts here: evaluation only, after proposals and measurements.
    auto eval = Csv(pooldiag::OutputPath(dir, "ground_truth_checks.csv"));
    eval << "gt_label,x,y,in_bounds,h,s,v,raw_foreground_at_point,opened_foreground_at_point,"
            "opened_fraction_radius5,component_id,border_connected,nearest_interior_seed,interior_"
            "distance_px,nearest_border_seed,border_distance_px,nearest_broad_ellipse,ellipse_"
            "distance_px,nearest_hough,hough_distance_px\n";
    for (const PointRecord &gt : truth)
    {
        const bool valid = std::isfinite(gt.centre.x) && std::isfinite(gt.centre.y) &&
                           gt.centre.x >= 0 && gt.centre.x < image.cols && gt.centre.y >= 0 &&
                           gt.centre.y < image.rows;
        eval << Quote(gt.id) << ',' << gt.centre.x << ',' << gt.centre.y << ',' << valid;
        if (!valid)
        {
            for (int j = 0; j < 16; ++j)
                eval << ',';
            eval << '\n';
            continue;
        }
        const cv::Point p(std::clamp(cvRound(gt.centre.x), 0, image.cols - 1),
                          std::clamp(cvRound(gt.centre.y), 0, image.rows - 1));
        const cv::Vec3b v = hsv.at<cv::Vec3b>(p);
        const int c = labels.at<int>(p);
        int count = 0, fg = 0;
        for (int dy = -5; dy <= 5; ++dy)
            for (int dx = -5; dx <= 5; ++dx)
            {
                const cv::Point q = p + cv::Point(dx, dy);
                if (dx * dx + dy * dy <= 25 && q.x >= 0 && q.x < image.cols && q.y >= 0 &&
                    q.y < image.rows)
                {
                    ++count;
                    fg += opened.at<uchar>(q) != 0;
                }
            }
        eval << ',' << int(v[0]) << ',' << int(v[1]) << ',' << int(v[2]) << ','
             << (raw.at<uchar>(p) != 0) << ',' << (opened.at<uchar>(p) != 0) << ','
             << double(fg) / count << ',' << c << ',' << (c > 0 && Border(stats, c, image.size()));
        for (int kind = 0; kind < 4; ++kind)
        {
            double best = std::numeric_limits<double>::infinity();
            std::string id;
            auto consider = [&](const std::string &name, cv::Point2f pt)
            {
                const double d = cv::norm(pt - gt.centre);
                if (d < best)
                {
                    best = d;
                    id = name;
                }
            };
            if (kind < 2)
                for (const Seed &seed : seeds)
                {
                    if (seed.border == (kind == 1))
                        consider(seed.id, cv::Point2f(seed.point));
                }
            if (kind == 2)
                for (const Ellipse &e : ellipses)
                {
                    if (e.broad)
                        consider(e.id, e.fit.center);
                }
            if (kind == 3)
                for (const PointRecord &h : hough)
                    consider(h.id, h.centre);
            eval << ',' << id << ',';
            if (!id.empty())
                eval << best;
        }
        eval << '\n';
    }
    auto hc = Csv(pooldiag::OutputPath(dir, "hough_centres.csv"));
    hc << "candidate_id,x,y\n";
    for (const PointRecord &h : hough)
        hc << h.id << ',' << h.centre.x << ',' << h.centre.y << '\n';
    Save(pooldiag::OutputPath(dir, "ellipses_numbered.png"), numbered);
    Save(pooldiag::OutputPath(dir, "ellipses_edge_support.png"), supported);
    Save(pooldiag::OutputPath(dir, "seeds_all.png"), seed_overlay);
    Save(pooldiag::OutputPath(dir, "edges.png"), edges);
    Save(pooldiag::OutputPath(dir, "foreground_raw.png"), raw);
    Save(pooldiag::OutputPath(dir, "foreground_opened.png"), opened);
    Save(pooldiag::OutputPath(dir, "rectified.png"), image);
    Save(pooldiag::OutputPath(dir, "smoothed_gray.png"), gray);
    const int perpage = 12, cols = 3, tw = 390, th = 210;
    for (std::size_t offset = 0; offset < broad.size(); offset += perpage)
    {
        cv::Mat sheet(4 * th, cols * tw, CV_8UC3, cv::Scalar(25, 25, 25));
        for (std::size_t j = offset; j < std::min(broad.size(), offset + perpage); ++j)
        {
            const Ellipse &e = *broad[j];
            const int tile = int(j - offset), ox = tile % cols * tw, oy = tile / cols * th;
            const double pad = e.major * 0.65 + 8;
            const int l = std::max(0, int(std::floor(e.fit.center.x - pad))),
                      t = std::max(0, int(std::floor(e.fit.center.y - pad)));
            const int r = std::min(image.cols, int(std::ceil(e.fit.center.x + pad))),
                      b = std::min(image.rows, int(std::ceil(e.fit.center.y + pad)));
            const cv::Rect roi(l, t, std::max(1, r - l), std::max(1, b - t));
            cv::Mat patch = image(roi).clone();
            cv::RotatedRect local = e.fit;
            local.center -= cv::Point2f(float(l), float(t));
            cv::ellipse(patch, local, cv::Scalar(0, 255, 255), 1);
            for (int k = 0; k < 3; ++k)
            {
                cv::Mat p;
                if (k == 0)
                    p = patch;
                else
                    cv::cvtColor(k == 1 ? edges(roi) : opened(roi), p, cv::COLOR_GRAY2BGR);
                const double scale = std::min(122.0 / p.cols, 130.0 / p.rows);
                cv::Mat small;
                cv::resize(
                    p,
                    small,
                    cv::Size(std::max(1, int(p.cols * scale)), std::max(1, int(p.rows * scale))),
                    0,
                    0,
                    cv::INTER_NEAREST);
                small.copyTo(sheet(cv::Rect(ox + 5 + k * 128, oy + 24, small.cols, small.rows)));
            }
            std::ostringstream a, bline;
            a << e.id << " minor=" << std::fixed << std::setprecision(1) << e.minor
              << " major=" << e.major;
            bline << "edge2=" << std::setprecision(2) << e.support2 << " vis=" << e.visible
                  << " seeds=" << e.interior_seeds << "/" << e.border_seeds;
            Text(sheet, a.str(), cv::Point(ox + 5, oy + 17));
            Text(sheet, "RGB / Canny / opened mask", cv::Point(ox + 5, oy + 170));
            Text(sheet, bline.str(), cv::Point(ox + 5, oy + 190));
        }
        Save(pooldiag::OutputPath(
                 dir, ("ellipse_crops_" + std::to_string(offset / perpage + 1)) + ".png"),
             sheet);
    }
    cv::FileStorage manifest((pooldiag::OutputPath(dir, "settings.yml")).string(),
                             cv::FileStorage::WRITE);
    if (!manifest.isOpened())
        throw std::runtime_error("Cannot save diagnostics settings.");
    manifest << "diagnostics_version" << "measurements_v1" << "image" << image_name
             << "opencv_version" << CV_VERSION;
    manifest << "source_width" << source_size.width << "source_height" << source_size.height;
    manifest << "width" << image.cols << "height" << image.rows << "perspective_transform"
             << transform;
    manifest << "median_kernel" << s.median_kernel << "hough_dp" << s.hough_dp << "hough_spacing"
             << s.hough_spacing << "hough_canny" << s.hough_canny << "hough_votes" << s.hough_votes
             << "hough_min_radius" << s.hough_min_radius << "hough_max_radius"
             << s.hough_max_radius;
    manifest << "canny_low" << s.canny_low << "canny_high" << s.canny_high;
    manifest << "cloth_h_min" << s.cloth_h_min << "cloth_h_max" << s.cloth_h_max << "cloth_s_min"
             << s.cloth_s_min << "cloth_s_max" << s.cloth_s_max << "cloth_v_min" << s.cloth_v_min;
    manifest << "opening_kernel" << s.opening_kernel << "opening_iterations"
             << s.opening_iterations;
    manifest << "peak_neighbourhood" << s.peak_neighbourhood << "minimum_interior_distance"
             << s.minimum_interior_distance << "minimum_peak_spacing" << s.minimum_peak_spacing
             << "distance_smooth_kernel" << s.distance_smooth_kernel << "distance_smooth_sigma"
             << s.distance_smooth_sigma;
    manifest << "ellipse_min_points" << s.ellipse_min_points << "ellipse_min_minor"
             << s.ellipse_min_minor << "ellipse_max_major" << s.ellipse_max_major
             << "perimeter_samples" << s.perimeter_samples << "angular_sectors" << s.angular_sectors
             << "sector_support_fraction" << s.sector_support_fraction << "duplicate_report_iou"
             << s.duplicate_report_iou;
    auto summary = Csv(pooldiag::OutputPath(dir, "summary.txt"));
    summary
        << "Image: " << image_name << "\nDiagnostics: measurements_v1\nOpenCV: " << CV_VERSION
        << "\nValid ellipse fits: " << ellipses.size()
        << "\nBroad ellipse proposals (NOT detections): " << broad_count
        << "\nInterior retained seeds: " << cv::countNonZero(peaks)
        << "\nBorder retained seeds: " << cv::countNonZero(border_peaks)
        << "\nHough proposals: " << hough.size()
        << "\nAnnotations for evaluation only: " << truth.size()
        << "\nRaw foreground pixels: " << cv::countNonZero(raw)
        << "\nOpened foreground pixels: " << cv::countNonZero(opened)
        << "\nThis summary describes proposals before selector rejection and duplicate "
           "suppression.\n"
        << "Support is relative to visible perimeter samples only; inspect visible_fraction "
           "separately.\n"
        << "Sector support: at least 50% of visible samples within 2px of ANY Canny edge.\n"
        << "Contour residual is mean abs(normalized radius - 1), not pixel error.\n"
        << "Ground-truth nearest distances are independent, NOT one-to-one matches, precision or "
           "recall.\n"
        << "Radius-5 annotation patches are local mask checks, NOT full-ball mask coverage.\n"
        << "Hue means are linear summaries, not circular hue statistics.\n"
        << "SI/SB seed IDs are scan-order; legacy P/B overlays are strength-order.\n"
        << "E IDs belong to RETR_TREE diagnostics, not the unnumbered legacy RETR_LIST overlay.\n";
    return ellipses;
}
} // namespace pooldiag

// -----------------------------------------------------------------------------
// Candidate validation v2. No ground truth is accepted by this detector.
// -----------------------------------------------------------------------------
namespace ballvision
{
struct SelectorSettings
{
    // Physical prior from the brief; silhouette calibration uses AllBalls only.
    double diameter_mm = 43.0;

    // Visible-perimeter evidence, not a complete-binary-mask requirement.
    int boundary_samples = 96, sectors = 12;
    double edge_tolerance_px = 2.0;
    double minimum_support = 0.50, minimum_sector_fraction = 0.50;
    double minimum_visible = 0.55;

    // Broad geometry limits relative to the shared training shape.
    double minimum_minor_scale = 0.70, maximum_minor_scale = 1.45;
    double minimum_major_scale = 0.75, maximum_major_scale = 1.40;
    double maximum_axis_ratio = 3.0;

    // Duplicate hypotheses and normalised appearance output.
    double duplicate_iou = 0.50, duplicate_centre_radius = 0.65;
    int crop_size = 96;

    // Stationary-camera pocket reference, learned only from AllBalls.
    double pocket_half_width_d = 1.5, pocket_depth_d = 1.75;
    double background_difference = 18.0, minimum_changed_fraction = 0.20;
};
const SelectorSettings parameters{};
struct ShapeModel
{
    double minor = 0, major = 0;
    double angle_intercept = 0, angle_slope = 0;
    int training_outlines = 0;
    double Angle(double x) const
    {
        return angle_intercept + angle_slope * x;
    }
};
struct Boundary
{
    double support = 0, sector_fraction = 0, visible = 0;
    int visible_sectors = 0, supported_sectors = 0;
};
struct Candidate
{
    std::string id, source, reason, suppressed_by;
    cv::RotatedRect ellipse;
    Boundary boundary;
    double foreground = 0, ring_foreground = 0, score = 0;
    int interior_seeds = 0, border_seeds = 0;
    bool accepted = false, near_pocket = false;
    double changed_fraction = 1.0;
};

// The six search neighbourhoods come from table geometry, not test detections.
// Reject unchanged pocket texture, not every object near a pocket.
inline bool NearPocket(const cv::Point2f &q, cv::Size size)
{
    const double d = parameters.diameter_mm;
    const double half_width = parameters.pocket_half_width_d * d;
    const double depth = parameters.pocket_depth_d * d;
    const bool near_end = q.y < depth || q.y > size.height - 1 - depth;
    return near_end && (q.x < half_width || q.x > size.width - 1 - half_width ||
                       std::abs(q.x - 0.5 * (size.width - 1)) < half_width);
}

inline cv::Mat PocketChange(const cv::Mat &image, const cv::Mat &reference)
{
    cv::Mat current, background, difference;
    cv::GaussianBlur(image, current, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(reference, background, cv::Size(5, 5), 1.0);
    cv::cvtColor(current, current, cv::COLOR_BGR2Lab);
    cv::cvtColor(background, background, cv::COLOR_BGR2Lab);
    cv::absdiff(current, background, difference);
    cv::Mat changed = cv::Mat::zeros(image.size(), CV_8U);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
        {
            const auto v = difference.at<cv::Vec3b>(y, x);
            if (NearPocket(cv::Point2f(float(x), float(y)), image.size()) &&
                std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]) >
                    parameters.background_difference)
                changed.at<uchar>(y, x) = 255;
        }
    return changed;
}

inline double ChangedCore(const cv::RotatedRect &ellipse, const cv::Mat &changed)
{
    const cv::Rect box = ellipse.boundingRect() & cv::Rect(0, 0, changed.cols, changed.rows);
    int visible = 0, different = 0;
    for (int y = box.y; y < box.y + box.height; ++y)
        for (int x = box.x; x < box.x + box.width; ++x)
        {
            const auto u = pooldiag::Unit(ellipse, cv::Point2f(float(x), float(y)));
            if (std::hypot(u.x, u.y) <= 0.80)
            {
                ++visible;
                different += changed.at<uchar>(y, x) != 0;
            }
        }
    return visible ? double(different) / visible : 0;
}
inline double Median(std::vector<double> values)
{
    if (values.empty())
        throw std::runtime_error("Empty shape calibration.");
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}
inline cv::RotatedRect Canonical(cv::RotatedRect e)
{
    if (e.size.width > e.size.height)
    {
        std::swap(e.size.width, e.size.height);
        e.angle += 90.0f;
    }
    while (e.angle > 90.0f)
        e.angle -= 180.0f;
    while (e.angle <= -90.0f)
        e.angle += 180.0f;
    return e;
}
inline cv::Mat EdgeDistance(const cv::Mat &edges)
{
    cv::Mat inverse, distance;
    cv::bitwise_not(edges, inverse);
    cv::distanceTransform(inverse, distance, cv::DIST_L2, 5);
    return distance;
}
inline Boundary MeasureBoundary(const cv::RotatedRect &e, const cv::Mat &edge_distance)
{
    const auto &p = parameters;
    std::vector<int> counts(p.sectors, 0), hits(p.sectors, 0);
    int visible = 0, supported = 0;
    const double a = e.angle * CV_PI / 180.0, ca = std::cos(a), sa = std::sin(a);
    for (int k = 0; k < p.boundary_samples; ++k)
    {
        const double t = 2.0 * CV_PI * k / p.boundary_samples;
        const double u = 0.5 * e.size.width * std::cos(t);
        const double v = 0.5 * e.size.height * std::sin(t);
        const double x = e.center.x + ca * u - sa * v, y = e.center.y + sa * u + ca * v;
        if (x < 0 || y < 0 || x > edge_distance.cols - 1 || y > edge_distance.rows - 1)
            continue;
        const int sector = k * p.sectors / p.boundary_samples;
        ++visible;
        ++counts[sector];
        if (edge_distance.at<float>(cvRound(y), cvRound(x)) <= p.edge_tolerance_px)
        {
            ++supported;
            ++hits[sector];
        }
    }
    Boundary b;
    b.visible = double(visible) / p.boundary_samples;
    b.support = visible ? double(supported) / visible : 0;
    for (int i = 0; i < p.sectors; ++i)
        if (counts[i])
        {
            ++b.visible_sectors;
            b.supported_sectors += double(hits[i]) / counts[i] >= 0.5;
        }
    b.sector_fraction = b.visible_sectors ? double(b.supported_sectors) / b.visible_sectors : 0;
    return b;
}
inline bool Finite(const cv::RotatedRect &e)
{
    return std::isfinite(e.center.x) && std::isfinite(e.center.y) && std::isfinite(e.angle) &&
           std::isfinite(e.size.width) && std::isfinite(e.size.height) && e.size.width > 0 &&
           e.size.height > 0;
}
// Learns projected silhouette scale/orientation from AllBalls only.
// No annotations, target count, test images or fixed ball coordinates are used.
inline ShapeModel LearnShape(const cv::Mat &training_rectified)
{
    cv::Mat gray, smooth, edges;
    cv::cvtColor(training_rectified, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, smooth, pooldiag::settings.median_kernel);
    cv::Canny(smooth, edges, pooldiag::settings.canny_low, pooldiag::settings.canny_high);
    const cv::Mat distance = EdgeDistance(edges);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    std::vector<cv::RotatedRect> examples;
    const double d = parameters.diameter_mm; // Nominal 1 pixel/mm on the table plane.
    for (const auto &contour : contours)
    {
        if (contour.size() < 60)
            continue;
        cv::RotatedRect e;
        try
        {
            e = Canonical(cv::fitEllipse(contour));
        }
        catch (const cv::Exception &)
        {
            continue;
        }
        if (!Finite(e) || e.size.width < 0.8 * d || e.size.width > 1.5 * d ||
            e.size.height < 1.4 * d || e.size.height > 2.5 * d ||
            e.size.height / e.size.width < 1.3 || e.size.height / e.size.width > 2.7)
            continue;
        const double margin = 0.5 * e.size.height + 3.0;
        if (e.center.x < margin || e.center.y < margin || e.center.x > edges.cols - 1 - margin ||
            e.center.y > edges.rows - 1 - margin)
            continue;
        const Boundary b = MeasureBoundary(e, distance);
        if (b.support < 0.85 || b.sector_fraction < 0.75)
            continue;
        bool duplicate = false;
        for (const auto &previous : examples)
            if (cv::norm(e.center - previous.center) < 0.5 * d)
            {
                duplicate = true;
                break;
            }
        if (!duplicate)
            examples.push_back(e);
    }
    if (examples.size() < 3)
        throw std::runtime_error(
            "Too few reliable AllBalls outlines for shape calibration; inspect training edges.");
    std::vector<double> minors, majors;
    for (const auto &e : examples)
    {
        minors.push_back(e.size.width);
        majors.push_back(e.size.height);
    }
    ShapeModel model;
    model.minor = Median(minors);
    model.major = Median(majors);
    std::vector<cv::RotatedRect> clean;
    for (const auto &e : examples)
        if (std::abs(e.size.width / model.minor - 1) < 0.25 &&
            std::abs(e.size.height / model.major - 1) < 0.25)
            clean.push_back(e);
    double mx = 0, ma = 0;
    for (const auto &e : clean)
    {
        mx += e.center.x;
        ma += e.angle;
    }
    mx /= clean.size();
    ma /= clean.size();
    double xx = 0, xa = 0;
    for (const auto &e : clean)
    {
        xx += (e.center.x - mx) * (e.center.x - mx);
        xa += (e.center.x - mx) * (e.angle - ma);
    }
    model.angle_slope = xx > 1e-6 ? xa / xx : 0;
    model.angle_intercept = ma - model.angle_slope * mx;
    model.training_outlines = static_cast<int>(clean.size());
    return model;
}
inline void MaskEvidence(Candidate &c, const cv::Mat &opened)
{
    cv::Rect box = c.ellipse.boundingRect();
    const int expand = cvCeil(c.ellipse.size.height * 0.2);
    box.x -= expand;
    box.y -= expand;
    box.width += 2 * expand;
    box.height += 2 * expand;
    box &= cv::Rect(0, 0, opened.cols, opened.rows);
    int inside = 0, fg = 0, ring = 0, ringfg = 0;
    for (int y = box.y; y < box.y + box.height; y += 2)
        for (int x = box.x; x < box.x + box.width; x += 2)
        {
            const cv::Point2f u = pooldiag::Unit(c.ellipse, cv::Point2f(float(x), float(y)));
            const double r = std::hypot(u.x, u.y);
            if (r <= 0.95)
            {
                ++inside;
                fg += opened.at<uchar>(y, x) != 0;
            }
            else if (r >= 1.08 && r <= 1.25)
            {
                ++ring;
                ringfg += opened.at<uchar>(y, x) != 0;
            }
        }
    c.foreground = inside ? double(fg) / inside : 0;
    c.ring_foreground = ring ? double(ringfg) / ring : 0;
}
inline double ShapeQuality(const cv::RotatedRect &e, const ShapeModel &model)
{
    return std::exp(-2.0 * (std::abs(std::log(e.size.width / model.minor)) +
                            std::abs(std::log(e.size.height / model.major))));
}
inline double SearchScore(const cv::RotatedRect &e,
                          const ShapeModel &model,
                          const cv::Mat &distance)
{
    const Boundary b = MeasureBoundary(e, distance);
    if (b.visible < parameters.minimum_visible)
        return -1;
    return 0.70 * b.support + 0.20 * b.sector_fraction + 0.10 * ShapeQuality(e, model);
}
// Fit a whole-ball template near a seed; do not mistake distance maxima for centres.
inline cv::RotatedRect Recover(const cv::Point2f &seed,
                               const ShapeModel &model,
                               const cv::Mat &distance)
{
    cv::RotatedRect best(
        seed, cv::Size2f(float(model.minor), float(model.major)), float(model.Angle(seed.x)));
    double best_score = -1;
    const int rx = cvCeil(model.minor * 0.30), ry = cvCeil(model.major * 0.35);
    for (int dy = -ry; dy <= ry; dy += 4)
        for (int dx = -rx; dx <= rx; dx += 4)
            for (const double scale : {0.90, 1.0, 1.10})
                for (const double da : {-10.0, 0.0, 10.0})
                {
                    cv::RotatedRect e(
                        seed + cv::Point2f(float(dx), float(dy)),
                        cv::Size2f(float(model.minor * scale), float(model.major * scale)),
                        float(model.Angle(seed.x + dx) + da));
                    if (e.center.x < 0 || e.center.y < 0 || e.center.x >= distance.cols ||
                        e.center.y >= distance.rows)
                        continue;
                    const double score =
                        SearchScore(e, model, distance) -
                        0.015 * std::hypot(double(dx) / model.minor, double(dy) / model.major);
                    if (score > best_score)
                    {
                        best_score = score;
                        best = e;
                    }
                }
    for (int iteration = 0; iteration < 2; ++iteration)
    {
        const cv::RotatedRect start = best;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
                for (const double scale : {0.97, 1.0, 1.03})
                {
                    cv::RotatedRect e = start;
                    e.center += cv::Point2f(float(dx), float(dy));
                    e.size.width *= float(scale);
                    e.size.height *= float(scale);
                    if (e.center.x < 0 || e.center.y < 0 || e.center.x >= distance.cols ||
                        e.center.y >= distance.rows)
                        continue;
                    const double score = SearchScore(e, model, distance) -
                                         0.015 * cv::norm(e.center - seed) / model.major;
                    if (score > best_score)
                    {
                        best_score = score;
                        best = e;
                    }
                }
    }
    return best;
}
inline double EllipseOverlap(const cv::RotatedRect &a, const cv::RotatedRect &b,
                         const cv::Size &visible = cv::Size())
{
    cv::Rect roi = (a.boundingRect() | b.boundingRect());
    // At the frame edge, compare observed pixels, not off-image geometry.
    if (!visible.empty())
        roi &= cv::Rect(cv::Point(), visible);
    if (roi.area() <= 0)
        return 0;
    cv::Mat am = cv::Mat::zeros(roi.size(), CV_8U), bm = am.clone(), intersection, united;
    cv::RotatedRect ar = a, br = b;
    ar.center -= cv::Point2f(roi.tl());
    br.center -= cv::Point2f(roi.tl());
    cv::ellipse(am, ar, 255, cv::FILLED);
    cv::ellipse(bm, br, 255, cv::FILLED);
    cv::bitwise_and(am, bm, intersection);
    cv::bitwise_or(am, bm, united);
    const int total = visible.empty() ? cv::countNonZero(united)
        : std::min(cv::countNonZero(am), cv::countNonZero(bm));
    return total ? double(cv::countNonZero(intersection)) / total : 0;
}
inline void ExportCrops(const cv::Mat &image,
                        const cv::Mat &opened,
                        const std::vector<Candidate> &balls,
                        const std::filesystem::path &dir)
{
    const auto crops = dir / "06_ball_crops";
    std::filesystem::create_directories(crops);
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    auto csv = pooldiag::Csv(pooldiag::OutputPath(dir, "crop_quality.csv"));
    csv << "detection_id,visible_core_pixels,owned_core_pixels,appearance_pixels,white_fraction,"
           "dark_fraction,colour_fraction\n";
    for (const auto &ball : balls)
    {
        const auto ball_dir = crops / ball.id;
        std::filesystem::create_directories(ball_dir);
        const int n = parameters.crop_size;
        cv::Mat mx(n, n, CV_32F), my(n, n, CV_32F);
        cv::Mat owned = cv::Mat::zeros(n, n, CV_8U), appearance = owned.clone();
        const double a = ball.ellipse.angle * CV_PI / 180.0;
        int visible = 0, kept = 0, used = 0, white = 0, dark = 0, colour = 0;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x)
            {
                const double u = (x + 0.5 - n * 0.5) / (n * 0.5),
                             v = (y + 0.5 - n * 0.5) / (n * 0.5);
                const double px = ball.ellipse.center.x +
                                  std::cos(a) * u * ball.ellipse.size.width * 0.5 -
                                  std::sin(a) * v * ball.ellipse.size.height * 0.5;
                const double py = ball.ellipse.center.y +
                                  std::sin(a) * u * ball.ellipse.size.width * 0.5 +
                                  std::cos(a) * v * ball.ellipse.size.height * 0.5;
                mx.at<float>(y, x) = float(px);
                my.at<float>(y, x) = float(py);
                const double ownr = std::hypot(u, v);
                if (ownr > 0.88 || px < 0 || py < 0 || px > image.cols - 1 || py > image.rows - 1)
                    continue;
                ++visible;
                bool belongs = true;
                for (const auto &other : balls)
                    if (other.id != ball.id)
                    {
                        const cv::Point2f q =
                            pooldiag::Unit(other.ellipse, cv::Point2f(float(px), float(py)));
                        if (std::hypot(q.x, q.y) < ownr)
                        {
                            belongs = false;
                            break;
                        }
                    }
                if (!belongs)
                    continue;
                owned.at<uchar>(y, x) = 255;
                ++kept;
                const cv::Vec3b value = hsv.at<cv::Vec3b>(cvRound(py), cvRound(px));
                // Preserve bright/desaturated green-ball pixels that failed the cloth mask.
                // Only discard pixels that ALSO have cloth-like brightness/saturation.
                const bool cloth_like = opened.at<uchar>(cvRound(py), cvRound(px)) == 0 &&
                                        value[0] >= pooldiag::settings.cloth_h_min &&
                                        value[0] <= pooldiag::settings.cloth_h_max &&
                                        value[1] >= 25 && value[1] <= 95 && value[2] >= 25 &&
                                        value[2] < 180;
                if (cloth_like)
                    continue;
                appearance.at<uchar>(y, x) = 255;
                ++used;
                if (value[2] < 65)
                    ++dark;
                else if (value[1] < 60 && value[2] > 140)
                    ++white;
                else
                    ++colour;
            }
        cv::Mat patch, core, clean;
        cv::remap(image, patch, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        patch.copyTo(core, owned);
        patch.copyTo(clean, appearance);
        pooldiag::Save(ball_dir / "rgb.png", patch);
        pooldiag::Save(ball_dir / "core.png", core);
        pooldiag::Save(ball_dir / "core_mask.png", owned);
        pooldiag::Save(ball_dir / "appearance.png", clean);
        pooldiag::Save(ball_dir / "appearance_mask.png", appearance);
        csv << ball.id << ',' << visible << ',' << kept << ',' << used << ','
            << (used ? double(white) / used : 0) << ',' << (used ? double(dark) / used : 0) << ','
            << (used ? double(colour) / used : 0) << '\n';
    }
}
inline std::vector<Candidate> Select(const cv::Mat &image,
                                     const cv::Mat &edges,
                                     const cv::Mat &opened,
                                     const cv::Mat &interior_seeds,
                                     const cv::Mat &border_seeds,
                                     const ShapeModel &model,
                                     const std::vector<pooldiag::Ellipse> &ellipses,
                                     const cv::Mat &pocket_reference,
                                     const std::filesystem::path &dir)
{
    const auto &p = parameters;
    CV_Assert(image.size() == pocket_reference.size());
    const cv::Mat changed = PocketChange(image, pocket_reference);
    pooldiag::Save(pooldiag::OutputPath(dir, "pocket_change.png"), changed);
    const cv::Mat edge_distance = EdgeDistance(edges);
    cv::Mat reference_gray, reference_smooth, reference_edges;
    cv::cvtColor(pocket_reference, reference_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(reference_gray, reference_smooth, pooldiag::settings.median_kernel);
    cv::Canny(reference_smooth, reference_edges,
              pooldiag::settings.canny_low, pooldiag::settings.canny_high);
    const cv::Mat reference_distance = EdgeDistance(reference_edges);
    std::vector<cv::Point> interior, border;
    cv::findNonZero(interior_seeds, interior);
    cv::findNonZero(border_seeds, border);
    std::vector<Candidate> candidates;
    for (const auto &e : ellipses)
        if (e.broad)
        {
            Candidate c;
            c.id = e.id;
            c.source = "contour";
            c.ellipse = Canonical(e.fit);
            candidates.push_back(c);
            // A numeral or partial arc can locate a dark ball even when the
            // foreground distance map has no separate maximum for its body.
            if (c.ellipse.size.width < p.minimum_minor_scale * model.minor ||
                c.ellipse.size.height < p.minimum_major_scale * model.major)
            {
                Candidate recovered;
                recovered.id = "RC" + e.id;
                recovered.source = "contour_centre_recovery";
                recovered.ellipse = Recover(c.ellipse.center, model, edge_distance);
                candidates.push_back(recovered);
            }
        }
    const cv::Mat distance = pooldiag::Distance(opened);
    for (int group = 0; group < 2; ++group)
    {
        const auto &points = group ? border : interior;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            // Large rail interiors do not define a plausible local ball radius.
            // This is a scale check, NOT rejection for border connectivity.
            if (distance.at<float>(points[i]) > 0.85 * model.minor)
                continue;
            Candidate c;
            c.id = std::string(group ? "RB" : "RI") + std::to_string(i + 1);
            c.source = group ? "border_seed_recovery" : "interior_seed_recovery";
            c.ellipse = Recover(cv::Point2f(points[i]), model, edge_distance);
            candidates.push_back(c);
        }
    }
    for (auto &c : candidates)
    {
        const auto &e = c.ellipse;
        if (!Finite(e) || e.center.x < 0 || e.center.y < 0 || e.center.x >= image.cols ||
            e.center.y >= image.rows)
        {
            c.reason = "invalid_geometry";
            continue;
        }
        c.boundary = MeasureBoundary(e, edge_distance);
        c.near_pocket = NearPocket(e.center, image.size());
        c.changed_fraction = c.near_pocket ? ChangedCore(e, changed) : 1.0;
        if (c.near_pocket && c.changed_fraction < p.minimum_changed_fraction)
            c.reason += "unchanged_pocket_background;";
        const double reference_support = MeasureBoundary(e, reference_distance).support;
        if (c.near_pocket &&
            c.boundary.support - reference_support < 0.50 * c.boundary.support)
            c.reason += "static_reference_boundary;";
        MaskEvidence(c, opened);
        if (c.source == "contour_centre_recovery" && c.boundary.visible < 1.0 &&
            c.boundary.support < p.minimum_visible &&
            c.foreground - c.ring_foreground < 0.10)
            c.reason += "weak_clipped_recovery;";
        for (const auto &q : interior)
        {
            const auto u = pooldiag::Unit(e, cv::Point2f(q));
            c.interior_seeds += std::hypot(u.x, u.y) <= 1;
        }
        for (const auto &q : border)
        {
            const auto u = pooldiag::Unit(e, cv::Point2f(q));
            c.border_seeds += std::hypot(u.x, u.y) <= 1;
        }
        if (e.size.width < p.minimum_minor_scale * model.minor ||
            e.size.width > p.maximum_minor_scale * model.minor ||
            e.size.height < p.minimum_major_scale * model.major ||
            e.size.height > p.maximum_major_scale * model.major ||
            e.size.height / e.size.width > p.maximum_axis_ratio)
            c.reason += "whole_ball_scale;";
        if (c.boundary.visible < p.minimum_visible)
            c.reason += "too_little_visible;";
        if (c.boundary.support < p.minimum_support)
            c.reason += "weak_boundary;";
        if (c.boundary.sector_fraction < p.minimum_sector_fraction)
            c.reason += "short_supported_arc;";
        // Mask evidence is a soft score only: cloth-coloured balls can be incomplete.
        const double contrast = std::max(0.0, c.foreground - c.ring_foreground);
        c.score = 0.60 * c.boundary.support + 0.20 * c.boundary.sector_fraction +
                  0.10 * ShapeQuality(e, model) + 0.10 * contrast;
    }
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (candidates[i].reason.empty())
            order.push_back(i);
    std::stable_sort(order.begin(),
                     order.end(),
                     [&](std::size_t a, std::size_t b)
                     { return candidates[a].score > candidates[b].score; });
    std::vector<Candidate> accepted;
    for (const std::size_t i : order)
    {
        Candidate &c = candidates[i];
        for (const auto &keep : accepted)
        {
            const auto u = pooldiag::Unit(keep.ellipse, c.ellipse.center);
            const auto v = pooldiag::Unit(c.ellipse, keep.ellipse.center);
            const double separation = std::min(std::hypot(u.x, u.y), std::hypot(v.x, v.y));
            // A clipped duplicate can have low full-ellipse IoU because most
            // of its extrapolated area lies outside the frame.
            const bool clipped_duplicate = c.boundary.visible < 1.0 &&
                EllipseOverlap(c.ellipse, keep.ellipse, image.size()) > p.duplicate_iou;
            if (separation < p.duplicate_centre_radius || clipped_duplicate ||
                EllipseOverlap(c.ellipse, keep.ellipse) > p.duplicate_iou)
            {
                c.reason = clipped_duplicate ? "clipped_duplicate" : "duplicate";
                c.suppressed_by = keep.id;
                break;
            }
        }
        if (c.reason.empty())
        {
            c.accepted = true;
            c.reason = "accepted";
            accepted.push_back(c);
        }
    }
    auto audit = pooldiag::Csv(pooldiag::OutputPath(dir, "selection.csv"));
    audit << "candidate_id,source,x,y,minor_px,major_px,angle,edge_support,sector_fraction,visible_"
             "fraction,foreground_inside,foreground_ring,interior_seeds,border_seeds,score,"
             "accepted,reason,suppressed_by,near_pocket,changed_core_fraction,reference_edge_support,boundary_support_gain\n";
    cv::Mat selected = image.clone(), rejected = image.clone();
    for (const auto &c : candidates)
    {
        audit << c.id << ',' << c.source << ',' << c.ellipse.center.x << ',' << c.ellipse.center.y
              << ',' << c.ellipse.size.width << ',' << c.ellipse.size.height << ','
              << c.ellipse.angle << ',' << c.boundary.support << ',' << c.boundary.sector_fraction
              << ',' << c.boundary.visible << ',' << c.foreground << ',' << c.ring_foreground << ','
              << c.interior_seeds << ',' << c.border_seeds << ',' << c.score << ',' << c.accepted
              << ',' << pooldiag::Quote(c.reason) << ',' << c.suppressed_by << ',' << c.near_pocket
              << ',' << c.changed_fraction << ','
              << MeasureBoundary(c.ellipse, reference_distance).support << ','
              << c.boundary.support - MeasureBoundary(c.ellipse, reference_distance).support << '\n';
        cv::Mat &display = c.accepted ? selected : rejected;
        cv::ellipse(
            display, c.ellipse, c.accepted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
        pooldiag::Text(display, c.id, c.ellipse.center + cv::Point2f(4, -4));
    }
    auto positions = pooldiag::Csv(pooldiag::OutputPath(dir, "detections.csv"));
    positions
        << "detection_id,x_table_px,y_table_px,minor_px,major_px,angle_degrees,score,source\n";
    for (const auto &c : accepted)
        positions << c.id << ',' << c.ellipse.center.x << ',' << c.ellipse.center.y << ','
                  << c.ellipse.size.width << ',' << c.ellipse.size.height << ',' << c.ellipse.angle
                  << ',' << c.score << ',' << c.source << '\n';
    pooldiag::Save(pooldiag::OutputPath(dir, "validated_detections.png"), selected);
    pooldiag::Save(pooldiag::OutputPath(dir, "rejected_candidates.png"), rejected);
    ExportCrops(image, opened, accepted, dir);
    cv::FileStorage manifest((pooldiag::OutputPath(dir, "selector_settings.yml")).string(),
                             cv::FileStorage::WRITE);
    manifest << "version" << "candidate_validation_v4_clipped_overlap"
             << "clipped_overlap_fraction" << p.duplicate_iou
             << "shape_training_image" << "AllBalls.jpeg"
             << "training_outlines" << model.training_outlines << "model_minor_px" << model.minor
             << "model_major_px" << model.major << "angle_intercept" << model.angle_intercept
             << "angle_slope" << model.angle_slope << "nominal_ball_diameter_mm" << p.diameter_mm
             << "minimum_support" << p.minimum_support << "minimum_sector_fraction"
             << p.minimum_sector_fraction << "minimum_visible" << p.minimum_visible
             << "minimum_minor_scale" << p.minimum_minor_scale << "maximum_minor_scale"
             << p.maximum_minor_scale << "minimum_major_scale" << p.minimum_major_scale
             << "maximum_major_scale" << p.maximum_major_scale << "maximum_axis_ratio"
             << p.maximum_axis_ratio << "duplicate_iou" << p.duplicate_iou
             << "duplicate_centre_radius" << p.duplicate_centre_radius << "edge_tolerance_px"
             << p.edge_tolerance_px << "boundary_samples" << p.boundary_samples << "sectors"
             << p.sectors << "pocket_half_width_d" << p.pocket_half_width_d
             << "pocket_depth_d" << p.pocket_depth_d << "background_difference_Lab8"
             << p.background_difference << "minimum_changed_fraction" << p.minimum_changed_fraction;
    auto summary = pooldiag::Csv(pooldiag::OutputPath(dir, "selector_summary.txt"));
    summary << "Candidate validation v3 (static pocket reference)\nAllBalls-only outline calibration: "
            << model.training_outlines << "\nModel axes: " << model.minor << ", " << model.major
            << "\nContour plus seed-recovery proposals: " << candidates.size()
            << "\nRetained detections (NOT recognised identities): " << accepted.size()
            << "\nBorder connectivity is not a rejection rule.\nMask evidence is soft, not "
               "required foreground coverage.\n"
            << "Positions are apparent rectified silhouette centres, not corrected table-plane "
               "contact points.\n"
            << "Crops are geometric estimates; overlap ownership is not a physical occlusion "
               "reconstruction.\n"
            << "00_settings/summary.txt describes proposal measurements, not selector decisions.\n";
    std::cout << "Candidate validation v3: " << accepted.size() << " retained from "
              << candidates.size() << " proposals.\n";
    return accepted;
}
} // namespace ballvision

// -----------------------------------------------------------------------------
// Ground-truth loading and downstream localisation evaluation
// -----------------------------------------------------------------------------
struct GroundTruthBall
{
    std::string label;
    cv::Point2f centre;
};

std::vector<GroundTruthBall> LoadGroundTruth(const std::string &csv_path,
                                             const std::string &image_name)
{
    std::ifstream file(csv_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open ground-truth file: " + csv_path);
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
            throw std::runtime_error("Incomplete ground-truth coordinate for " + image + ", ball " +
                                     label);
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

cv::Mat DrawGroundTruth(const cv::Mat &rectified_image,
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

        if (x < 0 || x >= rectified_image.cols || y < 0 || y >= rectified_image.rows)
        {
            std::cerr << "Ground-truth warning: " << image_name << ", ball " << ball.label
                      << ", row=" << y << ", column=" << x << " is outside the rectified image.\n";

            ++invalid_count;
            continue;
        }

        const cv::Point centre(cvRound(x), cvRound(y));

        cv::drawMarker(display, centre, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 2);

        cv::putText(display,
                    ball.label,
                    centre + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 255),
                    1);

        ++drawn_count;
    }

    std::cout << image_name << ": " << balls.size() << " nonblank annotations, " << drawn_count
              << " drawn, " << invalid_count << " outside image bounds.\n";

    return display;
}

// Evaluation is strictly downstream of detection. Annotation labels never
// participate in localisation matching or candidate acceptance.
std::vector<int> EvaluateLocalisation(const cv::Mat &image,
                          const std::vector<ballvision::Candidate> &detections,
                          const std::vector<GroundTruthBall> &truth,
                          const std::filesystem::path &dir)
{
    if (truth.empty())
        return std::vector<int>(detections.size(), -1);
    const double tolerance = 0.5 * ballvision::parameters.diameter_mm;
    const int nd = static_cast<int>(detections.size()), ng = static_cast<int>(truth.size());
    const int n = nd + ng;
    // Each unmatched endpoint pays penalty; choose penalty large enough that
    // cardinality dominates the total possible localisation error.
    const double penalty = (n + 1) * tolerance + 1;
    const double forbidden = 4 * (n + 1) * penalty;
    std::vector<std::vector<double>> cost(n + 1, std::vector<double>(n + 1, 0));
    auto in_bounds = [&](const cv::Point2f &q)
    {
        return std::isfinite(q.x) && std::isfinite(q.y) && q.x >= 0 && q.y >= 0 &&
               q.x < image.cols && q.y < image.rows;
    };
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
        {
            if (i <= nd && j <= ng)
            {
                const double d = cv::norm(detections[i - 1].ellipse.center - truth[j - 1].centre);
                cost[i][j] = in_bounds(truth[j - 1].centre) && d <= tolerance ? d : forbidden;
            }
            else if (i <= nd || j <= ng)
                cost[i][j] = penalty;
        }
    // Hungarian assignment with dummy rows/columns: maximum valid cardinality,
    // then minimum total Euclidean distance. Not greedy nearest-neighbour.
    std::vector<double> u(n + 1, 0), v(n + 1, 0);
    std::vector<int> assigned(n + 1, 0), previous(n + 1, 0);
    for (int i = 1; i <= n; ++i)
    {
        assigned[0] = i;
        int j0 = 0;
        std::vector<double> minimum(n + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(n + 1, false);
        do
        {
            used[j0] = true;
            const int i0 = assigned[j0];
            int j1 = 0;
            double delta = std::numeric_limits<double>::infinity();
            for (int j = 1; j <= n; ++j)
                if (!used[j])
                {
                    const double current = cost[i0][j] - u[i0] - v[j];
                    if (current < minimum[j])
                    {
                        minimum[j] = current;
                        previous[j] = j0;
                    }
                    if (minimum[j] < delta)
                    {
                        delta = minimum[j];
                        j1 = j;
                    }
                }
            for (int j = 0; j <= n; ++j)
                if (used[j])
                {
                    u[assigned[j]] += delta;
                    v[j] -= delta;
                }
                else
                    minimum[j] -= delta;
            j0 = j1;
        } while (assigned[j0] != 0);
        do
        {
            const int j1 = previous[j0];
            assigned[j0] = assigned[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int> match(nd, -1);
    std::vector<bool> matched_truth(ng, false);
    for (int j = 1; j <= ng; ++j)
        if (assigned[j] > 0 && assigned[j] <= nd && cost[assigned[j]][j] <= tolerance)
        {
            match[assigned[j] - 1] = j - 1;
            matched_truth[j - 1] = true;
        }
    cv::Mat overlay = image.clone();
    auto audit = pooldiag::Csv(pooldiag::OutputPath(dir, "localisation_matches.csv"));
    audit << "detection_id,annotation_label,status,detection_x,detection_y,annotation_x,annotation_"
             "y,error_px\n";
    std::vector<double> errors;
    int tp = 0, outside = 0;
    for (const auto &g : truth)
        outside += !in_bounds(g.centre);
    for (int i = 0; i < nd; ++i)
    {
        const auto &d = detections[i];
        const bool hit = match[i] >= 0;
        cv::ellipse(overlay, d.ellipse, hit ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        pooldiag::Text(overlay, d.id, d.ellipse.center);
        if (hit)
        {
            const auto &g = truth[match[i]];
            const double error = cv::norm(d.ellipse.center - g.centre);
            ++tp;
            errors.push_back(error);
            cv::line(overlay, d.ellipse.center, g.centre, cv::Scalar(255, 255, 0), 1);
            cv::drawMarker(overlay, g.centre, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 10, 1);
            audit << d.id << ',' << pooldiag::Quote(g.label) << ",TP," << d.ellipse.center.x << ','
                  << d.ellipse.center.y << ',' << g.centre.x << ',' << g.centre.y << ',' << error
                  << '\n';
        }
        else
            audit << d.id << ",,FP," << d.ellipse.center.x << ',' << d.ellipse.center.y << ",,,\n";
    }
    for (int j = 0; j < ng; ++j)
        if (!matched_truth[j])
        {
            const auto &g = truth[j];
            if (in_bounds(g.centre))
            {
                cv::drawMarker(overlay, g.centre, cv::Scalar(0, 165, 255), cv::MARKER_CROSS, 14, 2);
                pooldiag::Text(overlay, "miss " + g.label, g.centre);
            }
            audit << ',' << pooldiag::Quote(g.label) << ",FN,,," << g.centre.x << ',' << g.centre.y
                  << ",\n";
        }
    const int fp = nd - tp, fn = ng - tp;
    auto metrics = pooldiag::Csv(pooldiag::OutputPath(dir, "localisation_metrics.csv"));
    metrics << "metric,value\nmatching_tolerance_px," << tolerance << "\nTP," << tp << "\nFP," << fp
            << "\nFN," << fn << "\nannotations_outside_image," << outside << "\nprecision,";
    if (nd)
        metrics << double(tp) / nd;
    metrics << "\nrecall," << double(tp) / ng << "\nF1," << double(2 * tp) / (nd + ng)
            << "\nmean_error_px,";
    if (!errors.empty())
    {
        double sum = 0, squared = 0;
        for (double e : errors)
        {
            sum += e;
            squared += e * e;
        }
        metrics << sum / errors.size() << "\nmedian_error_px," << ballvision::Median(errors)
                << "\nRMSE_px," << std::sqrt(squared / errors.size());
    }
    metrics << "\n";
    pooldiag::Save(pooldiag::OutputPath(dir, "localisation_evaluation.png"), overlay);
    auto notes = pooldiag::Csv(pooldiag::OutputPath(dir, "evaluation_notes.txt"));
    notes << "Reference: supplied annotations, unchanged.\n"
          << "Matching: maximise valid one-to-one matches, then minimise summed centre distance.\n"
          << "Tolerance: 21.5 rectified pixels (half the nominal 43mm diameter), fixed across "
             "images.\n"
          << "Out-of-image annotations remain in the denominator and cannot match: inspect "
             "separately.\n"
          << "Absent annotations do not imply an image is a negative example; evaluation is "
             "skipped.\n"
          << "Units are rectified pixels; silhouette centres are not corrected ball/table contact "
             "points.\n"
          << "No ground truth is passed to candidate selection. AllBalls metrics are training "
             "diagnostics, not held-out performance.\n"
          << "Identity evaluation reuses these class-independent matches; reference labels are unchanged.\n";
    std::cout << "Localisation at " << tolerance << " px: TP=" << tp << ", FP=" << fp
              << ", FN=" << fn << '\n';
    return match;
}

// Ball Dectection
// -----------------------------------------------------------------------------
// Image preprocessing and diagnostic proposal generation
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// AllBalls-only appearance learning and independent ball identity recognition
// -----------------------------------------------------------------------------
namespace ballid
{
struct Settings
{
    int hue_bins = 36;
    double core_radius = 0.88;
    int white_saturation = 90, white_value = 140, dark_value = 65;
    int colour_saturation = 100, colour_value = 65;
    double white_weight = 1.5, dark_weight = 0.5, saturation_weight = 0.15;
    // Paint brightness separates red/maroon and similar hue pairs.
    double colour_fraction_weight = 1.0, colour_brightness_weight = 1.0;
    double minimum_colour_fraction = 0.02;
};
const Settings parameters{};

struct Appearance
{
    cv::Mat hue;
    double white = 0, dark = 0, saturation = 0;
    double colour_fraction = 0, colour_brightness = 0;
    int pixels = 0, colour_pixels = 0;
};
struct Prototype
{
    int identity = -1;
    Appearance appearance;
};
struct Prediction
{
    int identity = -1, alternative = -1;
    double distance = 0, margin = 0;
    Appearance appearance;
};

std::string Label(int identity)
{
    return identity == 0 ? "White" : identity > 0 ? std::to_string(identity) : "Unknown";
}
int Identity(const std::string &label)
{
    if (label == "White" || label == "white")
        return 0;
    try
    {
        const int value = std::stoi(label);
        return value >= 1 && value <= 15 ? value : -1;
    }
    catch (const std::exception &)
    {
        return -1;
    }
}

int AuditedTrainingIdentity(int supplied_identity)
{
    // AllBalls-only supervised label audit. Never call this for Table1--Table6.
    // The manifest records each supplied label and its visible-number correction.
    switch (supplied_identity)
    {
    case 11: return 13;
    case 12: return 11;
    case 13: return 12;
    case 14: return 15;
    case 15: return 14;
    default: return supplied_identity;
    }
}

Appearance Describe(const cv::Mat &hsv,
                    const cv::RotatedRect &ellipse,
                    const std::vector<ballvision::Candidate> &neighbours,
                    const std::string &id,
                    cv::Mat &mask)
{
    const auto &p = parameters;
    Appearance result;
    result.hue = cv::Mat::zeros(1, p.hue_bins, CV_32F);
    mask = cv::Mat::zeros(hsv.size(), CV_8U);
    const cv::Rect box = ellipse.boundingRect() & cv::Rect(0, 0, hsv.cols, hsv.rows);
    double saturation = 0;
    for (int y = box.y; y < box.y + box.height; ++y)
    {
        for (int x = box.x; x < box.x + box.width; ++x)
        {
            const cv::Point2f point(static_cast<float>(x), static_cast<float>(y));
            const auto u = pooldiag::Unit(ellipse, point);
            const double radius = std::hypot(u.x, u.y);
            if (radius > p.core_radius)
                continue;
            bool owned = true;
            for (const auto &other : neighbours)
            {
                if (other.id == id)
                    continue;
                const auto v = pooldiag::Unit(other.ellipse, point);
                if (std::hypot(v.x, v.y) < radius)
                {
                    owned = false;
                    break;
                }
            }
            if (!owned)
                continue;
            mask.at<uchar>(y, x) = 255;
            const auto v = hsv.at<cv::Vec3b>(y, x);
            ++result.pixels;
            result.dark += v[2] < p.dark_value;
            result.white += v[1] < p.white_saturation && v[2] > p.white_value;
            saturation += v[1] / 255.0;
            // Hue describes coloured paint, not white stripes, dark numerals or glare.
            // Do not discard green paint solely because it shares the cloth hue.
            if (v[1] >= p.colour_saturation && v[2] >= p.colour_value)
            {
                ++result.colour_pixels;
                result.colour_brightness += v[2] / 255.0;
                const double bin = v[0] * p.hue_bins / 180.0;
                const int first = static_cast<int>(bin) % p.hue_bins;
                const float fraction = static_cast<float>(bin - std::floor(bin));
                const float weight = v[1] / 255.0f;
                result.hue.at<float>(0, first) += weight * (1 - fraction);
                result.hue.at<float>(0, (first + 1) % p.hue_bins) += weight * fraction;
            }
        }
    }
    if (result.pixels)
    {
        result.white /= result.pixels;
        result.dark /= result.pixels;
        result.saturation = saturation / result.pixels;
        result.colour_fraction = double(result.colour_pixels) / result.pixels;
    }
    if (result.colour_pixels)
        result.colour_brightness /= result.colour_pixels;
    // Circular smoothing respects the wrap from red hue 179 back to hue zero.
    const cv::Mat unsmoothed = result.hue.clone();
    for (int i = 0; i < p.hue_bins; ++i)
        result.hue.at<float>(0, i) =
            0.25f * unsmoothed.at<float>(0, (i + p.hue_bins - 1) % p.hue_bins) +
            0.50f * unsmoothed.at<float>(0, i) +
            0.25f * unsmoothed.at<float>(0, (i + 1) % p.hue_bins);
    const double total = cv::sum(result.hue)[0];
    if (total > 0)
        result.hue /= total;
    return result;
}

double Distance(const Appearance &a, const Appearance &b)
{
    const auto &p = parameters;
    const bool colour_a = a.colour_fraction >= p.minimum_colour_fraction;
    const bool colour_b = b.colour_fraction >= p.minimum_colour_fraction;
    const double hue = colour_a && colour_b
        ? cv::compareHist(a.hue, b.hue, cv::HISTCMP_BHATTACHARYYA) +
              p.colour_brightness_weight * std::abs(a.colour_brightness - b.colour_brightness)
        : 0.0;
    return hue + p.colour_fraction_weight * std::abs(a.colour_fraction - b.colour_fraction) +
           p.white_weight * std::abs(a.white - b.white) +
           p.dark_weight * std::abs(a.dark - b.dark) +
           p.saturation_weight * std::abs(a.saturation - b.saturation);
}

std::vector<Prototype> Train(const cv::Mat &training,
                             const ballvision::ShapeModel &shape,
                             const std::vector<GroundTruthBall> &annotations,
                             const std::filesystem::path &dir)
{
    if (annotations.size() != 16)
        throw std::runtime_error("Recognition needs 16 labelled AllBalls training examples.");
    std::filesystem::create_directories(dir);
    cv::Mat gray, smooth, edges, hsv;
    cv::cvtColor(training, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, smooth, pooldiag::settings.median_kernel);
    cv::Canny(smooth, edges, pooldiag::settings.canny_low, pooldiag::settings.canny_high);
    cv::cvtColor(training, hsv, cv::COLOR_BGR2HSV);
    const cv::Mat edge_distance = ballvision::EdgeDistance(edges);
    std::vector<ballvision::Candidate> balls;
    for (const auto &g : annotations)
    {
        ballvision::Candidate ball;
        ball.id = g.label;
        ball.ellipse = ballvision::Recover(g.centre, shape, edge_distance);
        balls.push_back(ball);
    }
    auto manifest = pooldiag::Csv(dir / "training_manifest.csv");
    manifest << "supplied_label,training_label,annotation_x,annotation_y,ellipse_x,ellipse_y,"
                "white_fraction,dark_fraction,saturation,colour_pixels,used_pixels,label_note\n";
    auto histograms = pooldiag::Csv(dir / "training_hue_histograms.csv");
    histograms << "training_label,hue_bin,probability\n";
    std::vector<Prototype> model;
    std::vector<bool> seen(16, false);
    cv::Mat overlay = training.clone();
    for (std::size_t i = 0; i < balls.size(); ++i)
    {
        const auto &g = annotations[i];
        const int supplied_identity = Identity(g.label);
        const int identity = AuditedTrainingIdentity(supplied_identity);
        if (identity < 0 || identity >= 16)
            throw std::runtime_error("Unrecognised AllBalls training label: " + g.label);
        if (seen[identity])
            throw std::runtime_error("Duplicate AllBalls training identity: " + Label(identity));
        seen[identity] = true;
        cv::Mat mask, cutout;
        const Appearance appearance = Describe(hsv, balls[i].ellipse, balls, balls[i].id, mask);
        if (!appearance.pixels)
            throw std::runtime_error("Empty training appearance for " + g.label);
        model.push_back({identity, appearance});
        training.copyTo(cutout, mask);
        const cv::Rect box = balls[i].ellipse.boundingRect() & cv::Rect(0, 0, training.cols, training.rows);
        pooldiag::Save(dir / (Label(identity) + "_appearance.png"), cutout(box));
        cv::ellipse(overlay, balls[i].ellipse, cv::Scalar(0, 255, 0), 1);
        pooldiag::Text(overlay, Label(identity), balls[i].ellipse.center);
        manifest << g.label << ',' << Label(identity) << ',' << g.centre.x << ',' << g.centre.y
                 << ',' << balls[i].ellipse.center.x << ',' << balls[i].ellipse.center.y << ','
                 << appearance.white << ',' << appearance.dark << ',' << appearance.saturation
                 << ',' << appearance.colour_pixels << ',' << appearance.pixels << ','
                 << (identity == supplied_identity ? "supplied_label_retained"
                     : "visible_AllBalls_number_audit_training_only") << '\n';
        for (int b = 0; b < parameters.hue_bins; ++b)
            histograms << Label(identity) << ',' << b << ',' << appearance.hue.at<float>(0, b) << '\n';
    }
    pooldiag::Save(dir / "training_labels.png", overlay);
    cv::FileStorage settings((dir / "recognition_settings.yml").string(), cv::FileStorage::WRITE);
    settings << "training_image" << "AllBalls.jpeg" << "hue_bins" << parameters.hue_bins
             << "core_radius" << parameters.core_radius << "white_saturation" << parameters.white_saturation
             << "white_value" << parameters.white_value << "dark_value" << parameters.dark_value
             << "colour_saturation" << parameters.colour_saturation << "colour_value" << parameters.colour_value
             << "white_weight" << parameters.white_weight << "dark_weight" << parameters.dark_weight
             << "saturation_weight" << parameters.saturation_weight
             << "identity_assignment" << "independent_nearest_prototype_no_forced_uniqueness"
             << "confidence" << "runner_up_distance_minus_best_distance_not_probability"
             << "training_label_audit" << "11_to_13;12_to_11;13_to_12;14_to_15;15_to_14"
             << "evaluation_annotations" << "unchanged";
    return model;
}

std::vector<Prediction> Recognise(const cv::Mat &image,
                                  const std::vector<ballvision::Candidate> &balls,
                                  const std::vector<Prototype> &model,
                                  const std::filesystem::path &dir)
{
    const auto folder = dir / "08_recognition";
    std::filesystem::create_directories(folder);
    cv::Mat hsv, overlay = image.clone();
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    auto csv = pooldiag::Csv(folder / "identities.csv");
    csv << "detection_id,predicted_identity,alternative_identity,best_distance,margin,white_fraction,"
           "dark_fraction,saturation,colour_pixels,used_pixels,x_table_px,y_table_px\n";
    auto scores = pooldiag::Csv(folder / "identity_scores.csv");
    scores << "detection_id,identity,distance\n";
    std::vector<Prediction> predictions;
    for (const auto &ball : balls)
    {
        Prediction result;
        cv::Mat mask;
        result.appearance = Describe(hsv, ball.ellipse, balls, ball.id, mask);
        std::vector<std::pair<double, int>> ranking;
        for (const auto &example : model)
        {
            const double distance = Distance(result.appearance, example.appearance);
            ranking.push_back({distance, example.identity});
            scores << ball.id << ',' << Label(example.identity) << ',' << distance << '\n';
        }
        std::sort(ranking.begin(), ranking.end());
        if (result.appearance.pixels && ranking.size() >= 2)
        {
            result.identity = ranking[0].second;
            result.alternative = ranking[1].second;
            result.distance = ranking[0].first;
            result.margin = ranking[1].first - ranking[0].first;
        }
        predictions.push_back(result);
        cv::ellipse(overlay, ball.ellipse, cv::Scalar(0, 255, 0), 1);
        pooldiag::Text(overlay, Label(result.identity), ball.ellipse.center);
        const cv::Rect box = ball.ellipse.boundingRect() & cv::Rect(0, 0, image.cols, image.rows);
        pooldiag::Save(folder / (ball.id + "_descriptor_mask.png"), mask(box));
        csv << ball.id << ',' << Label(result.identity) << ',' << Label(result.alternative) << ','
            << result.distance << ',' << result.margin << ',' << result.appearance.white << ','
            << result.appearance.dark << ',' << result.appearance.saturation << ','
            << result.appearance.colour_pixels << ',' << result.appearance.pixels << ','
            << ball.ellipse.center.x << ',' << ball.ellipse.center.y << '\n';
    }
    pooldiag::Save(folder / "recognised_balls.png", overlay);
    return predictions;
}

void Evaluate(const std::vector<ballvision::Candidate> &detections,
              const std::vector<Prediction> &predictions,
              const std::vector<GroundTruthBall> &truth,
              const std::vector<int> &matches,
              const std::filesystem::path &dir)
{
    if (truth.empty())
        return;
    auto audit = pooldiag::Csv(dir / "07_evaluation/recognition_matches.csv");
    audit << "detection_id,predicted_identity,supplied_identity,status,centre_error_px\n";
    std::vector<bool> matched(truth.size(), false);
    int located = 0, correct = 0;
    int confusion[16][17] = {}; // Last column: missed/unknown.
    int extras[16] = {};
    for (std::size_t i = 0; i < predictions.size(); ++i)
    {
        const int predicted = predictions[i].identity, j = matches[i];
        audit << detections[i].id << ',' << Label(predicted) << ',';
        if (j < 0)
        {
            audit << ",unmatched_detection,\n";
            if (predicted >= 0) ++extras[predicted];
            continue;
        }
        matched[j] = true;
        ++located;
        const int actual = Identity(truth[j].label);
        const bool hit = predicted == actual && actual >= 0;
        correct += hit;
        if (actual >= 0) ++confusion[actual][predicted >= 0 ? predicted : 16];
        audit << truth[j].label << ',' << (hit ? "correct" : "wrong_identity") << ','
              << cv::norm(detections[i].ellipse.center - truth[j].centre) << '\n';
    }
    for (std::size_t j = 0; j < truth.size(); ++j)
    {
        if (matched[j]) continue;
        const int actual = Identity(truth[j].label);
        if (actual >= 0) ++confusion[actual][16];
        audit << ",," << truth[j].label << ",missed_ball,\n";
    }
    auto matrix = pooldiag::Csv(dir / "07_evaluation/recognition_confusion.csv");
    matrix << "supplied_identity";
    for (int i = 0; i < 16; ++i) matrix << ',' << Label(i);
    matrix << ",missed_or_unknown\n";
    for (int i = 0; i < 16; ++i)
    {
        matrix << Label(i);
        for (int j = 0; j < 17; ++j) matrix << ',' << confusion[i][j];
        matrix << '\n';
    }
    matrix << "unmatched_detection";
    for (int i = 0; i < 16; ++i) matrix << ',' << extras[i];
    matrix << ",0\n";
    auto metrics = pooldiag::Csv(dir / "07_evaluation/recognition_metrics.csv");
    metrics << "metric,value\nannotated_balls," << truth.size() << "\ndetections," << predictions.size()
            << "\nmatched_balls," << located << "\ncorrect_identity," << correct
            << "\nmatched_identity_accuracy,";
    if (located) metrics << double(correct) / located;
    metrics << "\nend_to_end_precision,";
    if (!predictions.empty()) metrics << double(correct) / predictions.size();
    metrics << "\nend_to_end_recall," << double(correct) / truth.size()
            << "\nend_to_end_F1," << 2.0 * correct / (truth.size() + predictions.size()) << '\n';
}

void CheckTrainingLabels(const std::vector<Prediction> &predictions,
                         const std::vector<GroundTruthBall> &truth,
                         const std::vector<int> &matches,
                         const std::filesystem::path &dir)
{
    CV_Assert(predictions.size() == matches.size());
    auto audit = pooldiag::Csv(dir / "07_evaluation/training_label_audit.csv");
    audit << "detection_index,supplied_identity,audited_training_identity,predicted_identity,correct\n";
    int located = 0, correct = 0;
    for (std::size_t i = 0; i < predictions.size(); ++i)
    {
        const int j = matches[i];
        if (j < 0)
            continue;
        CV_Assert(j < static_cast<int>(truth.size()));
        const int expected = AuditedTrainingIdentity(Identity(truth[j].label));
        const bool hit = expected >= 0 && predictions[i].identity == expected;
        ++located;
        correct += hit;
        audit << i + 1 << ',' << truth[j].label << ',' << Label(expected) << ','
              << Label(predictions[i].identity) << ',' << hit << '\n';
    }
    auto metrics = pooldiag::Csv(dir / "07_evaluation/training_label_audit_metrics.csv");
    metrics << "metric,value\nmatched_balls," << located << "\ncorrect_identity," << correct
            << "\nmatched_identity_accuracy,";
    if (located)
        metrics << double(correct) / located;
    metrics << '\n';
    auto notes = pooldiag::Csv(dir / "07_evaluation/training_label_audit_notes.txt");
    notes << "AllBalls-only resubstitution check using the explicit training-label audit.\n"
          << "The same image supplied the prototypes: this is NOT held-out accuracy.\n"
          << "Primary recognition_metrics.csv still uses the supplied labels unchanged.\n"
          << "No training-label correction is applied to Table1--Table6.\n"
          << "Inspect training/training_manifest.csv and training/training_labels.png.\n";
}
} // namespace ballid

struct DetectedBall
{
    cv::Point2f centre;
    float radius;
};

std::vector<DetectedBall> DetectBalls(const cv::Mat &rectified_image, cv::Mat &smoothed_gray)
{
    if (rectified_image.empty())
    {
        throw std::runtime_error("DetectBalls received an empty image.");
    }

    cv::Mat gray;

    cv::cvtColor(rectified_image, gray, cv::COLOR_BGR2GRAY);

    const auto &settings = pooldiag::settings;
    cv::medianBlur(gray, smoothed_gray, settings.median_kernel);

    // Baseline values are unchanged; manifest uses this same settings object.
    const double dp = settings.hough_dp;
    const double minimum_centre_distance = settings.hough_spacing;
    const double canny_high_threshold = settings.hough_canny;
    const double centre_vote_threshold = settings.hough_votes;

    const int minimum_radius = settings.hough_min_radius;
    const int maximum_radius = settings.hough_max_radius;

    std::vector<cv::Vec3f> circles;

    cv::HoughCircles(smoothed_gray,
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
        detections.push_back({cv::Point2f(circle[0], circle[1]), circle[2]});
    }

    return detections;
}

// Contour-based ellipse-fitting diagnostic
cv::Mat DrawEllipseCandidates(const cv::Mat &rectified_image, const cv::Mat &edge_image)
{
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(edge_image.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::Mat display = rectified_image.clone();

    int candidate_count = 0;

    for (const std::vector<cv::Point> &contour : contours)
    {
        // A diagnostic support filter, not a final detection rule.
        if (contour.size() < static_cast<std::size_t>(pooldiag::settings.ellipse_min_points))
        {
            continue;
        }

        const cv::RotatedRect fitted = cv::fitEllipse(contour);

        const float minor_axis = std::min(fitted.size.width, fitted.size.height);

        const float major_axis = std::max(fitted.size.width, fitted.size.height);

        // Broad diagnostic limits, not validated ball parameters.
        if (!std::isfinite(minor_axis) || !std::isfinite(major_axis) ||
            !std::isfinite(fitted.center.x) || !std::isfinite(fitted.center.y))
        {
            continue;
        }

        if (minor_axis < pooldiag::settings.ellipse_min_minor ||
            major_axis > pooldiag::settings.ellipse_max_major)
        {
            continue;
        }

        if (fitted.center.x < 0 || fitted.center.x >= rectified_image.cols || fitted.center.y < 0 ||
            fitted.center.y >= rectified_image.rows)
        {
            continue;
        }

        ++candidate_count;

        cv::ellipse(display, fitted, cv::Scalar(0, 255, 255), 1);

        const cv::Point centre(cvRound(fitted.center.x), cvRound(fitted.center.y));

        cv::drawMarker(display, centre, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 8, 1);

        std::cout << "Ellipse candidate " << candidate_count << ": x=" << fitted.center.x
                  << ", y=" << fitted.center.y << ", minor axis=" << minor_axis
                  << ", major axis=" << major_axis << '\n';
    }

    return display;
}

// Foreground Mask
cv::Mat BuildForegroundMask(const cv::Mat &rectified_image,
                            cv::Mat &cloth_mask,
                            int maximum_cloth_saturation)
{
    if (maximum_cloth_saturation < 25 || maximum_cloth_saturation > 255)
    {
        throw std::runtime_error("Maximum cloth saturation must be between 25 and 255.");
    }

    cv::Mat hsv;

    cv::cvtColor(rectified_image, hsv, cv::COLOR_BGR2HSV);

    const auto &settings = pooldiag::settings;
    const cv::Scalar cloth_lower(settings.cloth_h_min, settings.cloth_s_min, settings.cloth_v_min);

    const cv::Scalar cloth_upper(settings.cloth_h_max, maximum_cloth_saturation, 255);

    cv::inRange(hsv, cloth_lower, cloth_upper, cloth_mask);

    cv::Mat foreground_mask;
    cv::bitwise_not(cloth_mask, foreground_mask);

    return foreground_mask;
}

cv::Mat DrawMaskCandidates(const cv::Mat &rectified_image, const cv::Mat &foreground_mask)
{
    CV_Assert(foreground_mask.type() == CV_8UC1);
    CV_Assert(foreground_mask.size() == rectified_image.size());

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    const int number_of_labels =
        cv::connectedComponentsWithStats(foreground_mask, labels, stats, centroids, 8, CV_32S);

    cv::Mat display = rectified_image.clone();

    int shown_count = 0;

    // Label 0 is background.
    for (int label = 1; label < number_of_labels; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);

        // Diagnostic noise filter, not a ball-size rule.
        if (area < 100)
        {
            continue;
        }

        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);

        const int y = stats.at<int>(label, cv::CC_STAT_TOP);

        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);

        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        const bool touches_border = x == 0 || y == 0 || x + width == foreground_mask.cols ||
                                    y + height == foreground_mask.rows;

        const cv::Scalar colour = touches_border ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255);

        cv::rectangle(display, cv::Rect(x, y, width, height), colour, 2);

        cv::putText(display,
                    "C" + std::to_string(label),
                    cv::Point(x, std::max(15, y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colour,
                    1);

        std::cout << "Component C" << label << ": area=" << area << ", width=" << width
                  << ", height=" << height << ", touches border=" << (touches_border ? "yes" : "no")
                  << '\n';

        ++shown_count;
    }

    std::cout << shown_count << " foreground components above the diagnostic area limit.\n";

    return display;
}

cv::Mat DrawDistanceMarkers(const cv::Mat &rectified_image,
                            const cv::Mat &opened_mask,
                            cv::Mat &distance_display,
                            cv::Mat &peak_mask,
                            cv::Mat &border_peak_mask,
                            cv::Mat &border_marker_display,
                            const std::filesystem::path &audit_dir = {})
{
    CV_Assert(opened_mask.type() == CV_8UC1);
    CV_Assert(opened_mask.size() == rectified_image.size());

    // Training-image diagnostic settings, not final detection rules.
    const int peak_neighbourhood = pooldiag::settings.peak_neighbourhood;
    const float minimum_interior_distance = pooldiag::settings.minimum_interior_distance;
    const double minimum_peak_spacing = pooldiag::settings.minimum_peak_spacing;

    cv::Mat component_labels, component_stats, component_centroids;
    const int component_count = cv::connectedComponentsWithStats(
        opened_mask, component_labels, component_stats, component_centroids, 8, CV_32S);

    std::vector<bool> touches_border(component_count, false);
    for (int label = 1; label < component_count; ++label)
    {
        const int x = component_stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = component_stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = component_stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = component_stats.at<int>(label, cv::CC_STAT_HEIGHT);
        touches_border[label] =
            x == 0 || y == 0 || x + width == opened_mask.cols || y + height == opened_mask.rows;
    }

    cv::Mat padded_mask, padded_distance;
    cv::copyMakeBorder(opened_mask, padded_mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::distanceTransform(padded_mask, padded_distance, cv::DIST_L2, 5);
    const cv::Mat distance =
        padded_distance(cv::Rect(1, 1, opened_mask.cols, opened_mask.rows)).clone();

    // Normalization is for viewing only; all thresholds remain in pixels.
    cv::Mat normalized_distance;
    cv::normalize(distance, normalized_distance, 0, 255, cv::NORM_MINMAX);
    normalized_distance.convertTo(distance_display, CV_8U);

    // Smooth the distance surface, NOT the binary segmentation.
    cv::Mat smooth_distance;
    cv::GaussianBlur(distance,
                     smooth_distance,
                     cv::Size(pooldiag::settings.distance_smooth_kernel,
                              pooldiag::settings.distance_smooth_kernel),
                     pooldiag::settings.distance_smooth_sigma);

    cv::Mat local_maximum, maxima, sufficiently_thick, candidate_mask;
    const cv::Mat peak_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(peak_neighbourhood, peak_neighbourhood));
    cv::dilate(smooth_distance, local_maximum, peak_kernel);
    cv::compare(smooth_distance, local_maximum, maxima, cv::CMP_EQ);
    cv::compare(distance, minimum_interior_distance, sufficiently_thick, cv::CMP_GT);
    cv::bitwise_and(maxima, sufficiently_thick, candidate_mask);

    cv::Mat peak_labels, peak_stats, peak_centroids;
    const int peak_count = cv::connectedComponentsWithStats(
        candidate_mask, peak_labels, peak_stats, peak_centroids, 8, CV_32S);

    struct Peak
    {
        cv::Point point;
        float distance_value;
        int component;
    };

    std::vector<Peak> candidates;
    for (int label = 1; label < peak_count; ++label)
    {
        const cv::Rect box(peak_stats.at<int>(label, cv::CC_STAT_LEFT),
                           peak_stats.at<int>(label, cv::CC_STAT_TOP),
                           peak_stats.at<int>(label, cv::CC_STAT_WIDTH),
                           peak_stats.at<int>(label, cv::CC_STAT_HEIGHT));

        const cv::Mat this_peak = peak_labels(box) == label;
        cv::Point point;
        cv::minMaxLoc(smooth_distance(box), nullptr, nullptr, nullptr, &point, this_peak);
        point += box.tl();

        const int component = component_labels.at<int>(point);
        if (component > 0)
        {
            candidates.push_back({point, distance.at<float>(point), component});
        }
    }

    // Strongest first. Do not assume one seed per connected component:
    // a connected component may contain two genuinely touching balls.
    std::sort(candidates.begin(),
              candidates.end(),
              [](const Peak &a, const Peak &b)
              {
                  if (a.distance_value != b.distance_value)
                      return a.distance_value > b.distance_value;
                  if (a.point.y != b.point.y)
                      return a.point.y < b.point.y;
                  return a.point.x < b.point.x;
              });

    std::ofstream peak_audit;
    if (!audit_dir.empty())
    {
        std::filesystem::create_directories(audit_dir);
        peak_audit = pooldiag::Csv(pooldiag::OutputPath(audit_dir, "peak_selection.csv"));
        peak_audit << "candidate_rank,x,y,component_id,border_connected,raw_distance_px,smoothed_"
                      "distance_px,kept,suppressor_x,suppressor_y,suppressor_distance_px\n";
        pooldiag::Save(pooldiag::OutputPath(audit_dir, "local_maxima_all.png"), maxima);
        pooldiag::Save(pooldiag::OutputPath(audit_dir, "local_maxima_above_distance_floor.png"),
                       candidate_mask);
    }
    std::size_t candidate_rank = 0;
    std::vector<Peak> interior_peaks, border_peaks;
    for (const Peak &candidate : candidates)
    {
        ++candidate_rank;
        std::vector<Peak> &retained =
            touches_border[candidate.component] ? border_peaks : interior_peaks;

        bool duplicate = false;
        cv::Point suppressor(-1, -1);
        double separation = -1;
        for (const Peak &existing : retained)
        {
            if (cv::norm(candidate.point - existing.point) < minimum_peak_spacing)
            {
                duplicate = true;
                suppressor = existing.point;
                separation = cv::norm(candidate.point - existing.point);
                break;
            }
        }
        if (peak_audit.is_open())
        {
            peak_audit << candidate_rank << ',' << candidate.point.x << ',' << candidate.point.y
                       << ',' << candidate.component << ',' << touches_border[candidate.component]
                       << ',' << candidate.distance_value << ','
                       << smooth_distance.at<float>(candidate.point) << ',' << !duplicate << ',';
            if (duplicate)
                peak_audit << suppressor.x << ',' << suppressor.y << ',' << separation;
            else
                peak_audit << ",,";
            peak_audit << '\n';
        }
        if (!duplicate)
            retained.push_back(candidate);
    }

    peak_mask = cv::Mat::zeros(opened_mask.size(), CV_8UC1);
    border_peak_mask = cv::Mat::zeros(opened_mask.size(), CV_8UC1);
    cv::Mat display = rectified_image.clone();
    border_marker_display = rectified_image.clone();

    for (std::size_t i = 0; i < interior_peaks.size(); ++i)
    {
        const Peak &peak = interior_peaks[i];
        peak_mask.at<uchar>(peak.point) = 255;
        const cv::Scalar colour(0, 0, 255);
        cv::drawMarker(display, peak.point, colour, cv::MARKER_CROSS, 12, 2);
        cv::putText(display,
                    "P" + std::to_string(i + 1),
                    peak.point + cv::Point(7, -7),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.4,
                    colour,
                    1);
        std::cout << "Interior seed P" << i + 1 << ": x=" << peak.point.x << ", y=" << peak.point.y
                  << ", distance=" << peak.distance_value << ", component=" << peak.component
                  << '\n';
    }

    // Review these separately. Border contact is NOT a final rejection rule:
    // a legitimate ball may touch a rail-connected foreground component.
    for (std::size_t i = 0; i < border_peaks.size(); ++i)
    {
        const Peak &peak = border_peaks[i];
        border_peak_mask.at<uchar>(peak.point) = 255;
        const cv::Scalar colour(0, 165, 255);
        cv::drawMarker(border_marker_display, peak.point, colour, cv::MARKER_CROSS, 10, 1);
        cv::putText(border_marker_display,
                    "B" + std::to_string(i + 1),
                    peak.point + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    colour,
                    1);
    }

    std::cout << interior_peaks.size() << " interior seed candidates, " << border_peaks.size()
              << " border-connected seeds for review.\n";
    return display;
}

// -----------------------------------------------------------------------------
// Application entry point: shared calibration, fixed processing, then evaluation
// -----------------------------------------------------------------------------
void MyApplication()
{
    const char *file_location = "data/images/";
    const char *selected_image_env = std::getenv("POOL_IMAGE");
    const std::string selected_image = selected_image_env ? selected_image_env : "";
    const char *gui_env = std::getenv("POOL_GUI");
    const bool show_windows = gui_env && std::string(gui_env) == "1";
    int processed_images = 0;
    std::vector<std::string> processed_names;

    const char *pool_files[] = {"AllBalls.jpeg",
                                "Table1.jpeg",
                                "Table2.jpeg",
                                "Table3.jpeg",
                                "Table4.jpeg",
                                "Table5.jpeg",
                                "Table6.jpeg"};

    const int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);

    // Reject typos before allocating a run folder or loading any input images.
    bool image_selection_valid = selected_image.empty();
    for (const char *image_filename : pool_files)
    {
        if (selected_image == image_filename ||
            selected_image == std::filesystem::path(image_filename).stem().string())
            image_selection_valid = true;
    }
    if (!image_selection_valid)
        throw std::runtime_error("Unknown POOL_IMAGE: " + selected_image +
                                 ". Use AllBalls, Table1--Table6, or leave it unset.");

    const std::string diagnostic_run = pooldiag::RunTag();
    const std::filesystem::path diagnostic_root =
        std::filesystem::path("diagnostics") / "runs" / diagnostic_run;
    if (std::filesystem::exists(diagnostic_root))
        throw std::runtime_error("Diagnostic run already exists; choose another POOL_RUN: " +
                                 diagnostic_root.string());
    std::filesystem::create_directories(diagnostic_root);

    std::vector<cv::Mat> pool_image(number_of_images);

    // Table dimensions: width = columns, height = rows.
    const int table_width = 1434;
    const int table_height = 683;

    // Order: top left, top right, bottom right, bottom left.
    // Coordinates: (column, row).
    const cv::Point2f source_points[4] = {cv::Point2f(337.0f, 35.0f),
                                          cv::Point2f(1570.0f, 21.0f),
                                          cv::Point2f(1888.0f, 451.0f),
                                          cv::Point2f(38.0f, 472.0f)};

    const cv::Point2f destination_points[4] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(table_width - 1), 0.0f),
        cv::Point2f(static_cast<float>(table_width - 1), static_cast<float>(table_height - 1)),
        cv::Point2f(0.0f, static_cast<float>(table_height - 1))};

    // Use the same perspective transformation for all table images.
    const cv::Mat perspective_matrix =
        cv::getPerspectiveTransform(source_points, destination_points);

    // All settings and this one calibration are shared across every table.
    // Evaluation images and their annotations do not enter LearnShape().
    const cv::Mat training_source =
        cv::imread(std::string(file_location) + "AllBalls.jpeg", cv::IMREAD_COLOR);
    if (training_source.empty())
        throw std::runtime_error("Cannot load AllBalls training image.");
    cv::Mat training_rectified;
    cv::warpPerspective(training_source,
                        training_rectified,
                        perspective_matrix,
                        cv::Size(table_width, table_height),
                        cv::INTER_LINEAR);
    const ballvision::ShapeModel shape_model = ballvision::LearnShape(training_rectified);
    std::cout << "AllBalls shape calibration: " << shape_model.training_outlines
              << " reliable outlines, axes " << shape_model.minor << " x " << shape_model.major
              << " pixels.\n";

    // Identity training labels are audited separately; supplied evaluation truth is immutable.
    const auto identity_model = ballid::Train(
        training_rectified, shape_model,
        LoadGroundTruth("data/ground_truth/ground_truth.csv", "AllBalls.jpeg"),
        diagnostic_root / "training");

    for (int file_no = 0; file_no < number_of_images; ++file_no)
    {
        const std::string image_filename = pool_files[file_no];
        if (!selected_image.empty() && selected_image != image_filename &&
            selected_image != std::filesystem::path(image_filename).stem().string())
            continue;
        ++processed_images;

        std::cout << "\nProcessing: " << image_filename << '\n';

        std::string filename(file_location);
        filename.append(image_filename);

        pool_image[file_no] = cv::imread(filename, cv::IMREAD_COLOR);

        if (pool_image[file_no].empty())
        {
            throw std::runtime_error("Could not open or find the image: " + filename);
        }

        if (show_windows)
        {
            cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
            cv::imshow("Original Image", pool_image[file_no]);
        }

        cv::Mat rectified_image;

        cv::warpPerspective(pool_image[file_no],
                            rectified_image,
                            perspective_matrix,
                            cv::Size(table_width, table_height),
                            cv::INTER_LINEAR);

        const std::string image_name = std::filesystem::path(image_filename).stem().string();

        processed_names.push_back(image_name);
        const std::filesystem::path diagnostic_dir = diagnostic_root / image_name;

        const std::string annotation_path = "data/ground_truth/ground_truth.csv";
        const std::vector<GroundTruthBall> ground_truth =
            std::filesystem::exists(annotation_path)
                ? LoadGroundTruth(annotation_path, image_filename)
                : std::vector<GroundTruthBall>{};
        if (ground_truth.empty())
            std::cerr << "No annotations for " << image_filename
                      << "; detection still runs, evaluation skipped.\n";

        const cv::Mat ground_truth_display =
            DrawGroundTruth(rectified_image, ground_truth, image_filename);

        cv::Mat smoothed_gray;

        const std::vector<DetectedBall> detected_balls =
            DetectBalls(rectified_image, smoothed_gray);

        cv::Mat detection_display = rectified_image.clone();

        for (std::size_t i = 0; i < detected_balls.size(); ++i)
        {
            const DetectedBall &ball = detected_balls[i];

            const cv::Point centre(cvRound(ball.centre.x), cvRound(ball.centre.y));

            circle(detection_display, centre, cvRound(ball.radius), cv::Scalar(0, 255, 0), 2);

            drawMarker(detection_display, centre, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 8, 1);

            // Candidate index only: not a predicted ball identity.
            putText(detection_display,
                    "D" + std::to_string(i + 1),
                    centre + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    cv::Scalar(0, 255, 0),
                    1);

            std::cout << image_filename << ", D" << (i + 1) << ": x=" << ball.centre.x
                      << ", y=" << ball.centre.y << ", radius=" << ball.radius << '\n';
        }

        cv::Mat diagnostic_edges;

        // Mirrors the threshold pair associated with param1 = 100.
        // This is a diagnostic image, not an input to HoughCircles.
        cv::Canny(smoothed_gray,
                  diagnostic_edges,
                  pooldiag::settings.canny_low,
                  pooldiag::settings.canny_high);

        const cv::Mat ellipse_display = DrawEllipseCandidates(rectified_image, diagnostic_edges);

        cv::Mat cloth_mask;
        int maximum_cloth_saturation = pooldiag::settings.cloth_s_max;

        const cv::Mat foreground_mask =
            BuildForegroundMask(rectified_image, cloth_mask, maximum_cloth_saturation);

        const cv::Mat opening_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(pooldiag::settings.opening_kernel, pooldiag::settings.opening_kernel));

        cv::Mat opened_mask;

        cv::morphologyEx(foreground_mask,
                         opened_mask,
                         cv::MORPH_OPEN,
                         opening_kernel,
                         cv::Point(-1, -1),
                         pooldiag::settings.opening_iterations);

        cv::Mat distance_display;
        cv::Mat peak_mask;
        cv::Mat border_peak_mask;
        cv::Mat border_marker_display;

        const cv::Mat marker_display = DrawDistanceMarkers(rectified_image,
                                                           opened_mask,
                                                           distance_display,
                                                           peak_mask,
                                                           border_peak_mask,
                                                           border_marker_display,
                                                           diagnostic_dir);

        std::vector<pooldiag::PointRecord> hough_records, truth_records;
        for (std::size_t i = 0; i < detected_balls.size(); ++i)
            hough_records.push_back({"D" + std::to_string(i + 1), detected_balls[i].centre});
        for (const GroundTruthBall &ball : ground_truth)
            truth_records.push_back({ball.label, ball.centre});
        const auto measured_ellipses = pooldiag::Write(diagnostic_dir,
                                                       image_filename,
                                                       rectified_image,
                                                       smoothed_gray,
                                                       diagnostic_edges,
                                                       foreground_mask,
                                                       opened_mask,
                                                       peak_mask,
                                                       border_peak_mask,
                                                       hough_records,
                                                       truth_records,
                                                       perspective_matrix,
                                                       pool_image[file_no].size());
        const auto validated_balls = ballvision::Select(rectified_image,
                                                        diagnostic_edges,
                                                        opened_mask,
                                                        peak_mask,
                                                        border_peak_mask,
                                                        shape_model,
                                                        measured_ellipses,
                                                        training_rectified,
                                                        diagnostic_dir);
        const auto matches = EvaluateLocalisation(
            rectified_image, validated_balls, ground_truth, diagnostic_dir);
        const auto identities = ballid::Recognise(
            rectified_image, validated_balls, identity_model, diagnostic_dir);
        ballid::Evaluate(validated_balls, identities, ground_truth, matches, diagnostic_dir);
        if (image_filename == "AllBalls.jpeg")
            ballid::CheckTrainingLabels(identities, ground_truth, matches, diagnostic_dir);
        auto hough_csv = pooldiag::Csv(pooldiag::OutputPath(diagnostic_dir, "hough.csv"));
        hough_csv << "candidate_id,x,y,radius_px\n";
        for (std::size_t i = 0; i < detected_balls.size(); ++i)
            hough_csv << "D" << i + 1 << ',' << detected_balls[i].centre.x << ','
                      << detected_balls[i].centre.y << ',' << detected_balls[i].radius << '\n';
        const cv::Mat raw_distance = pooldiag::Distance(opened_mask);
        cv::Mat smooth_distance;
        cv::GaussianBlur(raw_distance,
                         smooth_distance,
                         cv::Size(pooldiag::settings.distance_smooth_kernel,
                                  pooldiag::settings.distance_smooth_kernel),
                         pooldiag::settings.distance_smooth_sigma);
        cv::FileStorage numeric((pooldiag::OutputPath(diagnostic_dir, "distance.yml.gz")).string(),
                                cv::FileStorage::WRITE);
        if (!numeric.isOpened())
            throw std::runtime_error("Cannot save numeric distance diagnostics.");
        numeric << "distance_px" << raw_distance << "smoothed_distance_px" << smooth_distance;
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "distance_display.png"),
                       distance_display);
        std::cout << "Structured diagnostics: " << diagnostic_dir << '\n';

        cv::Mat foreground_preview;

        rectified_image.copyTo(foreground_preview, foreground_mask);

        const cv::Mat mask_candidates = DrawMaskCandidates(rectified_image, foreground_mask);

        const cv::Mat opened_regions = DrawMaskCandidates(rectified_image, opened_mask);

        cv::Mat opened_preview;

        rectified_image.copyTo(opened_preview, opened_mask);

        cv::Mat hsv;
        cv::cvtColor(rectified_image, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);

        // Stretch hue's 0–179 range for visualisation only.
        // Do not use this stretched image for HSV thresholding.
        cv::Mat hue_display;
        hsv_channels[0].convertTo(hue_display, CV_8U, 255.0 / 179.0);

        // Save supplementary views once, alongside their corresponding stage.
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "ground_truth.png"),
                       ground_truth_display);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "hough_detections.png"),
                       detection_display);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "ellipse_candidates.png"),
                       ellipse_display);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "cloth.png"), cloth_mask);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "foreground_preview.png"),
                       foreground_preview);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "regions_raw.png"), mask_candidates);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "regions_opened.png"), opened_regions);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "opened_preview.png"), opened_preview);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "hue.png"), hue_display);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "saturation.png"), hsv_channels[1]);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "value.png"), hsv_channels[2]);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "peaks.png"), peak_mask);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "markers.png"), marker_display);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "border_peaks.png"), border_peak_mask);
        pooldiag::Save(pooldiag::OutputPath(diagnostic_dir, "border_markers.png"),
                       border_marker_display);

        std::cout << image_filename << ": " << validated_balls.size() << " retained detections; "
                  << detected_balls.size() << " diagnostic Hough proposals.\n";

        if (show_windows)
            cv::destroyAllWindows();
    }
    if (processed_images == 0)
        throw std::runtime_error("POOL_IMAGE matched no input: " + selected_image);
    pooldiag::WriteRunOverview(diagnostic_root, processed_names);
    std::cout << "Diagnostics run complete: " << diagnostic_root << '\n';
}
