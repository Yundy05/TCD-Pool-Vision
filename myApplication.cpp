// Pool-ball detection and recognition
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace fs = std::filesystem;
namespace prep {
struct Settings {
    // Shared preprocessing; no experimental Hough branch is used.
    int median_kernel = 5;
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

};
inline const Settings settings{};
inline cv::Mat Distance(const cv::Mat &mask) {
    cv::Mat pad, d;
    cv::copyMakeBorder(mask, pad, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::distanceTransform(pad, d, cv::DIST_L2, 5);
    return d(cv::Rect(1, 1, mask.cols, mask.rows)).clone();
}
struct Ellipse { std::string id; cv::RotatedRect fit; bool broad = false; };
inline cv::Point2f Unit(const cv::RotatedRect &r, cv::Point2f p) {
    const double a = r.angle * CV_PI / 180.0;
    const double dx = p.x - r.center.x, dy = p.y - r.center.y;
    return cv::Point2f(
        static_cast<float>((std::cos(a) * dx + std::sin(a) * dy) / (r.size.width * 0.5)),
        static_cast<float>((-std::sin(a) * dx + std::cos(a) * dy) / (r.size.height * 0.5)));
}

// Preserve RETR_TREE contour order and IDs: only measurements unused by selection are removed.
std::vector<Ellipse> Ellipses(const cv::Mat &edges) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges.clone(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    std::vector<Ellipse> result;
    for (const auto &c : contours) {
        if (c.size() < 5) continue;
        Ellipse e;
        e.id = "E" + std::to_string(result.size() + 1);
        try { e.fit = cv::fitEllipse(c); }
        catch (const cv::Exception &) { continue; }
        const double minor = std::min(e.fit.size.width, e.fit.size.height);
        const double major = std::max(e.fit.size.width, e.fit.size.height);
        if (!std::isfinite(e.fit.center.x) || !std::isfinite(e.fit.center.y) ||
            !std::isfinite(e.fit.angle) || !std::isfinite(minor) || !std::isfinite(major) ||
            minor <= 0 || major <= 0) continue;
        e.broad = c.size() >= static_cast<std::size_t>(settings.ellipse_min_points) &&
            minor >= settings.ellipse_min_minor && major <= settings.ellipse_max_major &&
            e.fit.center.x >= 0 && e.fit.center.x < edges.cols &&
            e.fit.center.y >= 0 && e.fit.center.y < edges.rows;
        result.push_back(e);
    }
    return result;
}
} // namespace prep
// Calibrated whole-ball geometry, static-pocket rejection and duplicate suppression.
namespace ballvision {
struct SelectorSettings {
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

    // Duplicate hypotheses in normalised ellipse coordinates.
    double duplicate_iou = 0.50, duplicate_centre_radius = 0.65;

    // Stationary-camera pocket reference, learned only from AllBalls.
    double pocket_half_width_d = 1.5, pocket_depth_d = 1.75;
    double background_difference = 18.0, minimum_changed_fraction = 0.20;
};
const SelectorSettings parameters{};
struct ShapeModel {
    double minor = 0, major = 0;
    double angle_intercept = 0, angle_slope = 0;
    int training_outlines = 0;
    double Angle(double x) const {
        return angle_intercept + angle_slope * x;
    }
};
struct Boundary {
    double support = 0, sector_fraction = 0, visible = 0;
    int visible_sectors = 0, supported_sectors = 0;
};
struct Candidate {
    std::string id, source, reason, suppressed_by;
    cv::RotatedRect ellipse;
    Boundary boundary;
    double foreground = 0, ring_foreground = 0, score = 0;
    bool accepted = false, near_pocket = false;
    double changed_fraction = 1.0;
};

// The six search neighbourhoods come from table geometry, not test detections.
// Reject unchanged pocket texture, not every object near a pocket.
inline bool NearPocket(const cv::Point2f &q, cv::Size size) {
    const double d = parameters.diameter_mm;
    const double half_width = parameters.pocket_half_width_d * d;
    const double depth = parameters.pocket_depth_d * d;
    const bool near_end = q.y < depth || q.y > size.height - 1 - depth;
    return near_end && (q.x < half_width || q.x > size.width - 1 - half_width ||
                       std::abs(q.x - 0.5 * (size.width - 1)) < half_width);
}

inline cv::Mat PocketChange(const cv::Mat &image, const cv::Mat &reference) {
    cv::Mat current, background, difference;
    cv::GaussianBlur(image, current, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(reference, background, cv::Size(5, 5), 1.0);
    cv::cvtColor(current, current, cv::COLOR_BGR2Lab);
    cv::cvtColor(background, background, cv::COLOR_BGR2Lab);
    cv::absdiff(current, background, difference);
    cv::Mat changed = cv::Mat::zeros(image.size(), CV_8U);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x) {
            const auto v = difference.at<cv::Vec3b>(y, x);
            if (NearPocket(cv::Point2f(float(x), float(y)), image.size()) &&
                std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]) >
                    parameters.background_difference)
                changed.at<uchar>(y, x) = 255;
        }
    return changed;
}

inline double ChangedCore(const cv::RotatedRect &ellipse, const cv::Mat &changed) {
    const cv::Rect box = ellipse.boundingRect() & cv::Rect(0, 0, changed.cols, changed.rows);
    int visible = 0, different = 0;
    for (int y = box.y; y < box.y + box.height; ++y)
        for (int x = box.x; x < box.x + box.width; ++x) {
            const auto u = prep::Unit(ellipse, cv::Point2f(float(x), float(y)));
            if (std::hypot(u.x, u.y) <= 0.80) {
                ++visible;
                different += changed.at<uchar>(y, x) != 0;
            }
        }
    return visible ? double(different) / visible : 0;
}
inline double Median(std::vector<double> values) {
    if (values.empty())
        throw std::runtime_error("Empty shape calibration.");
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}
inline cv::RotatedRect Canonical(cv::RotatedRect e) {
    if (e.size.width > e.size.height) {
        std::swap(e.size.width, e.size.height);
        e.angle += 90.0f;
    }
    while (e.angle > 90.0f)
        e.angle -= 180.0f;
    while (e.angle <= -90.0f)
        e.angle += 180.0f;
    return e;
}
inline cv::Mat EdgeDistance(const cv::Mat &edges) {
    cv::Mat inverse, distance;
    cv::bitwise_not(edges, inverse);
    cv::distanceTransform(inverse, distance, cv::DIST_L2, 5);
    return distance;
}
inline Boundary MeasureBoundary(const cv::RotatedRect &e, const cv::Mat &edge_distance) {
    const auto &p = parameters;
    std::vector<int> counts(p.sectors, 0), hits(p.sectors, 0);
    int visible = 0, supported = 0;
    const double a = e.angle * CV_PI / 180.0, ca = std::cos(a), sa = std::sin(a);
    for (int k = 0; k < p.boundary_samples; ++k) {
        const double t = 2.0 * CV_PI * k / p.boundary_samples;
        const double u = 0.5 * e.size.width * std::cos(t);
        const double v = 0.5 * e.size.height * std::sin(t);
        const double x = e.center.x + ca * u - sa * v, y = e.center.y + sa * u + ca * v;
        if (x < 0 || y < 0 || x > edge_distance.cols - 1 || y > edge_distance.rows - 1)
            continue;
        const int sector = k * p.sectors / p.boundary_samples;
        ++visible;
        ++counts[sector];
        if (edge_distance.at<float>(cvRound(y), cvRound(x)) <= p.edge_tolerance_px) {
            ++supported;
            ++hits[sector];
        }
    }
    Boundary b;
    b.visible = double(visible) / p.boundary_samples;
    b.support = visible ? double(supported) / visible : 0;
    for (int i = 0; i < p.sectors; ++i)
        if (counts[i]) {
            ++b.visible_sectors;
            b.supported_sectors += double(hits[i]) / counts[i] >= 0.5;
        }
    b.sector_fraction = b.visible_sectors ? double(b.supported_sectors) / b.visible_sectors : 0;
    return b;
}
inline bool Finite(const cv::RotatedRect &e) {
    return std::isfinite(e.center.x) && std::isfinite(e.center.y) && std::isfinite(e.angle) &&
           std::isfinite(e.size.width) && std::isfinite(e.size.height) && e.size.width > 0 &&
           e.size.height > 0;
}
// Learns projected silhouette scale/orientation from AllBalls only.
// No annotations, target count, test images or fixed ball coordinates are used.
inline ShapeModel LearnShape(const cv::Mat &training_rectified) {
    cv::Mat gray, smooth, edges;
    cv::cvtColor(training_rectified, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, smooth, prep::settings.median_kernel);
    cv::Canny(smooth, edges, prep::settings.canny_low, prep::settings.canny_high);
    const cv::Mat distance = EdgeDistance(edges);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    std::vector<cv::RotatedRect> examples;
    const double d = parameters.diameter_mm; // Nominal 1 pixel/mm on the table plane.
    for (const auto &contour : contours) {
        if (contour.size() < 60)
            continue;
        cv::RotatedRect e;
        try {
            e = Canonical(cv::fitEllipse(contour));
        }
        catch (const cv::Exception &) {
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
            if (cv::norm(e.center - previous.center) < 0.5 * d) {
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
    for (const auto &e : examples) {
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
    for (const auto &e : clean) {
        mx += e.center.x;
        ma += e.angle;
    }
    mx /= clean.size();
    ma /= clean.size();
    double xx = 0, xa = 0;
    for (const auto &e : clean) {
        xx += (e.center.x - mx) * (e.center.x - mx);
        xa += (e.center.x - mx) * (e.angle - ma);
    }
    model.angle_slope = xx > 1e-6 ? xa / xx : 0;
    model.angle_intercept = ma - model.angle_slope * mx;
    model.training_outlines = static_cast<int>(clean.size());
    return model;
}
inline void MaskEvidence(Candidate &c, const cv::Mat &opened) {
    cv::Rect box = c.ellipse.boundingRect();
    const int expand = cvCeil(c.ellipse.size.height * 0.2);
    box.x -= expand;
    box.y -= expand;
    box.width += 2 * expand;
    box.height += 2 * expand;
    box &= cv::Rect(0, 0, opened.cols, opened.rows);
    int inside = 0, fg = 0, ring = 0, ringfg = 0;
    for (int y = box.y; y < box.y + box.height; y += 2)
        for (int x = box.x; x < box.x + box.width; x += 2) {
            const cv::Point2f u = prep::Unit(c.ellipse, cv::Point2f(float(x), float(y)));
            const double r = std::hypot(u.x, u.y);
            if (r <= 0.95) {
                ++inside;
                fg += opened.at<uchar>(y, x) != 0;
            }
            else if (r >= 1.08 && r <= 1.25) {
                ++ring;
                ringfg += opened.at<uchar>(y, x) != 0;
            }
        }
    c.foreground = inside ? double(fg) / inside : 0;
    c.ring_foreground = ring ? double(ringfg) / ring : 0;
}
inline double ShapeQuality(const cv::RotatedRect &e, const ShapeModel &model) {
    return std::exp(-2.0 * (std::abs(std::log(e.size.width / model.minor)) +
                            std::abs(std::log(e.size.height / model.major))));
}
inline double SearchScore(const cv::RotatedRect &e,
                          const ShapeModel &model,
                          const cv::Mat &distance) {
    const Boundary b = MeasureBoundary(e, distance);
    if (b.visible < parameters.minimum_visible)
        return -1;
    return 0.70 * b.support + 0.20 * b.sector_fraction + 0.10 * ShapeQuality(e, model);
}
// Fit a whole-ball template near a seed; do not mistake distance maxima for centres.
inline cv::RotatedRect Recover(const cv::Point2f &seed,
                               const ShapeModel &model,
                               const cv::Mat &distance) {
    cv::RotatedRect best(
        seed, cv::Size2f(float(model.minor), float(model.major)), float(model.Angle(seed.x)));
    double best_score = -1;
    const int rx = cvCeil(model.minor * 0.30), ry = cvCeil(model.major * 0.35);
    for (int dy = -ry; dy <= ry; dy += 4)
        for (int dx = -rx; dx <= rx; dx += 4)
            for (const double scale : {0.90, 1.0, 1.10})
                for (const double da : {-10.0, 0.0, 10.0}) {
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
                    if (score > best_score) {
                        best_score = score;
                        best = e;
                    }
                }
    for (int iteration = 0; iteration < 2; ++iteration) {
        const cv::RotatedRect start = best;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
                for (const double scale : {0.97, 1.0, 1.03}) {
                    cv::RotatedRect e = start;
                    e.center += cv::Point2f(float(dx), float(dy));
                    e.size.width *= float(scale);
                    e.size.height *= float(scale);
                    if (e.center.x < 0 || e.center.y < 0 || e.center.x >= distance.cols ||
                        e.center.y >= distance.rows)
                        continue;
                    const double score = SearchScore(e, model, distance) -
                                         0.015 * cv::norm(e.center - seed) / model.major;
                    if (score > best_score) {
                        best_score = score;
                        best = e;
                    }
                }
    }
    return best;
}
inline double EllipseOverlap(const cv::RotatedRect &a, const cv::RotatedRect &b,
                         const cv::Size &visible = cv::Size()) {
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
inline std::vector<Candidate> Select(const cv::Mat &image,
                                     const cv::Mat &edges,
                                     const cv::Mat &opened,
                                     const cv::Mat &interior_seeds,
                                     const cv::Mat &border_seeds,
                                     const ShapeModel &model,
                                     const std::vector<prep::Ellipse> &ellipses,
                                     const cv::Mat &pocket_reference) {
    const auto &p = parameters;
    CV_Assert(image.size() == pocket_reference.size());
    const cv::Mat changed = PocketChange(image, pocket_reference);
    const cv::Mat edge_distance = EdgeDistance(edges);
    cv::Mat reference_gray, reference_smooth, reference_edges;
    cv::cvtColor(pocket_reference, reference_gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(reference_gray, reference_smooth, prep::settings.median_kernel);
    cv::Canny(reference_smooth, reference_edges,
              prep::settings.canny_low, prep::settings.canny_high);
    const cv::Mat reference_distance = EdgeDistance(reference_edges);
    std::vector<cv::Point> interior, border;
    cv::findNonZero(interior_seeds, interior);
    cv::findNonZero(border_seeds, border);
    std::vector<Candidate> candidates;
    for (const auto &e : ellipses)
        if (e.broad) {
            Candidate c;
            c.id = e.id;
            c.source = "contour";
            c.ellipse = Canonical(e.fit);
            candidates.push_back(c);
            // A numeral or partial arc can locate a dark ball even when the
            // foreground distance map has no separate maximum for its body.
            if (c.ellipse.size.width < p.minimum_minor_scale * model.minor ||
                c.ellipse.size.height < p.minimum_major_scale * model.major) {
                Candidate recovered;
                recovered.id = "RC" + e.id;
                recovered.source = "contour_centre_recovery";
                recovered.ellipse = Recover(c.ellipse.center, model, edge_distance);
                candidates.push_back(recovered);
            }
        }
    const cv::Mat distance = prep::Distance(opened);
    for (int group = 0; group < 2; ++group) {
        const auto &points = group ? border : interior;
        for (std::size_t i = 0; i < points.size(); ++i) {
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
    for (auto &c : candidates) {
        const auto &e = c.ellipse;
        if (!Finite(e) || e.center.x < 0 || e.center.y < 0 || e.center.x >= image.cols ||
            e.center.y >= image.rows) {
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
    for (const std::size_t i : order) {
        Candidate &c = candidates[i];
        for (const auto &keep : accepted) {
            const auto u = prep::Unit(keep.ellipse, c.ellipse.center);
            const auto v = prep::Unit(c.ellipse, keep.ellipse.center);
            const double separation = std::min(std::hypot(u.x, u.y), std::hypot(v.x, v.y));
            // A clipped duplicate can have low full-ellipse IoU because most
            // of its extrapolated area lies outside the frame.
            const bool clipped_duplicate = c.boundary.visible < 1.0 &&
                EllipseOverlap(c.ellipse, keep.ellipse, image.size()) > p.duplicate_iou;
            if (separation < p.duplicate_centre_radius || clipped_duplicate ||
                EllipseOverlap(c.ellipse, keep.ellipse) > p.duplicate_iou) {
                c.reason = clipped_duplicate ? "clipped_duplicate" : "duplicate";
                c.suppressed_by = keep.id;
                break;
            }
        }
        if (c.reason.empty()) {
            c.accepted = true;
            c.reason = "accepted";
            accepted.push_back(c);
        }
    }
    return accepted;
}
} // namespace ballvision

// The CSV stores image,label,row,column. Empty pairs are absent balls, not (0,0).
struct GroundTruthBall {
    std::string label;
    cv::Point2f centre;
};

std::vector<GroundTruthBall> LoadGroundTruth(const std::string &csv_path,
                                           const std::string &image_name) {
    std::ifstream file(csv_path);
    if (!file) throw std::runtime_error("Could not open ground-truth file: " + csv_path);
    std::vector<GroundTruthBall> balls;
    std::string line;
    std::getline(file,line); // Header: image,label,row,column.
    while (std::getline(file,line)) {
        if (line.empty()) continue;
        std::stringstream stream(line);
        std::string image,label,row,column;
        std::getline(stream,image,','); std::getline(stream,label,',');
        std::getline(stream,row,','); std::getline(stream,column,',');
        if (image != image_name || (row.empty() && column.empty())) continue;
        if (row.empty() || column.empty())
            throw std::runtime_error("Incomplete ground-truth coordinate for " + image + ", ball " + label);
        const float y = std::stof(row), x = std::stof(column);
        balls.push_back({label,cv::Point2f(x,y)}); // CSV row/column becomes OpenCV x/y.
    }
    return balls;
}

// One-to-one assignment: maximise valid matches first, then minimise centre error.
std::vector<int> MatchLocalisation(const cv::Mat &image,
                          const std::vector<ballvision::Candidate> &detections,
                          const std::vector<GroundTruthBall> &truth) {
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
    auto in_bounds = [&](const cv::Point2f &q) {
        return std::isfinite(q.x) && std::isfinite(q.y) && q.x >= 0 && q.y >= 0 &&
               q.x < image.cols && q.y < image.rows;
    };
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) {
            if (i <= nd && j <= ng) {
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
    for (int i = 1; i <= n; ++i) {
        assigned[0] = i;
        int j0 = 0;
        std::vector<double> minimum(n + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = assigned[j0];
            int j1 = 0;
            double delta = std::numeric_limits<double>::infinity();
            for (int j = 1; j <= n; ++j)
                if (!used[j]) {
                    const double current = cost[i0][j] - u[i0] - v[j];
                    if (current < minimum[j]) {
                        minimum[j] = current;
                        previous[j] = j0;
                    }
                    if (minimum[j] < delta) {
                        delta = minimum[j];
                        j1 = j;
                    }
                }
            for (int j = 0; j <= n; ++j)
                if (used[j]) {
                    u[assigned[j]] += delta;
                    v[j] -= delta;
                }
                else
                    minimum[j] -= delta;
            j0 = j1;
        } while (assigned[j0] != 0);
        do {
            const int j1 = previous[j0];
            assigned[j0] = assigned[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int> match(nd, -1);
    std::vector<bool> matched_truth(ng, false);
    for (int j = 1; j <= ng; ++j)
        if (assigned[j] > 0 && assigned[j] <= nd && cost[assigned[j]][j] <= tolerance) {
            match[assigned[j] - 1] = j - 1;
            matched_truth[j - 1] = true;
        }
    return match;
}

// Appearance descriptors and independent nearest-prototype identity assignment.
namespace ballid {
struct Settings {
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

struct Appearance {
    cv::Mat hue;
    double white = 0, dark = 0, saturation = 0;
    double colour_fraction = 0, colour_brightness = 0;
    int pixels = 0, colour_pixels = 0;
};
struct Prototype {
    int identity = -1;
    Appearance appearance;
};
struct Prediction {
    int identity = -1, alternative = -1;
    double distance = 0, margin = 0;
    Appearance appearance;
};

std::string Label(int identity) {
    return identity == 0 ? "White" : identity > 0 ? std::to_string(identity) : "Unknown";
}
int Identity(const std::string &label) {
    if (label == "White" || label == "white")
        return 0;
    try {
        const int value = std::stoi(label);
        return value >= 1 && value <= 15 ? value : -1;
    }
    catch (const std::exception &) {
        return -1;
    }
}

int AuditedTrainingIdentity(int supplied_identity) {
    // AllBalls-only supervised label audit. Never call this for Table1--Table6.
    // Preserve the original AllBalls corrections; evaluation labels remain untouched.
    switch (supplied_identity) {
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
                    cv::Mat &mask) {
    const auto &p = parameters;
    Appearance result;
    result.hue = cv::Mat::zeros(1, p.hue_bins, CV_32F);
    mask = cv::Mat::zeros(hsv.size(), CV_8U);
    const cv::Rect box = ellipse.boundingRect() & cv::Rect(0, 0, hsv.cols, hsv.rows);
    double saturation = 0;
    for (int y = box.y; y < box.y + box.height; ++y) {
        for (int x = box.x; x < box.x + box.width; ++x) {
            const cv::Point2f point(static_cast<float>(x), static_cast<float>(y));
            const auto u = prep::Unit(ellipse, point);
            const double radius = std::hypot(u.x, u.y);
            if (radius > p.core_radius)
                continue;
            bool owned = true;
            for (const auto &other : neighbours) {
                if (other.id == id)
                    continue;
                const auto v = prep::Unit(other.ellipse, point);
                if (std::hypot(v.x, v.y) < radius) {
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
            if (v[1] >= p.colour_saturation && v[2] >= p.colour_value) {
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
    if (result.pixels) {
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

double Distance(const Appearance &a, const Appearance &b) {
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
                             const std::vector<GroundTruthBall> &annotations) {
    if (annotations.size() != 16)
        throw std::runtime_error("Recognition needs 16 labelled AllBalls training examples.");
    cv::Mat gray, smooth, edges, hsv;
    cv::cvtColor(training, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, smooth, prep::settings.median_kernel);
    cv::Canny(smooth, edges, prep::settings.canny_low, prep::settings.canny_high);
    cv::cvtColor(training, hsv, cv::COLOR_BGR2HSV);
    const cv::Mat edge_distance = ballvision::EdgeDistance(edges);
    std::vector<ballvision::Candidate> balls;
    for (const auto &g : annotations) {
        ballvision::Candidate ball;
        ball.id = g.label;
        ball.ellipse = ballvision::Recover(g.centre, shape, edge_distance);
        balls.push_back(ball);
    }
    std::vector<Prototype> model;
    std::vector<bool> seen(16, false);
    for (std::size_t i = 0; i < balls.size(); ++i) {
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
    }
    return model;
}
std::vector<Prediction> Recognise(const cv::Mat &image,
                                  const std::vector<ballvision::Candidate> &balls,
                                  const std::vector<Prototype> &model) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<Prediction> predictions;
    for (const auto &ball : balls) {
        Prediction result;
        cv::Mat mask;
        result.appearance = Describe(hsv, ball.ellipse, balls, ball.id, mask);
        std::vector<std::pair<double, int>> ranking;
        for (const auto &example : model) {
            const double distance = Distance(result.appearance, example.appearance);
            ranking.push_back({distance, example.identity});
        }
        std::sort(ranking.begin(), ranking.end());
        if (result.appearance.pixels && ranking.size() >= 2) {
            result.identity = ranking[0].second;
            result.alternative = ranking[1].second;
            result.distance = ranking[0].first;
            result.margin = ranking[1].first - ranking[0].first;
        }
        predictions.push_back(result);
    }
    return predictions;
}
} // namespace ballid

// Cloth segmentation is supporting evidence, not a complete-ball requirement.
cv::Mat BuildForegroundMask(const cv::Mat &rectified_image,
                            cv::Mat &cloth_mask,
                            int maximum_cloth_saturation) {
    if (maximum_cloth_saturation < 25 || maximum_cloth_saturation > 255) {
        throw std::runtime_error("Maximum cloth saturation must be between 25 and 255.");
    }

    cv::Mat hsv;

    cv::cvtColor(rectified_image, hsv, cv::COLOR_BGR2HSV);

    const auto &settings = prep::settings;
    const cv::Scalar cloth_lower(settings.cloth_h_min, settings.cloth_s_min, settings.cloth_v_min);

    const cv::Scalar cloth_upper(settings.cloth_h_max, maximum_cloth_saturation, 255);

    cv::inRange(hsv, cloth_lower, cloth_upper, cloth_mask);

    cv::Mat foreground_mask;
    cv::bitwise_not(cloth_mask, foreground_mask);

    return foreground_mask;
}

// Local maxima retain their original tie-breaking and component-border treatment.
void FindSeeds(const cv::Mat &opened_mask, cv::Mat &peak_mask, cv::Mat &border_peak_mask) {
    const int peak_neighbourhood = prep::settings.peak_neighbourhood;
    const float minimum_interior_distance = prep::settings.minimum_interior_distance;
    const double minimum_peak_spacing = prep::settings.minimum_peak_spacing;

    cv::Mat component_labels, component_stats, component_centroids;
    const int component_count = cv::connectedComponentsWithStats(
        opened_mask, component_labels, component_stats, component_centroids, 8, CV_32S);

    std::vector<bool> touches_border(component_count, false);
    for (int label = 1; label < component_count; ++label) {
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
    // Smooth the distance surface, NOT the binary segmentation.
    cv::Mat smooth_distance;
    cv::GaussianBlur(distance,
                     smooth_distance,
                     cv::Size(prep::settings.distance_smooth_kernel,
                              prep::settings.distance_smooth_kernel),
                     prep::settings.distance_smooth_sigma);

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

    struct Peak {
        cv::Point point;
        float distance_value;
        int component;
    };

    std::vector<Peak> candidates;
    for (int label = 1; label < peak_count; ++label) {
        const cv::Rect box(peak_stats.at<int>(label, cv::CC_STAT_LEFT),
                           peak_stats.at<int>(label, cv::CC_STAT_TOP),
                           peak_stats.at<int>(label, cv::CC_STAT_WIDTH),
                           peak_stats.at<int>(label, cv::CC_STAT_HEIGHT));

        const cv::Mat this_peak = peak_labels(box) == label;
        cv::Point point;
        cv::minMaxLoc(smooth_distance(box), nullptr, nullptr, nullptr, &point, this_peak);
        point += box.tl();

        const int component = component_labels.at<int>(point);
        if (component > 0) {
            candidates.push_back({point, distance.at<float>(point), component});
        }
    }

    // Strongest first. Do not assume one seed per connected component:
    // a connected component may contain two genuinely touching balls.
    std::sort(candidates.begin(),
              candidates.end(),
              [](const Peak &a, const Peak &b) {
                  if (a.distance_value != b.distance_value)
                      return a.distance_value > b.distance_value;
                  if (a.point.y != b.point.y)
                      return a.point.y < b.point.y;
                  return a.point.x < b.point.x;
              });
    std::vector<Peak> interior_peaks, border_peaks;
    for (const Peak &candidate : candidates) {
        std::vector<Peak> &retained =
            touches_border[candidate.component] ? border_peaks : interior_peaks;

        bool duplicate = false;
        for (const Peak &existing : retained) {
            if (cv::norm(candidate.point - existing.point) < minimum_peak_spacing) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            retained.push_back(candidate);
    }

    peak_mask = cv::Mat::zeros(opened_mask.size(), CV_8UC1);
    border_peak_mask = cv::Mat::zeros(opened_mask.size(), CV_8UC1);

    for (const auto &p : interior_peaks) peak_mask.at<uchar>(p.point) = 255;
    for (const auto &p : border_peaks) border_peak_mask.at<uchar>(p.point) = 255;
}

// Final outputs only: two overlays per scene, detection coordinates, and a shared metric table.
inline void Text(cv::Mat &image,
                 const std::string &text,
                 cv::Point p,
                 cv::Scalar colour = cv::Scalar(255, 255, 255),
                 double scale = 0.4) {
    p.x = std::clamp(p.x, 0, std::max(0, image.cols - 100));
    p.y = std::clamp(p.y, 13, std::max(13, image.rows - 3));
    cv::putText(image, text, p, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3);
    cv::putText(image, text, p, cv::FONT_HERSHEY_SIMPLEX, scale, colour, 1);
}

void SaveImage(const fs::path &path, const cv::Mat &image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Cannot write image: " + path.string());
}
std::ofstream OpenCsv(const fs::path &path) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write CSV: " + path.string());
    file << std::fixed << std::setprecision(5);
    return file;
}
struct Metrics {
    int detections = 0, annotations = 0, tp = 0, correct = 0, outside = 0;
    std::vector<double> errors;
    void Add(const Metrics &m) {
        detections += m.detections; annotations += m.annotations; tp += m.tp;
        correct += m.correct; outside += m.outside;
        errors.insert(errors.end(), m.errors.begin(), m.errors.end());
    }
};
bool InBounds(const cv::Point2f &p, cv::Size size) {
    return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 &&
           p.x < size.width && p.y < size.height;
}
Metrics Evaluate(const cv::Mat &image, const std::vector<ballvision::Candidate> &balls,
                 const std::vector<ballid::Prediction> &identities,
                 const std::vector<GroundTruthBall> &truth, const fs::path &output) {
    Metrics m;
    m.detections = static_cast<int>(balls.size());
    m.annotations = static_cast<int>(truth.size());
    const auto match = MatchLocalisation(image, balls, truth);
    std::vector<bool> seen(truth.size(), false);
    cv::Mat overlay = image.clone();
    for (const auto &g : truth) m.outside += !InBounds(g.centre, image.size());
    for (std::size_t i = 0; i < balls.size(); ++i) {
        const bool hit = match[i] >= 0;
        const auto &b = balls[i];
        cv::ellipse(overlay, b.ellipse, hit ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
        Text(overlay, b.id, b.ellipse.center);
        if (hit) {
            const auto &g = truth[match[i]];
            seen[match[i]] = true; ++m.tp;
            m.errors.push_back(cv::norm(b.ellipse.center - g.centre));
            m.correct += identities[i].identity == ballid::Identity(g.label);
            cv::line(overlay, b.ellipse.center, g.centre, cv::Scalar(255,255,0), 1);
            cv::drawMarker(overlay, g.centre, cv::Scalar(255,0,255), cv::MARKER_CROSS, 10, 1);
        }
    }
    for (std::size_t j = 0; j < truth.size(); ++j)
        if (!seen[j] && InBounds(truth[j].centre, image.size())) {
            cv::drawMarker(overlay, truth[j].centre, cv::Scalar(0,165,255), cv::MARKER_CROSS, 14, 2);
            Text(overlay, "miss " + truth[j].label, truth[j].centre);
        }
    SaveImage(output, overlay);
    return m;
}
// Missing denominators are reported as NaN rather than implying a measured zero.
double Ratio(double n, double d) {
    return d ? n / d : std::numeric_limits<double>::quiet_NaN();
}
void WriteMetrics(std::ostream &out, const std::string &name, const Metrics &m) {
    double sum = 0, squared = 0;
    for (double e : m.errors) { sum += e; squared += e * e; }
    const double median = m.errors.empty() ? std::numeric_limits<double>::quiet_NaN()
                                         : ballvision::Median(m.errors);
    out << name << ',' << m.tp << ',' << m.detections-m.tp << ',' << m.annotations-m.tp
        << ',' << m.outside << ',' << Ratio(m.tp,m.detections) << ',' << Ratio(m.tp,m.annotations)
        << ',' << Ratio(2.0*m.tp,m.detections+m.annotations)
        << ',' << Ratio(sum,m.errors.size()) << ',' << median
        << ',' << std::sqrt(Ratio(squared,m.errors.size())) << ',' << m.correct
        << ',' << Ratio(m.correct,m.tp) << ',' << Ratio(m.correct,m.detections)
        << ',' << Ratio(m.correct,m.annotations)
        << ',' << Ratio(2.0*m.correct,m.detections+m.annotations) << '\n';
}
int Run(int argc, char **argv) {
    if (argc > 3) throw std::runtime_error("Usage: poolvision [data_directory] [output_directory]");
    const fs::path data = argc > 1 ? argv[1] : "data";
    const fs::path output = argc > 2 ? argv[2] : "results";
    const fs::path annotations = data / "ground_truth/ground_truth.csv";
    const std::vector<std::string> names = {"AllBalls","Table1","Table2","Table3","Table4","Table5","Table6"};
    const char *chosen = std::getenv("POOL_IMAGE");
    const std::string selected = chosen ? fs::path(chosen).stem().string() : "";
    if (!selected.empty() && std::find(names.begin(),names.end(),selected) == names.end())
        throw std::runtime_error("POOL_IMAGE must be AllBalls or Table1--Table6.");
    const char *gui = std::getenv("POOL_GUI");
    const bool show = gui && std::string(gui) == "1";
    const cv::Size size(1434,683);
    const cv::Point2f source[] = {{337,35},{1570,21},{1888,451},{38,472}};
    const cv::Point2f target[] = {{0,0},{1433,0},{1433,682},{0,682}};
    const cv::Mat homography = cv::getPerspectiveTransform(source,target);
    auto load = [&](const std::string &name) {
        const auto path = data / "images" / (name + ".jpeg");
        const cv::Mat image = cv::imread(path.string(),cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot load image: " + path.string());
        cv::Mat result;
        cv::warpPerspective(image,result,homography,size,cv::INTER_LINEAR);
        return result;
    };
    const cv::Mat training = load("AllBalls");
    const auto shape = ballvision::LearnShape(training);
    const auto model = ballid::Train(training,shape,
        LoadGroundTruth(annotations.string(),"AllBalls.jpeg"));
    std::cout << "AllBalls calibration: " << shape.training_outlines << " outlines; axes "
              << shape.minor << " x " << shape.major << " px.\n";
    fs::create_directories(output);
    auto summary = OpenCsv(output / "metrics.csv");
    summary << "image,TP,FP,FN,annotations_outside_image,precision,recall,F1,mean_error_px,"
               "median_error_px,RMSE_px,correct_identity,matched_identity_accuracy,"
               "end_to_end_precision,end_to_end_recall,end_to_end_F1\n";
    Metrics total;
    for (const auto &name : names) {
        if (!selected.empty() && selected != name) continue;
        const cv::Mat image = name == "AllBalls" ? training : load(name);
        cv::Mat gray,smooth,edges,cloth,opened,interior,border;
        cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
        cv::medianBlur(gray,smooth,prep::settings.median_kernel);
        cv::Canny(smooth,edges,prep::settings.canny_low,prep::settings.canny_high);
        const cv::Mat foreground = BuildForegroundMask(image,cloth,prep::settings.cloth_s_max);
        cv::morphologyEx(foreground,opened,cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(prep::settings.opening_kernel,
                prep::settings.opening_kernel)),cv::Point(-1,-1),prep::settings.opening_iterations);
        FindSeeds(opened,interior,border);
        const auto balls = ballvision::Select(image,edges,opened,interior,border,shape,
                                             prep::Ellipses(edges),training);
        const auto identities = ballid::Recognise(image,balls,model);
        cv::Mat recognised = image.clone();
        auto positions = OpenCsv(output / (name + "_balls.csv"));
        positions << "detection_id,x_table_px,y_table_px,minor_px,major_px,angle_degrees,score,"
                     "predicted_identity,alternative_identity,appearance_distance,margin\n";
        for (std::size_t i = 0; i < balls.size(); ++i) {
            const auto &b = balls[i]; const auto &p = identities[i];
            cv::ellipse(recognised,b.ellipse,cv::Scalar(0,255,0),1);
            Text(recognised,ballid::Label(p.identity),b.ellipse.center);
            positions << b.id << ',' << b.ellipse.center.x << ',' << b.ellipse.center.y
                << ',' << b.ellipse.size.width << ',' << b.ellipse.size.height << ',' << b.ellipse.angle
                << ',' << b.score << ',' << ballid::Label(p.identity) << ',' << ballid::Label(p.alternative)
                << ',' << p.distance << ',' << p.margin << '\n';
        }
        SaveImage(output / (name + "_recognised.png"),recognised);
        const auto truth = LoadGroundTruth(annotations.string(),name + ".jpeg");
        if (!truth.empty()) {
            const Metrics m = Evaluate(image,balls,identities,truth,output / (name + "_evaluation.png"));
            WriteMetrics(summary,name,m);
            if (name != "AllBalls") total.Add(m); // Training is never pooled into test scores.
            std::cout << name << ": TP=" << m.tp << " FP=" << m.detections-m.tp
                      << " FN=" << m.annotations-m.tp << " correct identities=" << m.correct << '\n';
        } else std::cout << name << ": " << balls.size() << " detections; no evaluation annotations.\n";
        if (show) { cv::imshow("Recognised balls",recognised); cv::waitKey(0); }
    }
    if (total.annotations) WriteMetrics(summary,"TestTotal",total);
    if (show) cv::destroyAllWindows();
    std::cout << "Final outputs: " << output << '\n';
    return EXIT_SUCCESS;
}
int main(int argc, char **argv) {
    try { return Run(argc,argv); }
    catch (const std::exception &e) {
        std::cerr << "Pool vision error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
