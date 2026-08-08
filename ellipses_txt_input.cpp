#define _USE_MATH_DEFINES  // Windows/MSVC: enables M_PI from <cmath>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// Used by boundary_inliers() for voting-level OpenMP.
// 0 means use OpenMP default/max threads.
static int g_vote_threads = 0;

static constexpr int SPACE_SIZE = 100;
static constexpr int SEMI_MAJOR_MIN = 12;
static constexpr int SEMI_MAJOR_MAX = 22;
static constexpr double ASPECT_RATIO_MIN = 0.8;
static constexpr double ASPECT_RATIO_MAX = 1.0;

// Plain integer/double 2D points, replacing cv::Point / cv::Point2d so this
// file has no OpenCV dependency (needed for the eventual Vitis/VCK190 build).
struct Pt2i {
    int x = 0;
    int y = 0;
};

struct Pt2d {
    double x = 0.0;
    double y = 0.0;
};

struct EllipseParams {
    double cx = 0.0;
    double cy = 0.0;
    double a = 0.0;
    double b = 0.0;
    double theta = 0.0;
};

struct ImagePoints {
    int width = 0;
    int height = 0;
    std::vector<Pt2i> xy;
};

struct MatchInfo {
    int pred_index = -1;
    int gt_index = -1;
    double iou = 0.0;
};

struct DetectOneInfo {
    double dght_time_s = 0.0;
    double trials = 0.0;
    double best_inliers = -1.0;
};

struct PeelingInfo {
    double total_detection_time_s = 0.0;
    double sum_dght_time_s = 0.0;
    double sum_inlier_remove_time_s = 0.0;
    double num_found = 0.0;
    double remaining_points = 0.0;
    double total_trials = 0.0;
};

struct MemoryEstimate {
    double cx_bins = 0.0;
    double cy_bins = 0.0;
    double a_bins = 0.0;
    double b_bins = 0.0;
    double theta_bins = 0.0;
    double hist_total_mib = 0.0;
};

struct Args {
    std::string ann;
    std::string hits_dir = "./txt";
    std::string mode = "first";
    int n = 10;
    int start = 0;
    int end = 1000000000;
    std::string ids = "";
    int img_width = 50;
    int img_height = 50;
    double center_step_px = 1.0;
    double a_step_px = 0.5;
    double b_step_px = 0.5;
    double angle_step_deg = 10.0;
    int seg_half_px = 24;
    int max_trials = 8000;
    int check_every = 250;
    int min_votes_mode = 60;
    int stable_checks_needed = 2;
    double boundary_tol = 0.50;
    int max_ellipses = 3;
    int min_inliers_removed = 10;
    int eval_downsample = 2;
    double iou_match_min = 0.30;
    std::string out_csv = "batch_results_dght_txt_input.csv";
    // 0 means: use OpenMP default/max threads for voting.
    int vote_threads = 0;
    // CHANGE THIS LINE only if you want a hard-coded default thread count.
    // 0 means: use OpenMP default. Otherwise run with: --num_threads 8
    int num_threads = 0;
};

struct RowResult {
    int img_id = 0;
    int hits = 0;
    int gt_count = 0;
    int pred_count = 0;
    int matched_pairs = 0;
    int correct_matches_iou_ge_thr = 0;
    int fp = 0;
    int fn = 0;
    double mean_iou = 0.0;
    double mean_center_dist_px = 0.0;
    double total_detection_time_s = 0.0;
    double sum_dght_time_s = 0.0;
    double sum_inlier_remove_time_s = 0.0;
    int num_found = 0;
    int remaining_points = 0;
    double total_trials = 0.0;
};

static void print_usage() {
    std::cout << "Usage: Batch_Evaluation_DGHT_txt --ann annotations.json [options]\n"
              << "Options:\n"
              << "  --hits_dir PATH                folder of id_<N>.txt hit-point files (default ./txt)\n"
              << "  --img_width INT                image width in px, default 50\n"
              << "  --img_height INT               image height in px, default 50\n"
              << "  --mode first|range|random\n"
              << "  --n INT\n"
              << "  --start INT\n"
              << "  --end INT\n"
              << "  --ids comma,separated,list\n"
              << "  --center_step_px FLOAT\n"
              << "  --a_step_px FLOAT\n"
              << "  --b_step_px FLOAT\n"
              << "  --angle_step_deg FLOAT\n"
              << "  --seg_half_px INT\n"
              << "  --max_trials INT\n"
              << "  --check_every INT\n"
              << "  --min_votes_mode INT\n"
              << "  --stable_checks_needed INT\n"
              << "  --boundary_tol FLOAT\n"
              << "  --max_ellipses INT\n"
              << "  --min_inliers_removed INT\n"
              << "  --eval_downsample INT\n"
              << "  --iou_match_min FLOAT\n"
              << "  --out_csv PATH\n"
              << "  --num_threads INT              OpenMP threads for parallel image processing, default OpenMP setting\n"
              << "  --vote_threads INT             OpenMP threads for voting/inlier checks, default OpenMP setting\n";
}

static bool starts_with_dash(const std::string& s) {
    return !s.empty() && s[0] == '-';
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc || starts_with_dash(argv[i + 1])) {
                throw std::runtime_error("Missing value for argument: " + name);
            }
            return std::string(argv[++i]);
        };

        if (key == "--ann") args.ann = need_value(key);
        else if (key == "--hits_dir") args.hits_dir = need_value(key);
        else if (key == "--img_width") args.img_width = std::stoi(need_value(key));
        else if (key == "--img_height") args.img_height = std::stoi(need_value(key));
        else if (key == "--mode") args.mode = need_value(key);
        else if (key == "--n") args.n = std::stoi(need_value(key));
        else if (key == "--start") args.start = std::stoi(need_value(key));
        else if (key == "--end") args.end = std::stoi(need_value(key));
        else if (key == "--ids") args.ids = need_value(key);
        else if (key == "--center_step_px") args.center_step_px = std::stod(need_value(key));
        else if (key == "--a_step_px") args.a_step_px = std::stod(need_value(key));
        else if (key == "--b_step_px") args.b_step_px = std::stod(need_value(key));
        else if (key == "--angle_step_deg") args.angle_step_deg = std::stod(need_value(key));
        else if (key == "--seg_half_px") args.seg_half_px = std::stoi(need_value(key));
        else if (key == "--max_trials") args.max_trials = std::stoi(need_value(key));
        else if (key == "--check_every") args.check_every = std::stoi(need_value(key));
        else if (key == "--min_votes_mode") args.min_votes_mode = std::stoi(need_value(key));
        else if (key == "--stable_checks_needed") args.stable_checks_needed = std::stoi(need_value(key));
        else if (key == "--boundary_tol") args.boundary_tol = std::stod(need_value(key));
        else if (key == "--max_ellipses") args.max_ellipses = std::stoi(need_value(key));
        else if (key == "--min_inliers_removed") args.min_inliers_removed = std::stoi(need_value(key));
        else if (key == "--eval_downsample") args.eval_downsample = std::stoi(need_value(key));
        else if (key == "--iou_match_min") args.iou_match_min = std::stod(need_value(key));
        else if (key == "--out_csv") args.out_csv = need_value(key);
        else if (key == "--vote_threads") args.vote_threads = std::stoi(need_value(key));
        else if (key == "--num_threads") args.num_threads = std::stoi(need_value(key));
        else if (key == "-h" || key == "--help") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.ann.empty()) {
        throw std::runtime_error("--ann is required");
    }
    return args;
}

// Reads a padded "x y value" hit-point txt file (headers already stripped
// upstream). Padding rows are written as "0 0 0"; since real hit rows always
// have value > 0 (the PNG->txt conversion only keeps pixels above threshold),
// filtering on value <= 0 cleanly discards padding without touching a
// legitimate hit that happens to land at pixel (0,0).
static std::vector<Pt2i> load_hit_points_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open hits file: " + path);
    }
    std::vector<Pt2i> pts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int x, y, v;
        if (!(iss >> x >> y >> v)) continue;
        if (v <= 0) continue;
        pts.push_back(Pt2i{x, y});
    }
    return pts;
}

static std::map<int, std::vector<EllipseParams>> load_annotations(const std::string& annotation_json_path) {
    std::ifstream f(annotation_json_path);
    if (!f) {
        throw std::runtime_error("Failed to open annotations: " + annotation_json_path);
    }
    json data;
    f >> data;

    std::map<int, std::vector<EllipseParams>> ann_map;
    for (const auto& ann : data.value("annotations", json::array())) {
        int img_id = ann.at("image_id").get<int>();
        std::vector<EllipseParams> ell_list;
        for (const auto& el : ann.value("ellipses", json::array())) {
            EllipseParams e;
            e.cx = el.at("center_x").get<double>();
            e.cy = el.at("center_y").get<double>();
            e.a = el.at("semi_major_axis").get<double>();
            e.b = el.at("semi_minor_axis").get<double>();
            e.theta = el.at("orientation_angle_rad").get<double>();
            ell_list.push_back(e);
        }
        ann_map[img_id] = std::move(ell_list);
    }
    return ann_map;
}

static EllipseParams scale_ellipse(const EllipseParams& e, double sx, double sy) {
    double s = (sx + sy) / 2.0;
    return EllipseParams{e.cx * sx, e.cy * sy, e.a * s, e.b * s, e.theta};
}

static std::vector<uint8_t> boundary_inliers(const std::vector<Pt2i>& xy, const EllipseParams& e, double boundary_tol) {
    std::vector<uint8_t> out(xy.size(), 0);

    const double c = std::cos(e.theta);
    const double s = std::sin(e.theta);
    const double a2 = std::max(e.a * e.a, 1e-9);
    const double b2 = std::max(e.b * e.b, 1e-9);

#ifdef _OPENMP
    const int threads = (g_vote_threads > 0) ? g_vote_threads : omp_get_max_threads();
#pragma omp parallel for schedule(static) if(xy.size() >= 64) num_threads(threads)
#endif
    for (int i = 0; i < static_cast<int>(xy.size()); ++i) {
        const auto& p = xy[static_cast<size_t>(i)];

        const double x = static_cast<double>(p.x) - e.cx;
        const double y = static_cast<double>(p.y) - e.cy;

        const double xp = x * c + y * s;
        const double yp = -x * s + y * c;

        const double v = (xp * xp) / a2 + (yp * yp) / b2;

        if (std::abs(v - 1.0) <= boundary_tol) {
            out[static_cast<size_t>(i)] = 1;
        }
    }

    return out;
}

static std::vector<uint8_t> ellipse_mask_downsampled(const EllipseParams& e, int H, int W, int ds) {
    ds = std::max(1, ds);
    int Hd = std::max(1, H / ds);
    int Wd = std::max(1, W / ds);
    std::vector<uint8_t> mask(Hd * Wd, 0);

    double c = std::cos(e.theta);
    double s = std::sin(e.theta);
    double a2 = std::max(e.a * e.a, 1e-9);
    double b2 = std::max(e.b * e.b, 1e-9);

    for (int y = 0; y < Hd; ++y) {
        for (int x = 0; x < Wd; ++x) {
            double px = static_cast<double>(x * ds) + (ds - 1) / 2.0;
            double py = static_cast<double>(y * ds) + (ds - 1) / 2.0;
            double dx = px - e.cx;
            double dy = py - e.cy;
            double xp = dx * c + dy * s;
            double yp = -dx * s + dy * c;
            double v = (xp * xp) / a2 + (yp * yp) / b2;
            mask[y * Wd + x] = (v <= 1.0) ? 1 : 0;
        }
    }
    return mask;
}

static double iou_mask(const std::vector<uint8_t>& m1, const std::vector<uint8_t>& m2) {
    uint64_t inter = 0;
    uint64_t uni = 0;
    size_t n = std::min(m1.size(), m2.size());
    for (size_t i = 0; i < n; ++i) {
        const bool a = m1[i] != 0;
        const bool b = m2[i] != 0;
        if (a && b) ++inter;
        if (a || b) ++uni;
    }
    return uni ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
}

static std::tuple<std::vector<MatchInfo>, std::vector<int>, std::vector<int>> greedy_match_iou(
    const std::vector<EllipseParams>& preds,
    const std::vector<EllipseParams>& gts,
    int H,
    int W,
    int ds) {

    int nP = static_cast<int>(preds.size());
    int nG = static_cast<int>(gts.size());
    if (nP == 0 || nG == 0) {
        std::vector<int> um_p(nP), um_g(nG);
        std::iota(um_p.begin(), um_p.end(), 0);
        std::iota(um_g.begin(), um_g.end(), 0);
        return { {}, um_p, um_g };
    }

    std::vector<std::vector<uint8_t>> pred_masks, gt_masks;
    pred_masks.reserve(preds.size());
    gt_masks.reserve(gts.size());
    for (const auto& p : preds) pred_masks.push_back(ellipse_mask_downsampled(p, H, W, ds));
    for (const auto& g : gts) gt_masks.push_back(ellipse_mask_downsampled(g, H, W, ds));

    std::vector<MatchInfo> pairs;
    for (int i = 0; i < nP; ++i) {
        for (int j = 0; j < nG; ++j) {
            pairs.push_back(MatchInfo{i, j, iou_mask(pred_masks[i], gt_masks[j])});
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const MatchInfo& a, const MatchInfo& b) {
        return a.iou > b.iou;
    });

    std::set<int> used_p, used_g;
    std::vector<MatchInfo> matches;
    for (const auto& p : pairs) {
        if (used_p.count(p.pred_index) || used_g.count(p.gt_index)) continue;
        used_p.insert(p.pred_index);
        used_g.insert(p.gt_index);
        matches.push_back(p);
    }

    std::vector<int> um_p, um_g;
    for (int i = 0; i < nP; ++i) if (!used_p.count(i)) um_p.push_back(i);
    for (int j = 0; j < nG; ++j) if (!used_g.count(j)) um_g.push_back(j);
    return {matches, um_p, um_g};
}

static double mean_center_dist_for_matches(
    const std::vector<EllipseParams>& preds,
    const std::vector<EllipseParams>& gts,
    const std::vector<MatchInfo>& matches) {
    if (matches.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& m : matches) {
        const auto& p = preds[m.pred_index];
        const auto& g = gts[m.gt_index];
        sum += std::hypot(p.cx - g.cx, p.cy - g.cy);
    }
    return sum / static_cast<double>(matches.size());
}

static std::vector<uint8_t> build_occupancy_grid(const std::vector<Pt2i>& points_xy, int H, int W) {
    std::vector<uint8_t> grid(static_cast<size_t>(H) * static_cast<size_t>(W), 0);
    for (const auto& p : points_xy) {
        if (0 <= p.x && p.x < W && 0 <= p.y && p.y < H) {
            grid[static_cast<size_t>(p.y) * W + p.x] = 1;
        }
    }
    return grid;
}

static std::optional<Pt2i> find_connected_seed(const std::vector<Pt2i>& points_xy, int H, int W) {
    if (points_xy.empty()) return std::nullopt;
    std::vector<uint8_t> grid = build_occupancy_grid(points_xy, H, W);
    std::vector<Pt2i> pts = points_xy;
    std::sort(pts.begin(), pts.end(), [](const Pt2i& a, const Pt2i& b) {
        return (a.y == b.y) ? (a.x < b.x) : (a.y < b.y);
    });

    for (const auto& p : pts) {
        int x = p.x, y = p.y;
        int x0 = std::max(0, x - 1), x1 = std::min(W - 1, x + 1);
        int y0 = std::max(0, y - 1), y1 = std::min(H - 1, y + 1);
        int neigh = 0;
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                neigh += static_cast<int>(grid[static_cast<size_t>(yy) * W + xx]);
            }
        }
        neigh -= 1;
        if (neigh >= 1) return p;
    }
    return pts.front();
}

static std::vector<Pt2i> segment_points(const std::vector<Pt2i>& points_xy, const Pt2i& seed, int W, int H, int seg_half) {
    int sx = seed.x, sy = seed.y;
    int x0 = std::max(0, sx - seg_half);
    int x1 = std::min(W - 1, sx + seg_half);
    int y0 = std::max(0, sy - seg_half);
    int y1 = std::min(H - 1, sy + seg_half);
    std::vector<Pt2i> out;
    out.reserve(points_xy.size());
    for (const auto& p : points_xy) {
        if (p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1) {
            out.push_back(p);
        }
    }
    return (out.size() >= 10) ? out : points_xy;
}

static std::optional<Eigen::Vector<double, 6>> conic_from_5_points_svd(const std::array<Pt2d, 5>& points5) {
    Eigen::Matrix<double, 5, 6> M;
    for (int i = 0; i < 5; ++i) {
        double x = points5[i].x;
        double y = points5[i].y;
        M(i, 0) = x * x;
        M(i, 1) = x * y;
        M(i, 2) = y * y;
        M(i, 3) = x;
        M(i, 4) = y;
        M(i, 5) = 1.0;
    }
    Eigen::JacobiSVD<Eigen::Matrix<double, 5, 6>> svd(M, Eigen::ComputeFullV);
    Eigen::Matrix<double, 6, 6> V = svd.matrixV();
    Eigen::Vector<double, 6> p = V.col(5);
    if (p.cwiseAbs().maxCoeff() < 1e-12) return std::nullopt;
    return p;
}

static std::optional<EllipseParams> ellipse_from_conic(const Eigen::Vector<double, 6>& p) {
    double A = p(0), B = p(1), C = p(2), D = p(3), E = p(4), F = p(5);
    if ((B * B - 4.0 * A * C) >= 0.0) return std::nullopt;

    Eigen::Matrix2d G;
    G << 2.0 * A, B,
         B, 2.0 * C;
    Eigen::Vector2d g(-D, -E);

    if (std::abs(G.determinant()) < 1e-12) return std::nullopt;
    Eigen::Vector2d center = G.fullPivLu().solve(g);
    double cx = center(0);
    double cy = center(1);

    double theta = 0.5 * std::atan2(B, (A - C));
    double c = std::cos(theta);
    double s = std::sin(theta);

    double Ap = A * c * c + B * c * s + C * s * s;
    double Cp = A * s * s - B * c * s + C * c * c;
    double Fc = A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F;

    if (std::abs(Ap) < 1e-12 || std::abs(Cp) < 1e-12) return std::nullopt;
    double a2 = -Fc / Ap;
    double b2 = -Fc / Cp;
    if (a2 <= 0.0 || b2 <= 0.0) return std::nullopt;

    double a, b, theta2;
    if (a2 >= b2) {
        a = std::sqrt(a2);
        b = std::sqrt(b2);
        theta2 = theta;
    } else {
        a = std::sqrt(b2);
        b = std::sqrt(a2);
        theta2 = std::fmod(theta + M_PI / 2.0, M_PI);
    }

    return EllipseParams{cx, cy, a, b, theta2};
}

static std::optional<EllipseParams> fit_ellipse_from_5_points(const std::array<Pt2d, 5>& points5) {
    auto p = conic_from_5_points_svd(points5);
    if (!p) return std::nullopt;
    return ellipse_from_conic(*p);
}

static void merge_histograms(const std::vector<int>& src, std::vector<int>& dst) {
    if (src.size() != dst.size()) return;
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] += src[i];
    }
}

static std::pair<std::optional<EllipseParams>, DetectOneInfo> dght_detect_one_ellipse(
    const std::vector<Pt2i>& points_xy,
    int W,
    int H,
    double center_step,
    double a_step,
    double b_step,
    double theta_step_rad,
    double boundary_tol,
    int seg_half,
    int max_trials,
    int check_every,
    int min_votes_mode,
    int stable_checks_needed,
    int min_inliers_accept) {

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    auto seed = find_connected_seed(points_xy, H, W);
    if (!seed) {
        return {std::nullopt, DetectOneInfo{0.0, 0.0, -1.0}};
    }

    auto seg_pts = segment_points(points_xy, *seed, W, H, seg_half);
    if (seg_pts.size() < 5) {
        auto dt = std::chrono::duration<double>(clock::now() - t0).count();
        return {std::nullopt, DetectOneInfo{dt, 0.0, -1.0}};
    }

    double scale = static_cast<double>(W) / static_cast<double>(SPACE_SIZE);
    double a_min = static_cast<double>(SEMI_MAJOR_MIN) * scale;
    double a_max = static_cast<double>(SEMI_MAJOR_MAX) * scale;
    double b_min = a_min * ASPECT_RATIO_MIN;
    double b_max = a_max * ASPECT_RATIO_MAX;

    int cx_bins = static_cast<int>(std::floor((W - 1) / center_step)) + 1;
    int cy_bins = static_cast<int>(std::floor((H - 1) / center_step)) + 1;
    int a_bins = static_cast<int>(std::floor((a_max - a_min) / a_step)) + 1;
    int b_bins = static_cast<int>(std::floor((b_max - b_min) / b_step)) + 1;
    int th_bins = static_cast<int>(std::floor(M_PI / theta_step_rad));

    std::vector<int> Hcx(cx_bins, 0), Hcy(cy_bins, 0), Ha(a_bins, 0), Hb(b_bins, 0), Hth(th_bins, 0);
    int trials_done = 0;
    std::optional<EllipseParams> best_candidate;
    int best_inliers = -1;

#ifdef _OPENMP
    const int nthreads = (g_vote_threads > 0) ? g_vote_threads : omp_get_max_threads();
    std::vector<std::vector<int>> local_Hcx(nthreads, std::vector<int>(cx_bins, 0));
    std::vector<std::vector<int>> local_Hcy(nthreads, std::vector<int>(cy_bins, 0));
    std::vector<std::vector<int>> local_Ha(nthreads, std::vector<int>(a_bins, 0));
    std::vector<std::vector<int>> local_Hb(nthreads, std::vector<int>(b_bins, 0));
    std::vector<std::vector<int>> local_Hth(nthreads, std::vector<int>(th_bins, 0));
    std::vector<std::optional<EllipseParams>> local_best(nthreads);
    std::vector<int> local_best_inliers(nthreads, -1);
    std::vector<int> local_trials_done(nthreads, 0);

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto& lcx = local_Hcx[tid];
        auto& lcy = local_Hcy[tid];
        auto& la = local_Ha[tid];
        auto& lb = local_Hb[tid];
        auto& lth = local_Hth[tid];
        std::mt19937 rng(123 + tid * 17);

#pragma omp for schedule(dynamic)
        for (int t = 1; t <= max_trials; ++t) {
            if (seg_pts.size() < 4) continue;
            std::vector<int> indices(seg_pts.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);

            std::array<Pt2d, 5> pts5 = {
                Pt2d{static_cast<double>(seed->x), static_cast<double>(seed->y)},
                Pt2d{static_cast<double>(seg_pts[indices[0]].x), static_cast<double>(seg_pts[indices[0]].y)},
                Pt2d{static_cast<double>(seg_pts[indices[1]].x), static_cast<double>(seg_pts[indices[1]].y)},
                Pt2d{static_cast<double>(seg_pts[indices[2]].x), static_cast<double>(seg_pts[indices[2]].y)},
                Pt2d{static_cast<double>(seg_pts[indices[3]].x), static_cast<double>(seg_pts[indices[3]].y)}
            };

            auto e_opt = fit_ellipse_from_5_points(pts5);
            if (!e_opt) continue;
            const auto& e = *e_opt;

            if (!(0.0 <= e.cx && e.cx < W && 0.0 <= e.cy && e.cy < H)) continue;
            if (e.a <= 0.0 || e.b <= 0.0) continue;

            double ratio = (e.a > 0.0) ? (e.b / e.a) : 0.0;
            if (ratio < ASPECT_RATIO_MIN || ratio > ASPECT_RATIO_MAX) continue;
            if (e.a < a_min || e.a > a_max) continue;
            if (e.b < b_min || e.b > b_max) continue;

            auto inl = boundary_inliers(points_xy, e, boundary_tol);
            int inl_cnt = std::accumulate(inl.begin(), inl.end(), 0);
            if (inl_cnt > local_best_inliers[tid]) {
                local_best_inliers[tid] = inl_cnt;
                local_best[tid] = e;
            }

            if (inl_cnt < std::max(6, min_inliers_accept / 2)) continue;

            int cxb = static_cast<int>(std::floor(e.cx / center_step));
            int cyb = static_cast<int>(std::floor(e.cy / center_step));
            int ab = static_cast<int>(std::floor((e.a - a_min) / a_step));
            int bb = static_cast<int>(std::floor((e.b - b_min) / b_step));
            double th = std::fmod(e.theta, M_PI);
            if (th < 0.0) th += M_PI;
            int thb = static_cast<int>(std::floor(th / theta_step_rad));
            if (thb >= th_bins) thb = th_bins - 1;

            if (!(0 <= cxb && cxb < cx_bins && 0 <= cyb && cyb < cy_bins &&
                  0 <= ab && ab < a_bins && 0 <= bb && bb < b_bins && 0 <= thb && thb < th_bins)) {
                continue;
            }

            lcx[cxb] += inl_cnt;
            lcy[cyb] += inl_cnt;
            la[ab] += inl_cnt;
            lb[bb] += inl_cnt;
            lth[thb] += inl_cnt;
            local_trials_done[tid] = t;
        }
    }

    for (int tid = 0; tid < nthreads; ++tid) {
        merge_histograms(local_Hcx[tid], Hcx);
        merge_histograms(local_Hcy[tid], Hcy);
        merge_histograms(local_Ha[tid], Ha);
        merge_histograms(local_Hb[tid], Hb);
        merge_histograms(local_Hth[tid], Hth);
        if (local_best_inliers[tid] > best_inliers) {
            best_inliers = local_best_inliers[tid];
            best_candidate = local_best[tid];
        }
        trials_done = std::max(trials_done, local_trials_done[tid]);
    }
#else
    std::mt19937 rng(123);
    for (int t = 1; t <= max_trials; ++t) {
        if (seg_pts.size() < 4) break;
        std::vector<int> indices(seg_pts.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::array<Pt2d, 5> pts5 = {
            Pt2d{static_cast<double>(seed->x), static_cast<double>(seed->y)},
            Pt2d{static_cast<double>(seg_pts[indices[0]].x), static_cast<double>(seg_pts[indices[0]].y)},
            Pt2d{static_cast<double>(seg_pts[indices[1]].x), static_cast<double>(seg_pts[indices[1]].y)},
            Pt2d{static_cast<double>(seg_pts[indices[2]].x), static_cast<double>(seg_pts[indices[2]].y)},
            Pt2d{static_cast<double>(seg_pts[indices[3]].x), static_cast<double>(seg_pts[indices[3]].y)}
        };

        auto e_opt = fit_ellipse_from_5_points(pts5);
        if (!e_opt) continue;
        const auto& e = *e_opt;

        if (!(0.0 <= e.cx && e.cx < W && 0.0 <= e.cy && e.cy < H)) continue;
        if (e.a <= 0.0 || e.b <= 0.0) continue;

        double ratio = (e.a > 0.0) ? (e.b / e.a) : 0.0;
        if (ratio < ASPECT_RATIO_MIN || ratio > ASPECT_RATIO_MAX) continue;
        if (e.a < a_min || e.a > a_max) continue;
        if (e.b < b_min || e.b > b_max) continue;

        auto inl = boundary_inliers(points_xy, e, boundary_tol);
        int inl_cnt = std::accumulate(inl.begin(), inl.end(), 0);
        if (inl_cnt > best_inliers) {
            best_inliers = inl_cnt;
            best_candidate = e;
        }

        if (inl_cnt < std::max(6, min_inliers_accept / 2)) continue;

        int cxb = static_cast<int>(std::floor(e.cx / center_step));
        int cyb = static_cast<int>(std::floor(e.cy / center_step));
        int ab = static_cast<int>(std::floor((e.a - a_min) / a_step));
        int bb = static_cast<int>(std::floor((e.b - b_min) / b_step));
        double th = std::fmod(e.theta, M_PI);
        if (th < 0.0) th += M_PI;
        int thb = static_cast<int>(std::floor(th / theta_step_rad));
        if (thb >= th_bins) thb = th_bins - 1;

        if (!(0 <= cxb && cxb < cx_bins && 0 <= cyb && cyb < cy_bins &&
              0 <= ab && ab < a_bins && 0 <= bb && bb < b_bins && 0 <= thb && thb < th_bins)) {
            continue;
        }

        Hcx[cxb] += inl_cnt;
        Hcy[cyb] += inl_cnt;
        Ha[ab] += inl_cnt;
        Hb[bb] += inl_cnt;
        Hth[thb] += inl_cnt;
        trials_done = t;
    }
#endif

    auto elapsed = std::chrono::duration<double>(clock::now() - t0).count();
    if (!best_candidate || best_inliers < min_inliers_accept) {
        return {std::nullopt, DetectOneInfo{elapsed, static_cast<double>(trials_done), static_cast<double>(best_inliers)}};
    }

    auto argmax = [](const std::vector<int>& v) {
        return static_cast<int>(std::distance(v.begin(), std::max_element(v.begin(), v.end())));
    };

    if (trials_done > 0 && *std::max_element(Hcx.begin(), Hcx.end()) > 0 &&
        *std::max_element(Ha.begin(), Ha.end()) > 0 && *std::max_element(Hth.begin(), Hth.end()) > 0) {

        int mcx = argmax(Hcx), mcy = argmax(Hcy), ma = argmax(Ha), mb = argmax(Hb), mth = argmax(Hth);
        EllipseParams cand;
        cand.cx = (mcx + 0.5) * center_step;
        cand.cy = (mcy + 0.5) * center_step;
        cand.a = a_min + (ma + 0.5) * a_step;
        cand.b = b_min + (mb + 0.5) * b_step;
        cand.theta = (mth + 0.5) * theta_step_rad;

        auto inl = boundary_inliers(points_xy, cand, boundary_tol);
        int cnt = std::accumulate(inl.begin(), inl.end(), 0);
        if (cnt >= min_inliers_accept) {
            return {cand, DetectOneInfo{elapsed, static_cast<double>(trials_done), static_cast<double>(best_inliers)}};
        }
    }

    return {best_candidate, DetectOneInfo{elapsed, static_cast<double>(trials_done), static_cast<double>(best_inliers)}};
}

static std::pair<std::vector<std::pair<EllipseParams, int>>, PeelingInfo> detect_dght_with_peeling(
    const std::vector<Pt2i>& points_xy,
    int W,
    int H,
    double center_step,
    double a_step,
    double b_step,
    double theta_step_rad,
    double boundary_tol,
    int max_ellipses,
    int min_inliers_removed,
    int seg_half,
    int max_trials,
    int check_every,
    int min_votes_mode,
    int stable_checks_needed) {

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    std::vector<Pt2i> remaining = points_xy;
    std::vector<std::pair<EllipseParams, int>> found;
    double sum_dght = 0.0;
    double sum_inlier = 0.0;
    double total_trials = 0.0;

    for (int k = 0; k < max_ellipses; ++k) {
        if (static_cast<int>(remaining.size()) < min_inliers_removed) break;

        auto [e_opt, info] = dght_detect_one_ellipse(
            remaining, W, H, center_step, a_step, b_step, theta_step_rad,
            boundary_tol, seg_half, max_trials, check_every,
            min_votes_mode, stable_checks_needed, min_inliers_removed);

        sum_dght += info.dght_time_s;
        total_trials += info.trials;
        if (!e_opt) break;

        auto tin0 = clock::now();
        auto inl = boundary_inliers(remaining, *e_opt, boundary_tol);
        int removed = std::accumulate(inl.begin(), inl.end(), 0);
        if (removed < min_inliers_removed) break;

        std::vector<Pt2i> next_remaining;
        next_remaining.reserve(remaining.size());
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (!inl[i]) next_remaining.push_back(remaining[i]);
        }
        remaining.swap(next_remaining);
        sum_inlier += std::chrono::duration<double>(clock::now() - tin0).count();
        found.push_back({*e_opt, removed});
    }

    double total_t = std::chrono::duration<double>(clock::now() - t0).count();
    return {found, PeelingInfo{total_t, sum_dght, sum_inlier, static_cast<double>(found.size()), static_cast<double>(remaining.size()), total_trials}};
}

static MemoryEstimate dght_memory_estimate_1d(int W, int H, double center_step, double a_step, double b_step, double theta_step_rad) {
    double scale = static_cast<double>(W) / static_cast<double>(SPACE_SIZE);
    double a_min = static_cast<double>(SEMI_MAJOR_MIN) * scale;
    double a_max = static_cast<double>(SEMI_MAJOR_MAX) * scale;
    double b_min = a_min * ASPECT_RATIO_MIN;
    double b_max = a_max * ASPECT_RATIO_MAX;

    int cx_bins = static_cast<int>(std::floor((W - 1) / center_step)) + 1;
    int cy_bins = static_cast<int>(std::floor((H - 1) / center_step)) + 1;
    int a_bins = static_cast<int>(std::floor((a_max - a_min) / a_step)) + 1;
    int b_bins = static_cast<int>(std::floor((b_max - b_min) / b_step)) + 1;
    int th_bins = static_cast<int>(std::floor(M_PI / theta_step_rad));

    double bytes_total = 4.0 * (cx_bins + cy_bins + a_bins + b_bins + th_bins);
    return MemoryEstimate{
        static_cast<double>(cx_bins), static_cast<double>(cy_bins), static_cast<double>(a_bins),
        static_cast<double>(b_bins), static_cast<double>(th_bins), bytes_total / (1024.0 * 1024.0)
    };
}

static std::vector<int> split_ids(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        try {
            out.push_back(std::stoi(item));
        } catch (...) {}
    }
    return out;
}

static std::vector<int> pick_image_ids(
    const std::map<int, std::vector<EllipseParams>>& ann_map,
    const std::string& mode,
    int n,
    int start,
    int end,
    const std::vector<int>& ids) {

    std::vector<int> all_ids;
    all_ids.reserve(ann_map.size());
    for (const auto& [id, _] : ann_map) all_ids.push_back(id);

    if (!ids.empty()) {
        std::vector<int> chosen;
        for (int id : ids) if (ann_map.count(id)) chosen.push_back(id);
        return chosen;
    }

    if (mode == "first") {
        if (n > 0 && n < static_cast<int>(all_ids.size())) all_ids.resize(n);
        return all_ids;
    }
    if (mode == "range") {
        std::vector<int> chosen;
        for (int id : all_ids) if (start <= id && id <= end) chosen.push_back(id);
        if (n > 0 && n < static_cast<int>(chosen.size())) chosen.resize(n);
        return chosen;
    }
    if (mode == "random") {
        std::vector<int> chosen = all_ids;
        std::mt19937 rng(123);
        std::shuffle(chosen.begin(), chosen.end(), rng);
        if (n <= 0 || n > static_cast<int>(chosen.size())) n = static_cast<int>(chosen.size());
        chosen.resize(n);
        return chosen;
    }
    if (n > 0 && n < static_cast<int>(all_ids.size())) all_ids.resize(n);
    return all_ids;
}

static double safe_mean(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

static double safe_std(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;
    double m = safe_mean(xs);
    double acc = 0.0;
    for (double x : xs) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(xs.size()));
}

static void write_csv(const std::string& path, const std::vector<RowResult>& rows) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Failed to write CSV: " + path);
    f << "img_id,hits,gt_count,pred_count,matched_pairs,correct_matches_iou_ge_thr,fp,fn,mean_iou,mean_center_dist_px,total_detection_time_s,sum_dght_time_s,sum_inlier_remove_time_s,num_found,remaining_points,total_trials\n";
    f << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        f << r.img_id << ',' << r.hits << ',' << r.gt_count << ',' << r.pred_count << ','
          << r.matched_pairs << ',' << r.correct_matches_iou_ge_thr << ',' << r.fp << ',' << r.fn << ','
          << r.mean_iou << ',' << r.mean_center_dist_px << ',' << r.total_detection_time_s << ','
          << r.sum_dght_time_s << ',' << r.sum_inlier_remove_time_s << ',' << r.num_found << ','
          << r.remaining_points << ',' << r.total_trials << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        auto ann_map = load_annotations(args.ann);
        auto ids_list = split_ids(args.ids);
        auto image_ids = pick_image_ids(ann_map, args.mode, args.n, args.start, args.end, ids_list);
        if (image_ids.empty()) {
            throw std::runtime_error("No image ids selected. Check your annotations or selection arguments.");
        }

        std::cout << "[INFO] Selected " << image_ids.size() << " images.\n";
        g_vote_threads = args.vote_threads;
#ifdef _OPENMP
        if (args.num_threads > 0) omp_set_num_threads(args.num_threads);
        std::cout << "[INFO] OpenMP enabled. Threads: " << omp_get_max_threads() << "\n";
#else
        std::cout << "[INFO] OpenMP not enabled at compile time. Build with -fopenmp for parallel execution.\n";
#endif
        std::cout << "[INFO] Input: txt hit-point files (no OpenCV) from " << args.hits_dir << "\n";
        std::cout << "[INFO] Method: DGHT-style 1D projected accumulators (weighted by inliers) + peeling\n";
        std::cout << "[INFO] boundary_tol=" << args.boundary_tol << ", max_ellipses=" << args.max_ellipses << "\n";
        std::cout << "[INFO] Saving per-image results to: " << args.out_csv << "\n";

        double theta_step_rad = args.angle_step_deg * M_PI / 180.0;
        auto mem_est = dght_memory_estimate_1d(args.img_width, args.img_height, args.center_step_px, args.a_step_px, args.b_step_px, theta_step_rad);

        std::vector<std::optional<RowResult>> row_slots(image_ids.size());
        std::vector<double> times_total, times_dght, times_inlier, iou_means, center_dist_means;
        int total_gt = 0, total_pred = 0, total_matched = 0, total_fp = 0, total_fn = 0, total_correct_matches = 0;
        std::atomic<int> skipped{0};
        std::atomic<int> processed_counter{0};

        auto t_batch0 = std::chrono::steady_clock::now();

        for (int idx = 0; idx < static_cast<int>(image_ids.size()); ++idx) {
            int img_id = image_ids[idx];
            std::string hits_path = (fs::path(args.hits_dir) / ("id_" + std::to_string(img_id) + ".txt")).string();
            std::vector<Pt2i> xy;
            try {
                xy = load_hit_points_txt(hits_path);
            } catch (...) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            int H = args.img_height;
            int W = args.img_width;
            double sx = static_cast<double>(W) / static_cast<double>(SPACE_SIZE);
            double sy = static_cast<double>(H) / static_cast<double>(SPACE_SIZE);

            auto [det, tinfo] = detect_dght_with_peeling(
                xy, W, H,
                args.center_step_px, args.a_step_px, args.b_step_px, theta_step_rad,
                args.boundary_tol, args.max_ellipses, args.min_inliers_removed, args.seg_half_px,
                args.max_trials, args.check_every, args.min_votes_mode, args.stable_checks_needed);

            std::vector<EllipseParams> preds;
            preds.reserve(det.size());
            for (const auto& item : det) preds.push_back(item.first);

            std::vector<EllipseParams> gt_scaled;
            if (auto it = ann_map.find(img_id); it != ann_map.end()) {
                for (const auto& e : it->second) gt_scaled.push_back(scale_ellipse(e, sx, sy));
            }

            auto [matches, um_p, um_g] = greedy_match_iou(preds, gt_scaled, H, W, args.eval_downsample);
            std::vector<double> match_ious;
            match_ious.reserve(matches.size());
            for (const auto& m : matches) match_ious.push_back(m.iou);
            double mean_iou = safe_mean(match_ious);
            double mean_cd = mean_center_dist_for_matches(preds, gt_scaled, matches);
            int correct_matches = 0;
            for (const auto& m : matches) if (m.iou >= args.iou_match_min) correct_matches++;

            row_slots[idx] = RowResult{
                img_id,
                static_cast<int>(xy.size()),
                static_cast<int>(gt_scaled.size()),
                static_cast<int>(preds.size()),
                static_cast<int>(matches.size()),
                correct_matches,
                static_cast<int>(um_p.size()),
                static_cast<int>(um_g.size()),
                mean_iou,
                mean_cd,
                tinfo.total_detection_time_s,
                tinfo.sum_dght_time_s,
                tinfo.sum_inlier_remove_time_s,
                static_cast<int>(tinfo.num_found),
                static_cast<int>(tinfo.remaining_points),
                tinfo.total_trials
            };

            int done = processed_counter.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((done % 10) == 0) {
                std::cout << "[INFO] processed " << done << "/" << image_ids.size() << "...\n";
            }
        }

        auto t_batch1 = std::chrono::steady_clock::now();

        std::vector<RowResult> rows;
        rows.reserve(row_slots.size());
        for (const auto& r : row_slots) {
            if (r) rows.push_back(*r);
        }

        if (rows.empty()) {
            std::cout << "[WARN] No rows written (no images processed). Check paths and ids.\n";
            return 0;
        }

        std::sort(rows.begin(), rows.end(), [](const RowResult& a, const RowResult& b) { return a.img_id < b.img_id; });

        for (const auto& r : rows) {
            total_gt += r.gt_count;
            total_pred += r.pred_count;
            total_matched += r.matched_pairs;
            total_fp += r.fp;
            total_fn += r.fn;
            total_correct_matches += r.correct_matches_iou_ge_thr;
            times_total.push_back(r.total_detection_time_s);
            times_dght.push_back(r.sum_dght_time_s);
            times_inlier.push_back(r.sum_inlier_remove_time_s);
            iou_means.push_back(r.mean_iou);
            center_dist_means.push_back(r.mean_center_dist_px);
        }

        write_csv(args.out_csv, rows);

        double batch_wall_time_s = std::chrono::duration<double>(t_batch1 - t_batch0).count();
        int processed = static_cast<int>(rows.size());

        std::cout << "\n====================\nBATCH SUMMARY\n====================\n";
        std::cout << "requested=" << image_ids.size() << ", processed=" << processed << ", skipped_missing_files=" << skipped.load() << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "batch_wall_time_s: " << batch_wall_time_s << "\n";
        std::cout << "avg_time_per_image_s: " << safe_mean(times_total) << "  std: " << safe_std(times_total) << "\n";

        std::cout << "\n====================\nTHEORETICAL MEMORY (DGHT 1D histograms)\n====================\n";
        std::cout << std::setprecision(6);
        std::cout << "hist_total_mib: " << mem_est.hist_total_mib << " MiB (int32)\n";
        std::cout << "bins: cx=" << mem_est.cx_bins << ", cy=" << mem_est.cy_bins << ", a=" << mem_est.a_bins
                  << ", b=" << mem_est.b_bins << ", theta=" << mem_est.theta_bins << "\n";

        std::cout << "\n====================\nTIMING BREAKDOWN (means over images)\n====================\n";
        std::cout << std::setprecision(4);
        std::cout << "mean sum_dght_time_s:          " << safe_mean(times_dght) << "\n";
        std::cout << "mean sum_inlier_remove_time_s: " << safe_mean(times_inlier) << "\n";

        std::cout << "\n====================\nACCURACY SUMMARY (order-invariant, greedy IoU matching)\n====================\n";
        std::cout << "total_gt_ellipses:   " << total_gt << "\n";
        std::cout << "total_pred_ellipses: " << total_pred << "\n";
        std::cout << "matched_pairs:       " << total_matched << "\n";
        std::cout << "FP (unmatched pred): " << total_fp << "\n";
        std::cout << "FN (unmatched gt):   " << total_fn << "\n";

        double precision = total_pred > 0 ? static_cast<double>(total_correct_matches) / static_cast<double>(total_pred) : 0.0;
        double recall = total_gt > 0 ? static_cast<double>(total_correct_matches) / static_cast<double>(total_gt) : 0.0;
        double f1 = (precision + recall) > 0.0 ? (2.0 * precision * recall / (precision + recall)) : 0.0;

        std::cout << "\nIoU threshold for 'correct match': " << args.iou_match_min << "\n";
        std::cout << "correct_matches: " << total_correct_matches << "\n";
        std::cout << "precision: " << precision << "\n";
        std::cout << "recall:    " << recall << "\n";
        std::cout << "F1:        " << f1 << "\n";

        std::cout << "\nMean metrics per image:\n";
        std::cout << "mean IoU:              " << safe_mean(iou_means) << "  std: " << safe_std(iou_means) << "\n";
        std::cout << "mean center dist (px): " << safe_mean(center_dist_means) << "  std: " << safe_std(center_dist_means) << "\n";
        std::cout << "\nPer-image results saved to: " << args.out_csv << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage();
        return 1;
    }
}
