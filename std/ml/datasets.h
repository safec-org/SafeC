#pragma once
// SafeC Standard Library — datasets and mini-batch loading for std::ml.
//
// A Dataset is two row-aligned tensors: x [n, features] and y [n, targets].
// For classification y is [n, 1] holding class indices as floats and
// 'classes' is the class count; for regression y holds the targets and
// 'classes' is 0; an unlabeled set (autoencoders, GANs) has y [n, 1] of
// zeros.
//
// Synthetic generators (all seeded through an Rng, so reproducible): the
// usual 2D toy problems for classifiers (moons, circles, spirals, blobs,
// XOR), a sine regression, a ring of Gaussians (the standard GAN mode-
// collapse test) and "shapes", small binary images of four shape classes
// for autoencoders and GANs. Loaders read MNIST / Fashion-MNIST IDX files
// and numeric CSV.
#include <std/ml/tensor.h>
#include <std/ml/nn.h>

namespace std {

struct Dataset {
    &Tensor x;               // [n, features]
    &Tensor y;               // [n, targets]
    unsigned long n;
    unsigned long features;
    unsigned long targets;
    unsigned long classes;   // 0 for regression / unlabeled
};

// ── Synthetic data ───────────────────────────────────────────────────────────
// Two interleaved half circles, 2 classes (scikit-learn's make_moons).
struct Dataset dataset_moons(unsigned long n, float noise, &Rng rng);
// A circle inside a larger one, radius ratio 'factor', 2 classes.
struct Dataset dataset_circles(unsigned long n, float noise, float factor, &Rng rng);
// 'classes' interleaved spiral arms.
struct Dataset dataset_spirals(unsigned long n, unsigned long classes, float noise, &Rng rng);
// Isotropic Gaussian clusters in 'features' dimensions, centers in [-5, 5].
struct Dataset dataset_blobs(unsigned long n, unsigned long classes, unsigned long features,
                             float spread, &Rng rng);
// The four XOR quadrants of [-1, 1]^2, 2 classes.
struct Dataset dataset_xor(unsigned long n, float noise, &Rng rng);
// Regression: x uniform in [-pi, pi], y = sin(x) + noise.
struct Dataset dataset_sine(unsigned long n, float noise, &Rng rng);
// 'modes' Gaussians (std 'std') evenly spaced on a circle of 'radius';
// the label is the mode.
struct Dataset dataset_gaussian_ring(unsigned long n, unsigned long modes, float radius, float std, &Rng rng);
// side x side binary images, flattened row-major, pixels 0 or 1. Classes:
// 0 square outline, 1 plus sign, 2 filled disc, 3 diagonal cross; random
// size and position. side >= 8.
struct Dataset dataset_shapes(unsigned long n, unsigned long side, &Rng rng);

// ── Files ────────────────────────────────────────────────────────────────────
// MNIST / Fashion-MNIST IDX files (uncompressed): images as [n, 784]
// scaled to [0, 1], 10 classes. 'limit' 0 reads every image. On any error
// the result has n == 0.
struct Dataset dataset_load_mnist(const &char imagesPath, const &char labelsPath, unsigned long limit);
// A numeric CSV: one row per line, comma-separated. 'labelColumn' -1 makes
// the set unlabeled; otherwise that column becomes y, and with
// 'classification' the labels are class indices (classes = max + 1). The
// first line is skipped when 'hasHeader'. n == 0 on error.
struct Dataset dataset_load_csv(const &char path, long labelColumn, int hasHeader, int classification);

// ── Utilities ────────────────────────────────────────────────────────────────
struct DatasetSplit {
    struct Dataset train;
    struct Dataset test;
};
// Shuffles, then puts 'testFraction' of the rows in 'test'.
struct DatasetSplit dataset_split(const &Dataset ds, float testFraction, &Rng rng);
// Mean 0, std 1 per feature column, from ds's own statistics.
void dataset_standardize(&Dataset ds);
// The same with train's statistics applied to both (no test leakage).
void dataset_standardize_pair(&Dataset train, &Dataset test);
void dataset_free(&Dataset ds);

// ── Mini-batches ─────────────────────────────────────────────────────────────
struct Batch {
    &Tensor x;   // [rows, features]
    &Tensor y;   // [rows, targets]
    unsigned long rows;
};

// Iterates a dataset in batches of 'batchSize' (the last one may be
// smaller), in a fresh random order every epoch when 'shuffle'.
struct DataLoader {
    &Tensor x;
    &Tensor y;
    &heap unsigned long order;
    unsigned long n;
    unsigned long batchSize;
    unsigned long pos;
    int shuffle;
    struct Rng rng;
};
struct DataLoader dataloader_new(const &Dataset ds, unsigned long batchSize, int shuffle, unsigned long seed);
// Fills 'out' with the next batch (new tensors, in the active tensor arena
// if any) and returns 1, or returns 0 at the end of an epoch and rewinds
// (reshuffling) for the next one.
int  dataloader_next(&DataLoader dl, &Batch out);
unsigned long dataloader_batches(const &DataLoader dl);   // batches per epoch
void dataloader_free(&DataLoader dl);

} // namespace std
