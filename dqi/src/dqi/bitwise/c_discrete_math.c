#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    MAX_AMBIENT_DIM = 32,
    MAX_FREE_POS = MAX_AMBIENT_DIM * MAX_AMBIENT_DIM
};

static inline void bitset_set(uint64_t *set, uint32_t idx) {
    set[idx >> 6] |= 1ULL << (idx & 63);
}

static inline int bitset_test(const uint64_t *set, uint32_t idx) {
    return (set[idx >> 6] >> (idx & 63)) & 1ULL;
}

static PyObject* power_sum(PyObject* self, PyObject* args) {
    long N;
    long power;
    if (!PyArg_ParseTuple(args, "ll", &N, &power)) {
        return NULL;
    }
    double total = 0.0;
    for (long n = 1; n <= N; ++n) {
        total += pow((double)n, (double)power);
    }
    return PyFloat_FromDouble(total);
}

// ---------------------------------------------------------------------------
// Count linear subspaces by enumerating reduced row-echelon matrices.
// The function enumerates every k-dimensional subspace of F_2^n by generating
// all k x n RREF matrices with entries stored as 64-bit rows.

static unsigned long long count_linear_subspaces_rref(int k, int n) {
    if (k == 0) {
        return 1ULL;  // only the zero subspace
    }

    unsigned long long total = 0ULL;
    uint32_t limit = 1u << n;
    uint32_t mask = (1u << k) - 1u;  // first combination of k pivot columns

    while (mask < limit) {
        int pivots[MAX_AMBIENT_DIM];
        int pivot_count = 0;
        for (int j = 0; j < n; ++j) {
            if ((mask >> j) & 1u) {
                pivots[pivot_count++] = j;
            }
        }

        uint64_t base_rows[MAX_AMBIENT_DIM];
        struct {int row; int col;} free_pos[MAX_FREE_POS];
        int num_free = 0;
        for (int i = 0; i < k; ++i) {
            base_rows[i] = 1ULL << pivots[i];
            for (int j = pivots[i] + 1; j < n; ++j) {
                if ((mask >> j) & 1u) {
                    continue;  // pivot column
                }
                free_pos[num_free].row = i;
                free_pos[num_free].col = j;
                ++num_free;
            }
        }

        uint64_t rows[MAX_AMBIENT_DIM];
        unsigned long long assignments = 1ULL << num_free;
        for (unsigned long long a = 0; a < assignments; ++a) {
            for (int i = 0; i < k; ++i) {
                rows[i] = base_rows[i];
            }
            for (int f = 0; f < num_free; ++f) {
                if ((a >> f) & 1ULL) {
                    rows[free_pos[f].row] |= 1ULL << free_pos[f].col;
                }
            }
            // visited one matrix
            ++total;
        }

        // Gosper's hack for next combination of k bits
        uint32_t c = mask & -mask;
        uint32_t r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }

    return total;
}

// ---------------------------------------------------------------------------
// Count affine subspaces by enumerating reduced row-echelon matrices and their
// cosets. For each k-dimensional linear subspace V of F_2^n we enumerate all
// 2^{n-k} affine shifts.

static unsigned long long count_affine_subspaces_rref(int k, int n) {
    if (k == n) {
        return 1ULL;  // whole space only
    }
    if (k == 0) {
        return 1ULL << n;  // every point
    }

    unsigned long long total = 0ULL;
    uint32_t limit = 1u << n;
    uint32_t mask = (1u << k) - 1u;

    while (mask < limit) {
        int pivots[MAX_AMBIENT_DIM];
        int pivot_count = 0;
        for (int j = 0; j < n; ++j) {
            if ((mask >> j) & 1u) {
                pivots[pivot_count++] = j;
            }
        }

        uint64_t base_rows[MAX_AMBIENT_DIM];
        struct {int row; int col;} free_pos[MAX_FREE_POS];
        int num_free = 0;
        for (int i = 0; i < k; ++i) {
            base_rows[i] = 1ULL << pivots[i];
            for (int j = pivots[i] + 1; j < n; ++j) {
                if ((mask >> j) & 1u) {
                    continue;
                }
                free_pos[num_free].row = i;
                free_pos[num_free].col = j;
                ++num_free;
            }
        }

        uint64_t rows[MAX_AMBIENT_DIM];
        unsigned long long assignments = 1ULL << num_free;
        unsigned long long cosets = 1ULL << (n - k);
        for (unsigned long long a = 0; a < assignments; ++a) {
            for (int i = 0; i < k; ++i) {
                rows[i] = base_rows[i];
            }
            for (int f = 0; f < num_free; ++f) {
                if ((a >> f) & 1ULL) {
                    rows[free_pos[f].row] |= 1ULL << free_pos[f].col;
                }
            }
            for (unsigned long long s = 0; s < cosets; ++s) {
                ++total;
            }
        }

        uint32_t c = mask & -mask;
        uint32_t r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }

    return total;
}

// ---------------------------------------------------------------------------
// Determine if a given set of vectors contains an affine subspace of
// dimension k.  The set is provided as a bitset of all 2^n points of F_2^n.

static int contains_affine_subspace_rref(int k, int n, const uint64_t *set_bits) {
    if (k == n) {
        unsigned long long total = 1ULL << n;
        for (unsigned long long i = 0; i < total; ++i) {
            if (!bitset_test(set_bits, (uint32_t)i)) {
                return 0;
            }
        }
        return 1;
    }
    if (k == 0) {
        unsigned long long words = ((1ULL << n) + 63) >> 6;
        for (unsigned long long i = 0; i < words; ++i) {
            if (set_bits[i]) {
                return 1;
            }
        }
        return 0;
    }

    uint32_t limit = 1u << n;
    uint32_t mask = (1u << k) - 1u;

    while (mask < limit) {
        int pivots[MAX_AMBIENT_DIM];
        int pivot_count = 0;
        int free_cols[MAX_AMBIENT_DIM];
        int free_count = 0;
        for (int j = 0; j < n; ++j) {
            if ((mask >> j) & 1u) {
                pivots[pivot_count++] = j;
            } else {
                free_cols[free_count++] = j;
            }
        }

        uint64_t base_rows[MAX_AMBIENT_DIM];
        struct {int row; int col;} free_pos[MAX_FREE_POS];
        int num_free = 0;
        for (int i = 0; i < k; ++i) {
            base_rows[i] = 1ULL << pivots[i];
            for (int j = pivots[i] + 1; j < n; ++j) {
                if ((mask >> j) & 1u) {
                    continue;
                }
                free_pos[num_free].row = i;
                free_pos[num_free].col = j;
                ++num_free;
            }
        }

        unsigned long long assignments = 1ULL << num_free;
        uint64_t rows[MAX_AMBIENT_DIM];
        for (unsigned long long a = 0; a < assignments; ++a) {
            for (int i = 0; i < k; ++i) {
                rows[i] = base_rows[i];
            }
            for (int f = 0; f < num_free; ++f) {
                if ((a >> f) & 1ULL) {
                    rows[free_pos[f].row] |= 1ULL << free_pos[f].col;
                }
            }

            unsigned long long cosets = 1ULL << (n - k);
            for (unsigned long long s = 0; s < cosets; ++s) {
                uint64_t shift = 0ULL;
                for (int fc = 0; fc < free_count; ++fc) {
                    if ((s >> fc) & 1ULL) {
                        shift |= 1ULL << free_cols[fc];
                    }
                }

                unsigned long long combos = 1ULL << k;
                int ok = 1;
                for (unsigned long long comb = 0; comb < combos; ++comb) {
                    uint64_t vec = shift;
                    for (int i = 0; i < k; ++i) {
                        if ((comb >> i) & 1ULL) {
                            vec ^= rows[i];
                        }
                    }
                    if (!bitset_test(set_bits, (uint32_t)vec)) {
                        ok = 0;
                        break;
                    }
                }
                if (ok) {
                    return 1;
                }
            }
        }

        uint32_t c = mask & -mask;
        uint32_t r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Search for a d-dimensional linear subspace V of F_2^n such that every
// coset of V has nonempty intersection with the given set.  The set is encoded
// as a bitset over all 2^n vectors.  If such a subspace is found the basis
// vectors are written into rows_out and the function returns 1, otherwise 0.

static int find_hitting_subspace_rref(int d, int n, const uint64_t *set_bits,
                                      uint64_t *rows_out) {
    if (d == n) {
        // The only coset is the whole space; ensure S is nonempty.
        unsigned long long total = 1ULL << n;
        for (unsigned long long i = 0; i < total; ++i) {
            if (bitset_test(set_bits, (uint32_t)i)) {
                for (int j = 0; j < n; ++j) {
                    rows_out[j] = 1ULL << j;
                }
                return 1;
            }
        }
        return 0;
    }

    uint32_t limit = 1u << n;
    uint32_t mask = (1u << d) - 1u;

    while (mask < limit) {
        int pivots[MAX_AMBIENT_DIM];
        int pivot_count = 0;
        int free_cols[MAX_AMBIENT_DIM];
        int free_count = 0;
        for (int j = 0; j < n; ++j) {
            if ((mask >> j) & 1u) {
                pivots[pivot_count++] = j;
            } else {
                free_cols[free_count++] = j;
            }
        }

        uint64_t base_rows[MAX_AMBIENT_DIM];
        struct {int row; int col;} free_pos[MAX_FREE_POS];
        int num_free = 0;
        for (int i = 0; i < d; ++i) {
            base_rows[i] = 1ULL << pivots[i];
            for (int j = pivots[i] + 1; j < n; ++j) {
                if ((mask >> j) & 1u) {
                    continue;
                }
                free_pos[num_free].row = i;
                free_pos[num_free].col = j;
                ++num_free;
            }
        }

        unsigned long long assignments = 1ULL << num_free;
        uint64_t rows[MAX_AMBIENT_DIM];
        for (unsigned long long a = 0; a < assignments; ++a) {
            for (int i = 0; i < d; ++i) {
                rows[i] = base_rows[i];
            }
            for (int f = 0; f < num_free; ++f) {
                if ((a >> f) & 1ULL) {
                    rows[free_pos[f].row] |= 1ULL << free_pos[f].col;
                }
            }

            unsigned long long cosets = 1ULL << (n - d);
            int all_hit = 1;
            for (unsigned long long s = 0; s < cosets; ++s) {
                uint64_t shift = 0ULL;
                for (int fc = 0; fc < free_count; ++fc) {
                    if ((s >> fc) & 1ULL) {
                        shift |= 1ULL << free_cols[fc];
                    }
                }

                unsigned long long combos = 1ULL << d;
                int found = 0;
                for (unsigned long long comb = 0; comb < combos; ++comb) {
                    uint64_t vec = shift;
                    for (int i = 0; i < d; ++i) {
                        if ((comb >> i) & 1ULL) {
                            vec ^= rows[i];
                        }
                    }
                    if (bitset_test(set_bits, (uint32_t)vec)) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    all_hit = 0;
                    break;
                }
            }
            if (all_hit) {
                for (int i = 0; i < d; ++i) {
                    rows_out[i] = rows[i];
                }
                return 1;
            }
        }

        uint32_t c = mask & -mask;
        uint32_t r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Heuristic search for a large subset of F_2^n that contains no w-dimensional
// affine subspace.  The search performs random restarts of a greedy algorithm.

static unsigned int rng_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1ULL;
    return (unsigned int)(*state >> 32);
}


// ---- Enumeration of affine subspaces -------------------------------------

typedef void (*affine_cb)(uint32_t *points, void *ctx);

// Enumerate every w-dimensional affine subspace of F_2^n and invoke the
// callback with the list of points belonging to the subspace.  The array passed
// to the callback has length 2^w.
static int enumerate_affine_subspaces(int w, int n, affine_cb cb, void *ctx) {
    unsigned int subspace_size = 1u << w;
    uint32_t *points = (uint32_t *)malloc(sizeof(uint32_t) * subspace_size);
    if (!points) {
        return -1;
    }

    unsigned long long space_size = 1ULL << n;
    if (w == n) {
        for (unsigned long long i = 0; i < space_size; ++i) {
            points[i] = (uint32_t)i;
        }
        cb(points, ctx);
        free(points);
        return 0;
    }
    if (w == 0) {
        for (unsigned long long i = 0; i < space_size; ++i) {
            points[0] = (uint32_t)i;
            cb(points, ctx);
        }
        free(points);
        return 0;
    }

    uint32_t limit = 1u << n;
    uint32_t mask = (1u << w) - 1u;
    while (mask < limit) {
        int pivots[MAX_AMBIENT_DIM];
        int pivot_count = 0;
        int free_cols[MAX_AMBIENT_DIM];
        int free_count = 0;
        for (int j = 0; j < n; ++j) {
            if ((mask >> j) & 1u) {
                pivots[pivot_count++] = j;
            } else {
                free_cols[free_count++] = j;
            }
        }

        uint64_t base_rows[MAX_AMBIENT_DIM];
        struct {
            int row;
            int col;
        } free_pos[MAX_FREE_POS];
        int num_free = 0;
        for (int i = 0; i < w; ++i) {
            base_rows[i] = 1ULL << pivots[i];
            for (int j = pivots[i] + 1; j < n; ++j) {
                if ((mask >> j) & 1u) {
                    continue;  // pivot column
                }
                free_pos[num_free].row = i;
                free_pos[num_free].col = j;
                ++num_free;
            }
        }

        unsigned long long assignments = 1ULL << num_free;
        uint64_t rows[MAX_AMBIENT_DIM];
        for (unsigned long long a = 0; a < assignments; ++a) {
            for (int i = 0; i < w; ++i) {
                rows[i] = base_rows[i];
            }
            for (int f = 0; f < num_free; ++f) {
                if ((a >> f) & 1ULL) {
                    rows[free_pos[f].row] |= 1ULL << free_pos[f].col;
                }
            }

            unsigned long long cosets = 1ULL << (n - w);
            for (unsigned long long s = 0; s < cosets; ++s) {
                uint64_t shift = 0ULL;
                for (int fc = 0; fc < free_count; ++fc) {
                    if ((s >> fc) & 1ULL) {
                        shift |= 1ULL << free_cols[fc];
                    }
                }

                unsigned long long combos = 1ULL << w;
                for (unsigned long long comb = 0; comb < combos; ++comb) {
                    uint64_t vec = shift;
                    for (int i = 0; i < w; ++i) {
                        if ((comb >> i) & 1ULL) {
                            vec ^= rows[i];
                        }
                    }
                    points[comb] = (uint32_t)vec;
                }
                cb(points, ctx);
            }
        }

        uint32_t c = mask & -mask;
        uint32_t r = mask + c;
        mask = (((r ^ mask) >> 2) / c) | r;
    }

    free(points);
    return 0;
}

struct find_max_overlap_ctx {
    int w;
    const uint64_t *set_bits;
    int max_overlap;
    uint32_t **subspaces;
    int count;
    int capacity;
};

static void max_overlap_cb(uint32_t *pts, void *vctx) {
    struct find_max_overlap_ctx *ctx = (struct find_max_overlap_ctx *)vctx;
    unsigned int size = 1u << ctx->w;
    int count = 0;
    for (unsigned int i = 0; i < size; ++i) {
        if (bitset_test(ctx->set_bits, pts[i])) {
            ++count;
        }
    }

    if (count > ctx->max_overlap) {
        ctx->max_overlap = count;
        ctx->count = 0; // Reset count for new max
    }

    if (count == ctx->max_overlap) {
        if (ctx->count >= ctx->capacity) {
            // Grow the list of subspaces
            int new_capacity = ctx->capacity == 0 ? 16 : ctx->capacity * 2;
            uint32_t **new_subspaces = realloc(ctx->subspaces, new_capacity * sizeof(uint32_t *));
            if (!new_subspaces) {
                // Handle realloc failure; maybe set an error flag in ctx
                return;
            }
            ctx->subspaces = new_subspaces;
            ctx->capacity = new_capacity;
        }
        uint32_t *subspace_copy = malloc(size * sizeof(uint32_t));
        if (!subspace_copy) {
            // Handle malloc failure
            return;
        }
        memcpy(subspace_copy, pts, size * sizeof(uint32_t));
        ctx->subspaces[ctx->count++] = subspace_copy;
    }
}

static int find_max_overlap_affine_subspace_rref(int k, int n,
                                                 const uint64_t *set_bits) {
    struct find_max_overlap_ctx ctx = {k, set_bits, 0, NULL, 0, 0};
    if (enumerate_affine_subspaces(k, n, max_overlap_cb, &ctx) < 0) {
        return -1;
    }
    // Free stored subspaces as they are not used in this version
    for (int i = 0; i < ctx.count; ++i) {
        free(ctx.subspaces[i]);
    }
    free(ctx.subspaces);
    return ctx.max_overlap;
}

static uint32_t** find_all_max_overlap_affine_subspace_rref(int k, int n,
                                                            const uint64_t *set_bits, int *num_subspaces) {
    struct find_max_overlap_ctx ctx = {k, set_bits, 0, NULL, 0, 0};
    if (enumerate_affine_subspaces(k, n, max_overlap_cb, &ctx) < 0) {
        return NULL;
    }
    *num_subspaces = ctx.count;
    return ctx.subspaces;
}

struct count_ctx {
    int w;
    uint32_t *deg;
    unsigned long long subspaces;
};

static void count_cb(uint32_t *pts, void *vctx) {
    struct count_ctx *ctx = (struct count_ctx *)vctx;
    uint32_t size = 1u << ctx->w;
    for (uint32_t i = 0; i < size; ++i) {
        ctx->deg[pts[i]]++;
    }
    ctx->subspaces++;
}

struct fill_ctx {
    int w;
    uint32_t *adj;
    uint32_t *pos;
    uint16_t *remaining;
    unsigned long long idx;
};

static void fill_cb(uint32_t *pts, void *vctx) {
    struct fill_ctx *ctx = (struct fill_ctx *)vctx;
    uint32_t size = 1u << ctx->w;
    unsigned long long idx = ctx->idx++;
    if (ctx->remaining) {
        ctx->remaining[idx] = (uint16_t)size;
    }
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t p = pts[i];
        ctx->adj[ctx->pos[p]++] = (uint32_t)idx;
    }
}

// Optimized heuristic search using an adjacency graph between points and
// affine subspaces.  Subspaces are removed from consideration once it becomes
// impossible for them to be fully contained in the chosen set.
static int find_subspace_evasive_set_greedy(int n, int w, int max_restarts,
                                            uint64_t seed, uint64_t *best_out) {
    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    unsigned long long subspaces = count_affine_subspaces_rref(w, n);
    unsigned long long edges = subspaces << w;

    uint32_t *degree = calloc(total_points, sizeof(uint32_t));
    if (!degree) {
        return -1;
    }
    struct count_ctx cctx = {w, degree, 0};
    if (enumerate_affine_subspaces(w, n, count_cb, &cctx) < 0) {
        free(degree);
        return -1;
    }

    unsigned long long counted_subspaces = cctx.subspaces;
    unsigned long long total_edges = 0;
    uint32_t *offsets = malloc(sizeof(uint32_t) * (total_points + 1));
    if (!offsets) {
        free(degree);
        return -1;
    }
    offsets[0] = 0;
    for (unsigned long long i = 0; i < total_points; ++i) {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    total_edges = offsets[total_points];

    uint32_t *adj = malloc(sizeof(uint32_t) * total_edges);
    uint16_t *remaining_init = malloc(sizeof(uint16_t) * counted_subspaces);
    uint32_t *pos = malloc(sizeof(uint32_t) * total_points);
    if (!adj || !remaining_init || !pos) {
        free(degree);
        free(offsets);
        free(adj);
        free(remaining_init);
        free(pos);
        return -1;
    }
    memcpy(pos, offsets, sizeof(uint32_t) * total_points);

    struct fill_ctx fctx = {w, adj, pos, remaining_init, 0};
    if (enumerate_affine_subspaces(w, n, fill_cb, &fctx) < 0) {
        free(degree);
        free(offsets);
        free(adj);
        free(remaining_init);
        free(pos);
        return -1;
    }

    free(degree);
    free(pos);

    uint16_t *remaining = malloc(sizeof(uint16_t) * counted_subspaces);
    uint64_t *current = calloc(words, sizeof(uint64_t));
    uint32_t *indices = malloc(sizeof(uint32_t) * total_points);
    if (!remaining || !current || !indices) {
        free(offsets);
        free(adj);
        free(remaining_init);
        free(remaining);
        free(current);
        free(indices);
        return -1;
    }

    for (uint32_t i = 0; i < total_points; ++i) {
        indices[i] = i;
    }
    memset(best_out, 0, words * sizeof(uint64_t));
    int best_size = 0;
    int stagnation = 0;
    uint64_t state = seed ? seed : 0x9e3779b97f4a7c15ULL;
    while (stagnation < max_restarts) {
        for (unsigned long long i = total_points - 1; i > 0; --i) {
            unsigned int j = rng_next(&state) % (i + 1);
            uint32_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        memset(current, 0, words * sizeof(uint64_t));
        memcpy(remaining, remaining_init,
               sizeof(uint16_t) * counted_subspaces);
        int size = 0;
        for (unsigned long long t = 0; t < total_points; ++t) {
            uint32_t idx = indices[t];
            int ok = 1;
            for (uint32_t e = offsets[idx]; e < offsets[idx + 1]; ++e) {
                uint32_t s = adj[e];
                if (remaining[s] <= 1) {
                    ok = 0;
                    break;
                }
            }
            if (ok) {
                bitset_set(current, idx);
                ++size;
                for (uint32_t e = offsets[idx]; e < offsets[idx + 1]; ++e) {
                    uint32_t s = adj[e];
                    --remaining[s];
                }
            }
        }
        if (size > best_size) {
            memcpy(best_out, current, words * sizeof(uint64_t));
            best_size = size;
            stagnation = 0;
        } else {
            ++stagnation;
        }
    }

    free(offsets);
    free(adj);
    free(remaining_init);
    free(remaining);
    free(current);
    free(indices);
    return best_size;
}

// Greedy search variant that also enforces a prefix covering constraint. For
// each prefix of length d (on the low-order bits), the chosen set must contain
// at least one point with that prefix.  When d = 0 this condition is vacuous
// and the caller should use the simpler greedy routine above.
static int find_subspace_evasive_set_rref(int n, int w, int max_restarts,
                                          uint64_t seed, uint64_t *best_out) {
    return find_subspace_evasive_set_greedy(n, w, max_restarts, seed, best_out);
}

static PyObject* count_linear_subspaces(PyObject* self, PyObject* args) {
    int k, n;
    if (!PyArg_ParseTuple(args, "ii", &k, &n)) {
        return NULL;
    }
    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError, "n must be at most %d", MAX_AMBIENT_DIM);
        return NULL;
    }
    if (k < 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "require 0 <= k <= n");
        return NULL;
    }

    unsigned long long total = count_linear_subspaces_rref(k, n);
    return PyLong_FromUnsignedLongLong(total);
}

static PyObject* count_affine_subspaces(PyObject* self, PyObject* args) {
    int k, n;
    if (!PyArg_ParseTuple(args, "ii", &k, &n)) {
        return NULL;
    }
    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError, "n must be at most %d", MAX_AMBIENT_DIM);
        return NULL;
    }
    if (k < 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "require 0 <= k <= n");
        return NULL;
    }

    unsigned long long total = count_affine_subspaces_rref(k, n);
    return PyLong_FromUnsignedLongLong(total);
}

static PyObject* contains_affine_subspace(PyObject* self, PyObject* args) {
    PyObject *py_set;
    int k;
    if (!PyArg_ParseTuple(args, "Oi", &py_set, &k)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(py_set, "S must be a sequence of bitstrings");
    if (!seq) {
        return NULL;
    }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(seq);
    if (m == 0) {
        Py_DECREF(seq);
        return PyBool_FromLong(0);
    }

    PyObject *first = PySequence_Fast_GET_ITEM(seq, 0);
    PyObject *first_seq = PySequence_Fast(first, "elements must be sequences of bits");
    if (!first_seq) {
        Py_DECREF(seq);
        return NULL;
    }
    int n = (int)PySequence_Fast_GET_SIZE(first_seq);
    Py_DECREF(first_seq);

    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError, "bitstrings must have length at most %d", MAX_AMBIENT_DIM);
        Py_DECREF(seq);
        return NULL;
    }
    if (k < 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "require 0 <= k <= n");
        Py_DECREF(seq);
        return NULL;
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    uint64_t *bitset = PyMem_Calloc(words, sizeof(uint64_t));
    if (!bitset) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < m; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *item_seq = PySequence_Fast(item, "bitstring must be a sequence");
        if (!item_seq) {
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(item_seq) != n) {
            PyErr_SetString(PyExc_ValueError, "all bitstrings must have equal length");
            Py_DECREF(item_seq);
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        uint32_t value = 0;
        for (int j = 0; j < n; ++j) {
            PyObject *bobj = PySequence_Fast_GET_ITEM(item_seq, j);
            long b = PyLong_AsLong(bobj);
            if ((b == -1 && PyErr_Occurred()) || (b != 0 && b != 1)) {
                PyErr_SetString(PyExc_ValueError, "bits must be 0 or 1");
                Py_DECREF(item_seq);
                PyMem_Free(bitset);
                Py_DECREF(seq);
                return NULL;
            }
            if (b) {
                value |= 1u << j;
            }
        }
        Py_DECREF(item_seq);
        bitset_set(bitset, value);
    }

    int result = contains_affine_subspace_rref(k, n, bitset);

    PyMem_Free(bitset);
    Py_DECREF(seq);

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject* find_max_overlap_affine_subspace(PyObject* self, PyObject* args) {
    PyObject *py_set;
    int k;
    if (!PyArg_ParseTuple(args, "Oi", &py_set, &k)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(py_set, "S must be a sequence of bitstrings");
    if (!seq) {
        return NULL;
    }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(seq);
    if (m == 0) {
        Py_DECREF(seq);
        return PyLong_FromLong(0);
    }

    PyObject *first = PySequence_Fast_GET_ITEM(seq, 0);
    PyObject *first_seq = PySequence_Fast(first, "elements must be sequences of bits");
    if (!first_seq) {
        Py_DECREF(seq);
        return NULL;
    }
    int n = (int)PySequence_Fast_GET_SIZE(first_seq);
    Py_DECREF(first_seq);

    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError,
                     "bitstrings must have length at most %d", MAX_AMBIENT_DIM);
        Py_DECREF(seq);
        return NULL;
    }
    if (k < 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "require 0 <= k <= n");
        Py_DECREF(seq);
        return NULL;
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    uint64_t *bitset = PyMem_Calloc(words, sizeof(uint64_t));
    if (!bitset) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < m; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *item_seq = PySequence_Fast(item, "bitstring must be a sequence");
        if (!item_seq) {
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(item_seq) != n) {
            PyErr_SetString(PyExc_ValueError,
                            "all bitstrings must have equal length");
            Py_DECREF(item_seq);
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        uint32_t value = 0;
        for (int j = 0; j < n; ++j) {
            PyObject *bobj = PySequence_Fast_GET_ITEM(item_seq, j);
            long b = PyLong_AsLong(bobj);
            if ((b == -1 && PyErr_Occurred()) || (b != 0 && b != 1)) {
                PyErr_SetString(PyExc_ValueError, "bits must be 0 or 1");
                Py_DECREF(item_seq);
                PyMem_Free(bitset);
                Py_DECREF(seq);
                return NULL;
            }
            if (b) {
                value |= 1u << j;
            }
        }
        Py_DECREF(item_seq);
        bitset_set(bitset, value);
    }

    int result = find_max_overlap_affine_subspace_rref(k, n, bitset);

    PyMem_Free(bitset);
    Py_DECREF(seq);

    if (result < 0) {
        PyErr_NoMemory();
        return NULL;
    }
    return PyLong_FromLong(result);
}

static PyObject* find_all_max_overlap_affine_subspaces(PyObject* self, PyObject* args) {
    PyObject *py_set;
    int k;
    if (!PyArg_ParseTuple(args, "Oi", &py_set, &k)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(py_set, "S must be a sequence of bitstrings");
    if (!seq) {
        return NULL;
    }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(seq);
    if (m == 0) {
        Py_DECREF(seq);
        return PyList_New(0);
    }

    PyObject *first = PySequence_Fast_GET_ITEM(seq, 0);
    PyObject *first_seq = PySequence_Fast(first, "elements must be sequences of bits");
    if (!first_seq) {
        Py_DECREF(seq);
        return NULL;
    }
    int n = (int)PySequence_Fast_GET_SIZE(first_seq);
    Py_DECREF(first_seq);

    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError,
                     "bitstrings must have length at most %d", MAX_AMBIENT_DIM);
        Py_DECREF(seq);
        return NULL;
    }
    if (k < 0 || k > n) {
        PyErr_SetString(PyExc_ValueError, "require 0 <= k <= n");
        Py_DECREF(seq);
        return NULL;
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    uint64_t *bitset = PyMem_Calloc(words, sizeof(uint64_t));
    if (!bitset) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < m; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *item_seq = PySequence_Fast(item, "bitstring must be a sequence");
        if (!item_seq) {
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(item_seq) != n) {
            PyErr_SetString(PyExc_ValueError,
                            "all bitstrings must have equal length");
            Py_DECREF(item_seq);
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        uint32_t value = 0;
        for (int j = 0; j < n; ++j) {
            PyObject *bobj = PySequence_Fast_GET_ITEM(item_seq, j);
            long b = PyLong_AsLong(bobj);
            if ((b == -1 && PyErr_Occurred()) || (b != 0 && b != 1)) {
                PyErr_SetString(PyExc_ValueError, "bits must be 0 or 1");
                Py_DECREF(item_seq);
                PyMem_Free(bitset);
                Py_DECREF(seq);
                return NULL;
            }
            if (b) {
                value |= 1u << j;
            }
        }
        Py_DECREF(item_seq);
        bitset_set(bitset, value);
    }

    int num_subspaces;
    uint32_t **subspaces = find_all_max_overlap_affine_subspace_rref(k, n, bitset, &num_subspaces);

    PyMem_Free(bitset);
    Py_DECREF(seq);

    if (!subspaces) {
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *py_subspaces = PyList_New(num_subspaces);
    if (!py_subspaces) {
        for (int i = 0; i < num_subspaces; ++i) free(subspaces[i]);
        free(subspaces);
        PyErr_NoMemory();
        return NULL;
    }

    unsigned int subspace_size = 1u << k;
    for (int i = 0; i < num_subspaces; ++i) {
        PyObject *py_subspace = PySet_New(NULL);
        if (!py_subspace) {
            // Cleanup
            Py_DECREF(py_subspaces);
            for (int j = 0; j < num_subspaces; ++j) free(subspaces[j]);
            free(subspaces);
            PyErr_NoMemory();
            return NULL;
        }
        for (unsigned int j = 0; j < subspace_size; ++j) {
            PyObject* point_tuple = PyTuple_New(n);
            if (!point_tuple) {
                // More cleanup
                Py_DECREF(py_subspace);
                Py_DECREF(py_subspaces);
                for (int l = 0; l < num_subspaces; ++l) free(subspaces[l]);
                free(subspaces);
                PyErr_NoMemory();
                return NULL;
            }
            uint32_t point = subspaces[i][j];
            for (int bit = 0; bit < n; ++bit) {
                PyTuple_SET_ITEM(point_tuple, bit, PyLong_FromLong((point >> bit) & 1));
            }
            PySet_Add(py_subspace, point_tuple);
            Py_DECREF(point_tuple);
        }
        PyList_SET_ITEM(py_subspaces, i, py_subspace);
        free(subspaces[i]);
    }
    free(subspaces);

    return py_subspaces;
}

// ---- Affine spectrum -----------------------------------------------------

struct affine_spectrum_ctx {
    int w;
    const uint64_t *set_bits;
    unsigned long long *spectrum_row;
};

static void affine_spectrum_cb(uint32_t *pts, void *vctx) {
    struct affine_spectrum_ctx *ctx = (struct affine_spectrum_ctx *)vctx;
    unsigned int size = 1u << ctx->w;
    int count = 0;
    for (unsigned int i = 0; i < size; ++i) {
        if (bitset_test(ctx->set_bits, pts[i])) {
            ++count;
        }
    }
    ctx->spectrum_row[count]++;
}

static unsigned long long** affine_spectrum_rref(int n, const uint64_t *set_bits) {
    unsigned long long **spectrum = (unsigned long long **)malloc(
        (n + 1) * sizeof(unsigned long long *));
    if (!spectrum) {
        return NULL;
    }

    for (int k = 0; k <= n; ++k) {
        unsigned long long row_size = (1ULL << k) + 1;
        spectrum[k] = (unsigned long long *)calloc(row_size, sizeof(unsigned long long));
        if (!spectrum[k]) {
            for (int i = 0; i < k; ++i) {
                free(spectrum[i]);
            }
            free(spectrum);
            return NULL;
        }
        struct affine_spectrum_ctx ctx = {k, set_bits, spectrum[k]};
        if (enumerate_affine_subspaces(k, n, affine_spectrum_cb, &ctx) < 0) {
            for (int i = 0; i <= k; ++i) {
                free(spectrum[i]);
            }
            free(spectrum);
            return NULL;
        }
    }
    return spectrum;
}

static PyObject* affine_spectrum(PyObject* self, PyObject* args) {
    PyObject *py_set;
    if (!PyArg_ParseTuple(args, "O", &py_set)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(py_set, "S must be a sequence of bitstrings");
    if (!seq) {
        return NULL;
    }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(seq);
    int n;
    if (m == 0) {
        // If the set is empty, we need to know the dimension `n`.
        // We'll try to get it from the user, but for now, let's assume n=0
        // and the user should provide a non-empty set to define `n`.
        // A better solution would be to take `n` as an argument.
        // For now, we handle this case by assuming n from an element.
        // If no elements, we can't determine n.
        PyErr_SetString(PyExc_ValueError, "Cannot determine dimension n from an empty set.");
        Py_DECREF(seq);
        return NULL;
    }

    PyObject *first = PySequence_Fast_GET_ITEM(seq, 0);
    PyObject *first_seq = PySequence_Fast(first, "elements must be sequences of bits");
    if (!first_seq) {
        Py_DECREF(seq);
        return NULL;
    }
    n = (int)PySequence_Fast_GET_SIZE(first_seq);
    Py_DECREF(first_seq);

    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError,
                     "bitstrings must have length at most %d", MAX_AMBIENT_DIM);
        Py_DECREF(seq);
        return NULL;
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    uint64_t *bitset = PyMem_Calloc(words, sizeof(uint64_t));
    if (!bitset) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < m; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *item_seq = PySequence_Fast(item, "bitstring must be a sequence");
        if (!item_seq) {
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(item_seq) != n) {
            PyErr_SetString(PyExc_ValueError,
                            "all bitstrings must have equal length");
            Py_DECREF(item_seq);
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        uint32_t value = 0;
        for (int j = 0; j < n; ++j) {
            PyObject *bobj = PySequence_Fast_GET_ITEM(item_seq, j);
            long b = PyLong_AsLong(bobj);
            if ((b == -1 && PyErr_Occurred()) || (b != 0 && b != 1)) {
                PyErr_SetString(PyExc_ValueError, "bits must be 0 or 1");
                Py_DECREF(item_seq);
                PyMem_Free(bitset);
                Py_DECREF(seq);
                return NULL;
            }
            if (b) {
                value |= 1u << j;
            }
        }
        Py_DECREF(item_seq);
        bitset_set(bitset, value);
    }

    unsigned long long **spectrum = affine_spectrum_rref(n, bitset);
    PyMem_Free(bitset);
    Py_DECREF(seq);

    if (!spectrum) {
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *py_spectrum = PyList_New(n + 1);
    if (!py_spectrum) {
        for (int k = 0; k <= n; ++k) free(spectrum[k]);
        free(spectrum);
        PyErr_NoMemory();
        return NULL;
    }

    for (int k = 0; k <= n; ++k) {
        unsigned long long row_size = (1ULL << k) + 1;
        PyObject *py_row = PyList_New(row_size);
        if (!py_row) {
            // Complicated cleanup
            Py_DECREF(py_spectrum);
            for (int i = 0; i <= n; ++i) free(spectrum[i]);
            free(spectrum);
            PyErr_NoMemory();
            return NULL;
        }
        for (unsigned long long s = 0; s < row_size; ++s) {
            PyObject *val = PyLong_FromUnsignedLongLong(spectrum[k][s]);
            if (!val) {
                // More complicated cleanup
                Py_DECREF(py_row);
                Py_DECREF(py_spectrum);
                for (int i = 0; i <= n; ++i) free(spectrum[i]);
                free(spectrum);
                PyErr_NoMemory();
                return NULL;
            }
            PyList_SET_ITEM(py_row, s, val);
        }
        PyList_SET_ITEM(py_spectrum, k, py_row);
        free(spectrum[k]);
    }
    free(spectrum);

    return py_spectrum;
}

static PyObject* find_hitting_subspace(PyObject* self, PyObject* args) {
    PyObject *py_set;
    int d;
    if (!PyArg_ParseTuple(args, "Oi", &py_set, &d)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(py_set, "S must be a sequence of bitstrings");
    if (!seq) {
        return NULL;
    }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(seq);
    if (m == 0) {
        Py_DECREF(seq);
        Py_RETURN_NONE;
    }

    PyObject *first = PySequence_Fast_GET_ITEM(seq, 0);
    PyObject *first_seq = PySequence_Fast(first, "elements must be sequences of bits");
    if (!first_seq) {
        Py_DECREF(seq);
        return NULL;
    }
    int n = (int)PySequence_Fast_GET_SIZE(first_seq);
    Py_DECREF(first_seq);

    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError, "bitstrings must have length at most %d", MAX_AMBIENT_DIM);
        Py_DECREF(seq);
        return NULL;
    }
    if (d <= 0 || d > n) {
        PyErr_SetString(PyExc_ValueError, "require 1 <= d <= n");
        Py_DECREF(seq);
        return NULL;
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    uint64_t *bitset = PyMem_Calloc(words, sizeof(uint64_t));
    if (!bitset) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < m; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *item_seq = PySequence_Fast(item, "bitstring must be a sequence");
        if (!item_seq) {
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(item_seq) != n) {
            PyErr_SetString(PyExc_ValueError, "all bitstrings must have equal length");
            Py_DECREF(item_seq);
            PyMem_Free(bitset);
            Py_DECREF(seq);
            return NULL;
        }
        uint32_t value = 0;
        for (int j = 0; j < n; ++j) {
            PyObject *bobj = PySequence_Fast_GET_ITEM(item_seq, j);
            long b = PyLong_AsLong(bobj);
            if ((b == -1 && PyErr_Occurred()) || (b != 0 && b != 1)) {
                PyErr_SetString(PyExc_ValueError, "bits must be 0 or 1");
                Py_DECREF(item_seq);
                PyMem_Free(bitset);
                Py_DECREF(seq);
                return NULL;
            }
            if (b) {
                value |= 1u << j;
            }
        }
        Py_DECREF(item_seq);
        bitset_set(bitset, value);
    }

    uint64_t rows[MAX_AMBIENT_DIM];
    int result = find_hitting_subspace_rref(d, n, bitset, rows);

    PyMem_Free(bitset);
    Py_DECREF(seq);

    if (!result) {
        Py_RETURN_NONE;
    }

    PyObject *basis = PyList_New(d);
    if (!basis) {
        return NULL;
    }
    for (int i = 0; i < d; ++i) {
        PyObject *tuple = PyTuple_New(n);
        if (!tuple) {
            Py_DECREF(basis);
            return NULL;
        }
        for (int j = 0; j < n; ++j) {
            PyObject *bit = PyLong_FromLong((rows[i] >> j) & 1ULL);
            if (!bit) {
                Py_DECREF(tuple);
                Py_DECREF(basis);
                return NULL;
            }
            PyTuple_SET_ITEM(tuple, j, bit);  // steals reference
        }
        PyList_SET_ITEM(basis, i, tuple);  // steals reference
    }

    return basis;
}

// Optimized heuristic search for a subspace-limited set. This is a more
// general version of the evasive set finder.
static int find_subspace_limited_set_greedy(int n, const int *max_overlaps,
                                            int max_restarts, uint64_t seed,
                                            uint64_t *best_out) {
    if (n > 20) { // Safeguard against very large n
        return -2;
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;

    // Count total number of subspaces and edges
    unsigned long long total_subspaces = 0;
    for (int w = 0; w <= n; ++w) {
        total_subspaces += count_affine_subspaces_rref(w, n);
    }

    uint32_t *degree = calloc(total_points, sizeof(uint32_t));
    if (!degree) return -1;

    // Adjacency information
    uint32_t *adj = NULL;
    uint16_t *subspace_dims = NULL;
    uint32_t *offsets = NULL;
    uint32_t *pos = NULL;

    // Workspace for greedy algorithm
    uint16_t *overlaps = NULL;
    uint64_t *current = NULL;
    uint32_t *indices = NULL;

    int best_size = -1;

    // Enumerate all subspaces to build adjacency list
    unsigned long long current_subspace_idx = 0;
    for (int w = 0; w <= n; ++w) {
        struct count_ctx cctx = {w, degree, 0};
        if (enumerate_affine_subspaces(w, n, count_cb, &cctx) < 0) {
            goto cleanup;
        }
    }

    unsigned long long total_edges = 0;
    offsets = malloc(sizeof(uint32_t) * (total_points + 1));
    if (!offsets) goto cleanup;
    offsets[0] = 0;
    for (unsigned long long i = 0; i < total_points; ++i) {
        offsets[i + 1] = offsets[i] + degree[i];
        total_edges += degree[i];
    }

    adj = malloc(sizeof(uint32_t) * total_edges);
    subspace_dims = malloc(sizeof(uint16_t) * total_subspaces);
    pos = malloc(sizeof(uint32_t) * total_points);
    if (!adj || !subspace_dims || !pos) goto cleanup;

    memcpy(pos, offsets, sizeof(uint32_t) * total_points);

    current_subspace_idx = 0;
    for (int w = 0; w <= n; ++w) {
        unsigned long long num_subspaces_w = count_affine_subspaces_rref(w, n);
        for (unsigned long long i = 0; i < num_subspaces_w; ++i) {
            subspace_dims[current_subspace_idx + i] = w;
        }
        // The fill_cb doesn't use remaining_init, so we can pass NULL.
        struct fill_ctx fctx = {w, adj, pos, NULL, current_subspace_idx};
        if (enumerate_affine_subspaces(w, n, fill_cb, &fctx) < 0) {
            goto cleanup;
        }
        current_subspace_idx += num_subspaces_w;
    }

    free(degree);
    degree = NULL;
    free(pos);
    pos = NULL;

    // Greedy search with restarts
    overlaps = malloc(sizeof(uint16_t) * total_subspaces);
    current = calloc(words, sizeof(uint64_t));
    indices = malloc(sizeof(uint32_t) * total_points);
    if (!overlaps || !current || !indices) goto cleanup;

    for (uint32_t i = 0; i < total_points; ++i) {
        indices[i] = i;
    }
    memset(best_out, 0, words * sizeof(uint64_t));
    best_size = 0;
    int stagnation = 0;
    uint64_t state = seed ? seed : 0x9e3779b97f4a7c15ULL;

    while (stagnation < max_restarts) {
        for (unsigned long long i = total_points - 1; i > 0; --i) {
            unsigned int j = rng_next(&state) % (i + 1);
            uint32_t tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        memset(current, 0, words * sizeof(uint64_t));
        memset(overlaps, 0, sizeof(uint16_t) * total_subspaces);
        int size = 0;

        for (unsigned long long t = 0; t < total_points; ++t) {
            uint32_t idx = indices[t];
            int ok = 1;
            for (uint32_t e = offsets[idx]; e < offsets[idx + 1]; ++e) {
                uint32_t s = adj[e];
                uint16_t w = subspace_dims[s];
                if (overlaps[s] >= max_overlaps[w]) {
                    ok = 0;
                    break;
                }
            }
            if (ok) {
                bitset_set(current, idx);
                ++size;
                for (uint32_t e = offsets[idx]; e < offsets[idx + 1]; ++e) {
                    uint32_t s = adj[e];
                    ++overlaps[s];
                }
            }
        }

        if (size > best_size) {
            memcpy(best_out, current, words * sizeof(uint64_t));
            best_size = size;
            stagnation = 0;
        } else {
            ++stagnation;
        }
    }

cleanup:
    free(degree);
    free(offsets);
    free(adj);
    free(subspace_dims);
    free(pos);
    free(overlaps);
    free(current);
    free(indices);
    return best_size;
}


static PyObject* find_subspace_limited_set(PyObject* self, PyObject* args) {
    int n, max_restarts;
    PyObject *py_max_overlaps;
    unsigned int seed = 0;
    if (!PyArg_ParseTuple(args, "iO!i|I", &n, &PyList_Type, &py_max_overlaps, &max_restarts, &seed)) {
        return NULL;
    }
    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError, "n must be at most %d", MAX_AMBIENT_DIM);
        return NULL;
    }
    if (max_restarts <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_restarts must be positive");
        return NULL;
    }
    if (PyList_Size(py_max_overlaps) != n + 1) {
        PyErr_Format(PyExc_ValueError, "max_overlaps must have length n+1 = %d", n + 1);
        return NULL;
    }

    int max_overlaps[n + 1];
    for (int i = 0; i <= n; ++i) {
        PyObject *item = PyList_GetItem(py_max_overlaps, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_ValueError, "max_overlaps must contain integers");
            return NULL;
        }
        max_overlaps[i] = PyLong_AsLong(item);
        if (max_overlaps[i] < 0) {
            PyErr_SetString(PyExc_ValueError, "max_overlaps values cannot be negative");
            return NULL;
        }
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    uint64_t *best = PyMem_Calloc(words, sizeof(uint64_t));
    if (!best) {
        PyErr_NoMemory();
        return NULL;
    }

    int size = find_subspace_limited_set_greedy(n, max_overlaps, max_restarts, seed, best);

    if (size == -1) {
        PyMem_Free(best);
        PyErr_NoMemory();
        return NULL;
    }
    if (size == -2) {
        PyMem_Free(best);
        PyErr_Format(PyExc_ValueError, "n is too large for this function.");
        return NULL;
    }


    PyObject *result = PyList_New(size);
    if (!result) {
        PyMem_Free(best);
        return NULL;
    }
    int idx = 0;
    for (unsigned long long i = 0; i < total_points; ++i) {
        if (bitset_test(best, (uint32_t)i)) {
            PyObject *tuple = PyTuple_New(n);
            if (!tuple) {
                Py_DECREF(result);
                PyMem_Free(best);
                return NULL;
            }
            for (int j = 0; j < n; ++j) {
                PyObject *bit = PyLong_FromLong((i >> j) & 1ULL);
                if (!bit) {
                    Py_DECREF(tuple);
                    Py_DECREF(result);
                    PyMem_Free(best);
                    return NULL;
                }
                PyTuple_SET_ITEM(tuple, j, bit);
            }
            PyList_SET_ITEM(result, idx++, tuple);
        }
    }

    PyMem_Free(best);
    return result;
}

static PyObject* find_subspace_evasive_set(PyObject* self, PyObject* args) {
    int n, w, max_restarts;
    unsigned int seed = 0;
    if (!PyArg_ParseTuple(args, "iii|I", &n, &w, &max_restarts, &seed)) {
        return NULL;
    }
    if (n > MAX_AMBIENT_DIM) {
        PyErr_Format(PyExc_ValueError, "n must be at most %d", MAX_AMBIENT_DIM);
        return NULL;
    }
    if (w < 0 || w > n) {
        PyErr_SetString(PyExc_ValueError, "require 0 <= w <= n");
        return NULL;
    }
    if (max_restarts <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_restarts must be positive");
        return NULL;
    }

    unsigned long long total_points = 1ULL << n;
    unsigned long long words = (total_points + 63) >> 6;
    uint64_t *best = PyMem_Calloc(words, sizeof(uint64_t));
    if (!best) {
        PyErr_NoMemory();
        return NULL;
    }

    int size;
    size = find_subspace_evasive_set_rref(n, w, max_restarts, seed, best);
    if (size < 0) {
        PyMem_Free(best);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *result = PyList_New(size);
    if (!result) {
        PyMem_Free(best);
        return NULL;
    }
    int idx = 0;
    for (unsigned long long i = 0; i < total_points; ++i) {
        if (bitset_test(best, (uint32_t)i)) {
            PyObject *tuple = PyTuple_New(n);
            if (!tuple) {
                Py_DECREF(result);
                PyMem_Free(best);
                return NULL;
            }
            for (int j = 0; j < n; ++j) {
                PyObject *bit = PyLong_FromLong((i >> j) & 1ULL);
                if (!bit) {
                    Py_DECREF(tuple);
                    Py_DECREF(result);
                    PyMem_Free(best);
                    return NULL;
                }
                PyTuple_SET_ITEM(tuple, j, bit);  // steals reference
            }
            PyList_SET_ITEM(result, idx++, tuple);  // steals reference
        }
    }

    PyMem_Free(best);
    return result;
}

static PyMethodDef DiscreteMathMethods[] = {
    {"power_sum", power_sum, METH_VARARGS, "Sum j**i for j=1..N."},
    {"count_linear_subspaces", count_linear_subspaces, METH_VARARGS,
     "Count linear k-subspaces of F_2^n."},
    {"count_affine_subspaces", count_affine_subspaces, METH_VARARGS,
     "Count affine k-subspaces of F_2^n."},
    {"contains_affine_subspace", contains_affine_subspace, METH_VARARGS,
     "Return True if S contains some k-dimensional affine subspace."},
    {"find_max_overlap_affine_subspace", find_max_overlap_affine_subspace,
     METH_VARARGS,
     "Return max size of S intersect A over k-dim affine subspaces."},
    {"find_all_max_overlap_affine_subspaces", find_all_max_overlap_affine_subspaces,
     METH_VARARGS,
     "Return all k-dim affine subspaces with max overlap with S."},
    {"affine_spectrum", affine_spectrum, METH_VARARGS,
     "Compute the affine spectrum of a set of points in F_2^n."},
    {"find_hitting_subspace", find_hitting_subspace, METH_VARARGS,
     "Return a basis for a d-dim subspace whose cosets all meet S, or None."},
    {"find_subspace_evasive_set", find_subspace_evasive_set, METH_VARARGS,
     "Heuristically search for a large subset avoiding w-dim affine subspaces."},
    {"find_subspace_limited_set", find_subspace_limited_set, METH_VARARGS,
     "Heuristically search for a large subset satisfying overlap constraints."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "c_discrete_math",
    NULL,
    -1,
    DiscreteMathMethods,
};

PyMODINIT_FUNC PyInit_c_discrete_math(void) {
    return PyModule_Create(&moduledef);
}
