/* Exact 3-D KD-tree nearest-colour lookup.
 *
 * Build and query state lives entirely in the heap structures owned by one
 * kd_tree instance; there are no mutable globals, so building or querying
 * different trees from different threads is safe.  A built tree is never
 * modified, and kd_query only writes to the caller supplied output buffer and
 * to its own stack, so concurrent kd_query calls on the same tree are safe.
 *
 * Tie-breaking matches numpy.argmin over the squared Euclidean distance:
 * among equidistant palette entries the smallest palette index wins.
 */
#include <stdlib.h>
#include <string.h>

typedef struct {
    int left, right;  /* child node ids, -1 for leaves */
    int start, end;   /* range in pts/ids for leaves */
    double lo[3];
    double hi[3];
} kd_node;

typedef struct {
    int n;            /* palette size */
    int nnodes;
    int nseed;
    double *pts;      /* sorted points, 3 doubles per entry */
    int *ids;         /* palette index of each sorted point */
    kd_node *nodes;
    double *seed;     /* seed points used for the initial upper bound */
} kd_tree;

/* ------------------------------------------------------------------ */
/* context-carrying mergesort (no global comparator state)             */
/* ------------------------------------------------------------------ */
static int kd_less(int a, int b, int dim, const double *pts) {
    double x = pts[3 * a + dim], y = pts[3 * b + dim];
    if (x < y) return 1;
    if (x > y) return 0;
    return a < b;
}

static void kd_mergesort(int *ids, int *tmp, int begin, int end, int dim,
                         const double *pts) {
    int count = end - begin;
    if (count <= 24) {
        for (int i = begin + 1; i < end; i++) {
            int v = ids[i], j = i - 1;
            while (j >= begin && kd_less(v, ids[j], dim, pts)) {
                ids[j + 1] = ids[j];
                j--;
            }
            ids[j + 1] = v;
        }
        return;
    }
    int mid = begin + count / 2;
    kd_mergesort(ids, tmp, begin, mid, dim, pts);
    kd_mergesort(ids, tmp, mid, end, dim, pts);
    int i = begin, j = mid, k = begin;
    while (i < mid && j < end) {
        if (kd_less(ids[j], ids[i], dim, pts)) tmp[k++] = ids[j++];
        else tmp[k++] = ids[i++];
    }
    while (i < mid) tmp[k++] = ids[i++];
    while (j < end) tmp[k++] = ids[j++];
    memcpy(ids + begin, tmp + begin, sizeof(int) * (size_t)count);
}

/* ------------------------------------------------------------------ */
/* build                                                               */
/* ------------------------------------------------------------------ */
static int kd_build(kd_tree *t, int *ids, int *tmp, int begin, int end,
                    int leaf) {
    int node = t->nnodes++;
    int count = end - begin;
    double lo0 = 1e300, lo1 = 1e300, lo2 = 1e300;
    double hi0 = -1e300, hi1 = -1e300, hi2 = -1e300;
    for (int k = begin; k < end; k++) {
        const double *p = &t->pts[3 * ids[k]];
        if (p[0] < lo0) lo0 = p[0];
        if (p[0] > hi0) hi0 = p[0];
        if (p[1] < lo1) lo1 = p[1];
        if (p[1] > hi1) hi1 = p[1];
        if (p[2] < lo2) lo2 = p[2];
        if (p[2] > hi2) hi2 = p[2];
    }
    kd_node *nd = &t->nodes[node];
    nd->left = nd->right = -1;
    nd->lo[0] = lo0; nd->lo[1] = lo1; nd->lo[2] = lo2;
    nd->hi[0] = hi0; nd->hi[1] = hi1; nd->hi[2] = hi2;
    if (count <= leaf || (lo0 == hi0 && lo1 == hi1 && lo2 == hi2)) {
        /* Sorted ascending by palette index makes ties deterministic. */
        for (int a = begin + 1; a < end; a++) {
            int v = ids[a], b = a - 1;
            while (b >= begin && ids[b] > v) { ids[b + 1] = ids[b]; b--; }
            ids[b + 1] = v;
        }
        nd->start = begin;
        nd->end = end;
        return node;
    }
    double dx = hi0 - lo0, dy = hi1 - lo1, dz = hi2 - lo2;
    int dim = 0;
    double width = dx;
    if (dy > width) { width = dy; dim = 1; }
    if (dz > width) { dim = 2; }
    int mid = begin + count / 2;
    kd_mergesort(ids, tmp, begin, end, dim, t->pts);
    int left = kd_build(t, ids, tmp, begin, mid, leaf);
    int right = kd_build(t, ids, tmp, mid, end, leaf);
    t->nodes[node].left = left;
    t->nodes[node].right = right;
    return node;
}

/* Farthest-point sampling: a handful of well spread seed points give a tight
 * initial nearest-distance upper bound and cut the number of visited nodes. */
static void kd_make_seeds(kd_tree *t, int nseed) {
    if (nseed <= 0 || t->n <= 0) return;
    int ns = nseed < t->n ? nseed : t->n;
    double *seed = (double *)malloc(sizeof(double) * 3 * (size_t)ns);
    double *mind = (double *)malloc(sizeof(double) * (size_t)t->n);
    if (!seed || !mind) { free(seed); free(mind); return; }
    for (int i = 0; i < t->n; i++) mind[i] = 1e300;
    int best = 0;
    for (int s = 0; s < ns; s++) {
        memcpy(&seed[3 * s], &t->pts[3 * best], 3 * sizeof(double));
        double far = -1.0;
        int next = 0;
        for (int i = 0; i < t->n; i++) {
            double d0 = t->pts[3 * i] - t->pts[3 * best];
            double d1 = t->pts[3 * i + 1] - t->pts[3 * best + 1];
            double d2 = t->pts[3 * i + 2] - t->pts[3 * best + 2];
            double d = d0 * d0 + d1 * d1 + d2 * d2;
            if (d < mind[i]) mind[i] = d;
            if (mind[i] > far) { far = mind[i]; next = i; }
        }
        best = next;
    }
    free(mind);
    t->seed = seed;
    t->nseed = ns;
}

kd_tree *kd_create(const double *palette, int n, int leaf, int nseed) {
    if (!palette || n <= 0) return NULL;
    if (leaf < 1) leaf = 1;
    kd_tree *t = (kd_tree *)calloc(1, sizeof(kd_tree));
    int *ids = (int *)malloc(sizeof(int) * (size_t)n);
    int *tmp = (int *)malloc(sizeof(int) * (size_t)n);
    t->pts = (double *)malloc(sizeof(double) * 3 * (size_t)n);
    t->ids = (int *)malloc(sizeof(int) * (size_t)n);
    t->nodes = (kd_node *)malloc(sizeof(kd_node) * (2 * (size_t)n + 1));
    if (!t->pts || !t->ids || !t->nodes || !ids || !tmp) {
        free(ids); free(tmp);
        free(t->pts); free(t->ids); free(t->nodes); free(t);
        return NULL;
    }
    memcpy(t->pts, palette, sizeof(double) * 3 * (size_t)n);
    for (int i = 0; i < n; i++) ids[i] = i;
    t->n = n;
    t->nnodes = 0;
    kd_build(t, ids, tmp, 0, n, leaf);
    /* Materialise points in tree order. */
    for (int k = 0; k < n; k++) {
        t->ids[k] = ids[k];
        memcpy(&t->pts[3 * k], &palette[3 * ids[k]], 3 * sizeof(double));
    }
    free(ids);
    free(tmp);
    kd_make_seeds(t, nseed);
    return t;
}

void kd_destroy(kd_tree *t) {
    if (!t) return;
    free(t->pts);
    free(t->ids);
    free(t->nodes);
    free(t->seed);
    free(t);
}

/* ------------------------------------------------------------------ */
/* query                                                               */
/* ------------------------------------------------------------------ */
static double kd_box_dist2(const kd_node *nd, const double *p) {
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, v;
    v = nd->lo[0] - p[0]; if (v > 0) d0 = v;
    v = p[0] - nd->hi[0]; if (v > d0) d0 = v;
    v = nd->lo[1] - p[1]; if (v > 0) d1 = v;
    v = p[1] - nd->hi[1]; if (v > d1) d1 = v;
    v = nd->lo[2] - p[2]; if (v > 0) d2 = v;
    v = p[2] - nd->hi[2]; if (v > d2) d2 = v;
    return d0 * d0 + d1 * d1 + d2 * d2;
}

/* Returns 0 on success.  out[i] receives the nearest palette index. */
int kd_query(const kd_tree *t, const double *queries, int count, int *out) {
    if (!t || !queries || !out || count < 0) return -1;
    const kd_node *nodes = t->nodes;
    const double *pts = t->pts;
    int n = t->n;
    for (int qi = 0; qi < count; qi++) {
        const double *p = queries + 3 * (size_t)qi;
        double best_d = 1e300;
        int best_i = n;
        for (int s = 0; s < t->nseed; s++) {
            double a0 = t->seed[3 * s] - p[0];
            double a1 = t->seed[3 * s + 1] - p[1];
            double a2 = t->seed[3 * s + 2] - p[2];
            double d = a0 * a0 + a1 * a1 + a2 * a2;
            if (d < best_d) best_d = d;
        }
        int stack[256];
        int sp = 0;
        stack[sp++] = 0;
        while (sp) {
            int node = stack[--sp];
            const kd_node *nd = &nodes[node];
            if (kd_box_dist2(nd, p) > best_d) continue;
            if (nd->left < 0) {
                for (int k = nd->start; k < nd->end; k++) {
                    const double *s = &pts[3 * k];
                    double a0 = s[0] - p[0];
                    double a1 = s[1] - p[1];
                    double a2 = s[2] - p[2];
                    double d = a0 * a0 + a1 * a1 + a2 * a2;
                    int id = t->ids[k];
                    if (d < best_d || (d == best_d && id < best_i)) {
                        best_d = d;
                        best_i = id;
                    }
                }
            } else {
                const kd_node *nl = &nodes[nd->left];
                const kd_node *nr = &nodes[nd->right];
                double dl = kd_box_dist2(nl, p);
                double dr = kd_box_dist2(nr, p);
                if (dl <= dr) {
                    if (dr <= best_d) stack[sp++] = nd->right;
                    stack[sp++] = nd->left;
                } else {
                    if (dl <= best_d) stack[sp++] = nd->left;
                    stack[sp++] = nd->right;
                }
            }
        }
        out[qi] = best_i;
    }
    return 0;
}
