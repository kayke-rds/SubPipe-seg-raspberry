/* bench.c -- Fast-SCNN INT8 benchmark paralelo com composição visual.
 *
 * Uso:
 *   ./fastscnn_bench <dir_imgs> <dir_masks> <dir_out> [n_workers] [bilinear]
 *
 * bilinear = 0 (default) -> upsample nearest  (rápido, pixelado)
 * bilinear = 1           -> upsample bilinear (suave, +2 ms por imagem)
 *
 * Assume pesos em ./fastscnn_weights.bin e ./fastscnn_ppm_fp32.bin.
 *
 * Métricas reportadas:
 *   Global  : agregadas por pixel (TP/FP/TN/FN somados em todas as imagens)
 *   Per-img : média dos IoUs calculados imagem por imagem
 *   mIoU    : (IoU_pipe + IoU_bg) / 2
 *   Pipe = classe positiva (tubo); BG = classe negativa (fundo)
 *
 * Saída:
 *   {dir_out}/{stem}_composite.png     imagem 1088x640 por amostra
 *   {dir_out}/metrics_worker_N.csv     métricas por imagem, por worker
 *   stdout                             métricas agregadas
 */
#define _POSIX_C_SOURCE 200809L

#include "fastscnn_runtime.h"
#include "image_io.h"
#include "preprocess.h"
#include "postprocess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/mman.h>

#define IN_H    320
#define IN_W    544
#define IN_PIX  (IN_H * IN_W)
#define OUT_W   (2 * IN_W)         /* 1088 */
#define OUT_H   (2 * IN_H)         /* 640  */

static const float MEAN[3] = { 0.3551499f, 0.5187109f, 0.49404788f };
static const float STD [3] = { 0.11498363f, 0.07658384f, 0.14597918f };

/* ---------- helpers ---------- */

static int cmp_str(const void *a, const void *b)
{ return strcmp(*(const char *const *)a, *(const char *const *)b); }

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double cpu_sec(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_stime.tv_sec
         + ((double)ru.ru_utime.tv_usec + (double)ru.ru_stime.tv_usec) * 1e-6;
}

static double peak_ram_mb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1024.0;
}

static int has_ext(const char *n, const char *e) {
    size_t ln = strlen(n), le = strlen(e);
    if (ln < le) return 0;
    for (size_t i = 0; i < le; i++) {
        char a = n[ln - le + i];
        if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
        if (a != e[i]) return 0;
    }
    return 1;
}

static int is_image(const char *n) {
    return has_ext(n, ".jpg") || has_ext(n, ".jpeg") || has_ext(n, ".png")
        || has_ext(n, ".bmp");
}

static void stem_of(const char *n, char *o, size_t cap) {
    const char *d = strrchr(n, '.');
    size_t k = d ? (size_t)(d - n) : strlen(n);
    if (k >= cap) k = cap - 1;
    memcpy(o, n, k); o[k] = '\0';
}

/* ---------- estatísticas por worker (mmap compartilhado) ---------- */

typedef struct {
    long   tp, fp, tn, fn;
    size_t processed;
    size_t skipped;
    long   empty_gt_count;
    long   empty_gt_ok;
    double sum_iou_pipe_per_img;   /* soma dos IoU_pipe imagem-a-imagem */
    double sum_iou_bg_per_img;     /* soma dos IoU_bg   imagem-a-imagem */
    double cpu_sum_pct;
    double t_infer_total;
    double peak_rss_kb;
} Stats;

/* ---------- worker ---------- */

static void run_worker(int wid, int n_workers,
                       char **files, size_t n_files,
                       const char *dir_imgs, const char *dir_masks,
                       const char *dir_out, int use_bilinear, Stats *st)
{
    memset(st, 0, sizeof(*st));

    char csv_path[1024];
    snprintf(csv_path, sizeof(csv_path), "%s/metrics_worker_%d.csv",
             dir_out, wid);
    FILE *csv = fopen(csv_path, "w");
    if (csv)
        fprintf(csv,
                "stem,tp,fp,tn,fn,"
                "iou,iou_bg,miou,dice,precision,recall,infer_ms\n");

    static float   input_nchw[3 * IN_PIX];
    static uint8_t gt_320    [IN_PIX];
    static uint8_t pred_mask [IN_PIX];
    static uint8_t comp_rgb  [OUT_H * OUT_W * 3];

    double cpu_prev  = cpu_sec();
    double wall_prev = now_sec();

    for (size_t i = (size_t)wid; i < n_files; i += (size_t)n_workers) {
        char img_path[1024], mask_path[1024], out_path[1024], stem[256];

        stem_of(files[i], stem, sizeof(stem));
        snprintf(img_path,  sizeof(img_path),  "%s/%s",     dir_imgs,  files[i]);
        snprintf(mask_path, sizeof(mask_path), "%s/%s.png", dir_masks, stem);
        snprintf(out_path,  sizeof(out_path),  "%s/%s_composite.png",
                 dir_out, stem);

        int sw, sh;
        uint8_t *img = load_image_rgb(img_path, &sw, &sh);
        if (!img) { st->skipped++; continue; }

        int gw, gh;
        uint8_t *gt_rgb = load_image_rgb(mask_path, &gw, &gh);
        if (!gt_rgb) { free_image(img); st->skipped++; continue; }

        /* 1) Inferência */
        double t0 = now_sec();
        LetterboxInfo lb;
        letterbox_image_rgb_to_nchw(img, sh, sw, input_nchw, IN_H, IN_W,
                                    MEAN, STD, &lb);
        fastscnn_forward(input_nchw);
        int y_zp = fastscnn_output_zp();

        if (use_bilinear) {
            fastscnn_binarize_bilinear(fastscnn_logits(), 40, 68, y_zp,
                                       pred_mask, IN_H, IN_W);
        } else {
            fastscnn_binarize(fastscnn_output(), IN_PIX, y_zp, pred_mask);
        }
        double t1 = now_sec();
        st->t_infer_total += t1 - t0;

        free_image(img);

        /* 2) GT → 0/255 (tubo vermelho) → letterbox nearest */
        uint8_t *gt_gray = (uint8_t *)malloc((size_t)gw * gh);
        if (!gt_gray) { free_image(gt_rgb); continue; }
        for (int k = 0; k < gw * gh; k++) {
            uint8_t r = gt_rgb[k*3 + 0];
            uint8_t g = gt_rgb[k*3 + 1];
            uint8_t b = gt_rgb[k*3 + 2];
            gt_gray[k] = (r > 60 && r > g + 30 && r > b + 30) ? 255 : 0;
        }
        free_image(gt_rgb);
        letterbox_mask_nearest(gt_gray, gh, gw, gt_320, IN_H, IN_W, &lb);
        free(gt_gray);

        /* 3) Métricas por imagem */
        int tp=0, fp=0, tn=0, fn=0;
        for (int k = 0; k < IN_PIX; k++) {
            int p = pred_mask[k] != 0;
            int g = gt_320[k]    != 0;
            if      ( p &&  g) tp++;
            else if ( p && !g) fp++;
            else if (!p &&  g) fn++;
            else               tn++;
        }
        int gt_sum = tp + fn;
        int pr_sum = tp + fp;
        if (gt_sum == 0) {
            st->empty_gt_count++;
            if (pr_sum == 0) st->empty_gt_ok++;
        }

        /* IoU pipe: classe positiva (tubo) */
        float iou_pipe = (tp+fp+fn) == 0 ? 1.0f
                         : (float)tp / (float)(tp+fp+fn);

        /* IoU bg: negando as classes (tn do pipe vira tp do bg, etc.) */
        int bg_tp = tn, bg_fp = fn, bg_fn = fp;
        float iou_bg   = (bg_tp+bg_fp+bg_fn) == 0 ? 1.0f
                         : (float)bg_tp / (float)(bg_tp+bg_fp+bg_fn);

        float miou     = 0.5f * (iou_pipe + iou_bg);
        float dice     = (2*tp+fp+fn) == 0 ? 1.0f
                         : 2.0f * (float)tp / (float)(2*tp+fp+fn);
        float prec     = (tp+fp) == 0 ? 0.0f : (float)tp / (float)(tp+fp);
        float rec      = (tp+fn) == 0 ? 0.0f : (float)tp / (float)(tp+fn);

        st->tp += tp; st->fp += fp; st->tn += tn; st->fn += fn;
        st->sum_iou_pipe_per_img += iou_pipe;
        st->sum_iou_bg_per_img   += iou_bg;
        st->processed++;

        if (csv) {
            fprintf(csv,
                    "%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n",
                    stem, tp, fp, tn, fn,
                    iou_pipe, iou_bg, miou, dice, prec, rec,
                    (t1 - t0) * 1000.0);
        }

        /* 4) Composição visual */
        compose_output(gt_320, pred_mask, IN_W, IN_H, comp_rgb);
        save_rgb_png(out_path, comp_rgb, OUT_W, OUT_H);

        /* 5) CPU amostrado */
        double cpu_now = cpu_sec(), wall_now = now_sec();
        double wall_dt = wall_now - wall_prev;
        if (wall_dt > 0.0)
            st->cpu_sum_pct += 100.0 * (cpu_now - cpu_prev) / wall_dt;
        cpu_prev  = cpu_now;
        wall_prev = wall_now;
    }

    if (csv) fclose(csv);
    st->peak_rss_kb = peak_ram_mb() * 1024.0;
}

/* ---------- main ---------- */

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "uso: %s <dir_imgs> <dir_masks> <dir_out> [n_workers] [bilinear]\n"
                "  bilinear = 0 (nearest, default) ou 1 (bilinear)\n",
                argv[0]);
        return 1;
    }
    const char *dir_imgs  = argv[1];
    const char *dir_masks = argv[2];
    const char *dir_out   = argv[3];
    int n_workers = (argc > 4) ? atoi(argv[4])
                               : (int)sysconf(_SC_NPROCESSORS_ONLN);
    int use_bilinear = (argc > 5) ? atoi(argv[5]) : 0;

    if (n_workers < 1)  n_workers = 1;
    if (n_workers > 16) n_workers = 16;

    mkdir(dir_out, 0755);

    if (fastscnn_init("fastscnn_weights.bin", "fastscnn_ppm_fp32.bin") < 0) {
        fprintf(stderr, "falha em fastscnn_init\n");
        return 1;
    }

    /* Enumera imagens */
    DIR *d = opendir(dir_imgs);
    if (!d) { perror("opendir"); return 1; }
    char **files = NULL;
    size_t n_files = 0, cap = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!is_image(ent->d_name)) continue;
        if (n_files == cap) {
            cap = cap ? cap * 2 : 64;
            files = (char **)realloc(files, cap * sizeof(char *));
        }
        files[n_files++] = strdup(ent->d_name);
    }
    closedir(d);
    if (n_files == 0) { fprintf(stderr, "sem imagens em %s\n", dir_imgs); return 1; }
    qsort(files, n_files, sizeof(char *), cmp_str);

    if (n_workers > (int)n_files) n_workers = (int)n_files;

    printf("imagens : %zu\n", n_files);
    printf("workers : %d\n", n_workers);
    printf("upsample: %s\n", use_bilinear ? "BILINEAR" : "nearest");
    printf("saída   : %s\n\n", dir_out);

    Stats *stats = mmap(NULL, n_workers * sizeof(Stats),
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (stats == MAP_FAILED) { perror("mmap"); return 1; }

    double wall_start = now_sec();

    pid_t *pids = (pid_t *)malloc(n_workers * sizeof(pid_t));
    for (int w = 0; w < n_workers; w++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 1; }
        if (pid == 0) {
            run_worker(w, n_workers, files, n_files,
                       dir_imgs, dir_masks, dir_out, use_bilinear, &stats[w]);
            _exit(0);
        }
        pids[w] = pid;
    }

    int rc = 0;
    for (int w = 0; w < n_workers; w++) {
        int status;
        waitpid(pids[w], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) rc = 1;
    }
    double wall_total = now_sec() - wall_start;

    /* Agrega */
    long   tp = 0, fp = 0, tn = 0, fn = 0;
    size_t n_processed = 0, n_skipped = 0;
    long   empty_gt = 0, empty_gt_ok = 0;
    double sum_iou_pipe = 0.0, sum_iou_bg = 0.0;
    double t_infer = 0.0, cpu_sum = 0.0, peak_max_kb = 0.0;
    for (int w = 0; w < n_workers; w++) {
        tp += stats[w].tp; fp += stats[w].fp;
        tn += stats[w].tn; fn += stats[w].fn;
        n_processed   += stats[w].processed;
        n_skipped     += stats[w].skipped;
        empty_gt      += stats[w].empty_gt_count;
        empty_gt_ok   += stats[w].empty_gt_ok;
        sum_iou_pipe  += stats[w].sum_iou_pipe_per_img;
        sum_iou_bg    += stats[w].sum_iou_bg_per_img;
        t_infer       += stats[w].t_infer_total;
        cpu_sum       += stats[w].cpu_sum_pct;
        if (stats[w].peak_rss_kb > peak_max_kb)
            peak_max_kb = stats[w].peak_rss_kb;
    }

    for (size_t i = 0; i < n_files; i++) free(files[i]);
    free(files); free(pids);

    if (n_processed == 0) { fprintf(stderr, "nada processado\n"); return 1; }

    const double eps = 1e-7;

    /* --- Métricas GLOBAIS (agregadas por pixel) --- */
    double iou_pipe_glob = (double)tp / ((double)tp + fp + fn + eps);
    double iou_bg_glob   = (double)tn / ((double)tn + fn + fp + eps);
    double miou_glob     = 0.5 * (iou_pipe_glob + iou_bg_glob);

    /* --- Métricas PER-IMAGE (média dos IoUs imagem-a-imagem) --- */
    double iou_pipe_mean = sum_iou_pipe / (double)n_processed;
    double iou_bg_mean   = sum_iou_bg   / (double)n_processed;
    double miou_mean     = 0.5 * (iou_pipe_mean + iou_bg_mean);

    double dice = 2.0 * (double)tp / (2.0 * (double)tp + fp + fn + eps);
    double prec = (double)tp / ((double)tp + fp + eps);
    double rec  = (double)tp / ((double)tp + fn + eps);

    double fps = (double)n_processed / wall_total;
    double cpu = cpu_sum / (double)n_processed;
    double ms_per_img = 1000.0 * t_infer
                        / (double)n_processed;

    printf("\n=== Resultado (n=%zu processadas, %zu puladas, %d workers) ===\n",
           n_processed, n_skipped, n_workers);
    printf("upsample: %s\n", use_bilinear ? "BILINEAR" : "nearest");
    printf("\n[GLOBAL — agregado por pixel]\n");
    printf("  IoU_pipe = %.4f   IoU_bg = %.4f   mIoU = %.4f\n",
           iou_pipe_glob, iou_bg_glob, miou_glob);
    printf("\n[PER-IMAGE — média dos IoUs imagem-a-imagem]\n");
    printf("  IoU_pipe = %.4f   IoU_bg = %.4f   mIoU = %.4f\n",
           iou_pipe_mean, iou_bg_mean, miou_mean);
    printf("\n[PIPE — precisão/recall/dice]\n");
    printf("  Dice = %.4f   Prec = %.4f   Rec = %.4f\n", dice, prec, rec);
    printf("\n[CONFUSÃO]\n");
    printf("  TP=%ld  FP=%ld  TN=%ld  FN=%ld\n", tp, fp, tn, fn);
    printf("\n[PERFORMANCE]\n");
    printf("  FPS=%.2f  wall=%.2fs  RAM_pico_worker=%.1f MB  CPU_medio=%.1f%%\n",
           fps, wall_total, peak_max_kb / 1024.0, cpu);
    printf("  Tempo médio por imagem (por worker): %.2f ms\n", ms_per_img);
    if (empty_gt > 0)
        printf("\n[GT VAZIO]\n"
               "  %ld imagens, %ld corretamente previstas vazias (%.1f%%)\n",
               empty_gt, empty_gt_ok,
               100.0 * (double)empty_gt_ok / (double)empty_gt);

    return rc;
}
