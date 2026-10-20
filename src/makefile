# Makefile — YOLO26n-seg + Fast-SCNN INT8 unified bench
# Localização: src/makefile
#
# Uso:
#   make                          -> build/bench
#   make clean
#   make check
#   make run-yolo     IMGS=... MASKS=... OUT=... [WORKERS=8]
#   make run-fscnn    IMGS=... MASKS=... OUT=... [WORKERS=8] [BILINEAR=1]

CC       := gcc
CSTD     := -std=c11
WARN     := -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
OPT      := -O3 -funroll-loops -march=native -ftree-vectorize -fno-math-errno -fno-trapping-math -std=c11
INCLUDES := -I . -I common -I preprocess -I postprocess \
            -I fast_scnn_int8 -I generated
CFLAGS   := $(CSTD) $(WARN) $(OPT) $(INCLUDES)
LDFLAGS  := -lm

# ---------- fontes compiladas no bench ----------
YOLO_SRC   := blocks.c yolo26_forward.c head.c kernels.c yolo26n_seg_mem.c
FSCNN_SRC  := fast_scnn_int8/fastscnn_forward.c \
              fast_scnn_int8/kernels_int8.c
COMMON_SRC := common/image_io.c \
              preprocess/preprocess.c \
              postprocess/postprocess.c
BENCH_SRC  := main.c

# NÃO compilar:
#   fast_scnn_int8/bench.c    (versão antiga, substituída por main.c)
#   test/*.c                  (testes isolados)

ALL_SRC := $(BENCH_SRC) $(YOLO_SRC) $(FSCNN_SRC) $(COMMON_SRC)

OBJ_DIR := build
OBJ     := $(ALL_SRC:%.c=$(OBJ_DIR)/%.o)
BIN     := $(OBJ_DIR)/bench

# Parâmetros para os alvos run-* (sobrescreva na linha de comando)
IMGS     ?= ../dataset/images
MASKS    ?= ../dataset/masks
OUT      ?= ../out
WORKERS  ?= 8
BILINEAR ?= 1

.PHONY: all clean check run-yolo run-fscnn

all: $(BIN)

$(BIN): $(OBJ)
	@echo "  LD  $@"
	@$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo "  CC  $<"
	@$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)

check:
	@echo "Verificando fontes:"
	@for f in $(ALL_SRC); do \
	    if [ -f "$$f" ]; then echo "  ok      $$f"; \
	    else                echo "  FALTA   $$f"; fi; \
	done
	@echo "Verificando headers essenciais:"
	@for h in generated/yolo26n_seg.h fast_scnn_int8/fastscnn_runtime.h; do \
	    if [ -f "$$h" ]; then echo "  ok      $$h"; \
	    else                echo "  FALTA   $$h"; fi; \
	done

# ----- alvos de conveniência (evitam problema de cwd/weight path) -----
run-yolo: $(BIN)
	./$(BIN) yolo26 "$(IMGS)" "$(MASKS)" "$(OUT)" \
	         $(WORKERS) generated/yolo26n_seg_weights.bin

run-fscnn: $(BIN)
	@# Fast-SCNN abre "fastscnn_weights.bin" e "fastscnn_ppm_fp32.bin"
	@# por caminho relativo -> roda a partir de generated/
	@mkdir -p "$(OUT)"
	cd generated && ../$(BIN) fastscnn "../$(IMGS)" "../$(MASKS)" "../$(OUT)" \
	                         $(WORKERS) $(BILINEAR)
