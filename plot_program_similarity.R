#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(aplot)
})

robust_csv = "/mnt/sdb/xzh/Vproject/TCA/py3ca/out/programs_robust.csv"
mp_map_csv = "/mnt/sdb/xzh/Vproject/TCA/py3ca/out/meta_programs_by_sample.csv"
out_path = "/mnt/sdb/xzh/Vproject/TCA/py3ca/out/heatmap_program_jaccard.png"
plot_width = 8
plot_height = 8

robust = read.csv(robust_csv, stringsAsFactors = FALSE)
mp_map = read.csv(mp_map_csv, stringsAsFactors = FALSE)

programs = robust %>%
  group_by(program_id, sample_id) %>%
  summarize(genes = list(unique(gene)), .groups = "drop")

mp_map = mp_map %>%
  select(program_id, meta_program) %>%
  distinct()

programs = programs %>%
  left_join(mp_map, by = "program_id") %>%
  mutate(meta_program = ifelse(is.na(meta_program), "MP_unassigned", meta_program))

meta_levels = sort(unique(programs$meta_program))
if ("MP_unassigned" %in% meta_levels) {
  meta_levels = c(setdiff(meta_levels, "MP_unassigned"), "MP_unassigned")
}

programs = programs %>%
  mutate(meta_program = factor(meta_program, levels = meta_levels)) %>%
  arrange(meta_program, program_id)

prog_ids = programs$program_id
n = length(prog_ids)
if (n < 2) {
  stop("Not enough programs to plot")
}

jaccard = matrix(0, nrow = n, ncol = n, dimnames = list(prog_ids, prog_ids))

genes_list = programs$genes
for (i in seq_len(n)) {
  gi = genes_list[[i]]
  for (j in i:n) {
    gj = genes_list[[j]]
    inter = length(intersect(gi, gj))
    uni = length(union(gi, gj))
    val = if (uni == 0) 0 else inter / uni
    jaccard[i, j] = val
    jaccard[j, i] = val
  }
}

ord = seq_len(n)

jaccard_ord = jaccard[ord, ord]

df_heat = as.data.frame(jaccard_ord) %>%
  mutate(program_id = rownames(jaccard_ord)) %>%
  pivot_longer(-program_id, names_to = "program_id_col", values_to = "jaccard")

df_heat$program_id = factor(df_heat$program_id, levels = rev(rownames(jaccard_ord)))
df_heat$program_id_col = factor(df_heat$program_id_col, levels = rownames(jaccard_ord))


anno_ord = programs %>%
  select(program_id, sample_id, meta_program)
anno_ord = anno_ord[ord, , drop = FALSE]
anno_ord$program_id = factor(anno_ord$program_id, levels = rev(rownames(jaccard_ord)))

program_assigned = anno_ord |> 
  filter(meta_program != "MP_unassigned") |> 
  pull(program_id)


df_heat = df_heat |> 
  filter(program_id %in% program_assigned & program_id_col %in% program_assigned)


p_heat = ggplot(df_heat, aes(x = program_id_col, y = program_id, fill = jaccard)) +
  geom_tile() +
  scale_fill_gradientn(colors = c("#f7fbff", "#6baed6", "#08306b"), limits = c(0, 1)) +
  theme_minimal(base_size = 10) +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  ) +
  labs(fill = "Jaccard")


p_anno = ggplot(anno_ord, aes(x = 1, y = program_id, fill = meta_program)) +
  geom_tile() +
  theme_void() +
  theme(legend.position = "right") +
  labs(fill = "meta_program")

p_sample = ggplot(anno_ord, aes(x = 1, y = program_id, fill = sample_id)) +
  geom_tile() +
  theme_void() +
  theme(legend.position = "right") +
  labs(fill = "sample_id")

p_final = insert_left(p_heat, p_anno, width = 0.06)
p_final = insert_left(p_final, p_sample, width = 0.06)

p_final
 
# ggsave(out_path, p_final, width = plot_width, height = plot_height, dpi = 200)
# View(df_heat)
# View(anno_ord)

