# SPDX-License-Identifier: Apache-2.0
"""K2i, the index sources' FP4 scorer in one launch between K1 and K2 (``k2i``,
``score_fp4``), its per-step tasks (``index_plan``) and FULL walk plan on the
device (``walk_plan``), a candidate consumer's key table and selection map
(``candidate_table``), and the host side that feeds its logits to SGLang's top-k
kernels (``runner``).

Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 68506823 (MIT License,
Copyright (C) 2026, Advanced Micro Devices, Inc.) by way of sgl-project/sglang#43984;
each file names its origin.
"""
