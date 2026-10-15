# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel sources in every build's cache key: an edit to them rebuilds."""

from sgl_amd_mono.common.plan import source_digest

SOURCES = source_digest("common", "deepseek_v41/gfx950")
