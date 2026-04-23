# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Hugo static site studying LLM inference at the GPU kernel level, progressing through five architecture variants: Vanilla Transformer → Dense LLM (Llama) → MoE (Mixtral) → MLA (DeepSeek-V2) → Parallel MoE (DeepSeek-V3). Published to GitHub Pages at https://yongseungyu-etri.github.io/LLM-Inference-Study/.

Content is written in Korean with English technical terms. Each subsection follows a 4-step methodology: concept/motivation → math → operation decomposition → GPU kernel mapping (FlashInfer/cuBLASLt/CUTLASS/Triton on A100).

## Build & Dev Commands

```bash
# Local dev server (includes draft pages)
hugo server --buildDrafts
# → http://localhost:1313/LLM-Inference-Study/

# Production build
hugo --gc --minify
```

Requires Hugo Extended (v0.160.1 in CI). The theme `hugo-book` is a git submodule — clone with `--recurse-submodules` or run `git submodule update --init --recursive`.

## Deployment

Auto-deployed to GitHub Pages on push to `main` via `.github/workflows/deploy.yml`. No manual deploy step needed.

## Architecture

- **Hugo config**: `hugo.toml` — uses hugo-book theme, KaTeX math enabled via goldmark passthrough delimiters
- **Theme**: `themes/hugo-book` (git submodule, alex-shpak/hugo-book)
- **Content**: all study material lives under `content/docs/` organized as `S{n}_{TopicName}/{n}-{m}_{SubtopicName}/index.md`
- **Math rendering**: KaTeX loaded via custom partials in `layouts/partials/docs/inject/` (head.html for CSS, body.html for JS + auto-render)
- **Static assets**: `static/private/` holds encrypted onboarding doc; `content/private.md` is a hidden page (`bookHidden: true`)

## Content Conventions

- Section index files (`_index.md`) use Hugo `weight` frontmatter for ordering
- Subsection pages are `index.md` inside named directories (Hugo page bundles), enabling co-located images
- Math uses `$...$` (inline), `$$...$$` or `\[...\]` (display blocks) — Hugo's goldmark passthrough is configured to preserve these for KaTeX
- Hugo shortcodes like `{{< relref "..." >}}` are used for cross-references
- Goldmark `unsafe: true` is enabled, so raw HTML in markdown is rendered
