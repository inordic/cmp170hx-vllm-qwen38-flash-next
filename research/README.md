# Research and night reports

The reports from the way to this build, oldest first. They were written for us as morning reports, and they are kept as written apart from removed network details and, in the night-4 report, links to the data and one marked note added on publication.

| Report | Content |
|---|---|
| [`strata/2026-10-09-strata-report.md`](strata/2026-10-09-strata-report.md) | where we started: Strata 0.1.41 (a GGUF engine for Flash-Next) with Swift 1.5 IQ3_XXS, plus our three engine patches (`strata/patches/`). Prefill split parts gave +18–26% at 4–8K, and same-prefix parking gave 0.19 s follow-ups |
| [`2026-10-10-alternative-models.md`](2026-10-10-alternative-models.md) | which other open models could beat Swift 1.5 on this box, and why we stayed with Flash-Next |
| [`2026-10-14-night1-models-and-config.md`](2026-10-14-night1-models-and-config.md) | night 1 on vLLM: cyankiwi vs Swift 1.5 vs Intel AutoRound, the concurrency ladder, KV offload, the INT8 KV bug, a 75-minute soak |
| [`2026-10-15-night2-quality-and-performance.md`](2026-10-15-night2-quality-and-performance.md) | night 2: backports, the PLE read-ahead tuning (the big prefill win), n-gram lookup, and the full quality suite Swift vs Intel |
| [`2026-10-17-night3-chat-template.md`](2026-10-17-night3-chat-template.md) | night 3: building and A/B-testing the chat template that became [Qwen Asgard](https://github.com/leonbollerup/qwen-asgard-jinja), plus the xhigh vs medium and history-handling experiments |
| [`2026-10-20-night4-drafters-and-quality-builds.md`](2026-10-20-night4-drafters-and-quality-builds.md) | night 4: the int4 draft head + MTP 4 that became the current standard (+19% code decode), Swift-A16 and Q6x quality builds, the INT8 KV fix (patch K1), DFlash2, probabilistic drafting, terse, and the correctness checks behind patches 08–18 |
| [`quality-eval-design.md`](quality-eval-design.md) | the design of the quality harness: benchmark choice, statistics, sandbox |

The VBIOS cross-flash is described in [`../vbios/`](../vbios/).
