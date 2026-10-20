# Night 5 (2026-10-20/21): maximum quality and speed - R vs C1 (A16+INT8 KV), C2 (+probabilistic), C3 (+terse)
- 20:00:00 UTC start; production profile saved as /root/tune/night5/qfn-prod.env.baseline
- 20:00:19 UTC [host] clocks: 20:00:17 lock 1440: GPU clocks set to "(gpuClkMin 1440, gpuClkMax 1440)" for GPU 00000000:01:00.0 GPU clocks set to "(gp
- 20:04:05 UTC [R] up (ready after 226 s); GPU KV cache size: 838,024 tokens
- [R] prefill 1K 3169 8K 5690 32K 7693 | decode-at-size 1K 159 8K 151 32K 160 | code 191.1 tok/s (accept 3.55) | edit off 304.2 / low 289.4 | func PASS | multiturn PASS
- [Rmatrix] prefill 1K 3459 4K 5168 8K 6227 16K 6824 32K 7787 64K 8157 128K 8231 | decode-at-size 1K 166 4K 160 8K 152 16K 148 32K 150 64K 176 128K 153 | code 200.8 tok/s (accept 3.55) | edit off 0.0 / low 0.0 | quick (decode only)
- 20:16:19 UTC [R] conc u8-1024 | n5-R-u8-1024 | u8-1024 | 8 users | 403.6 tok/s total | 243 req | TTFT med 0.34 s p95 0.62 s | tg/stream 56.5 | errors 0 |
- 20:18:53 UTC [R] conc u16-1024 | n5-R-u16-1024 | u16-1024 | 16 users | 511.1 tok/s total | 309 req | TTFT med 0.43 s p95 0.79 s | tg/stream 34.6 | errors 0 |
- 20:21:26 UTC [R] conc mixed | n5-R-mixed | mixed | 5 users | 168.2 tok/s total | 103 req | TTFT med 0.23 s p95 5.29 s | tg/stream 36.3 | errors 0 | agent TTFT med 4.96 s | chat TTFT med 0.21 s p95 2.25 s |
- 21:13:10 UTC [n5-R] suite: bfcl n=3120 acc=0.8054 run=0.001;evalplus n=542 acc=0.8579 run=0.0018;longctx n=180 acc=0.95 run=0.0;
