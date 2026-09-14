# Gated residuals

The gated-residual mix requires an explicit `weights_independent` contract.
Blackwell HC4/H2560/R320 with 1–16 rows defaults to a single CuTe kernel with a
shared Down/Up weight TMA warp, K128 stages and fixed-order cluster16
reduction when its residency, alignment and independent-weight requirements
are met. This is the only fused mix implementation; all other cases use the
unfused PyTorch GEMMs plus Triton epilogues, with no derived weight cache.

`test/ops/test_hyperconnection.py` checks the fused CuTe path against an FP64
reference in eager execution and CUDA graphs, with focused synchronization and
dispatch regressions. Run it from `tokenspeed-kernel/`:

```bash
python -m pytest test/ops/test_hyperconnection.py -q -rs
```

Check skip reasons to confirm that the fused path ran on a supported GPU.
Runtime composition tests live in
`test/runtime/test_hyperconnection_kernel_boundary.py` at the repository root.
