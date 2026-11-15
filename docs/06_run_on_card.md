# 6. Run on the card

The hardware programs run on a U280 with the Vitis shell they were built for, the gen3x16 XDMA base shell. It reports itself as `xilinx_u280_gen3x16_xdma_base_1`, and the XRT drivers `xocl` and `xclmgmt` must be loaded. If the card runs another shell, [07_card_shell.md](07_card_shell.md) puts this one back (administrators only).

## Check the card

```bash
source env.sh
lspci -nn -d 10ee:                  # two functions, for example 04:00.0 [10ee:500c] and 04:00.1 [10ee:500d]
lsmod | grep -E 'xocl|xclmgmt'      # both XRT drivers
xbutil examine                      # the card, its shell and its PCIe address
xbutil validate --device 0000:04:00.1   # address of the second function from lspci; all tests pass
sycl-ls                             # lists the card as xilinx_u280_gen3x16_xdma_base_1
```

The PCIe IDs `10ee:500c` and `10ee:500d` belong to the Vitis gen3x16 XDMA base shell. Other IDs mean another shell; then XRT and `sycl-ls` find no card.

## Run the four examples

From the repository folder, with the programs built as in [05_hardware_build.md](05_hardware_build.md):

```bash
cd examples/answer_42  && time ./answer_42_hw; echo "exit $?"
cd ../vector_add       && time ./vector_add_hw; echo "exit $?"
cd ../matmul           && time ./matmul_hw; echo "exit $?"
cd ../gcn              && time ./gcn_hw data; echo "exit $?"
```

| Example | Pass condition | First run | Same program again |
|---|---|---|---|
| answer_42 | prints `The universal answer to the question is 42` | 7.2 s | 0.21 s |
| vector_add | exit status 0 (every value is checked with `assert`; nothing is printed after the device lines) | 7.3 s | 0.20 s |
| matmul | `C[0][0..3] = 256 183 210 264` and `PASSED: 0 errors in 256 values`, exit status 0 | 7.9 s | 0.31 s |
| gcn | `PASSED: 0 errors in 18956 values`, exit status 0; test accuracy 811 of 1000 | 8.05 s | 0.53 s |

Each program carries its own FPGA image. At the first kernel submission XRT loads it into the user part of the shell if the card holds a different image, which takes 7 to 8 s. If the card already holds this image, XRT skips the load, as in the second runs above. Running another program in between loads that program's image instead.

The `SYCL_PI_TRACE` lines at the start of each run come from `env.sh` and show the device the runtime picked. The line `warning: queue created with unhandled properties` is harmless: the XRT plugin ignores the out-of-order flag of the queue. Each example folder has a `card_run_log.md` with the full log of a run on the card and an explanation of every line.
