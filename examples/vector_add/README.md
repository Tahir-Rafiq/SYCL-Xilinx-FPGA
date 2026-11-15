# vector_add

One kernel adds two vectors of 300 integers on the FPGA, and the host checks every result.

The host fills two input buffers through host accessors: `b` with 0, 1, 2, ... and `c` with 5, 6, 7, .... The kernel copies `b` into a local array and then computes `a[i] = array[i] + c[i]`. The host checks every element with `assert(a[i] == 5 + 2*i)`, and the program exits with status 0 when all 300 are correct.

What it adds over [answer_42](../answer_42/): data copied from the host to the card, a kernel with two read-only inputs and one write-only output, a local array that HLS places in on-chip block RAM, two loops that HLS turns into pipelines, and a check of every output value.

`single_task` means that one work-item runs the whole loop. This is the usual style on FPGAs: the loop becomes a hardware pipeline that starts a new iteration before the previous one has finished. A GPU would instead run many threads at once.

`vector_add.cpp` is an unchanged copy of `sycl/test/vitis/simple_tests/single_task_vector_add.cpp` from the compiler (triSYCL/sycl, commit 35aa03e0; Apache License v2.0 with LLVM Exceptions, see [ACKNOWLEDGEMENTS.md](../../ACKNOWLEDGEMENTS.md)).

## Build and run

From the repository folder, after `source env.sh`:

```bash
cd examples/vector_add
sycl-build vector_add.cpp hw_emu         # about 6 minutes
./vector_add_hw_emu; echo "exit $?"      # about 30 seconds
sycl-build vector_add.cpp hw             # about 1 hour
./vector_add_hw; echo "exit $?"          # on the card
```

The program prints only the device lines. Exit status 0 means that all 300 values were correct. A wrong value stops it with `Assertion ... "invalid result from kernel" failed`, `Aborted` and exit status 134. [card_run_log.md](card_run_log.md) has the full log of a run on the card with an explanation of every line.

## Results

From the HLS report (`build_hw_emu/vector_add-<hash>/vxx_comp_report/<kernel>/hls_reports/<kernel>_csynth.rpt`, where `<kernel>` is the generated name `handlerEE_clES2_E3add_...`):

| Loop | Trip count | Iteration latency | II | Loop latency |
|---|---|---|---|---|
| copy `b` into the local array | 300 | 2 cycles | 1 | 300 cycles |
| `a[i] = array[i] + c[i]` | 300 | 3 cycles | 1 | 301 cycles |

Both loops are pipelined with an initiation interval (II) of 1: a new iteration starts every clock cycle, so 300 iterations take about 300 cycles and not 300 times the iteration latency. The whole kernel takes 819 cycles, 2.73 µs at 300 MHz; the cycles beyond the two loops go to memory access setup. HLS estimated a clock period of 2.43 ns against the 3.33 ns target. The kernel uses 3 BRAM_18K blocks (one holds the 300-entry local array), no DSP, 2112 flip-flops and 4087 LUTs, under 0.4% of the LUTs of the U280.

The hardware build took 64 minutes on a 32-core host and met timing at 300 MHz (Vivado estimated 373.6 MHz). On the card the first run took 7.3 s, almost all of it loading the FPGA image, and a second run 0.20 s.
