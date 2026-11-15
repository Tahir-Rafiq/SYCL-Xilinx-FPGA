# 7. Put the Vitis shell on the card (administrators)

This page is for the administrator of a U280 host. It needs root access, physical or remote power control of the host, and some care: a mistake here can take the host off the network until it is power-cycled.

The card loads its shell from its configuration flash at power-up. A shell from another project (for example the OpenNIC shell, PCIe IDs `10ee:903f` and `10ee:913f`) overwrites the Vitis shell there, and XRT then finds no card. The steps below write the Vitis gen3x16 XDMA base shell back into the flash through the card's USB JTAG port, following the JTAG flash procedure of AMD/Xilinx answer record 71757. The procedure does not use PCIe, so it works whatever shell the card runs. It needs one cold boot of the host.

You need Vivado 2023.2, the USB cable from the card to the host (`lsusb -d 0403:6011` shows the FT4232H JTAG chip), sudo for `setpci` and for the sysfs `remove` files, and power control of the host. The image is `partition.mcs` from the platform package `xilinx-u280-gen3x16-xdma-base`.

## 1. Find the card and the PCIe bridge above it

```bash
lspci -nn -d 10ee:                                   # the two functions of the card, for example 04:00.0 and 04:00.1
C=0000:04:00                                          # bus and device of the card, from lspci
B=$(basename $(dirname $(readlink -f /sys/bus/pci/devices/$C.0)))   # the bridge, for example 0000:03:02.0
echo $B
lsmod | grep -E 'onic|qdma|xocl|xclmgmt'              # expect nothing: no driver for the card
pgrep -a hw_server                                    # at most one, with no Vivado hardware manager open on it
md5sum /lib/firmware/xilinx/283bab8f654d8674968f4da57f7fa5d7/partition.mcs   # 0ef78238cddbfce65d13bc18d0261723
```

The procedure was tested with no driver loaded for the card; it was not tried with one loaded.

## 2. Mask PCIe errors on the bridge and remove the card from Linux

Step 3 loads a helper bitstream into the FPGA. That replaces the shell, so the card drops off the PCIe link, and without these commands the host can treat that as a fatal error and reset. The settings last until the next boot.

```bash
sudo setpci -s $B COMMAND                              # read and note the value
sudo setpci -s $B CAP_EXP+8.w                          # read and note the value
sudo setpci -s $B COMMAND=0000:0100                    # clear bit 8, SERR# enable
sudo setpci -s $B CAP_EXP+8.w=0000:0004                # clear bit 2, fatal error reporting
echo 1 | sudo tee /sys/bus/pci/devices/$B/$C.1/remove
echo 1 | sudo tee /sys/bus/pci/devices/$B/$C.0/remove
lspci -d 10ee:                                         # expect no output
```

Do not rescan the PCIe bus from here on. Only the cold boot in step 4 brings the card back.

## 3. Write the flash

Run it inside `screen`, so that it keeps running if the SSH connection drops (`screen -r flash` reattaches):

```bash
screen -S flash
mkdir -p ~/flash_u280 && cd ~/flash_u280
vivado -mode batch -source <repository>/scripts/flash_card.tcl \
  -tclargs /lib/firmware/xilinx/283bab8f654d8674968f4da57f7fa5d7/partition.mcs 2>&1 | tee flash.log
grep -nE 'ERROR|CRITICAL' flash.log                    # expect nothing
```

The script connects to hw_server on localhost:3121 and selects the FPGA `xcu280_u55c_0`. It loads the SPI helper bitstream into the FPGA, then erases, programs and verifies the flash sectors covered by the file: addresses 0x01000000 to 0x03E9FFFF, about 49 MB. The factory (golden) image at address 0 is not touched. Program and verify is the longest step; let it finish. If a step fails, do not reboot: JTAG works whatever the flash holds, so the script can be run again.

## 4. Cold boot

Power the host off and on (`sudo shutdown -h now`, then the power button or the remote power control). A warm reboot (`sudo reboot`) keeps the card powered, so the FPGA keeps the helper bitstream and does not read the flash.

## 5. Check the card after the boot

```bash
lspci -nn -d 10ee:                                     # 10ee:500c and 10ee:500d
lsmod | grep -E 'xocl|xclmgmt'                         # both loaded
sudo /opt/xilinx/xrt/bin/xbmgmt examine                # shell xilinx_u280_gen3x16_xdma_base_1
xbutil validate --device 0000:04:00.1                  # second function from lspci; all tests pass
```

`sudo` may need the full path of `xbmgmt`, as above, because sudo uses its own `PATH`. The card is then ready for [06_run_on_card.md](06_run_on_card.md); the programs need no rebuild.
