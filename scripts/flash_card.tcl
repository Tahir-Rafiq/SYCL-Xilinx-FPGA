# Write a shell image (.mcs) into the configuration flash of an Alveo U280 over JTAG.
# Follows the JTAG flash procedure of AMD/Xilinx answer record 71757.
# See docs/07_card_shell.md: the card must first be removed from the PCIe bus,
# and the host needs a cold boot afterwards.
#
# Usage: vivado -mode batch -source flash_card.tcl -tclargs <image.mcs>
if {[llength $argv] != 1} {
  puts "usage: vivado -mode batch -source flash_card.tcl -tclargs <image.mcs>"
  exit 1
}
set mcs [lindex $argv 0]

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set dev [lindex [get_hw_devices xcu280_u55c_0] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev

# Flash part of the U280
create_hw_cfgmem -hw_device $dev mt25qu01g-spi-x1_x2_x4
set cfg [get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.ADDRESS_RANGE {use_file} $cfg
set_property PROGRAM.FILES [list $mcs] $cfg
set_property PROGRAM.PRM_FILE {} $cfg
set_property PROGRAM.ERASE 1 $cfg
set_property PROGRAM.CFG_PROGRAM 1 $cfg
set_property PROGRAM.VERIFY 1 $cfg

# Load the SPI helper bitstream into the FPGA, then erase, program and verify the flash
create_hw_bitstream -hw_device $dev [get_property PROGRAM.HW_CFGMEM_BITFILE $dev]
program_hw_devices $dev
refresh_hw_device $dev
program_hw_cfgmem -hw_cfgmem $cfg
refresh_hw_device $dev
close_hw_target
