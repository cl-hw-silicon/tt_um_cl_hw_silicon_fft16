# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
# cocotb unit benches (the top-level template Makefile stays for the TT flow / GL test).
#   make -f units.mk UNIT=core|timing|twiddle [SIM=icarus|verilator] [N=16|32] [CLK_PS=62500]
# The RTL instantiates the PDK clock-gate cell, so the PDK functional cell models are compiled in RTL
# simulation too: -DFUNCTIONAL, unit delay, no power pins.
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
UNIT ?= core
SRC_DIR = $(PWD)/../src
CELLS = $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog
SIM_BUILD = sim_build/$(UNIT)_$(SIM)
CORE_SRC = sdf_stage.v twiddle_w32.v twiddle_w16.v twiddle_w8.v fft_core.v iq_tail.v clk_gate.v

N ?= 16
export N
ifeq ($(UNIT),core)
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(CORE_SRC)) $(PWD)/tb_core.v
TOPLEVEL = tb_core
COCOTB_TEST_MODULES = test_core
NEED_CELLS = 1
SIM_BUILD = sim_build/$(UNIT)$(N)_$(SIM)
ifeq ($(SIM),icarus)
COMPILE_ARGS += -Ptb_core.N=$(N)
else
EXTRA_ARGS += -GN=$(N)
endif
endif
ifeq ($(UNIT),timing)
VERILOG_SOURCES += $(SRC_DIR)/cp_timing.v $(PWD)/tb_timing.v
TOPLEVEL = tb_timing
COCOTB_TEST_MODULES = test_timing
SIM_BUILD = sim_build/$(UNIT)$(N)_$(SIM)
ifeq ($(SIM),icarus)
COMPILE_ARGS += -Ptb_timing.N=$(N)
else
EXTRA_ARGS += -GN=$(N)
endif
endif
ifeq ($(UNIT),twiddle)
VERILOG_SOURCES += $(SRC_DIR)/twiddle_w32.v $(SRC_DIR)/twiddle_w16.v $(SRC_DIR)/twiddle_w8.v $(PWD)/tb_twiddle.v
TOPLEVEL = tb_twiddle
COCOTB_TEST_MODULES = test_twiddle
endif

ifdef NEED_CELLS
COMPILE_ARGS    += -DFUNCTIONAL -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(CELLS)/primitives.v $(CELLS)/sky130_fd_sc_hd.v
endif
ifeq ($(SIM),verilator)
EXTRA_ARGS += --timing -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN
endif
COMPILE_ARGS += -I$(SRC_DIR)
# model/ on the python path for the bit-exact oracles
export PYTHONPATH := $(PWD)/../model:$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim
