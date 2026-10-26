# Copyright (c) 2026 Cable Television Laboratories, Inc.
# SPDX-License-Identifier: Apache-2.0
# LibreLane's own base constraints, plus a hold-only clock uncertainty of 0.1 ns, plus the capture-crossing
# rule at the end. Setup uncertainty stays at CLOCK_UNCERTAINTY_CONSTRAINT (0.25 ns).
# Used through PNR_SDC_FILE, which in LibreLane 3.1.0.dev3 also feeds the post-PnR signoff STA
# (SIGNOFF_SDC_FILE is unset). The FFT datapath clock clk_dp is `clk` propagated through the ICG
# (sky130_fd_sc_hd__dlclkp_1), not a separate SDC clock, so the clock-level setting below covers the
# clk_dp sinks as well (checked in the hold reports).
source $::env(FALLBACK_SDC)
set_clock_uncertainty -hold 0.1 [get_clocks [lindex $::env(CLOCK_PORT) 0]]
# --- extension of the 0.1 ns hold rule: the ADC capture -> gated-clock crossing --------------------------------
# The ADC capture register cap[*] (free-running clk_regs tree) feeds stage-1 flops on the gated clk_dp tree, which
# is ~0.28 ns later at ff; with the clock derate doubled (the sign-off hold check) this regs -> dp crossing is the
# worst hold path (+0.007 / -0.031 ns in two hardens without this rule). Hold-only uncertainty 0.2 ns (0.1 + 0.1)
# on exactly the clk_dp registers in cap's combinational fanout (72 at N = 16; the same set as the fanout of every
# clk_regs register), so the resizer buffers that crossing and nothing else. Stricter than the 0.1 ns rule, never
# looser. The sign-off hold check uses 0.1 ns on every register with the doubled derate (this line stands in for
# that derate and is not counted twice).
proc cap_dp_clock_pins {} {
    set front [get_pins -of_objects [get_nets {cap[*]}] -filter "direction==output"]
    set seen [dict create]
    set ck {}
    while {[llength $front]} {
        set next {}
        foreach op $front {
            foreach ld [get_pins -of_objects [get_nets -of_objects $op] -filter "direction==input"] {
                set c [get_cells -of_objects $ld]
                set cn [get_full_name $c]
                if {[dict exists $seen $cn]} { continue }
                dict set seen $cn 1
                set clkp [get_pins -quiet $cn/CLK]
                if {[llength $clkp]} {
                    if {[string match *clk_dp* [get_full_name [get_nets -of_objects $clkp]]]} { lappend ck $clkp }
                } else {
                    foreach o [get_pins -of_objects $c -filter "direction==output"] { lappend next $o }
                }
            }
        }
        set front $next
    }
    return $ck
}
set cap_dp_ck [cap_dp_clock_pins]
puts "\[INFO\] capture crossing: hold uncertainty 0.2 ns on [llength $cap_dp_ck] clk_dp clock pins in the cap fanout"
set_clock_uncertainty -hold 0.2 $cap_dp_ck
