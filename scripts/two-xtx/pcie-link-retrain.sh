#!/usr/bin/env bash
# pcie-link-retrain.sh [--dry-run] [--wait SECONDS]
# Retrain PCIe links that trained below what both ends support (2026-10-03: both DEG2 TB5 eGPU docks enumerate the
# JHL9480 -> Navi31 link with Target Link Speed = 2.5 GT/s = 0.8 GB/s; a retrain gives 16 GT/s x4 = 3.3+ GB/s).
# Method: the kernel's PCIe bandwidth controller exposes a thermal cooling device per port; stepping it 1 -> 0 makes the
# kernel reprogram the port's target speed to the maximum and retrain the link. No setpci, no guessing register offsets.
# Only touches ports whose downstream link is slower than min(port max, device max). Waits for late (Thunderbolt) devices.
set -u
DRY=0; WAIT=120
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1 ;; --wait) WAIT=$2; shift ;; esac; shift; done

gts() { awk '{print $1+0}' "$1" 2>/dev/null; }   # "16.0 GT/s PCIe" -> 16.0
log() { echo "pcie-link-retrain: $*"; }

fix_pass() {   # prints number of degraded links found
    local n=0
    for cd in /sys/class/thermal/cooling_device*; do
        local type; type=$(cat "$cd/type" 2>/dev/null) || continue
        case "$type" in PCIe_Port_Link_Speed_*) ;; *) continue ;; esac
        local port=${type#PCIe_Port_Link_Speed_}
        local pdir=/sys/bus/pci/devices/$port
        local child; child=$(ls -d "$pdir"/0000:* 2>/dev/null | head -1)
        [ -n "$child" ] || continue
        local cur pmax cmax want
        cur=$(gts "$child/current_link_speed"); pmax=$(gts "$pdir/max_link_speed"); cmax=$(gts "$child/max_link_speed")
        [ -n "$cur" ] && [ -n "$pmax" ] && [ -n "$cmax" ] || continue
        want=$(awk -v a="$pmax" -v b="$cmax" 'BEGIN{print (a<b)?a:b}')
        awk -v c="$cur" -v w="$want" 'BEGIN{exit !(c < w)}' || continue
        n=$((n + 1))
        local name; name=$(basename "$child")
        if [ $DRY = 1 ]; then
            log "would retrain $port -> $name: $cur GT/s, capable $want GT/s ($(basename "$cd"))"
            continue
        fi
        echo 1 > "$cd/cur_state" && echo 0 > "$cd/cur_state"
        sleep 1
        log "retrained $port -> $name: $cur -> $(gts "$child/current_link_speed") GT/s x$(cat "$child/current_link_width") (capable $want)"
    done
    echo "$n" > /run/pcie-link-retrain.count 2>/dev/null || echo "$n" > /tmp/pcie-link-retrain.count
}

# eGPU docks show up after boltd authorizes them: repeat until both XTXs are present and no link is degraded, or timeout
deadline=$(( $(date +%s) + WAIT ))
while :; do
    fix_pass
    left=$(cat /run/pcie-link-retrain.count 2>/dev/null || cat /tmp/pcie-link-retrain.count)
    n_gpu=$(lspci -d 1002: 2>/dev/null | grep -c 'VGA.*Navi 31')
    [ $DRY = 1 ] && break
    [ "$n_gpu" -ge 2 ] && [ "$left" = 0 ] && break
    [ $(date +%s) -ge $deadline ] && { log "timeout: $n_gpu XTX present, $left link(s) were still degraded"; break; }
    sleep 5
done
# final state of every eGPU link for the journal
for g in $(lspci -D -d 1002: 2>/dev/null | grep 'VGA.*Navi 31' | awk '{print $1}'); do
    up=$(basename "$(readlink -f /sys/bus/pci/devices/$g/../..)")
    [ -f "/sys/bus/pci/devices/$up/current_link_speed" ] || continue
    log "final: $up $(cat /sys/bus/pci/devices/$up/current_link_speed) x$(cat /sys/bus/pci/devices/$up/current_link_width) (above $g)"
done
exit 0
