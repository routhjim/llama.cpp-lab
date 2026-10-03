# eGPU docks that train their internal PCIe link at Gen1

Two Minisforum DEG2 docks (Intel JHL9480 Thunderbolt 5 controller), each holding a Radeon RX 7900 XTX, on the
USB4 ports of a Strix Halo board (Fedora 44, kernel 7.2). Every host <-> GPU transfer ran at about 0.7-0.8 GB/s,
on both docks, with pinned or pageable memory. The USB4 link itself was fine (2 lanes x 20 Gb/s).

## Cause

The link inside the dock, from the TB5 controller's downstream port to the GPU board's upstream switch, came up at
2.5 GT/s x4 (PCIe Gen1) although both ends support 16 GT/s:

```
$ sudo lspci -vv -s 02:00.0 | grep -E 'LnkCap:|LnkSta:|LnkCtl2:'     # JHL9480 downstream port
    LnkCap:  Port #0, Speed 16GT/s, Width x4
    LnkSta:  Speed 2.5GT/s, Width x4
    LnkCtl2: Target Link Speed: 2.5GT/s                               <- the limit
$ sudo lspci -vv -s 03:00.0 | grep -E 'LnkSta:|LnkCtl2:'               # Navi31 upstream switch
    LnkSta:  Speed 2.5GT/s (downgraded), Width x4 (downgraded)
    LnkCtl2: Target Link Speed: 16GT/s
```

The controller port's Target Link Speed was Gen1. There were no AER errors, and the kernel's PCIe bandwidth
controller was not throttling it (its cooling device sat at state 0). The `2.5 GT/s x1` that `lspci` and `dmesg`
show for the root ports in front of the docks is a virtual USB4 tunnel port reporting a nominal value: ignore it.

## Fix

Step the port's kernel bandwidth-controller cooling device away from 0 and back. The kernel then reprograms the
target speed to the maximum both ends support and retrains the link:

```sh
grep -l PCIe_Port_Link_Speed_0000:02:00.0 /sys/class/thermal/cooling_device*/type   # find the device
echo 1 | sudo tee /sys/class/thermal/cooling_deviceN/cur_state
echo 0 | sudo tee /sys/class/thermal/cooling_deviceN/cur_state
cat /sys/bus/pci/devices/0000:03:00.0/current_link_speed                             # 16.0 GT/s PCIe
```

[`scripts/two-xtx/pcie-link-retrain.sh`](../scripts/two-xtx/pcie-link-retrain.sh) does this for every port whose
link runs below what both ends support, waiting for Thunderbolt devices to be authorized;
[`pcie-link-retrain.service`](../scripts/two-xtx/pcie-link-retrain.service) runs it at boot. It does not cover a
dock plugged in later: run the script by hand then.

## Effect

151 MiB `ggml_backend_tensor_set` / `get` per card:

| | host -> GPU | GPU -> host |
|---|---|---|
| Gen1 (both docks) | 0.67 GB/s | 0.82 GB/s |
| Gen4, pageable | 3.26 GB/s | 3.29 GB/s |
| Gen4, pinned | 3.72-3.80 GB/s | 3.69-3.76 GB/s |

That is the 40 Gb/s USB4 tunnel's limit. On the two-GPU Qwen3.8-27B pipeline
([qwen38-two-xtx-pipeline.md](qwen38-two-xtx-pipeline.md)) the per-step logits readback and cross-card activation
copies sit on the critical path: 8 concurrent agent streams went from 185 to 259-269 t/s with no other change.
Every earlier measurement on these cards understates them.
