# loadtest/topology.py
"""
CPU topology helpers: pick the logical CPUs a load-test scenario may use on this edge machine.

A physical core on SMT hardware has two sibling threads. Pinning takes whole sibling pairs
first (both threads of a core) so a "4 CPU" layout is two full cores, not four threads on
four different cores (which would flatter the result on this chip).
"""
from __future__ import annotations

import os


def available_cpus() -> list[int]:
    """Logical CPUs this process may run on."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:                      # non-Linux
        return list(range(os.cpu_count() or 1))


def _siblings(cpu: int) -> tuple[int, ...]:
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return (cpu,)
    cpus: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return tuple(sorted(cpus))


def ordered_cpus(cpus: list[int] | None = None) -> list[int]:
    """CPUs ordered so hardware-thread siblings are adjacent (whole cores first)."""
    cpus = sorted(cpus if cpus is not None else available_cpus())
    allowed = set(cpus)
    out: list[int] = []
    seen: set[int] = set()
    for cpu in cpus:
        if cpu in seen:
            continue
        group = [c for c in _siblings(cpu) if c in allowed and c not in seen]
        out.extend(group)
        seen.update(group)
    return out


def cpu_set(n_cpus: int, cpus: list[int] | None = None) -> list[int]:
    """The first ``n_cpus`` logical CPUs in sibling-pair order (sorted for readability)."""
    order = ordered_cpus(cpus)
    if n_cpus <= 0 or n_cpus > len(order):
        raise ValueError(f"requested {n_cpus} CPUs but only {len(order)} are available")
    return sorted(order[:n_cpus])


def split_cpus(cpus: list[int], n_procs: int) -> list[list[int]]:
    """Split ``cpus`` into ``n_procs`` disjoint, near-equal groups (whole sibling pairs stay together)."""
    if n_procs <= 0 or n_procs > len(cpus):
        raise ValueError(f"cannot split {len(cpus)} CPUs across {n_procs} processes")
    order = ordered_cpus(cpus)
    base, rem = divmod(len(order), n_procs)
    groups, start = [], 0
    for i in range(n_procs):
        size = base + (1 if i < rem else 0)
        groups.append(sorted(order[start:start + size]))
        start += size
    return groups
