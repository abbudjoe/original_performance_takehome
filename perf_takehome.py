"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def get_slot_reads_writes(self, engine: str, slot: tuple) -> tuple[set, set]:
        """
        Analyze a slot to determine which scratch addresses it reads and writes.
        Returns (reads, writes) as sets of addresses.
        """
        reads = set()
        writes = set()

        if engine == "alu":
            # (op, dest, src1, src2)
            op, dest, src1, src2 = slot
            writes.add(dest)
            reads.add(src1)
            reads.add(src2)
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                # (vbroadcast, dest, src)
                _, dest, src = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                reads.add(src)
            elif slot[0] == "multiply_add":
                # (multiply_add, dest, a, b, c)
                _, dest, a, b, c = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(a + i)
                    reads.add(b + i)
                    reads.add(c + i)
            else:
                # (op, dest, src1, src2)
                op, dest, src1, src2 = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(src1 + i)
                    reads.add(src2 + i)
        elif engine == "load":
            if slot[0] == "load":
                # (load, dest, addr_reg)
                _, dest, addr_reg = slot
                writes.add(dest)
                reads.add(addr_reg)
            elif slot[0] == "vload":
                # (vload, dest, addr_reg) - loads VLEN elements
                _, dest, addr_reg = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                reads.add(addr_reg)
            elif slot[0] == "const":
                # (const, dest, value) - no reads
                _, dest, _ = slot
                writes.add(dest)
            elif slot[0] == "load_offset":
                # (load_offset, dest, addr, offset)
                _, dest, addr, offset = slot
                writes.add(dest + offset)
                reads.add(addr + offset)
        elif engine == "store":
            if slot[0] == "store":
                # (store, addr_reg, src)
                _, addr_reg, src = slot
                reads.add(addr_reg)
                reads.add(src)
            elif slot[0] == "vstore":
                # (vstore, addr_reg, src)
                _, addr_reg, src = slot
                reads.add(addr_reg)
                for i in range(VLEN):
                    reads.add(src + i)
        elif engine == "flow":
            if slot[0] == "select":
                # (select, dest, cond, a, b)
                _, dest, cond, a, b = slot
                writes.add(dest)
                reads.add(cond)
                reads.add(a)
                reads.add(b)
            elif slot[0] == "vselect":
                # (vselect, dest, cond, a, b)
                _, dest, cond, a, b = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(cond + i)
                    reads.add(a + i)
                    reads.add(b + i)
            elif slot[0] in ("pause", "halt"):
                pass  # No reads/writes
            elif slot[0] == "cond_jump":
                _, cond, _ = slot
                reads.add(cond)
            elif slot[0] == "cond_jump_rel":
                _, cond, _ = slot
                reads.add(cond)
            elif slot[0] == "add_imm":
                _, dest, a, _ = slot
                writes.add(dest)
                reads.add(a)
        elif engine == "debug":
            pass  # Debug instructions don't affect scheduling

        return reads, writes

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        """
        Pack slots into instruction bundles using VLIW scheduling.

        Respects dependencies (RAW, WAR, WAW) and slot limits per engine.
        Uses an efficient algorithm that tracks pending writes.
        """
        if not vliw or not slots:
            # Fall back to simple one-slot-per-bundle
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        # Analyze each slot for reads/writes
        slot_info = []
        for i, (engine, slot) in enumerate(slots):
            reads, writes = self.get_slot_reads_writes(engine, slot)
            slot_info.append({
                'idx': i,
                'engine': engine,
                'slot': slot,
                'reads': reads,
                'writes': writes,
                'scheduled': False,
                # Track which slot index this depends on (RAW dependency)
                'ready_after': -1,
            })

        # Build dependency graph: for each slot, find ALL slots it depends on
        # This is O(n) per slot using maps from address -> last writer/reader
        last_writer = {}  # address -> slot index that last wrote to it
        last_reader = {}  # address -> slot index that last read from it

        for i, info in enumerate(slot_info):
            deps = set()
            # RAW: we read something that was written earlier
            for addr in info['reads']:
                if addr in last_writer:
                    deps.add(last_writer[addr])
            # WAW: we write something that was written earlier
            for addr in info['writes']:
                if addr in last_writer:
                    deps.add(last_writer[addr])
            # WAR: we write something that was read earlier
            for addr in info['writes']:
                if addr in last_reader:
                    deps.add(last_reader[addr])
            info['deps'] = deps  # All dependencies

            # Update last_writer for our writes
            for addr in info['writes']:
                last_writer[addr] = i
            # Update last_reader for our reads
            for addr in info['reads']:
                last_reader[addr] = i

        # Schedule using a greedy algorithm
        instrs = []
        num_scheduled = 0
        total = len(slots)
        current_cycle_start = 0  # Index of first slot that could be in current bundle

        while num_scheduled < total:
            bundle = defaultdict(list)
            bundle_writes = set()
            bundle_reads = set()
            slot_counts = defaultdict(int)
            scheduled_this_cycle = []
            scheduled_this_cycle_set = set()

            for i in range(current_cycle_start, total):
                info = slot_info[i]
                if info['scheduled']:
                    continue

                engine = info['engine']
                slot = info['slot']
                reads = info['reads']
                writes = info['writes']

                # Check if ALL dependencies are satisfied
                # All deps must be scheduled in a PREVIOUS cycle (not this one)
                deps_satisfied = True
                for dep in info['deps']:
                    if not slot_info[dep]['scheduled']:
                        deps_satisfied = False
                        break
                    # If dep was scheduled this cycle, we can't be in same bundle
                    if dep in scheduled_this_cycle_set:
                        deps_satisfied = False
                        break
                if not deps_satisfied:
                    continue

                # Check slot limit
                if slot_counts[engine] >= SLOT_LIMITS.get(engine, 1):
                    continue

                # Check WAW within bundle
                if writes & bundle_writes:
                    continue

                # Check WAR within bundle
                if writes & bundle_reads:
                    continue

                # Check RAW within bundle: can't read from address written in this bundle
                if reads & bundle_writes:
                    continue

                # Schedule this slot
                bundle[engine].append(slot)
                bundle_writes |= writes
                bundle_reads |= reads
                slot_counts[engine] += 1
                scheduled_this_cycle.append(i)
                scheduled_this_cycle_set.add(i)

            # Mark scheduled
            for i in scheduled_this_cycle:
                slot_info[i]['scheduled'] = True
                num_scheduled += 1

            # Update current_cycle_start to skip already scheduled slots
            while current_cycle_start < total and slot_info[current_cycle_start]['scheduled']:
                current_cycle_start += 1

            if bundle:
                instrs.append(dict(bundle))
            elif num_scheduled < total:
                # Safety: force schedule next unscheduled slot
                for i in range(total):
                    if not slot_info[i]['scheduled']:
                        info = slot_info[i]
                        instrs.append({info['engine']: [info['slot']]})
                        info['scheduled'] = True
                        num_scheduled += 1
                        break

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def broadcast_const(self, val, name=None):
        """Allocate a vector constant by broadcasting a scalar value to VLEN elements."""
        key = ("vec", val)
        if key not in self.const_map:
            # First get scalar constant
            scalar_addr = self.scratch_const(val)
            # Allocate vector space
            vec_addr = self.alloc_scratch(name or f"v_const_{val}", length=VLEN)
            # Broadcast scalar to vector
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.const_map[key] = vec_addr
        return self.const_map[key]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vectorized(self, v_val, v_tmp1, v_tmp2):
        """Vectorized hash computation - operates on VLEN elements at once."""
        slots = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # Get or create broadcasted vector constants
            v_const1 = self.broadcast_const(val1)
            v_const3 = self.broadcast_const(val3)
            # v_tmp1 = v_val op1 const1
            slots.append(("valu", (op1, v_tmp1, v_val, v_const1)))
            # v_tmp2 = v_val op3 const3
            slots.append(("valu", (op3, v_tmp2, v_val, v_const3)))
            # v_val = v_tmp1 op2 v_tmp2
            slots.append(("valu", (op2, v_val, v_tmp1, v_tmp2)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Main kernel entry point - uses vectorized implementation.
        """
        return self.build_kernel_vectorized(forest_height, n_nodes, batch_size, rounds)

    def build_kernel_vectorized(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel - processes VLEN (8) batch elements at a time.
        """
        # Scalar temporaries
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_scalar = self.alloc_scratch("tmp_scalar")

        # Vector scratch registers (each is VLEN=8 words)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)

        # Load header values from memory
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_addr, i))
            self.add("load", ("load", self.scratch[v], tmp_addr))

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector constants (broadcasted)
        v_zero = self.broadcast_const(0, "v_zero")
        v_one = self.broadcast_const(1, "v_one")
        v_two = self.broadcast_const(2, "v_two")
        v_n_nodes = self.broadcast_const(n_nodes, "v_n_nodes")

        # Pause to match reference_kernel2's first yield
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting vectorized loop"))

        # Process each batch separately to prevent VLIW from interleaving
        # (batches share scratch variables like v_idx, v_val, so they can't overlap)
        for round in range(rounds):
            for i in range(0, batch_size, VLEN):  # Step by 8
                batch_body = []
                batch_offset = self.scratch_const(i)

                # Load 8 indices: v_idx = mem[inp_indices_p + i : inp_indices_p + i + 8]
                batch_body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], batch_offset)))
                batch_body.append(("load", ("vload", v_idx, tmp_addr)))

                # Load 8 values: v_val = mem[inp_values_p + i : inp_values_p + i + 8]
                batch_body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], batch_offset)))
                batch_body.append(("load", ("vload", v_val, tmp_addr)))

                # Gather 8 tree values (this is the bottleneck - non-contiguous access)
                # v_node_val[j] = mem[forest_values_p + v_idx[j]] for j in 0..7
                for j in range(VLEN):
                    batch_body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], v_idx + j)))
                    batch_body.append(("load", ("load", v_node_val + j, tmp_addr)))

                # XOR values with node values: v_val = v_val ^ v_node_val
                batch_body.append(("valu", ("^", v_val, v_val, v_node_val)))

                # Vectorized hash computation (18 valu operations)
                batch_body.extend(self.build_hash_vectorized(v_val, v_tmp1, v_tmp2))

                # Compute next index: idx = 2*idx + (1 if val%2==0 else 2)
                # v_tmp1 = v_val & 1 (faster than %)
                batch_body.append(("valu", ("&", v_tmp1, v_val, v_one)))
                # v_tmp1 = (v_tmp1 == 0) ? 1 : 0
                batch_body.append(("valu", ("==", v_tmp1, v_tmp1, v_zero)))
                # v_tmp3 = v_tmp1 ? 1 : 2 (if even, add 1; if odd, add 2)
                batch_body.append(("flow", ("vselect", v_tmp3, v_tmp1, v_one, v_two)))
                # v_idx = 2 * v_idx
                batch_body.append(("valu", ("*", v_idx, v_idx, v_two)))
                # v_idx = v_idx + v_tmp3
                batch_body.append(("valu", ("+", v_idx, v_idx, v_tmp3)))

                # Wrap index: idx = 0 if idx >= n_nodes else idx
                batch_body.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
                batch_body.append(("flow", ("vselect", v_idx, v_tmp1, v_idx, v_zero)))

                # Store 8 indices: mem[inp_indices_p + i : ...] = v_idx
                batch_body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], batch_offset)))
                batch_body.append(("store", ("vstore", tmp_addr, v_idx)))

                # Store 8 values: mem[inp_values_p + i : ...] = v_val
                batch_body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], batch_offset)))
                batch_body.append(("store", ("vstore", tmp_addr, v_val)))

                # Pack this batch's instructions (VLIW can reorder within batch only)
                batch_instrs = self.build(batch_body)
                self.instrs.extend(batch_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel_vectorized(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
