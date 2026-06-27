//! Alias analysis, consisting of a "last store" pass and a "memory
//! values" pass. These two passes operate as one fused pass, and so
//! are implemented together here.
//!
//! We partition memory state into several *disjoint pieces* of
//! "abstract state". There are a finite number of such pieces:
//! currently, we call them "heap", "table", "vmctx", and "other".Any
//! given address in memory belongs to exactly one disjoint piece.
//!
//! One never tracks which piece a concrete address belongs to at
//! runtime; this is a purely static concept. Instead, all
//! memory-accessing instructions (loads and stores) are labeled with
//! one of these four categories in the `MemFlags`. It is forbidden
//! for a load or store to access memory under one category and a
//! later load or store to access the same memory under a different
//! category. This is ensured to be true by construction during
//! frontend translation into CLIF and during legalization.
//!
//! Given that this non-aliasing property is ensured by the producer
//! of CLIF, we can compute a *may-alias* property: one load or store
//! may-alias another load or store if both access the same category
//! of abstract state.
//!
//! The "last store" pass helps to compute this aliasing: it scans the
//! code, finding at each program point the last instruction that
//! *might have* written to a given part of abstract state.
//!
//! We can't say for sure that the "last store" *did* actually write
//! that state, but we know for sure that no instruction *later* than
//! it (up to the current instruction) did. However, we can get a
//! must-alias property from this: if at a given load or store, we
//! look backward to the "last store", *AND* we find that it has
//! exactly the same address expression and type, then we know that
//! the current instruction's access *must* be to the same memory
//! location.
//!
//! To get this must-alias property, we compute a sparse table of
//! "memory values": these are known equivalences between SSA `Value`s
//! and particular locations in memory. The memory-values table is a
//! mapping from (last store, address expression, type) to SSA
//! value. At a store, we can insert into this table directly. At a
//! load, we can also insert, if we don't already have a value (from
//! the store that produced the load's value).
//!
//! Then we can do two optimizations at once given this table. If a
//! load accesses a location identified by a (last store, address,
//! type) key already in the table, we replace it with the SSA value
//! for that memory location. This is usually known as "redundant load
//! elimination" if the value came from an earlier load of the same
//! location, or "store-to-load forwarding" if the value came from an
//! earlier store to the same location.
//!
//! In theory we could also do *dead-store elimination*, where if a
//! store overwrites a key in the table, *and* if no other load/store
//! to the abstract state category occurred, *and* no other trapping
//! instruction occurred (at which point we need an up-to-date memory
//! state because post-trap-termination memory state can be observed),
//! *and* we can prove the original store could not have trapped, then
//! we can eliminate the original store. Because this is so complex,
//! and the conditions for doing it correctly when post-trap state
//! must be correct likely reduce the potential benefit, we don't yet
//! do this.

use crate::{
    cursor::{Cursor, FuncCursor},
    dominator_tree::DominatorTree,
    inst_predicates::{
        has_memory_fence_semantics, inst_addr_offset_type, inst_store_data, visit_block_succs,
    },
    ir::{AliasRegion, Block, Function, Inst, Opcode, Type, Value, immediates::Offset32},
    trace,
};
use alloc::vec::Vec;
use cranelift_entity::{EntityRef, packed_option::PackedOption};
use rustc_hash::{FxHashMap, FxHashSet};

/// For a given program point, the vector of last-store instruction
/// indices for each disjoint category of abstract state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LastStores {
    heap: PackedOption<Inst>,
    table: PackedOption<Inst>,
    vmctx: PackedOption<Inst>,
    other: PackedOption<Inst>,
}

impl LastStores {
    fn update(&mut self, func: &Function, inst: Inst) {
        let opcode = func.dfg.insts[inst].opcode();
        if has_memory_fence_semantics(opcode) {
            self.heap = inst.into();
            self.table = inst.into();
            self.vmctx = inst.into();
            self.other = inst.into();
        } else if opcode.can_store() {
            if let Some(memflags) = func.dfg.insts[inst].memflags() {
                match memflags.alias_region() {
                    None => self.other = inst.into(),
                    Some(AliasRegion::Heap) => self.heap = inst.into(),
                    Some(AliasRegion::Table) => self.table = inst.into(),
                    Some(AliasRegion::Vmctx) => self.vmctx = inst.into(),
                }
            } else {
                self.heap = inst.into();
                self.table = inst.into();
                self.vmctx = inst.into();
                self.other = inst.into();
            }
        }
    }

    fn get_last_store(&self, func: &Function, inst: Inst) -> PackedOption<Inst> {
        if let Some(memflags) = func.dfg.insts[inst].memflags() {
            match memflags.alias_region() {
                None => self.other,
                Some(AliasRegion::Heap) => self.heap,
                Some(AliasRegion::Table) => self.table,
                Some(AliasRegion::Vmctx) => self.vmctx,
            }
        } else if func.dfg.insts[inst].opcode().can_load()
            || func.dfg.insts[inst].opcode().can_store()
        {
            inst.into()
        } else {
            PackedOption::default()
        }
    }

    fn meet_from(&mut self, other: &LastStores, loc: Inst) {
        let meet = |a: PackedOption<Inst>, b: PackedOption<Inst>| -> PackedOption<Inst> {
            match (a.into(), b.into()) {
                (None, None) => None.into(),
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (Some(a), Some(b)) if a == b => a,
                _ => loc.into(),
            }
        };

        self.heap = meet(self.heap, other.heap);
        self.table = meet(self.table, other.table);
        self.vmctx = meet(self.vmctx, other.vmctx);
        self.other = meet(self.other, other.other);
    }
}

// LastStores is exposed publicly so the egraph mid-end can pass it through
// the ISLE context to `AliasAnalysis::find_dead_store_at`.

/// Per-`MemoryLoc` value entry. Tracks both the defining instruction and
/// whether a subsequent load has must-aliased to it (consuming its value).
/// The `observed` bit was added in Phase 1C to support redundant-store DCE.
#[derive(Clone, Copy, Debug)]
struct MemoryValueEntry {
    /// The instruction that defined this memory-value (a store, or a load
    /// whose result we recorded as the equivalent value at this location).
    def_inst: Inst,
    /// The SSA value present at this memory location.
    value: Value,
    /// `true` once any load has must-aliased to this entry and forwarded
    /// its value. Stays `false` if only stores have written here without
    /// any consumer reading the location. Used by DSE to verify that the
    /// prior store's value has never been observed before we delete it.
    observed: bool,
}

/// A key identifying a unique memory location.
///
/// For the result of a load to be equivalent to the result of another
/// load, or the store data from a store, we need for (i) the
/// "version" of memory (here ensured by having the same last store
/// instruction to touch the disjoint category of abstract state we're
/// accessing); (ii) the address must be the same (here ensured by
/// having the same SSA value, which doesn't change after computed);
/// (iii) the offset must be the same; and (iv) the accessed type and
/// extension mode (e.g., 8-to-32, signed) must be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MemoryLoc {
    last_store: PackedOption<Inst>,
    address: Value,
    offset: Offset32,
    ty: Type,
    /// We keep the *opcode* of the instruction that produced the
    /// value we record at this key if the opcode is anything other
    /// than an ordinary load or store. This is needed when we
    /// consider loads that extend the value: e.g., an 8-to-32
    /// sign-extending load will produce a 32-bit value from an 8-bit
    /// value in memory, so we can only reuse that (as part of RLE)
    /// for another load with the same extending opcode.
    ///
    /// We could improve the transform to insert explicit extend ops
    /// in place of extending loads when we know the memory value, but
    /// we haven't yet done this.
    extending_opcode: Option<Opcode>,
}

/// An alias-analysis pass.
pub struct AliasAnalysis<'a> {
    /// The domtree for the function.
    domtree: &'a DominatorTree,

    /// Input state to a basic block.
    block_input: FxHashMap<Block, LastStores>,

    /// Known memory-value equivalences. This is the result of the
    /// analysis. This is a mapping from (last store, address
    /// expression, offset, type) to SSA `Value`.
    ///
    /// We keep the defining inst around for quick dominance checks, and
    /// an `observed` bit (Phase 1C) that flips to `true` the first time
    /// a must-aliased load consumes the entry's value. DSE uses this to
    /// know whether the prior store's value has been read.
    mem_values: FxHashMap<MemoryLoc, MemoryValueEntry>,
}

impl<'a> AliasAnalysis<'a> {
    /// Perform an alias analysis pass.
    pub fn new(func: &Function, domtree: &'a DominatorTree) -> AliasAnalysis<'a> {
        trace!("alias analysis: input is:\n{:?}", func);
        let mut analysis = AliasAnalysis {
            domtree,
            block_input: FxHashMap::default(),
            mem_values: FxHashMap::default(),
        };

        analysis.compute_block_input_states(func);
        analysis
    }

    fn compute_block_input_states(&mut self, func: &Function) {
        let mut queue = vec![];
        let mut queue_set = FxHashSet::default();
        let entry = func.layout.entry_block().unwrap();
        queue.push(entry);
        queue_set.insert(entry);

        while let Some(block) = queue.pop() {
            queue_set.remove(&block);
            let mut state = *self
                .block_input
                .entry(block)
                .or_insert_with(|| LastStores::default());

            trace!(
                "alias analysis: input to block{} is {:?}",
                block.index(),
                state
            );

            for inst in func.layout.block_insts(block) {
                state.update(func, inst);
                trace!("after inst{}: state is {:?}", inst.index(), state);
            }

            visit_block_succs(func, block, |_inst, succ, _from_table| {
                let succ_first_inst = func.layout.block_insts(succ).next().unwrap();
                let updated = match self.block_input.get_mut(&succ) {
                    Some(succ_state) => {
                        let old = *succ_state;
                        succ_state.meet_from(&state, succ_first_inst);
                        *succ_state != old
                    }
                    None => {
                        self.block_input.insert(succ, state);
                        true
                    }
                };

                if updated && queue_set.insert(succ) {
                    queue.push(succ);
                }
            });
        }
    }

    /// Get the starting state for a block.
    pub fn block_starting_state(&self, block: Block) -> LastStores {
        self.block_input
            .get(&block)
            .cloned()
            .unwrap_or_else(|| LastStores::default())
    }

    /// Process one instruction. Meant to be invoked in program order
    /// within a block, and ideally in RPO or at least some domtree
    /// preorder for maximal reuse.
    ///
    /// Returns `true` if instruction was removed.
    pub fn process_inst(
        &mut self,
        func: &mut Function,
        state: &mut LastStores,
        inst: Inst,
    ) -> Option<Value> {
        trace!(
            "alias analysis: scanning at inst{} with state {:?} ({:?})",
            inst.index(),
            state,
            func.dfg.insts[inst],
        );

        let replacing_value = if let Some((address, offset, ty)) = inst_addr_offset_type(func, inst)
        {
            let address = func.dfg.resolve_aliases(address);
            let opcode = func.dfg.insts[inst].opcode();

            if opcode.can_store() {
                let store_data = inst_store_data(func, inst).unwrap();
                let store_data = func.dfg.resolve_aliases(store_data);
                let mem_loc = MemoryLoc {
                    last_store: inst.into(),
                    address,
                    offset,
                    ty,
                    extending_opcode: get_ext_opcode(opcode),
                };
                trace!(
                    "alias analysis: at inst{}: store with data v{} at loc {:?}",
                    inst.index(),
                    store_data.index(),
                    mem_loc
                );
                self.mem_values.insert(
                    mem_loc,
                    MemoryValueEntry {
                        def_inst: inst,
                        value: store_data,
                        observed: false,
                    },
                );

                None
            } else if opcode.can_load() {
                let last_store = state.get_last_store(func, inst);
                let load_result = func.dfg.inst_results(inst)[0];
                let mem_loc = MemoryLoc {
                    last_store,
                    address,
                    offset,
                    ty,
                    extending_opcode: get_ext_opcode(opcode),
                };
                trace!(
                    "alias analysis: at inst{}: load with last_store inst{} at loc {:?}",
                    inst.index(),
                    last_store.map(|inst| inst.index()).unwrap_or(usize::MAX),
                    mem_loc
                );

                // Is there a Value already known to be stored
                // at this specific memory location?  If so,
                // we can alias the load result to this
                // already-known Value.
                //
                // Check if the definition dominates this
                // location; it might not, if it comes from a
                // load (stores will always dominate though if
                // their `last_store` survives through
                // meet-points to this use-site).
                let aliased = if let Some(entry) = self.mem_values.get(&mem_loc).cloned() {
                    let MemoryValueEntry {
                        def_inst, value, ..
                    } = entry;
                    trace!(
                        " -> sees known value v{} from inst{}",
                        value.index(),
                        def_inst.index()
                    );
                    if self.domtree.dominates(def_inst, inst, &func.layout) {
                        trace!(
                            " -> dominates; value equiv from v{} to v{} inserted",
                            load_result.index(),
                            value.index()
                        );
                        // Phase 1C: mark the entry as observed by a load.
                        // This forwarding consumes the entry's value, so any
                        // subsequent DSE attempt against this `def_inst` must
                        // be rejected.
                        if let Some(e) = self.mem_values.get_mut(&mem_loc) {
                            e.observed = true;
                        }
                        Some(value)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Otherwise, we can keep *this* load around
                // as a new equivalent value.
                if aliased.is_none() {
                    trace!(
                        " -> inserting load result v{} at loc {:?}",
                        load_result.index(),
                        mem_loc
                    );
                    self.mem_values.insert(
                        mem_loc,
                        MemoryValueEntry {
                            def_inst: inst,
                            value: load_result,
                            // A miss-aliased load inserts itself as the
                            // equivalent value at this location, but no
                            // store's value has yet been observed via
                            // this entry — leave `observed = false`.
                            observed: false,
                        },
                    );
                }

                aliased
            } else {
                None
            }
        } else {
            None
        };

        state.update(func, inst);

        replacing_value
    }

    /// Phase 1C — redundant-store DCE query (luna v2.1 Path D).
    ///
    /// Given the egraph driver's current store instruction `current_inst`
    /// and the alias-analysis `state` at this program point (BEFORE the
    /// driver has called `process_inst` for `current_inst`), return the
    /// `Inst` handle of a PRIOR store at the same memory location that
    /// is now provably dead — meaning the egraph driver may safely
    /// remove it from the layout. Returns `None` if no such prior store
    /// exists or any safety precondition is unmet.
    ///
    /// Preconditions enforced here (in addition to the ISLE rule's
    /// notrap check on the CURRENT store):
    ///
    ///   1. Current store has plain-`Store` opcode (no extending variant).
    ///   2. `state.get_last_store` for the current store's alias region
    ///      points at some `prior_inst`.
    ///   3. `prior_inst` is itself a plain-`Store` opcode with notrap
    ///      MemFlags.
    ///   4. `prior_inst` and `current_inst` share the same `(address,
    ///      offset, ty)` MemoryLoc key — i.e., there is a
    ///      `mem_values` entry keyed under `last_store = prior_inst`
    ///      with `def_inst = prior_inst`.
    ///   5. The `mem_values` entry's `observed` bit is `false`: no
    ///      intervening load has must-aliased to the prior store's
    ///      value.
    ///   6. Block-scope discipline (Phase 1C / Phase 1G):
    ///        - Phase 1C (same-block): `prior_inst` and `current_inst`
    ///          live in the same basic block. Precondition (7) below
    ///          walks the layout forward to verify no `can_trap` insts
    ///          intervene.
    ///        - Phase 1G (cross-block, strict-chain — added 2026-06-27):
    ///          `prior_block` `block_dominates` `current_block`, and the
    ///          immediate-dominator chain from `current_block` up to
    ///          `prior_block` is BOTH (a) can_trap-clean across all
    ///          insts on the chain, AND (b) every chain block's
    ///          terminator's successors are themselves on the chain
    ///          (no off-chain branch could observe the prior store's
    ///          value before the current store overwrites it).
    ///          See `cross_block_dominator_chain_check` below.
    ///   7. No instruction between `prior_inst` (exclusive) and
    ///      `current_inst` (exclusive) in the layout `can_trap()`. The
    ///      upstream TODO at `alias_analysis.rs:53-62` explicitly calls
    ///      out post-trap-termination memory state observability as
    ///      the safety boundary; we exclude any trapping inst between
    ///      the two stores. Phase 1G subsumes precondition (7) into
    ///      the cross-block chain walk.
    ///   8. `prior_inst` dominates `current_inst` (cheap belt-and-
    ///      suspenders given precondition 6).
    pub fn find_dead_store_at(
        &self,
        func: &Function,
        state: &LastStores,
        current_inst: Inst,
    ) -> Option<Inst> {
        // (1) current must be a plain store (not an extending variant)
        let current_opcode = func.dfg.insts[current_inst].opcode();
        if !current_opcode.can_store() {
            return None;
        }
        if get_ext_opcode(current_opcode).is_some() {
            return None;
        }
        let (address, offset, ty) = inst_addr_offset_type(func, current_inst)?;
        let address = func.dfg.resolve_aliases(address);

        // (2) prior_inst = last store for current's region
        let prior_inst = state.get_last_store(func, current_inst).expand()?;
        if prior_inst == current_inst {
            return None;
        }

        // (3) prior must be a plain notrap store
        let prior_opcode = func.dfg.insts[prior_inst].opcode();
        if !prior_opcode.can_store() {
            return None;
        }
        if get_ext_opcode(prior_opcode).is_some() {
            return None;
        }
        let prior_flags = func.dfg.insts[prior_inst].memflags()?;
        if !prior_flags.notrap() {
            return None;
        }

        // (4) mem_values entry must exist under (last_store=prior_inst,
        //     address, offset, ty) and have def_inst == prior_inst.
        let key = MemoryLoc {
            last_store: prior_inst.into(),
            address,
            offset,
            ty,
            extending_opcode: None,
        };
        let entry = self.mem_values.get(&key)?;
        if entry.def_inst != prior_inst {
            return None;
        }

        // (5) no intervening load may have read the prior store's value
        if entry.observed {
            return None;
        }

        // (6 + 7) block-scope check fused with can_trap walk.
        //
        // Same-block (Phase 1C path) keeps the original layout-order
        // forward walk. Cross-block (Phase 1G strict-chain path) walks
        // the dominator chain from `current_block` up to `prior_block`,
        // verifying can_trap-clean insts and that every chain block's
        // terminator branches only into the chain.
        let prior_block = func.layout.inst_block(prior_inst)?;
        let current_block = func.layout.inst_block(current_inst)?;
        if prior_block == current_block {
            // Phase 1C — same-block forward walk.
            let mut walk = func.layout.next_inst(prior_inst);
            while let Some(inst) = walk {
                if inst == current_inst {
                    break;
                }
                if func.dfg.insts[inst].opcode().can_trap() {
                    return None;
                }
                walk = func.layout.next_inst(inst);
            }
            // If we didn't reach current_inst by walking forward from
            // prior, prior is NOT before current in this block — reject.
            if walk.is_none() {
                return None;
            }
        } else {
            // Phase 1G strict-chain (1G.B.2) — cross-block.
            //
            // Cheap O(1) block-dominance test first; the idom-chain walk
            // also catches non-domination but bails late.
            if !self.domtree.block_dominates(prior_block, current_block) {
                return None;
            }
            self.cross_block_dominator_chain_check(
                func,
                prior_inst,
                prior_block,
                current_inst,
                current_block,
            )?;
        }

        // (8) dominance (cheap after (6))
        if !self.domtree.dominates(prior_inst, current_inst, &func.layout) {
            return None;
        }

        Some(prior_inst)
    }

    /// Phase 1G.B.2 (strict-chain) — cross-block DSE safety check.
    ///
    /// Walks the immediate-dominator chain from `current_block` up to
    /// `prior_block`. Returns `Some(())` if BOTH:
    ///
    ///   (a) Every instruction on the chain (in `prior_block` after
    ///       `prior_inst`, in any intermediate block, and in
    ///       `current_block` up to but not including `current_inst`) is
    ///       `!can_trap()`. A trapping inst between the two stores
    ///       would create a program point at which the prior store's
    ///       memory value is externally observable, defeating DSE.
    ///
    ///   (b) For every chain block whose terminator we cross (i.e.,
    ///       every chain block except `current_block`), all of that
    ///       terminator's CFG successors are themselves on the chain.
    ///       An off-chain branch successor would be a side-exit block
    ///       reachable BEFORE `current_inst` overwrites the slot —
    ///       that side-exit could observe the prior store's memory
    ///       value, which DSE must forbid.
    ///
    /// Returns `None` (rejecting DSE) on any of:
    ///   - prior_block is unreachable from current_block via idom chain
    ///     (this is also implied by precondition `block_dominates`,
    ///     but the walk handles it defensively).
    ///   - any chain inst is can_trap.
    ///   - any chain block branches off-chain (the Phase 1G.B.3
    ///     deopt-safe relaxation lifts this case under restricted
    ///     conditions; see `successors_chain_or_deopt_safe`).
    ///
    /// Safety burden: this method is the entire correctness gate for
    /// cross-block DSE. The cranelift verifier does NOT enforce alias
    /// analysis correctness (see Phase 1G.A audit §3); a bug here would
    /// silently drop a memory write that an external observer can see.
    fn cross_block_dominator_chain_check(
        &self,
        func: &Function,
        prior_inst: Inst,
        prior_block: Block,
        current_inst: Inst,
        current_block: Block,
    ) -> Option<()> {
        // Walk the idom chain from `current_block` upward, collecting
        // every block we pass through until we land on `prior_block`.
        //
        // chain[0] = current_block, chain[N-1] = prior_block.
        let mut chain: Vec<Block> = Vec::new();
        chain.push(current_block);
        let mut cur = current_block;
        while cur != prior_block {
            cur = self.domtree.idom(cur)?;
            chain.push(cur);
        }
        let chain_set: FxHashSet<Block> = chain.iter().copied().collect();

        // Iterate prior → ... → current (CFG execution order).
        for &block in chain.iter().rev() {
            let is_prior_block = block == prior_block;
            let is_current_block = block == current_block;

            // Start past `prior_inst` in the first chain block, at
            // the block head otherwise.
            let start = if is_prior_block {
                // prior_inst is a non-terminator store, so next_inst is
                // always Some (worst case the block's terminator).
                func.layout.next_inst(prior_inst)?
            } else {
                func.layout.first_inst(block)?
            };

            let mut walk = Some(start);
            while let Some(inst) = walk {
                // Stop at current_inst in the final chain block.
                if is_current_block && inst == current_inst {
                    break;
                }
                let opcode = func.dfg.insts[inst].opcode();
                if opcode.can_trap() {
                    return None;
                }
                // Terminator handling. We only encounter terminators on
                // chain blocks BEFORE current_block — current_block's
                // terminator is past current_inst and we break above.
                if opcode.is_branch() {
                    debug_assert!(
                        !is_current_block,
                        "current_block terminator must be past current_inst"
                    );
                    if !self.successors_all_on_chain(func, block, &chain_set) {
                        return None;
                    }
                }
                walk = func.layout.next_inst(inst);
            }

            // If we hit the end of current_block without seeing
            // current_inst, layout order doesn't agree with dominance
            // — bail.
            if is_current_block && walk.is_none() {
                return None;
            }
        }

        Some(())
    }

    /// Phase 1G.B.2 helper — every CFG successor of `block`'s
    /// terminator must be in `chain_set`. Returns `true` if so.
    fn successors_all_on_chain(
        &self,
        func: &Function,
        block: Block,
        chain_set: &FxHashSet<Block>,
    ) -> bool {
        let mut all_on_chain = true;
        visit_block_succs(func, block, |_branch, succ, _from_table| {
            if !chain_set.contains(&succ) {
                all_on_chain = false;
            }
        });
        all_on_chain
    }

    /// Drop the `mem_values` entry that `store_inst` inserted for itself.
    /// Called by the egraph driver right before a DSE-RemoveOther erases
    /// `store_inst` from the layout, so later `dominates(store_inst,
    /// load_inst, layout)` queries do not panic on a removed inst.
    ///
    /// Only the single entry where `def_inst == store_inst` is removed;
    /// other entries keyed with `last_store = store_inst` (inserted by
    /// later miss-aliased loads) keep their own def_insts and stay valid
    /// since those def_insts are loads still present in the layout.
    pub fn invalidate_mem_value_for_store(&mut self, func: &Function, store_inst: Inst) {
        if let Some((address, offset, ty)) = inst_addr_offset_type(func, store_inst) {
            let address = func.dfg.resolve_aliases(address);
            let key = MemoryLoc {
                last_store: store_inst.into(),
                address,
                offset,
                ty,
                extending_opcode: None,
            };
            self.mem_values.remove(&key);
        }
    }

    /// Make a pass and update known-redundant loads to aliased
    /// values. We interleave the updates with the memory-location
    /// tracking because resolving some aliases may expose others
    /// (e.g. in cases of double-indirection with two separate chains
    /// of loads).
    pub fn compute_and_update_aliases(&mut self, func: &mut Function) {
        let mut pos = FuncCursor::new(func);

        while let Some(block) = pos.next_block() {
            let mut state = self.block_starting_state(block);
            while let Some(inst) = pos.next_inst() {
                if let Some(replaced_result) = self.process_inst(pos.func, &mut state, inst) {
                    let result = pos.func.dfg.inst_results(inst)[0];
                    pos.func.dfg.clear_results(inst);
                    pos.func.dfg.change_to_alias(result, replaced_result);
                    pos.remove_inst_and_step_back();
                }
            }
        }
    }
}

fn get_ext_opcode(op: Opcode) -> Option<Opcode> {
    debug_assert!(op.can_load() || op.can_store());
    match op {
        Opcode::Load | Opcode::Store => None,
        _ => Some(op),
    }
}
