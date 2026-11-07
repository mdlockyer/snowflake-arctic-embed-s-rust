//! Intra-op multi-core forward pass for a **single** sequence (latency mode), SME backend.
//!
//! Apple M4 Pro/Max have one SME matrix unit per P-core cluster, so a single thread can only use one
//! of them, and the NEON elementwise work (GELU, LayerNorm, softmax) runs on one core. Here `n` threads
//! run the same layer program on disjoint slices of each operator:
//!
//! | Operator        | Split                                                         |
//! |-----------------|---------------------------------------------------------------|
//! | Q/K/V GEMM      | by head pairs (64-column super-blocks of Q, K and V)          |
//! | Attention       | by heads (each thread consumes only its own Q/K/V columns)    |
//! | Other GEMMs     | by 64-column super-blocks                                     |
//! | GELU + quantize | by the FFN-up columns each thread produced, row max exchanged |
//! | LayerNorm etc.  | by rows                                                       |
//!
//! with a spin barrier between dependent stages (8 per layer; ~80-250 ns each on M4 Max).
//! Each stage's units are split into one contiguous share per thread; a thread drains its own share
//! first (static split, so consecutive stages reuse data on the same core) and then steals what is
//! left of the others' (so a slower or preempted thread does not stall the barrier).
//!
//! Measured on M4 Max (µs per embedding, 1 / 2 / 3 / 4 threads): 8 tokens 400 / 304 / 438 / 451,
//! 32 tokens 1164 / 725 / 886 / 880, 128 tokens 5027 / 3066 / 3431 / 3276, 512 tokens
//! 33341 / 19058 / 18270 / 18728. Two threads (one per SME unit) is the sweet spot; more threads
//! share an SME unit and add synchronization. [`ParallelContext::recommended`] picks 2 on SME.
//! Every thread packs the GEMM A operand into its own scratch, which is cheaper than an extra barrier.
//! The final layer's single-row (`[CLS]`) tail runs on the calling thread.
//!
//! Helper threads spin for a short time after each job (so back-to-back calls start instantly)
//! and then park, so an idle [`ParallelContext`] costs no CPU.

use crate::gemm::SimdBackend;
use crate::model::{PackedModel, Workspace};
use crate::quantize::{HIDDEN_SIZE, MAX_POSITIONS};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long an idle helper keeps spinning for the next job before parking.
const IDLE_SPIN: Duration = Duration::from_micros(300);

/// Sense-free generation spin barrier. Falls back to yielding, then short sleeps, if a participant
/// is descheduled.
struct SpinBarrier {
    n: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    fn new(n: usize) -> Self {
        Self {
            n,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn wait(&self) {
        if self.n == 1 {
            return;
        }
        let g = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            self.generation.store(g.wrapping_add(1), Ordering::Release);
        } else {
            // Spin (normal case: all participants are running), then yield, then sleep: when the
            // machine is oversubscribed a descheduled participant must get a core, and spinning or
            // yielding user-interactive threads can starve it almost indefinitely.
            let mut spins = 0u32;
            while self.generation.load(Ordering::Acquire) == g {
                spins = spins.saturating_add(1);
                if spins < 20_000 {
                    std::hint::spin_loop();
                } else if spins < 20_200 {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(Duration::from_micros(20));
                }
            }
        }
    }
}

/// Raw pointers to the buffers all threads share for one forward pass (owned by thread 0's workspace).
#[derive(Clone, Copy)]
struct Shared {
    hidden: *mut f32,
    tmp_hidden: *mut f32,
    qkv: *mut f32,
    ctx: *mut f32,
    inter: *mut f32,
    inter_q: *mut i8,
    act_q: *mut i8,
    act_scale: *mut f32,
    q_cls: *mut f32,
    /// `[INTERMEDIATE_SIZE / 64, MAX_POSITIONS]` per-super-block partial GELU row maxima.
    row_max: *mut f32,
}

struct Job {
    model: *const PackedModel,
    ids: *const u16,
    seq_len: usize,
    shared: Shared,
}

struct PoolState {
    n: usize,
    epoch: AtomicUsize,
    job: AtomicPtr<Job>,
    done: AtomicUsize,
    shutdown: AtomicBool,
    sleeping: Vec<AtomicBool>,
    barrier: SpinBarrier,
    /// Per (stage, owner thread) claim counters into that owner's contiguous share of the stage's
    /// units (static-first scheduling with work stealing), zeroed per job.
    stage_ctr: Vec<PaddedCounter>,
}

/// Upper bound on scheduling stages per forward pass (8 per layer x 11 layers + embeddings + layer 11).
const MAX_STAGES: usize = 128;

struct HelperWs(UnsafeCell<Workspace>);
// Each helper workspace is only ever touched by its own helper thread.
unsafe impl Sync for HelperWs {}

/// Reusable context for low-latency single-sequence inference on several cores
/// (see the module docs). Create once and reuse; `n_threads = 2` targets the two P-cluster SME
/// units of M4 Pro/Max. With a non-SME backend it falls back to the single-threaded forward pass.
pub struct ParallelContext {
    ws: Workspace,
    row_max: Vec<f32>,
    state: Arc<PoolState>,
    helpers: Vec<JoinHandle<()>>,
}

impl ParallelContext {
    pub fn new(n_threads: usize) -> Self {
        let n = n_threads.clamp(1, 8);
        let state = Arc::new(PoolState {
            n,
            epoch: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            done: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            sleeping: (0..n).map(|_| AtomicBool::new(false)).collect(),
            barrier: SpinBarrier::new(n),
            stage_ctr: (0..MAX_STAGES * n)
                .map(|_| PaddedCounter(AtomicUsize::new(0)))
                .collect(),
        });
        let helpers = (1..n)
            .map(|tid| {
                let st = Arc::clone(&state);
                std::thread::Builder::new()
                    .name(format!("arctic-embed-par-{tid}"))
                    .spawn(move || helper_main(tid, st, 0))
                    .expect("spawn intra-op helper thread")
            })
            .collect();
        Self {
            ws: Workspace::new(),
            row_max: vec![0.0; (crate::quantize::INTERMEDIATE_SIZE / 64) * MAX_POSITIONS],
            state,
            helpers,
        }
    }

    /// Recommended context for `engine`: 2 threads (one per P-cluster SME unit) when the SME
    /// backend is active, otherwise 1 (plain single-threaded forward pass, no helper threads).
    pub fn recommended(engine: &crate::ArcticEmbedS) -> Self {
        Self::new(if engine.model.backend == SimdBackend::ArmSme {
            2
        } else {
            1
        })
    }

    pub fn num_threads(&self) -> usize {
        self.state.n
    }

    /// Embeds pre-tokenized `input_ids` (truncated to 512) into `out_emb` using all threads.
    pub fn forward_cls(
        &mut self,
        model: &PackedModel,
        input_ids: &[u16],
        out_emb: &mut [f32; HIDDEN_SIZE],
    ) {
        let seq_len = input_ids.len().min(MAX_POSITIONS);
        assert!(seq_len >= 1, "input_ids must not be empty");
        if self.state.n == 1
            || model.backend != SimdBackend::ArmSme
            || !cfg!(target_arch = "aarch64")
        {
            model.forward_cls(input_ids, &mut self.ws, out_emb);
            return;
        }
        let ws = &mut self.ws;
        let shared = Shared {
            hidden: ws.hidden.as_mut_ptr(),
            tmp_hidden: ws.tmp_hidden.as_mut_ptr(),
            qkv: ws.qkv_buf.as_mut_ptr(),
            ctx: ws.attn_ctx.as_mut_ptr(),
            inter: ws.inter_buf.as_mut_ptr(),
            inter_q: ws.inter_q.as_mut_ptr(),
            act_q: ws.act_q.as_mut_ptr(),
            act_scale: ws.act_scale.as_mut_ptr(),
            q_cls: ws.q_cls.as_mut_ptr(),
            row_max: self.row_max.as_mut_ptr(),
        };
        let mut job = Job {
            model,
            ids: input_ids.as_ptr(),
            seq_len,
            shared,
        };
        let st = &*self.state;
        st.done.store(0, Ordering::Relaxed);
        for c in &st.stage_ctr {
            c.0.store(0, Ordering::Relaxed);
        }
        st.job.store(&mut job, Ordering::Release);
        st.epoch.fetch_add(1, Ordering::SeqCst);
        for (i, h) in self.helpers.iter().enumerate() {
            if st.sleeping[i + 1].load(Ordering::SeqCst) {
                h.thread().unpark();
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            worker_forward(0, st, &job, &mut self.ws as *mut Workspace);
            tail_last_layer(model, &mut self.ws, seq_len, out_emb);
        }
        // Helpers leave the job right after the last barrier; wait so `job` and the buffers stay valid.
        let mut spins = 0u32;
        while st.done.load(Ordering::Acquire) != st.n - 1 {
            spins = spins.saturating_add(1);
            if spins < 20_000 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        st.job.store(std::ptr::null_mut(), Ordering::Relaxed);
    }
}

impl Drop for ParallelContext {
    fn drop(&mut self) {
        self.state.shutdown.store(true, Ordering::SeqCst);
        self.state.epoch.fetch_add(1, Ordering::SeqCst);
        for h in &self.helpers {
            h.thread().unpark();
        }
        for h in self.helpers.drain(..) {
            let _ = h.join();
        }
    }
}

/// `start_epoch` is the epoch at creation: reading `st.epoch` here instead would lose the first
/// job if the OS starts this thread only after the caller has already published it.
fn helper_main(tid: usize, st: Arc<PoolState>, start_epoch: usize) {
    set_thread_qos_user_interactive();
    let ws = HelperWs(UnsafeCell::new(Workspace::new()));
    let mut seen = start_epoch;
    loop {
        let mut idle_since = Instant::now();
        let mut spins = 0u32;
        loop {
            let e = st.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            spins += 1;
            if spins % 1024 != 0 {
                std::hint::spin_loop();
                continue;
            }
            if idle_since.elapsed() < IDLE_SPIN {
                continue;
            }
            st.sleeping[tid].store(true, Ordering::SeqCst);
            if st.epoch.load(Ordering::SeqCst) == seen {
                std::thread::park();
            }
            st.sleeping[tid].store(false, Ordering::SeqCst);
            idle_since = Instant::now();
        }
        if st.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let job = st.job.load(Ordering::Acquire);
        if job.is_null() {
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            worker_forward(tid, &st, &*job, ws.0.get());
        }
        st.done.fetch_add(1, Ordering::Release);
    }
}

/// Puts the current thread in the user-interactive QoS class so macOS schedules it on P-cores.
fn set_thread_qos_user_interactive() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

/// Cache-line-padded atomic so owners' counters don't false-share.
#[repr(align(128))]
struct PaddedCounter(AtomicUsize);

/// Static-first scheduling with work stealing: every thread walks the same sequence of stages. Each
/// stage's units are split into one contiguous share per thread; a thread first drains its own share
/// (so with equally fast threads the split is static and consecutive stages keep their data on the
/// same core/cluster), then steals the remaining units of the other shares. A thread that is slower
/// (an E-core, or sharing its cluster's SME unit) therefore just ends up with fewer units instead of
/// stalling the barrier.
struct Sched<'a> {
    st: &'a PoolState,
    tid: usize,
    stage: usize,
}

impl Sched<'_> {
    /// Runs `f(unit)` for the units of the next stage that this thread claims.
    #[inline]
    fn run(&mut self, units: usize, mut f: impl FnMut(usize)) {
        let n = self.st.n;
        let ctrs = &self.st.stage_ctr[self.stage * n..(self.stage + 1) * n];
        self.stage += 1;
        debug_assert!(self.stage <= MAX_STAGES);
        for k in 0..n {
            let owner = (self.tid + k) % n;
            let (lo, hi) = (owner * units / n, (owner + 1) * units / n);
            loop {
                let u = lo + ctrs[owner].0.fetch_add(1, Ordering::Relaxed);
                if u >= hi {
                    break;
                }
                f(u);
            }
        }
    }
}

/// Rows per row-wise work unit: small enough to balance, large enough to amortize the atomic.
#[inline]
fn row_chunk(m: usize, n: usize) -> usize {
    m.div_ceil(4 * n).max(4)
}

/// Layers 0..10 plus the parallel part of layer 11 (Q for `[CLS]`, K/V, `[CLS]` attention) for thread `tid`.
#[cfg(target_arch = "aarch64")]
unsafe fn worker_forward(tid: usize, st: &PoolState, job: &Job, ws: *mut Workspace) {
    use crate::gemm::{sme_gemm_packed_superblocks, sme_pack_activations};
    use crate::quantize::{HEAD_DIM, INTERMEDIATE_SIZE, LAYER_NORM_EPS, NUM_LAYERS};
    use crate::simd_math::{
        attention_heads_sme, compute_scores_head32, dequant_add_embeddings_row, extract_kv_head32,
        gelu_segment_pass1, gelu_segment_quantize, layernorm_smooth_quantize_row,
        smooth_quantize_row, softmax_inplace, weighted_sum_head32,
    };
    use std::slice::{from_raw_parts, from_raw_parts_mut};

    let n = st.n;
    let model = &*job.model;
    let m = job.seq_len;
    let sh = job.shared;
    let ws = &mut *ws;
    let bar = &st.barrier;
    let mut sched = Sched { st, tid, stage: 0 };
    const H: usize = HIDDEN_SIZE;
    const I: usize = INTERMEDIATE_SIZE;
    const I_SB: usize = INTERMEDIATE_SIZE / 64;
    let rc = row_chunk(m, n);
    let row_units = m.div_ceil(rc);
    let rows_of = |u: usize| (u * rc, ((u + 1) * rc).min(m));
    let inv_sqrt_d = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut hid = sh.hidden;
    let mut tmp = sh.tmp_hidden;
    // Shared quantized activations: re-borrowed at each use, only after the barrier that published them.
    let aq = || from_raw_parts(sh.act_q as *const i8, m * H);
    let asc = || from_raw_parts(sh.act_scale as *const f32, m);

    // Embeddings + LN + quantize for layer 0's QKV (rows).
    let ids = from_raw_parts(job.ids, m);
    sched.run(row_units, |u| {
        let (r0, r1) = rows_of(u);
        for t in r0..r1 {
            let tok = ids[t] as usize;
            let h_row = from_raw_parts_mut(hid.add(t * H), H);
            dequant_add_embeddings_row(
                &model.word_emb_q[tok * H..(tok + 1) * H],
                model.word_emb_scale[tok],
                &model.pos_emb_q[t * H..(t + 1) * H],
                model.pos_emb_scale[t],
                &model.type0_emb_f32[..H],
                h_row,
            );
            *sh.act_scale.add(t) = layernorm_smooth_quantize_row(
                h_row,
                &model.emb_ln_weight,
                &model.emb_ln_bias,
                &model.layers[0].qkv.inv_smooth,
                from_raw_parts_mut(sh.act_q.add(t * H), H),
                LAYER_NORM_EPS,
            );
        }
    });
    bar.wait();

    let mut full_max = [0.0f32; MAX_POSITIONS];
    let mut inter_scale = [0.0f32; MAX_POSITIONS];

    for l in 0..(NUM_LAYERS - 1) {
        let layer = &model.layers[l];

        // Unit = head pair: its Q, K, V super-blocks, then attention for those two heads, on one thread.
        sme_pack_activations(m, H, aq(), &mut ws.act_packed);
        sched.run(NUM_HEADS_PAIRS, |hp| {
            for base in [0, NUM_HEADS_PAIRS, 2 * NUM_HEADS_PAIRS] {
                sme_gemm_packed_superblocks(
                    m,
                    &ws.act_packed,
                    asc(),
                    &layer.qkv,
                    std::ptr::null(),
                    sh.qkv,
                    base + hp,
                    1,
                );
            }
            let qkv = from_raw_parts(sh.qkv as *const f32, m * 3 * H);
            if m >= 16 {
                attention_heads_sme(
                    m,
                    qkv,
                    &mut ws.sme_q_t,
                    &mut ws.sme_k_t,
                    &mut ws.v_head,
                    &mut ws.sme_scores_blk,
                    &mut ws.sme_p_t,
                    sh.ctx,
                    None,
                    2 * hp..2 * hp + 2,
                );
            } else {
                for h in 2 * hp..2 * hp + 2 {
                    let h_off = h * HEAD_DIM;
                    extract_kv_head32(
                        qkv,
                        m,
                        3 * H,
                        H + h_off,
                        2 * H + h_off,
                        &mut ws.k_head,
                        &mut ws.v_head,
                    );
                    for tq in 0..m {
                        let scores = &mut ws.scores[..m];
                        let q = &qkv[tq * 3 * H + h_off..tq * 3 * H + h_off + HEAD_DIM];
                        compute_scores_head32(q, &ws.k_head[..m * HEAD_DIM], inv_sqrt_d, scores);
                        softmax_inplace(scores);
                        let out = from_raw_parts_mut(sh.ctx.add(tq * H + h_off), HEAD_DIM);
                        weighted_sum_head32(scores, &ws.v_head[..m * HEAD_DIM], out);
                    }
                }
            }
        });
        bar.wait();

        // Quantize attention context (rows).
        sched.run(row_units, |u| {
            let (r0, r1) = rows_of(u);
            for t in r0..r1 {
                *sh.act_scale.add(t) = smooth_quantize_row(
                    from_raw_parts(sh.ctx.add(t * H), H),
                    &layer.attn_out.inv_smooth,
                    from_raw_parts_mut(sh.act_q.add(t * H), H),
                );
            }
        });
        bar.wait();

        // Attention output projection + residual (super-blocks).
        sme_pack_activations(m, H, aq(), &mut ws.act_packed);
        sched.run(H / 64, |sb| {
            sme_gemm_packed_superblocks(m, &ws.act_packed, asc(), &layer.attn_out, hid, tmp, sb, 1);
        });
        bar.wait();
        std::mem::swap(&mut hid, &mut tmp);

        // LN1 + quantize (rows).
        sched.run(row_units, |u| {
            let (r0, r1) = rows_of(u);
            for t in r0..r1 {
                *sh.act_scale.add(t) = layernorm_smooth_quantize_row(
                    from_raw_parts_mut(hid.add(t * H), H),
                    &layer.ln1_weight,
                    &layer.ln1_bias,
                    &layer.ffn_inter.inv_smooth,
                    from_raw_parts_mut(sh.act_q.add(t * H), H),
                    LAYER_NORM_EPS,
                );
            }
        });
        bar.wait();

        // FFN up + GELU pass 1 (unit = 64-column super-block; partial row maxima per super-block).
        sme_pack_activations(m, H, aq(), &mut ws.act_packed);
        sched.run(I_SB / FFN_SB_PER_UNIT, |u| {
            let sb = u * FFN_SB_PER_UNIT;
            sme_gemm_packed_superblocks(
                m,
                &ws.act_packed,
                asc(),
                &layer.ffn_inter,
                std::ptr::null(),
                sh.inter,
                sb,
                FFN_SB_PER_UNIT,
            );
            for sb in sb..sb + FFN_SB_PER_UNIT {
                let rm = from_raw_parts_mut(sh.row_max.add(sb * MAX_POSITIONS), m);
                gelu_segment_pass1(m, sh.inter, I, sb * 64, 64, &layer.ffn_out.inv_smooth, rm);
            }
        });
        bar.wait();

        // Full-row maxima (every thread computes its own copy), then GELU quantize (super-blocks).
        for r in 0..m {
            let mut mx = 0.0f32;
            for sb in 0..I_SB {
                mx = mx.max(*sh.row_max.add(sb * MAX_POSITIONS + r));
            }
            full_max[r] = mx;
            inter_scale[r] = if mx.is_finite() {
                mx.max(1e-8) / 127.0
            } else {
                f32::MAX / 127.0
            };
        }
        sched.run(I_SB / FFN_SB_PER_UNIT, |u| {
            // Per 64-column segment: pass 1 packed each segment's fp16 values at that segment's start.
            for sb in u * FFN_SB_PER_UNIT..(u + 1) * FFN_SB_PER_UNIT {
                gelu_segment_quantize(m, sh.inter, I, sb * 64, 64, &full_max[..m], sh.inter_q);
            }
        });
        bar.wait();

        // FFN down + residual (super-blocks).
        sme_pack_activations(
            m,
            I,
            from_raw_parts(sh.inter_q as *const i8, m * I),
            &mut ws.inter_packed,
        );
        sched.run(H / 64, |sb| {
            sme_gemm_packed_superblocks(
                m,
                &ws.inter_packed,
                &inter_scale[..m],
                &layer.ffn_out,
                hid,
                tmp,
                sb,
                1,
            );
        });
        bar.wait();
        std::mem::swap(&mut hid, &mut tmp);

        // LN2 + quantize for the next layer's QKV (rows).
        let next_smooth = &model.layers[l + 1].qkv.inv_smooth;
        sched.run(row_units, |u| {
            let (r0, r1) = rows_of(u);
            for t in r0..r1 {
                *sh.act_scale.add(t) = layernorm_smooth_quantize_row(
                    from_raw_parts_mut(hid.add(t * H), H),
                    &layer.ln2_weight,
                    &layer.ln2_bias,
                    next_smooth,
                    from_raw_parts_mut(sh.act_q.add(t * H), H),
                    LAYER_NORM_EPS,
                );
            }
        });
        bar.wait();
    }

    // Layer 11 (unit = head pair): Q for [CLS] only, K/V for all tokens, then [CLS] attention.
    let last = &model.layers[NUM_LAYERS - 1];
    let q_only = last.q_only.as_ref().unwrap();
    let kv_only = last.kv_only.as_ref().unwrap();
    sme_pack_activations(1, H, &aq()[..H], &mut ws.inter_packed);
    sme_pack_activations(m, H, aq(), &mut ws.act_packed);
    sched.run(NUM_HEADS_PAIRS, |hp| {
        sme_gemm_packed_superblocks(
            1,
            &ws.inter_packed,
            &asc()[..1],
            q_only,
            std::ptr::null(),
            sh.q_cls,
            hp,
            1,
        );
        for base in [0, NUM_HEADS_PAIRS] {
            sme_gemm_packed_superblocks(
                m,
                &ws.act_packed,
                asc(),
                kv_only,
                std::ptr::null(),
                sh.qkv,
                base + hp,
                1,
            );
        }
        let kv = from_raw_parts(sh.qkv as *const f32, m * 2 * H);
        for h in 2 * hp..2 * hp + 2 {
            let h_off = h * HEAD_DIM;
            extract_kv_head32(
                kv,
                m,
                2 * H,
                h_off,
                H + h_off,
                &mut ws.k_head,
                &mut ws.v_head,
            );
            let scores = &mut ws.scores[..m];
            compute_scores_head32(
                from_raw_parts(sh.q_cls.add(h_off), HEAD_DIM),
                &ws.k_head[..m * HEAD_DIM],
                inv_sqrt_d,
                scores,
            );
            softmax_inplace(scores);
            weighted_sum_head32(
                scores,
                &ws.v_head[..m * HEAD_DIM],
                from_raw_parts_mut(sh.ctx.add(h_off), HEAD_DIM),
            );
        }
    });
    bar.wait();
    // Thread 0 continues with the single-row tail. Two swaps per layer leave the residual stream in
    // `ws.hidden`, which the tail reads.
    debug_assert_eq!(hid, sh.hidden);
}

const NUM_HEADS_PAIRS: usize = crate::quantize::NUM_HEADS / 2;
/// FFN-up / GELU work unit width in 64-column super-blocks (fewer GEMM calls, 12 units per stage).
const FFN_SB_PER_UNIT: usize = 2;

/// Layer 11's `[CLS]`-row tail (`M = 1`), run on thread 0 after [`worker_forward`].
#[cfg(target_arch = "aarch64")]
unsafe fn tail_last_layer(
    model: &PackedModel,
    ws: &mut Workspace,
    _seq_len: usize,
    out_emb: &mut [f32; HIDDEN_SIZE],
) {
    use crate::gemm::gemm_int8_fused;
    use crate::quantize::{INTERMEDIATE_SIZE, LAYER_NORM_EPS, NUM_LAYERS};
    use crate::simd_math::{
        gelu_smooth_quantize_batch, l2_normalize_inplace, layernorm_inplace_row,
        layernorm_smooth_quantize_row, smooth_quantize_row,
    };
    let last = &model.layers[NUM_LAYERS - 1];
    ws.act_scale[0] = smooth_quantize_row(
        &ws.attn_ctx[..HIDDEN_SIZE],
        &last.attn_out.inv_smooth,
        &mut ws.act_q[..HIDDEN_SIZE],
    );
    gemm_int8_fused(
        1,
        &ws.act_q[..HIDDEN_SIZE],
        &mut ws.act_packed,
        &ws.act_scale[..1],
        &last.attn_out,
        Some(&ws.hidden[..HIDDEN_SIZE]),
        &mut ws.tmp_hidden[..HIDDEN_SIZE],
    );
    ws.act_scale[0] = layernorm_smooth_quantize_row(
        &mut ws.tmp_hidden[..HIDDEN_SIZE],
        &last.ln1_weight,
        &last.ln1_bias,
        &last.ffn_inter.inv_smooth,
        &mut ws.act_q[..HIDDEN_SIZE],
        LAYER_NORM_EPS,
    );
    gemm_int8_fused(
        1,
        &ws.act_q[..HIDDEN_SIZE],
        &mut ws.act_packed,
        &ws.act_scale[..1],
        &last.ffn_inter,
        None,
        &mut ws.inter_buf[..INTERMEDIATE_SIZE],
    );
    gelu_smooth_quantize_batch(
        1,
        &mut ws.inter_buf[..INTERMEDIATE_SIZE],
        &last.ffn_out.inv_smooth,
        &mut ws.inter_q[..INTERMEDIATE_SIZE],
        &mut ws.act_scale[..1],
        true,
    );
    gemm_int8_fused(
        1,
        &ws.inter_q[..INTERMEDIATE_SIZE],
        &mut ws.inter_packed,
        &ws.act_scale[..1],
        &last.ffn_out,
        Some(&ws.tmp_hidden[..HIDDEN_SIZE]),
        out_emb,
    );
    layernorm_inplace_row(out_emb, &last.ln2_weight, &last.ln2_bias, LAYER_NORM_EPS);
    l2_normalize_inplace(out_emb);
}
