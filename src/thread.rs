use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
};

use crate::{
    board::Board,
    history::{ContinuationCorrectionHistory, ContinuationHistory, CorrectionHistory, NoisyHistory, QuietHistory},
    nnue::Network,
    numa::{NumaReplicator, NumaValue},
    stack::Stack,
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    transposition::TranspositionTable,
    types::{MAX_MOVES, MAX_PLY, Move, Score, normalize_to_cp},
};

#[repr(align(64))]
struct AlignedAtomicU64 {
    inner: AtomicU64,
}

pub struct Counter {
    shards: Box<[AlignedAtomicU64]>,
}

unsafe impl Sync for Counter {}

impl Counter {
    pub fn aggregate(&self) -> u64 {
        self.shards.iter().map(|shard| shard.inner.load(Ordering::Relaxed)).sum()
    }

    pub fn get(&self, id: usize) -> u64 {
        self.shards[id].inner.load(Ordering::Relaxed)
    }

    pub fn increment(&self, id: usize) {
        self.shards[id].inner.store(self.shards[id].inner.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        for shard in &self.shards {
            shard.inner.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self {
            shards: std::iter::from_fn(|| Some(AlignedAtomicU64 { inner: AtomicU64::new(0) }))
                .take(ThreadPool::available_threads())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

pub struct Status {
    inner: AtomicUsize,
}

impl Status {
    pub const STOPPED: usize = 0;
    pub const RUNNING: usize = 1;

    pub fn get(&self) -> usize {
        self.inner.load(Ordering::Acquire)
    }

    pub fn set(&self, status: usize) {
        self.inner.store(status, Ordering::Release);
    }
}

impl Clone for Status {
    fn clone(&self) -> Self {
        Self { inner: AtomicUsize::new(self.inner.load(Ordering::Relaxed)) }
    }
}

impl Default for Status {
    fn default() -> Self {
        Self { inner: AtomicUsize::new(Self::STOPPED) }
    }
}

#[derive(Default)]
pub struct SharedCorrectionHistory {
    pub pawn: CorrectionHistory,
    pub minor: CorrectionHistory,
    pub non_pawn: [CorrectionHistory; 2],
}

unsafe impl NumaValue for SharedCorrectionHistory {}

pub struct SharedContext {
    pub tt: TranspositionTable,
    pub status: Status,
    pub nodes: Counter,
    pub tb_hits: Counter,
    pub soft_stop_votes: AtomicUsize,
    pub best_stats: [AtomicU32; MAX_MOVES],
    pub history: *const SharedCorrectionHistory,
    pub replicator: NumaReplicator<SharedCorrectionHistory>,
}

impl Default for SharedContext {
    fn default() -> Self {
        let replicator = unsafe { NumaReplicator::new(SharedCorrectionHistory::default) };

        Self {
            tt: TranspositionTable::default(),
            status: Status::default(),
            nodes: Counter::default(),
            tb_hits: Counter::default(),
            soft_stop_votes: AtomicUsize::new(0),
            best_stats: [const { AtomicU32::new(0) }; MAX_MOVES],
            history: unsafe { replicator.get() },
            replicator,
        }
    }
}

unsafe impl Send for SharedContext {}
unsafe impl Sync for SharedContext {}

/// Per-thread search statistics. Accumulated during search, aggregated by thread 0,
/// then dumped as a single JSON object. Add `pub stats: SearchStats` to `ThreadData`.
///
/// Call `SearchStats::dump_json` from `start()` after all threads have joined, e.g.:
///
///   td.stats.dump_json("search_stats.json");
///
/// or aggregate across threads first:
///
///   let mut combined = SearchStats::new();
///   for t in threads { combined.aggregate(&t.stats); }
///   combined.dump_json("search_stats.json");

#[derive(Default, Clone)]
pub struct SearchStats {
    // -----------------------------------------------------------------------
    // Transposition table
    // -----------------------------------------------------------------------
    /// TT lookup called (every non-excluded node)
    pub tt_reads: u64,
    /// TT entry was present and depth/score valid
    pub tt_hits: u64,
    /// Non-PV node entered the early-cutoff branch
    pub tt_cutoff_attempts: u64,
    /// Cutoff actually returned (halfmove_clock < 90)
    pub tt_cutoffs_taken: u64,
    /// Cutoff suppressed by 50-move rule proximity
    pub tt_cutoffs_blocked_50mr: u64,

    // -----------------------------------------------------------------------
    // Aspiration windows
    // -----------------------------------------------------------------------
    pub aspiration_fail_low: u64,
    pub aspiration_fail_high: u64,

    // -----------------------------------------------------------------------
    // Pre-move pruning (bulk node savings)
    // -----------------------------------------------------------------------
    pub razoring_hits: u64,
    pub rfp_hits: u64,

    pub nmp_attempts: u64,
    /// NMP returned >= beta without verification
    pub nmp_cutoffs_direct: u64,
    /// NMP triggered the verification search path
    pub nmp_verifications: u64,
    /// Verification search also confirmed >= beta
    pub nmp_cutoffs_verified: u64,

    pub probcut_move_tried: u64,
    pub probcut_cutoffs: u64,

    // -----------------------------------------------------------------------
    // Singular Extensions
    // -----------------------------------------------------------------------
    /// Entered the SE candidate check
    pub se_candidates: u64,
    pub se_single_extension: u64,
    pub se_double_extension: u64,
    pub se_triple_extension: u64,
    /// Multi-cut early return
    pub se_multicut: u64,
    /// Negative extension applied (-2)
    pub se_negative_extension: u64,
    /// tt_move cleared because singular_score > tt_score
    pub se_tt_move_cleared: u64,

    // -----------------------------------------------------------------------
    // Per-move pruning
    // -----------------------------------------------------------------------
    pub lmp_hits: u64,
    pub fp_hits: u64,
    pub bnfp_hits: u64,
    /// SEE pruning fired (both quiet and noisy thresholds)
    pub see_prune_quiet: u64,
    pub see_prune_noisy: u64,

    // -----------------------------------------------------------------------
    // Move ordering quality
    // -----------------------------------------------------------------------
    /// Beta cutoff happened on the 1st move tried (ideal: ~80%+)
    pub beta_cutoff_move_1: u64,
    pub beta_cutoff_move_2: u64,
    pub beta_cutoff_move_3_to_5: u64,
    pub beta_cutoff_move_6_plus: u64,
    /// Total beta cutoffs observed
    pub beta_cutoffs_total: u64,

    // -----------------------------------------------------------------------
    // LMR
    // -----------------------------------------------------------------------
    /// LMR depth reduction was applied
    pub lmr_applied: u64,
    /// score > alpha after reduced search → full re-search triggered
    pub lmr_research_needed: u64,
    /// new_depth bumped up after re-search score was promising
    pub lmr_depth_extended: u64,

    // -----------------------------------------------------------------------
    // Node counts
    // -----------------------------------------------------------------------
    pub search_nodes: u64,
    pub search_nodes_pv: u64,
    pub search_nodes_nonpv: u64,
    pub qsearch_nodes: u64,

    // -----------------------------------------------------------------------
    // QSearch specifics
    // -----------------------------------------------------------------------
    pub qs_tt_cutoffs: u64,
    pub qs_stand_pat_cutoffs: u64,
    pub qs_lmp_hits: u64,
    pub qs_see_prune_hits: u64,
    pub qs_beta_cutoff_move_1: u64,
    pub qs_beta_cutoff_move_2_plus: u64,

    // -----------------------------------------------------------------------
    // Extensions / depth modifications (search())
    // -----------------------------------------------------------------------
    pub hindsight_depth_increase: u64,
    pub hindsight_depth_decrease: u64,
}

impl SearchStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge another thread's stats into self (call from thread 0 after join).
    pub fn aggregate(&mut self, other: &Self) {
        self.tt_reads += other.tt_reads;
        self.tt_hits += other.tt_hits;
        self.tt_cutoff_attempts += other.tt_cutoff_attempts;
        self.tt_cutoffs_taken += other.tt_cutoffs_taken;
        self.tt_cutoffs_blocked_50mr += other.tt_cutoffs_blocked_50mr;

        self.aspiration_fail_low += other.aspiration_fail_low;
        self.aspiration_fail_high += other.aspiration_fail_high;

        self.razoring_hits += other.razoring_hits;
        self.rfp_hits += other.rfp_hits;
        self.nmp_attempts += other.nmp_attempts;
        self.nmp_cutoffs_direct += other.nmp_cutoffs_direct;
        self.nmp_verifications += other.nmp_verifications;
        self.nmp_cutoffs_verified += other.nmp_cutoffs_verified;
        self.probcut_move_tried += other.probcut_move_tried;
        self.probcut_cutoffs += other.probcut_cutoffs;

        self.se_candidates += other.se_candidates;
        self.se_single_extension += other.se_single_extension;
        self.se_double_extension += other.se_double_extension;
        self.se_triple_extension += other.se_triple_extension;
        self.se_multicut += other.se_multicut;
        self.se_negative_extension += other.se_negative_extension;
        self.se_tt_move_cleared += other.se_tt_move_cleared;

        self.lmp_hits += other.lmp_hits;
        self.fp_hits += other.fp_hits;
        self.bnfp_hits += other.bnfp_hits;
        self.see_prune_quiet += other.see_prune_quiet;
        self.see_prune_noisy += other.see_prune_noisy;

        self.beta_cutoff_move_1 += other.beta_cutoff_move_1;
        self.beta_cutoff_move_2 += other.beta_cutoff_move_2;
        self.beta_cutoff_move_3_to_5 += other.beta_cutoff_move_3_to_5;
        self.beta_cutoff_move_6_plus += other.beta_cutoff_move_6_plus;
        self.beta_cutoffs_total += other.beta_cutoffs_total;

        self.lmr_applied += other.lmr_applied;
        self.lmr_research_needed += other.lmr_research_needed;
        self.lmr_depth_extended += other.lmr_depth_extended;

        self.search_nodes += other.search_nodes;
        self.search_nodes_pv += other.search_nodes_pv;
        self.search_nodes_nonpv += other.search_nodes_nonpv;
        self.qsearch_nodes += other.qsearch_nodes;

        self.qs_tt_cutoffs += other.qs_tt_cutoffs;
        self.qs_stand_pat_cutoffs += other.qs_stand_pat_cutoffs;
        self.qs_lmp_hits += other.qs_lmp_hits;
        self.qs_see_prune_hits += other.qs_see_prune_hits;
        self.qs_beta_cutoff_move_1 += other.qs_beta_cutoff_move_1;
        self.qs_beta_cutoff_move_2_plus += other.qs_beta_cutoff_move_2_plus;

        self.hindsight_depth_increase += other.hindsight_depth_increase;
        self.hindsight_depth_decrease += other.hindsight_depth_decrease;
    }

    /// Serialize to a single-line JSON object and append to `path`.
    /// Each search call appends one line → the file is newline-delimited JSON (ndjson),
    /// which the Python script reads with `pd.read_json(..., lines=True)`.
    pub fn dump_json(&self, path: &str) {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("Cannot open stats file");

        // Hand-rolled JSON — no serde dependency required.
        let json = format!(
            "{{\
            \"tt_reads\":{tt_reads},\
            \"tt_hits\":{tt_hits},\
            \"tt_cutoff_attempts\":{tt_cutoff_attempts},\
            \"tt_cutoffs_taken\":{tt_cutoffs_taken},\
            \"tt_cutoffs_blocked_50mr\":{tt_cutoffs_blocked_50mr},\
            \"aspiration_fail_low\":{aspiration_fail_low},\
            \"aspiration_fail_high\":{aspiration_fail_high},\
            \"razoring_hits\":{razoring_hits},\
            \"rfp_hits\":{rfp_hits},\
            \"nmp_attempts\":{nmp_attempts},\
            \"nmp_cutoffs_direct\":{nmp_cutoffs_direct},\
            \"nmp_verifications\":{nmp_verifications},\
            \"nmp_cutoffs_verified\":{nmp_cutoffs_verified},\
            \"probcut_move_tried\":{probcut_move_tried},\
            \"probcut_cutoffs\":{probcut_cutoffs},\
            \"se_candidates\":{se_candidates},\
            \"se_single_extension\":{se_single_extension},\
            \"se_double_extension\":{se_double_extension},\
            \"se_triple_extension\":{se_triple_extension},\
            \"se_multicut\":{se_multicut},\
            \"se_negative_extension\":{se_negative_extension},\
            \"se_tt_move_cleared\":{se_tt_move_cleared},\
            \"lmp_hits\":{lmp_hits},\
            \"fp_hits\":{fp_hits},\
            \"bnfp_hits\":{bnfp_hits},\
            \"see_prune_quiet\":{see_prune_quiet},\
            \"see_prune_noisy\":{see_prune_noisy},\
            \"beta_cutoff_move_1\":{beta_cutoff_move_1},\
            \"beta_cutoff_move_2\":{beta_cutoff_move_2},\
            \"beta_cutoff_move_3_to_5\":{beta_cutoff_move_3_to_5},\
            \"beta_cutoff_move_6_plus\":{beta_cutoff_move_6_plus},\
            \"beta_cutoffs_total\":{beta_cutoffs_total},\
            \"lmr_applied\":{lmr_applied},\
            \"lmr_research_needed\":{lmr_research_needed},\
            \"lmr_depth_extended\":{lmr_depth_extended},\
            \"search_nodes\":{search_nodes},\
            \"search_nodes_pv\":{search_nodes_pv},\
            \"search_nodes_nonpv\":{search_nodes_nonpv},\
            \"qsearch_nodes\":{qsearch_nodes},\
            \"qs_tt_cutoffs\":{qs_tt_cutoffs},\
            \"qs_stand_pat_cutoffs\":{qs_stand_pat_cutoffs},\
            \"qs_lmp_hits\":{qs_lmp_hits},\
            \"qs_see_prune_hits\":{qs_see_prune_hits},\
            \"qs_beta_cutoff_move_1\":{qs_beta_cutoff_move_1},\
            \"qs_beta_cutoff_move_2_plus\":{qs_beta_cutoff_move_2_plus},\
            \"hindsight_depth_increase\":{hindsight_depth_increase},\
            \"hindsight_depth_decrease\":{hindsight_depth_decrease}\
            }}",
            tt_reads = self.tt_reads,
            tt_hits = self.tt_hits,
            tt_cutoff_attempts = self.tt_cutoff_attempts,
            tt_cutoffs_taken = self.tt_cutoffs_taken,
            tt_cutoffs_blocked_50mr = self.tt_cutoffs_blocked_50mr,
            aspiration_fail_low = self.aspiration_fail_low,
            aspiration_fail_high = self.aspiration_fail_high,
            razoring_hits = self.razoring_hits,
            rfp_hits = self.rfp_hits,
            nmp_attempts = self.nmp_attempts,
            nmp_cutoffs_direct = self.nmp_cutoffs_direct,
            nmp_verifications = self.nmp_verifications,
            nmp_cutoffs_verified = self.nmp_cutoffs_verified,
            probcut_move_tried = self.probcut_move_tried,
            probcut_cutoffs = self.probcut_cutoffs,
            se_candidates = self.se_candidates,
            se_single_extension = self.se_single_extension,
            se_double_extension = self.se_double_extension,
            se_triple_extension = self.se_triple_extension,
            se_multicut = self.se_multicut,
            se_negative_extension = self.se_negative_extension,
            se_tt_move_cleared = self.se_tt_move_cleared,
            lmp_hits = self.lmp_hits,
            fp_hits = self.fp_hits,
            bnfp_hits = self.bnfp_hits,
            see_prune_quiet = self.see_prune_quiet,
            see_prune_noisy = self.see_prune_noisy,
            beta_cutoff_move_1 = self.beta_cutoff_move_1,
            beta_cutoff_move_2 = self.beta_cutoff_move_2,
            beta_cutoff_move_3_to_5 = self.beta_cutoff_move_3_to_5,
            beta_cutoff_move_6_plus = self.beta_cutoff_move_6_plus,
            beta_cutoffs_total = self.beta_cutoffs_total,
            lmr_applied = self.lmr_applied,
            lmr_research_needed = self.lmr_research_needed,
            lmr_depth_extended = self.lmr_depth_extended,
            search_nodes = self.search_nodes,
            search_nodes_pv = self.search_nodes_pv,
            search_nodes_nonpv = self.search_nodes_nonpv,
            qsearch_nodes = self.qsearch_nodes,
            qs_tt_cutoffs = self.qs_tt_cutoffs,
            qs_stand_pat_cutoffs = self.qs_stand_pat_cutoffs,
            qs_lmp_hits = self.qs_lmp_hits,
            qs_see_prune_hits = self.qs_see_prune_hits,
            qs_beta_cutoff_move_1 = self.qs_beta_cutoff_move_1,
            qs_beta_cutoff_move_2_plus = self.qs_beta_cutoff_move_2_plus,
            hindsight_depth_increase = self.hindsight_depth_increase,
            hindsight_depth_decrease = self.hindsight_depth_decrease,
        );

        writeln!(f, "{}", json).expect("Cannot write stats");
    }
}

pub struct ThreadData {
    pub id: usize,
    pub stats: SearchStats,
    pub shared: Arc<SharedContext>,
    pub board: Board,
    pub time_manager: TimeManager,
    pub stack: Box<Stack>,
    pub nnue: Network,
    pub root_moves: Vec<RootMove>,
    pub pv_table: PrincipalVariationTable,
    pub noisy_history: NoisyHistory,
    pub quiet_history: QuietHistory,
    pub continuation_history: ContinuationHistory,
    pub continuation_corrhist: ContinuationCorrectionHistory,
    pub best_move_changes: usize,
    pub optimism: [i32; 2],
    pub root_depth: i32,
    pub root_delta: i32,
    pub sel_depth: i32,
    pub completed_depth: i32,
    pub nmp_min_ply: i32,
    pub previous_best_score: i32,
    pub root_in_tb: bool,
    pub stop_probing_tb: bool,
    pub multi_pv: usize,
    pub pv_index: usize,
    pub pv_start: usize,
    pub pv_end: usize,
}

impl ThreadData {
    pub fn new(shared: Arc<SharedContext>) -> Self {
        Self {
            id: 0,
            stats: SearchStats::new(),
            shared,
            board: Board::starting_position(),
            time_manager: TimeManager::new(Limits::Infinite, 0, 0),
            stack: Stack::new(),
            nnue: Network::default(),
            root_moves: Vec::new(),
            pv_table: PrincipalVariationTable::default(),
            noisy_history: NoisyHistory::default(),
            quiet_history: QuietHistory::default(),
            continuation_history: ContinuationHistory::default(),
            continuation_corrhist: ContinuationCorrectionHistory::default(),
            best_move_changes: 0,
            optimism: [0; 2],
            root_depth: 0,
            root_delta: 0,
            sel_depth: 0,
            completed_depth: 0,
            nmp_min_ply: 0,
            previous_best_score: 0,
            root_in_tb: false,
            stop_probing_tb: false,
            multi_pv: 1,
            pv_index: 0,
            pv_start: 0,
            pv_end: 0,
        }
    }

    pub fn nodes(&self) -> u64 {
        self.shared.nodes.get(self.id)
    }

    pub fn corrhist(&self) -> &SharedCorrectionHistory {
        unsafe { &*self.shared.history }
    }

    pub fn conthist(&self, ply: isize, index: isize, mv: Move) -> i32 {
        self.continuation_history.get(self.stack[ply - index].conthist, self.board.piece_on(mv.from()), mv.to())
    }

    pub fn print_uci_info(&self, depth: i32) {
        let elapsed = self.time_manager.elapsed();
        let nps = self.shared.nodes.aggregate() as f64 / elapsed.as_secs_f64();
        let ms = elapsed.as_millis();

        for pv_index in 0..self.multi_pv {
            let root_move = &self.root_moves[pv_index];

            let updated = root_move.score != -Score::INFINITE;

            if depth == 1 && !updated && pv_index > 0 {
                continue;
            }

            let depth = if updated { depth } else { (depth - 1).max(1) };
            let mut score = if updated { root_move.display_score } else { root_move.previous_score };

            let mut upperbound = root_move.upperbound;
            let mut lowerbound = root_move.lowerbound;

            if self.root_in_tb && score.abs() <= Score::TB_WIN {
                score = root_move.tb_score;
                upperbound = false;
                lowerbound = false;
            }

            let mut formatted_score = match score.abs() {
                s if s < Score::TB_WIN_IN_MAX => {
                    format!("cp {}", normalize_to_cp(score, &self.board))
                }
                s if s <= Score::TB_WIN => {
                    let cp = 20_000 - Score::TB_WIN + score.abs();
                    format!("cp {}", if score.is_positive() { cp } else { -cp })
                }
                _ => {
                    let mate = (Score::MATE - score.abs() + score.is_positive() as i32) / 2;
                    format!("mate {}", if score.is_positive() { mate } else { -mate })
                }
            };

            if upperbound {
                formatted_score.push_str(" upperbound");
            } else if lowerbound {
                formatted_score.push_str(" lowerbound");
            }

            print!(
                "info depth {depth} seldepth {} multipv {} score {formatted_score} nodes {} time {ms} nps {nps:.0} hashfull {} tbhits {} pv",
                root_move.sel_depth,
                pv_index + 1,
                self.shared.nodes.aggregate(),
                self.shared.tt.hashfull(),
                self.shared.tb_hits.aggregate(),
            );

            print!(" {}", root_move.mv.to_uci(&self.board));
            for mv in root_move.pv.line() {
                print!(" {}", mv.to_uci(&self.board));
            }

            println!();
        }
    }
}

#[derive(Clone)]
pub struct RootMove {
    pub mv: Move,
    pub score: i32,
    pub previous_score: i32,
    pub display_score: i32,
    pub upperbound: bool,
    pub lowerbound: bool,
    pub sel_depth: i32,
    pub nodes: u64,
    pub pv: PrincipalVariationTable,
    pub tb_rank: i32,
    pub tb_score: i32,
}

impl Default for RootMove {
    fn default() -> Self {
        Self {
            mv: Move::NULL,
            score: -Score::INFINITE,
            previous_score: -Score::INFINITE,
            display_score: -Score::INFINITE,
            upperbound: false,
            lowerbound: false,
            sel_depth: 0,
            nodes: 0,
            pv: PrincipalVariationTable::default(),
            tb_rank: 0,
            tb_score: 0,
        }
    }
}

#[derive(Clone)]
pub struct PrincipalVariationTable {
    table: Box<[[Move; MAX_PLY + 1]]>,
    len: [usize; MAX_PLY + 1],
}

impl PrincipalVariationTable {
    pub fn line(&self) -> &[Move] {
        &self.table[0][..self.len[0]]
    }

    pub const fn clear(&mut self, ply: usize) {
        self.len[ply] = 0;
    }

    pub fn update(&mut self, ply: usize, mv: Move) {
        self.table[ply][0] = mv;
        self.len[ply] = self.len[ply + 1] + 1;

        for i in 0..self.len[ply + 1] {
            self.table[ply][i + 1] = self.table[ply + 1][i];
        }
    }

    pub fn commit_full_root_pv(&mut self, src: &Self, start_ply: usize) {
        let len = src.len[start_ply].min(MAX_PLY + 1);
        self.len[0] = len;
        self.table[0][..len].copy_from_slice(&src.table[start_ply][..len]);
    }
}

impl Default for PrincipalVariationTable {
    fn default() -> Self {
        Self {
            table: vec![[Move::NULL; MAX_PLY + 1]; MAX_PLY + 1].into_boxed_slice(),
            len: [0; MAX_PLY + 1],
        }
    }
}
