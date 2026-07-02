use core::ffi::{c_char, c_int, c_uint};

pub const TB_LOSS: c_uint = 0;
pub const TB_DRAW: c_uint = 2;
pub const TB_WIN: c_uint = 4;

pub const TB_MAX_MOVES: c_uint = 193;

const TB_MAX_PLY: usize = 240;

pub type TbMove = u16;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TbRootMove {
    pub move_: TbMove,
    pub pv: [TbMove; TB_MAX_PLY],
    pub pvSize: c_uint,
    pub tbScore: c_int,
    pub tbRank: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TbRootMoves {
    pub size: c_uint,
    pub moves: [TbRootMove; TB_MAX_MOVES as usize],
}

unsafe extern "C" {
    pub static mut TB_LARGEST: c_uint;

    pub fn tb_init(path: *const c_char) -> bool;

    pub fn tb_probe_wdl(
        white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64,
        rule50: c_uint, castling: c_uint, ep: c_uint, turn: bool,
    ) -> c_uint;

    pub fn tb_probe_root_dtz(
        white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64,
        rule50: c_uint, castling: c_uint, ep: c_uint, turn: bool, has_repeated: bool, use_rule50: bool,
        results: *mut TbRootMoves,
    ) -> c_int;

    pub fn tb_probe_root_wdl(
        white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64,
        rule50: c_uint, castling: c_uint, ep: c_uint, turn: bool, use_rule50: bool, results: *mut TbRootMoves,
    ) -> c_int;
}
