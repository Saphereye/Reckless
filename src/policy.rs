use std::sync::Arc;
use std::sync::OnceLock;
use crate::types::Move;

#[repr(align(64))]
pub struct Aligned<T>(pub T);

impl Move {
    pub const fn move_index(self) -> usize {
        (self.from() as usize) * 64 + (self.to() as usize)
    }
}

pub struct PolicyParameters {
    fc1_weight: Aligned<[[i8; 768]; 64]>,
    fc1_bias: Aligned<[i32; 64]>,
    fc2_weight: Aligned<[[f32; 64]; 4096]>,
    fc2_bias: Aligned<[f32; 4096]>,
    fc1_scale: f32,
}

pub struct PolicyHandle {
    inner: Arc<PolicyParameters>,
}

impl Clone for PolicyHandle {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl PolicyHandle {
    fn new() -> Option<Self> {
        let params = PolicyParameters::load()?;
        Some(Self { inner: Arc::new(params) })
    }

    pub fn score_moves(&self, input: &[f32; 768]) -> [f32; 4096] {
        let fc1_out = self.fc1_forward(input);
        self.fc2_forward(&fc1_out)
    }

    #[inline]
    fn fc1_forward(&self, input: &[f32; 768]) -> [f32; 64] {
        let mut output = [0.0f32; 64];

        for i in 0..64 {
            let bias = self.inner.fc1_bias.0[i] as f32;
            let dot = fc1_dot_i8_f32(&self.inner.fc1_weight.0[i], input);
            output[i] = ((dot + bias) / (128.0 * self.inner.fc1_scale)).max(0.0);
        }

        output
    }

    #[inline]
    fn fc2_forward(&self, fc1_out: &[f32; 64]) -> [f32; 4096] {
        let mut output = [0.0f32; 4096];

        for i in 0..4096 {
            output[i] = fc2_dot_f32(&self.inner.fc2_weight.0[i], fc1_out) + self.inner.fc2_bias.0[i];
        }

        output
    }
}

#[inline]
#[cfg(target_feature = "avx2")]
fn fc1_dot_i8_f32(weights: &[i8; 768], input: &[f32; 768]) -> f32 {
    let mut sum: i64 = 0;

    for chunk in 0..768 / 32 {
        let base = chunk * 32;
        let mut acc = 0i64;

        for j in 0..32 {
            let w = weights[base + j] as i64;
            let inp = (input[base + j] * 128.0) as i64;
            acc += w * inp;
        }

        sum += acc;
    }

    sum as f32
}

#[inline]
#[cfg(not(target_feature = "avx2"))]
fn fc1_dot_i8_f32(weights: &[i8; 768], input: &[f32; 768]) -> f32 {
    weights
        .iter()
        .zip(input)
        .map(|(w, i)| (*w as f32) * (i * 128.0))
        .sum()
}

#[inline]
#[cfg(target_feature = "avx2")]
fn fc2_dot_f32(weights: &[f32; 64], input: &[f32; 64]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in (0..64).step_by(8) {
            let w = _mm256_loadu_ps(weights.as_ptr().add(i));
            let inp = _mm256_loadu_ps(input.as_ptr().add(i));
            acc = _mm256_fmadd_ps(w, inp, acc);
        }

        let arr = std::mem::transmute::<__m256, [f32; 8]>(acc);
        arr.iter().sum()
    }
}

#[inline]
#[cfg(not(target_feature = "avx2"))]
fn fc2_dot_f32(weights: &[f32; 64], input: &[f32; 64]) -> f32 {
    weights.iter().zip(input).map(|(w, i)| w * i).sum()
}

impl PolicyParameters {
    fn load() -> Option<Self> {
        let fc1_w = std::fs::read("networks/policy_fc1_w_i8.bin").ok()?;
        let fc1_b = std::fs::read("networks/policy_fc1_b_i32.bin").ok()?;
        let fc1_scale_bytes = std::fs::read("networks/policy_fc1_scale.bin").ok()?;
        let fc2_w = std::fs::read("networks/policy_fc2_w_f32.bin").ok()?;
        let fc2_b = std::fs::read("networks/policy_fc2_b_f32.bin").ok()?;

        if fc1_w.len() != 64 * 768 || fc1_b.len() != 64 * 4 || fc1_scale_bytes.len() != 4
            || fc2_w.len() != 4096 * 64 * 4 || fc2_b.len() != 4096 * 4
        {
            return None;
        }

        let mut fc1_weight_arr = [[0i8; 768]; 64];
        for i in 0..64 {
            for j in 0..768 {
                fc1_weight_arr[i][j] = fc1_w[i * 768 + j] as i8;
            }
        }

        let mut fc1_bias_arr = [0i32; 64];
        for i in 0..64 {
            fc1_bias_arr[i] = i32::from_le_bytes([
                fc1_b[i * 4],
                fc1_b[i * 4 + 1],
                fc1_b[i * 4 + 2],
                fc1_b[i * 4 + 3],
            ]);
        }

        let fc1_scale = f32::from_le_bytes([
            fc1_scale_bytes[0],
            fc1_scale_bytes[1],
            fc1_scale_bytes[2],
            fc1_scale_bytes[3],
        ]);

        let mut fc2_weight_arr = [[0.0f32; 64]; 4096];
        for i in 0..4096 {
            for j in 0..64 {
                let offset = (i * 64 + j) * 4;
                fc2_weight_arr[i][j] = f32::from_le_bytes([fc2_w[offset], fc2_w[offset + 1], fc2_w[offset + 2], fc2_w[offset + 3]]);
            }
        }

        let mut fc2_bias_arr = [0.0f32; 4096];
        for i in 0..4096 {
            let offset = i * 4;
            fc2_bias_arr[i] = f32::from_le_bytes([fc2_b[offset], fc2_b[offset + 1], fc2_b[offset + 2], fc2_b[offset + 3]]);
        }

        Some(Self {
            fc1_weight: Aligned(fc1_weight_arr),
            fc1_bias: Aligned(fc1_bias_arr),
            fc2_weight: Aligned(fc2_weight_arr),
            fc2_bias: Aligned(fc2_bias_arr),
            fc1_scale,
        })
    }
}

static POLICY: OnceLock<Option<PolicyHandle>> = OnceLock::new();

pub fn initialize() {
    let _ = POLICY.set(PolicyHandle::new());
}

pub fn score_moves(input: &[f32; 768]) -> [f32; 4096] {
    POLICY
        .get_or_init(|| PolicyHandle::new())
        .as_ref()
        .map(|policy| policy.score_moves(input))
        .unwrap_or([0.0f32; 4096])
}
