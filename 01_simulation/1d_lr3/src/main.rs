use rand::{thread_rng, SeedableRng};
use rand_pcg::Pcg64;
use rand_distr::{Distribution, Normal};
use ndarray::{Array2, Array3};
use ndarray_npy::write_npy;
use std::path::Path;
use std::fs;
use log::info;
use env_logger;
use rayon::prelude::*;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    sigma: f64,
    #[arg(long, default_value_t = 0.0)]
    k_start: f64,
    #[arg(long, default_value_t = 3.0)]
    k_end: f64,
    #[arg(long, default_value_t = 0.1)]
    k_step: f64,
    #[arg(long, default_value_t = 500)]
    runs: usize,
    #[arg(long)]
    output_dir: String,
    #[arg(short, long)]
    n: usize,
}

const D: usize = 2;
const DELTA: f64 = 1.0;
const T: f64 = 200.0;
const DT: f64 = 0.01;
const STEPS: usize = (T / DT) as usize;
const WARMUP: usize = 1000;

fn generate_network(n: usize, sigma: f64, seed: Option<u64>) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut rng = if let Some(s) = seed {
        Pcg64::seed_from_u64(s)
    } else {
        Pcg64::from_entropy()
    };
    let uniform = rand_distr::Uniform::new(0.0, 1.0);
    let mut neighbors = vec![Vec::new(); n];
    let mut degrees = vec![0; n];
    for i in 0..n {
        for j in (i+1)..n {
            let dist_direct = (i as i32 - j as i32).abs() as f64;
            let dist_wrap = n as f64 - dist_direct;
            let dist = dist_direct.min(dist_wrap);
            if dist > 0.0 {
                let prob = 1.0 / dist.powf(1.0 + sigma);
                if uniform.sample(&mut rng) < prob {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                    degrees[i] += 1;
                    degrees[j] += 1;
                }
            }
        }
    }
    (neighbors, degrees)
}

struct KuramotoModel {
    sigma_state: Array2<f64>,
    w: Array3<f64>,
    neighbors: Vec<Vec<usize>>,
    degrees: Vec<usize>,
    n: usize,
    k: f64,
}

impl KuramotoModel {
    fn new(neighbors: Vec<Vec<usize>>, degrees: Vec<usize>, n: usize, k: f64) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let normal_w = Normal::new(0.0, DELTA).unwrap();
        let mut sigma_state = Array2::<f64>::zeros((n, D));
        for i in 0..n {
            let mut state = vec![0.0; D];
            for x in &mut state { *x = normal.sample(&mut rng); }
            let norm: f64 = state.iter().map(|x| x*x).sum::<f64>().sqrt();
            for (j, x) in state.iter().enumerate() { sigma_state[[i,j]] = x / norm; }
        }
        let mut w = Array3::<f64>::zeros((n, D, D));
        for i in 0..n {
            for j in 0..D {
                for ki in 0..j {
                    let val = normal_w.sample(&mut rng);
                    w[[i,j,ki]] = val;
                    w[[i,ki,j]] = -val;
                }
            }
        }
        Self { sigma_state, w, neighbors, degrees, n, k }
    }

    fn dynamics(&self, sigma_state: &Array2<f64>, dsigma_dt: &mut Array2<f64>) {
        dsigma_dt.fill(0.0);
        for i in 0..self.n {
            let mut a_i = vec![0.0; D];
            for &j in &self.neighbors[i] {
                for d in 0..D { a_i[d] += sigma_state[[j,d]]; }
            }
            let dot: f64 = a_i.iter().zip(sigma_state.row(i).iter()).map(|(a,s)| a*s).sum();
            let mut w_term = vec![0.0; D];
            for d1 in 0..D {
                for d2 in 0..D { w_term[d1] += self.w[[i,d1,d2]] * sigma_state[[i,d2]]; }
            }
            let k_i = self.degrees[i];
            let cs = if k_i > 0 { self.k / k_i as f64 } else { 0.0 };
            for d in 0..D {
                dsigma_dt[[i,d]] = w_term[d] + cs * (a_i[d] - sigma_state[[i,d]] * dot);
            }
        }
    }

    fn integrate(&mut self, steps: usize, dt: f64) -> Array2<f64> {
        let mut order_params = Array2::<f64>::zeros((steps, 1));
        let mut k1 = Array2::<f64>::zeros((self.n, D));
        let mut k2 = Array2::<f64>::zeros((self.n, D));
        let mut k3 = Array2::<f64>::zeros((self.n, D));
        let mut k4 = Array2::<f64>::zeros((self.n, D));
        let mut tmp = Array2::<f64>::zeros((self.n, D));
        for step in 0..steps {
            self.dynamics(&self.sigma_state, &mut k1);
            k1 *= dt;
            tmp.assign(&self.sigma_state); tmp += &(&k1*0.5);
            self.dynamics(&tmp, &mut k2); k2 *= dt;
            tmp.assign(&self.sigma_state); tmp += &(&k2*0.5);
            self.dynamics(&tmp, &mut k3); k3 *= dt;
            tmp.assign(&self.sigma_state); tmp += &k3;
            self.dynamics(&tmp, &mut k4); k4 *= dt;
            let delta = (&k1 + &k2*2.0 + &k3*2.0 + &k4) / 6.0;
            self.sigma_state += &delta;
            let mut mean = vec![0.0; D];
            for i in 0..self.n { for d in 0..D { mean[d] += self.sigma_state[[i,d]]; } }
            for x in &mut mean { *x /= self.n as f64; }
            order_params[[step,0]] = mean.iter().map(|x| x*x).sum::<f64>().sqrt();
        }
        order_params
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let n = args.n;
    let sigma = args.sigma;
    let runs = args.runs;

    let mut k_vals = Vec::new();
    let mut k = args.k_start;
    while k <= args.k_end + 1e-9 {
        k_vals.push(k);
        k += args.k_step;
    }

    let sigma_str = format!("{:.3}", sigma);
    let n_str = format!("{}", n);
    let n_dir = format!("{}/sigma_{}/N_{}", args.output_dir, sigma_str, n_str);

    info!("sigma={}, N={}, K=[{:.1},{:.1}] step={:.1}, runs={}, T={:.0}, DT={}", 
          sigma, n, args.k_start, args.k_end, args.k_step, runs, T, DT);

    for &k_val in &k_vals {
        let k_str = format!("{:.2}", k_val);
        let k_dir = format!("{}/K_{}", n_dir, k_str);
        fs::create_dir_all(&k_dir).unwrap();
        info!("K={} ({}/{})", k_val, k_vals.iter().position(|&x| x == k_val).unwrap()+1, k_vals.len());

        let results: Vec<(usize, Array2<f64>)> = (0..runs).into_par_iter().filter_map(|run| {
            let filename = format!("{}/kuramoto_1d_lr3_D2_N{}_K{}_run{}.npy", k_dir, n, k_str, run);
            if Path::new(&filename).exists() { return None; }
            let (neighbors, degrees) = generate_network(n, sigma, None);
            let mut model = KuramotoModel::new(neighbors, degrees, n, k_val);
            model.integrate(WARMUP, DT);
            let order_params = model.integrate(STEPS, DT);
            Some((run, order_params))
        }).collect();

        let new_count = results.len();
        for (run, order_params) in results {
            let filename = format!("{}/kuramoto_1d_lr3_D2_N{}_K{}_run{}.npy", k_dir, n, k_str, run);
            write_npy(&filename, &order_params).unwrap();
        }
        info!("K={:.2} done, saved {} new files", k_val, new_count);
    }

    info!("All done for sigma={}, N={}", sigma, n);
}
