use rand::{thread_rng, SeedableRng};
use rand_pcg::Pcg64;
use rand_distr::{Distribution, Normal};
use ndarray::{Array2, Array3};
use ndarray_npy::write_npy;
use std::fs;
use std::path::Path;
use log::{info, warn, error};
use env_logger;
use rayon::prelude::*;
use clap::Parser;

/// 2D LRDG网络上Kuramoto动力学模拟的命令行参数（通用版）
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Sigma值（长程衰减指数）
    #[arg(long)]
    sigma: f64,

    /// 粗扫描K起始值
    #[arg(long)]
    k_start: f64,

    /// 粗扫描K结束值
    #[arg(long)]
    k_end: f64,

    /// 粗扫描K步长
    #[arg(long, default_value = "0.1")]
    k_step: f64,

    /// 估计的临界Kc值（用于精细扫描）
    #[arg(long)]
    kc: f64,

    /// Kc前后扫描的步数（3表示±3步=±0.03）
    #[arg(long, default_value = "3")]
    kc_range: usize,

    /// 精细扫描K步长
    #[arg(long, default_value = "0.01")]
    kc_step: f64,

    /// 粗扫描重复次数
    #[arg(long, default_value = "500")]
    runs_coarse: usize,

    /// 精细扫描重复次数
    #[arg(long, default_value = "1000")]
    runs_fine: usize,

    /// 输出根目录
    #[arg(short, long)]
    output_dir: String,

    /// 并行线程数（0表示使用所有可用核心）
    #[arg(short, long, default_value = "0")]
    threads: usize,
}

// 固定参数设置
const D: usize = 2;
const DELTA: f64 = 1.0;
const T: f64 = 200.0;
const DT: f64 = 0.01;
const STEPS: usize = (T / DT) as usize;

/// 生成N值列表
fn get_n_values() -> Vec<usize> {
    vec![256, 400, 625, 900, 1296, 2401, 4096] // 16²..64²
}

/// 生成K值列表：粗扫描 + Kc精细扫描，去重排序
fn generate_k_values(args: &Args) -> Vec<(f64, usize)> {
    let mut k_tasks: Vec<(f64, usize)> = Vec::new();

    // 粗扫描
    let mut k = args.k_start;
    while k <= args.k_end + 1e-9 {
        k_tasks.push((k, args.runs_coarse));
        k += args.k_step;
    }

    // 精细扫描：Kc前后各kc_range步
    for i in -(args.kc_range as i64)..=(args.kc_range as i64) {
        let k_fine = args.kc + i as f64 * args.kc_step;
        // 去重：如果粗扫描已有此K值，则用精细扫描的runs数覆盖
        if let Some(entry) = k_tasks.iter_mut().find(|(kk, _)| (kk - k_fine).abs() < 1e-9) {
            entry.1 = args.runs_fine;
        } else {
            k_tasks.push((k_fine, args.runs_fine));
        }
    }

    k_tasks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    k_tasks
}

/// 生成2D LRDG网络的邻居列表
fn generate_2d_lrdg_neighbors(n: usize, sigma: f64, seed: Option<u64>) -> (Vec<Vec<usize>>, Vec<usize>) {
    let l_f64 = (n as f64).sqrt();
    if (l_f64.fract()).abs() > 1e-9 {
        panic!("节点总数 N ({}) 必须是完美平方数!", n);
    }
    let l = l_f64 as usize;

    let mut rng = if let Some(s) = seed {
        Pcg64::seed_from_u64(s)
    } else {
        Pcg64::from_entropy()
    };

    let mut neighbors = vec![Vec::new(); n];
    let mut degrees = vec![0; n];
    let mut edge_set = vec![vec![false; n]; n];

    // 二维周期性骨架
    for i in 0..n {
        let x_i = i % l;
        let y_i = i / l;
        let x_right = (x_i + 1) % l;
        let j_right = y_i * l + x_right;
        if !edge_set[i][j_right] {
            neighbors[i].push(j_right);
            neighbors[j_right].push(i);
            edge_set[i][j_right] = true;
            edge_set[j_right][i] = true;
        }
        let y_down = (y_i + 1) % l;
        let j_down = y_down * l + x_i;
        if !edge_set[i][j_down] {
            neighbors[i].push(j_down);
            neighbors[j_down].push(i);
            edge_set[i][j_down] = true;
            edge_set[j_down][i] = true;
        }
    }

    // 长程连接
    let rho = 2.0 + sigma;
    for i in 0..n {
        let x_i = i % l;
        let y_i = i / l;
        for j in (i + 1)..n {
            if edge_set[i][j] { continue; }
            let x_j = j % l;
            let y_j = j / l;
            let dx = (x_i as i32 - x_j as i32).abs();
            let dy = (y_i as i32 - y_j as i32).abs();
            let wrapped_dx = (l as i32 - dx).min(dx) as f64;
            let wrapped_dy = (l as i32 - dy).min(dy) as f64;
            let distance_sq = wrapped_dx.powi(2) + wrapped_dy.powi(2);
            if distance_sq > 1.0 {
                let distance = distance_sq.sqrt();
                let prob = distance.powf(-rho);
                let uniform = rand_distr::Uniform::new(0.0, 1.0);
                if uniform.sample(&mut rng) < prob {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                    edge_set[i][j] = true;
                    edge_set[j][i] = true;
                }
            }
        }
    }

    for i in 0..n { degrees[i] = neighbors[i].len(); }
    (neighbors, degrees)
}

struct KuramotoModel {
    sigma: Array2<f64>,
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

        let mut sigma = Array2::<f64>::zeros((n, D));
        for i in 0..n {
            let mut state = vec![0.0; D];
            for x in &mut state { *x = normal.sample(&mut rng); }
            let norm: f64 = state.iter().map(|x| x * x).sum::<f64>().sqrt();
            for (j, x) in state.iter().enumerate() { sigma[[i, j]] = x / norm; }
        }

        let mut w = Array3::<f64>::zeros((n, D, D));
        for i in 0..n {
            for j in 0..D {
                for k_idx in 0..j {
                    let val = normal_w.sample(&mut rng);
                    w[[i, j, k_idx]] = val;
                    w[[i, k_idx, j]] = -val;
                }
            }
        }

        Self { sigma, w, neighbors, degrees, n, k }
    }

    fn dynamics(&self, sigma: &Array2<f64>, dsigma_dt: &mut Array2<f64>) {
        dsigma_dt.fill(0.0);
        for i in 0..self.n {
            let mut a_i = vec![0.0; D];
            for &j in &self.neighbors[i] {
                for d_idx in 0..D { a_i[d_idx] += sigma[[j, d_idx]]; }
            }
            let dot_product: f64 = a_i.iter().zip(sigma.row(i).iter()).map(|(a, s)| a * s).sum();
            let mut w_term = vec![0.0; D];
            for d1 in 0..D { for d2 in 0..D { w_term[d1] += self.w[[i, d1, d2]] * sigma[[i, d2]]; } }
            let k_i = self.degrees[i];
            let coupling_strength = if k_i > 0 { self.k / k_i as f64 } else { 0.0 };
            for d_idx in 0..D {
                dsigma_dt[[i, d_idx]] = w_term[d_idx] + coupling_strength * (a_i[d_idx] - sigma[[i, d_idx]] * dot_product);
            }
        }
    }

    fn integrate(&mut self, steps: usize, dt: f64) -> Array2<f64> {
        let mut order_parameters = Array2::<f64>::zeros((steps, 1));
        let mut k1 = Array2::<f64>::zeros((self.n, D));
        let mut k2 = Array2::<f64>::zeros((self.n, D));
        let mut k3 = Array2::<f64>::zeros((self.n, D));
        let mut k4 = Array2::<f64>::zeros((self.n, D));
        let mut temp_sigma = Array2::<f64>::zeros((self.n, D));

        for step in 0..steps {
            self.dynamics(&self.sigma, &mut k1); k1 *= dt;
            temp_sigma.assign(&self.sigma); temp_sigma += &(&k1 * 0.5);
            self.dynamics(&temp_sigma, &mut k2); k2 *= dt;
            temp_sigma.assign(&self.sigma); temp_sigma += &(&k2 * 0.5);
            self.dynamics(&temp_sigma, &mut k3); k3 *= dt;
            temp_sigma.assign(&self.sigma); temp_sigma += &k3;
            self.dynamics(&temp_sigma, &mut k4); k4 *= dt;
            let delta = (&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) / 6.0;
            self.sigma += &delta;

            let mut mean_sigma = vec![0.0; D];
            for i in 0..self.n { for d_idx in 0..D { mean_sigma[d_idx] += self.sigma[[i, d_idx]]; } }
            for x in &mut mean_sigma { *x /= self.n as f64; }
            order_parameters[[step, 0]] = mean_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();
        }
        order_parameters
    }
}

#[derive(Clone)]
struct SimulationTask {
    n: usize,
    k: f64,
    run: usize,
}

fn simulate_single_point(
    n: usize,
    k: f64,
    run: usize,
    sigma_val: f64,
    output_dir: &str,
) -> Result<(), String> {
    const MAX_ATTEMPTS: usize = 10;

    // 与 spectral_dimension/2d_lrdg/D_2/raw 及作图用的 .2f 一致（sigma_0.78，不是 sigma_0.780）
    let sigma_dir = format!("{}/sigma_{:.2}", output_dir, sigma_val);
    let n_dir = format!("{}/N_{}", sigma_dir, n);
    let k_dir = format!("{}/K_{:.2}", n_dir, k);
    let filename = format!("{}/kuramoto_2d_lrdg_D{}_N{}_K{:.2}_run{}.npy",
                          k_dir, D, n, k, run);

    if Path::new(&filename).exists() { return Ok(()); }
    fs::create_dir_all(&k_dir).map_err(|e| format!("Failed to create directory: {}", e))?;

    for attempt in 1..=MAX_ATTEMPTS {
        let (neighbors, degrees) = generate_2d_lrdg_neighbors(n, sigma_val, None);
        let mut model = KuramotoModel::new(neighbors, degrees, n, k);
        model.integrate(1000, DT);
        let order_params = model.integrate(STEPS, DT);
        let has_nan = order_params.iter().any(|&x| x.is_nan() || x.is_infinite());

        if has_nan {
            if attempt < MAX_ATTEMPTS {
                warn!("Attempt {}/{} failed (NaN/Inf) for sigma={:.3}, N={}, K={:.2}, run={}", attempt, MAX_ATTEMPTS, sigma_val, n, k, run);
                continue;
            } else {
                return Err(format!("All {} attempts failed for sigma={:.3}, N={}, K={:.2}, run={}", MAX_ATTEMPTS, sigma_val, n, k, run));
            }
        }

        write_npy(&filename, &order_params).map_err(|e| format!("Failed to write npy: {}", e))?;
        if attempt > 1 {
            info!("Success after {} attempts: sigma={:.3}, N={}, K={:.2}, run={}", attempt, sigma_val, n, k, run);
        }
        return Ok(());
    }
    Err("Unexpected end of retry loop".into())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().unwrap();
    }
    let available_threads = rayon::current_num_threads();
    info!("Using {} threads", available_threads);

    let n_values = get_n_values();
    let k_tasks = generate_k_values(&args);

    info!("2D LRDG Kuramoto - sigma={:.2} (d_s={:.2})", args.sigma, 4.0 / args.sigma);
    info!("  N values: {:?}", n_values);
    info!("  K points: {} (coarse {} + fine around Kc={:.2})", k_tasks.len(),
          k_tasks.iter().filter(|(_, r)| *r == args.runs_coarse).count(), args.kc);
    info!("  Output: {}", args.output_dir);

    fs::create_dir_all(&args.output_dir).unwrap();

    let mut tasks = Vec::new();
    for &n in &n_values {
        for &(k, runs) in &k_tasks {
            for run in 0..runs {
                tasks.push(SimulationTask { n, k, run });
            }
        }
    }

    let total_tasks = tasks.len();
    info!("Total tasks: {} ({} N × {} K configs)", total_tasks, n_values.len(), k_tasks.len());

    let successful_count = std::sync::atomic::AtomicUsize::new(0);
    let failed_count = std::sync::atomic::AtomicUsize::new(0);
    let start_time = std::time::Instant::now();

    info!("Starting parallel computation...");
    tasks.par_iter().for_each(|task| {
        match simulate_single_point(task.n, task.k, task.run, args.sigma, &args.output_dir) {
            Ok(()) => {
                let count = successful_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if count % 500 == 0 || count == total_tasks {
                    let elapsed = start_time.elapsed().as_secs();
                    let rate = if elapsed > 0 { count as f64 / elapsed as f64 } else { 0.0 };
                    info!("Progress: {}/{} ({:.1}%) - {}m{}s - {:.1} tasks/s",
                          count, total_tasks, 100.0 * count as f64 / total_tasks as f64,
                          elapsed / 60, elapsed % 60, rate);
                }
            }
            Err(e) => {
                error!("Failed: N={}, K={:.2}, run={} - {}", task.n, task.k, task.run, e);
                failed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });

    let elapsed = start_time.elapsed();
    let final_ok = successful_count.load(std::sync::atomic::Ordering::Relaxed);
    let final_fail = failed_count.load(std::sync::atomic::Ordering::Relaxed);

    info!("============================================");
    info!("Completed! Time: {:.2}h | OK: {} | Fail: {}",
          elapsed.as_secs_f64() / 3600.0, final_ok, final_fail);
    info!("============================================");

    let param_info = format!(
        "2D LRDG Kuramoto - sigma={:.2}\n\
         D={}, DELTA={}, T={}, DT={}, STEPS={}\n\
         Sigma={} (d_s_theory={:.2})\n\
         K coarse: [{:.1}..{:.1}] step={:.1} runs={}\n\
         K fine: Kc={:.2} ±{:.2} step={:.3} runs={}\n\
         N values: {:?}\n\
         Total K configs: {}\n\
         Total tasks: {}\n\
         Successful: {}\n\
         Failed: {}\n\
         Time: {:.2}h\n",
        args.sigma, D, DELTA, T, DT, STEPS, args.sigma, 4.0/args.sigma,
        args.k_start, args.k_end, args.k_step, args.runs_coarse,
        args.kc, args.kc_range as f64 * args.kc_step, args.kc_step, args.runs_fine,
        n_values, k_tasks.len(), total_tasks, final_ok, final_fail,
        elapsed.as_secs_f64() / 3600.0
    );
    fs::write(format!("{}/simulation_info.txt", args.output_dir), &param_info).unwrap();

    if final_fail > 0 { std::process::exit(1); }
}
