use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ml_prefetcher::PredictivePrefetcher;
use rand::Rng;
use std::iter;
use tokio::runtime::Runtime;

fn benchmark_sequential_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sequential Pattern");
    let rt = Runtime::new().unwrap();
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut prefetcher = PredictivePrefetcher::new(4);
                rt.block_on(async {
                    for i in 0..size {
                        black_box(prefetcher.access(i as i32).await);
                    }
                })
            })
        });
    }
    
    group.finish();
}

fn benchmark_strided_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Strided Pattern");
    let rt = Runtime::new().unwrap();
    let strides = [2, 4, 8];
    
    for &size in &[100, 1000, 10000] {
        for &stride in &strides {
            group.bench_with_input(
                BenchmarkId::new("size", format!("{}_stride_{}", size, stride)), 
                &(size, stride), 
                |b, &(size, stride)| {
                    b.iter(|| {
                        let mut prefetcher = PredictivePrefetcher::new(4);
                        rt.block_on(async {
                            for i in (0..size).step_by(stride) {
                                black_box(prefetcher.access(i as i32).await);
                            }
                        })
                    })
                }
            );
        }
    }
    
    group.finish();
}

fn benchmark_repeated_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Repeated Pattern");
    let rt = Runtime::new().unwrap();
    
    let patterns = vec![
        vec![1_i32, 2, 3],
        vec![1, 2, 3, 4],
        vec![1, 2, 3, 4, 5]
    ];
    
    for pattern in patterns.iter() {
        let pattern_len = pattern.len();
        group.bench_with_input(
            BenchmarkId::new("pattern_size", pattern_len),
            pattern,
            |b, pattern| {
                b.iter(|| {
                    let mut prefetcher = PredictivePrefetcher::new(4);
                    let repeated = pattern.iter()
                        .cycle()
                        .take(pattern.len() * 100)
                        .copied()
                        .collect::<Vec<_>>();
                    rt.block_on(async {
                        for &addr in &repeated {
                            black_box(prefetcher.access(addr).await);
                        }
                    })
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_random_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Pattern");
    let rt = Runtime::new().unwrap();
    
    for &size in &[100, 1000, 10000] {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = rand::thread_rng();
            let addresses: Vec<i32> = (0..size)
                .map(|_| rng.gen_range(0..1000))
                .collect();
                
            b.iter(|| {
                let mut prefetcher = PredictivePrefetcher::new(4);
                rt.block_on(async {
                    for &addr in &addresses {
                        black_box(prefetcher.access(addr).await);
                    }
                })
            })
        });
    }
    
    group.finish();
}

fn benchmark_mixed_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Pattern");
    let rt = Runtime::new().unwrap();
    
    for &size in &[100, 1000, 10000] {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            // Create a mix of different patterns
            let sequential: Vec<i32> = (0..size/4).map(|x| x as i32).collect();
            let strided: Vec<i32> = (0..size/4).map(|x| (x * 2) as i32).collect();
            let repeated: Vec<i32> = iter::repeat(vec![1, 2, 3, 4])
                .take(size/16)
                .flatten()
                .collect();
            let mut rng = rand::thread_rng();
            let random: Vec<i32> = (0..size/4)
                .map(|_| rng.gen_range(0..1000))
                .collect();
            
            // Combine all patterns
            let mut mixed = Vec::new();
            mixed.extend(sequential);
            mixed.extend(strided);
            mixed.extend(repeated);
            mixed.extend(random);
            
            b.iter(|| {
                let mut prefetcher = PredictivePrefetcher::new(4);
                rt.block_on(async {
                    for &addr in &mixed {
                        black_box(prefetcher.access(addr).await);
                    }
                })
            })
        });
    }
    
    group.finish();
}

fn benchmark_pattern_transition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pattern Transition");
    let rt = Runtime::new().unwrap();
    
    for &size in &[100, 1000, 10000] {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let section_size = size / 4;
            
            // Create transitions between patterns
            let mut pattern = Vec::new();
            
            // Sequential
            pattern.extend((0..section_size).map(|x| x as i32));
            
            // Strided
            pattern.extend((0..section_size).map(|x| (x * 2) as i32));
            
            // Repeated
            pattern.extend(iter::repeat(vec![1, 2, 3, 4])
                .take(section_size/4)
                .flatten());
            
            // Random
            let mut rng = rand::thread_rng();
            pattern.extend((0..section_size)
                .map(|_| rng.gen_range(0..1000)));
            
            b.iter(|| {
                let mut prefetcher = PredictivePrefetcher::new(4);
                rt.block_on(async {
                    for &addr in &pattern {
                        black_box(prefetcher.access(addr).await);
                    }
                })
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_sequential_pattern,
    benchmark_strided_pattern,
    benchmark_repeated_pattern,
    benchmark_random_pattern,
    benchmark_mixed_pattern,
    benchmark_pattern_transition
);
criterion_main!(benches);