use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ml_prefetcher::PredictivePrefetcher;
use rand::Rng;
use std::iter;

fn benchmark_sequential_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sequential Pattern");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut prefetcher = PredictivePrefetcher::new(4);
                for i in 0..size {
                    black_box(prefetcher.access(i));
                }
            })
        });
    }
    
    group.finish();
}

fn benchmark_strided_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Strided Pattern");
    let strides = [2, 4, 8];
    
    for &size in &[100, 1000, 10000] {
        for &stride in &strides {
            group.bench_with_input(
                BenchmarkId::new("size", format!("{}_stride_{}", size, stride)), 
                &(size, stride), 
                |b, &(size, stride)| {
                    b.iter(|| {
                        let mut prefetcher = PredictivePrefetcher::new(4);
                        for i in (0..size).step_by(stride) {
                            black_box(prefetcher.access(i));
                        }
                    })
                }
            );
        }
    }
    
    group.finish();
}

fn benchmark_repeated_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Repeated Pattern");
    
    // Different pattern sizes
    let patterns = vec![
        vec![1, 2, 3],
        vec![1, 2, 3, 4],
        vec![1, 2, 3, 4, 5]
    ];
    
    for (idx, pattern) in patterns.iter().enumerate() {
        let pattern_len = pattern.len();
        group.bench_with_input(
            BenchmarkId::new("pattern_size", pattern_len),
            pattern,
            |b, pattern| {
                b.iter(|| {
                    let mut prefetcher = PredictivePrefetcher::new(4);
                    // Repeat pattern multiple times
                    let repeated = pattern.iter()
                        .cycle()
                        .take(pattern.len() * 100)
                        .copied()
                        .collect::<Vec<_>>();
                    for &addr in &repeated {
                        black_box(prefetcher.access(addr));
                    }
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_random_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Pattern");
    
    for &size in &[100, 1000, 10000] {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let mut rng = rand::thread_rng();
            let addresses: Vec<usize> = (0..size)
                .map(|_| rng.gen_range(0..1000))
                .collect();
                
            b.iter(|| {
                let mut prefetcher = PredictivePrefetcher::new(4);
                for &addr in &addresses {
                    black_box(prefetcher.access(addr));
                }
            })
        });
    }
    
    group.finish();
}

fn benchmark_mixed_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Pattern");
    
    for &size in &[100, 1000, 10000] {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            // Create a mix of different patterns
            let sequential: Vec<usize> = (0..size/4).collect();
            let strided: Vec<usize> = (0..size/4).map(|x| x * 2).collect();
            let repeated: Vec<usize> = iter::repeat(vec![1, 2, 3, 4])
                .take(size/16)
                .flatten()
                .collect();
            let mut rng = rand::thread_rng();
            let random: Vec<usize> = (0..size/4)
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
                for &addr in &mixed {
                    black_box(prefetcher.access(addr));
                }
            })
        });
    }
    
    group.finish();
}

fn benchmark_pattern_transition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pattern Transition");
    
    for &size in &[100, 1000, 10000] {
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            let section_size = size / 4;
            
            // Create transitions between patterns
            let mut pattern = Vec::new();
            
            // Sequential
            pattern.extend(0..section_size);
            
            // Strided
            pattern.extend((0..section_size).map(|x| x * 2));
            
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
                for &addr in &pattern {
                    black_box(prefetcher.access(addr));
                }
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