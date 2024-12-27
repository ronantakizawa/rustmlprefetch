use ml_prefetcher::PredictivePrefetcher;

fn main() {
    let mut prefetcher = PredictivePrefetcher::new(4);
    
    // Test with different access patterns
    
    // Sequential access pattern
    println!("Testing sequential access pattern:");
    for address in 1..10 {
        let prefetched = prefetcher.access(address);
        println!("Accessed: {}, Prefetched: {:?}", address, prefetched);
    }
    
    // Strided access pattern
    println!("\nTesting strided access pattern:");
    for address in (0..20).step_by(2) {
        let prefetched = prefetcher.access(address);
        println!("Accessed: {}, Prefetched: {:?}", address, prefetched);
    }
    
    // Random access with some repeated patterns
    println!("\nTesting pattern with repetitions:");
    let pattern = vec![1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3];
    for &address in &pattern {
        let prefetched = prefetcher.access(address);
        println!("Accessed: {}, Prefetched: {:?}", address, prefetched);
    }
    
    let (hits, misses, accuracy) = prefetcher.get_stats();
    println!("\nFinal Stats:");
    println!("Hits: {}", hits);
    println!("Misses: {}", misses);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}