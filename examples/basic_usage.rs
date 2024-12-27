use ml_prefetcher::PredictivePrefetcher;

fn main() {
    // Create a prefetcher with history size of 4
    let mut prefetcher = PredictivePrefetcher::new(4);

    println!("Testing sequential pattern...");
    for i in 1..=10 {
        let predictions = prefetcher.access(i);
        println!("Access: {}, Predicted next: {:?}", i, predictions);
    }

    println!("\nTesting strided pattern...");
    for i in (0..20).step_by(2) {
        let predictions = prefetcher.access(i);
        println!("Access: {}, Predicted next: {:?}", i, predictions);
    }

    println!("\nTesting repeated pattern...");
    let pattern = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
    for &addr in &pattern {
        let predictions = prefetcher.access(addr);
        println!("Access: {}, Predicted next: {:?}", addr, predictions);
    }

    let (hits, misses, accuracy) = prefetcher.get_stats();
    println!("\nFinal Stats:");
    println!("Hits: {}", hits);
    println!("Misses: {}", misses);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}