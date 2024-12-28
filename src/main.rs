use ml_prefetcher::PredictivePrefetcher;

#[tokio::main]
async fn main() {
    let mut prefetcher = PredictivePrefetcher::new(4);
    
    // Start async predictor
    let mut rx = prefetcher.start_async_predictor().await;
    
    // Spawn task to handle predictions
    let prediction_handler = tokio::spawn(async move {
        while let Some(batch) = rx.recv().await {
            println!("Pattern {:?} ({:.2} confidence): {} -> {:?}", 
                    batch.pattern_type, batch.confidence, 
                    batch.address, batch.predictions);
        }
    });

    // Test sequential pattern
    println!("\nTesting sequential access pattern:");
    for i in 1..10 {
        let predictions = prefetcher.access(i).await;
        println!("Accessed: {}, Prefetched: {:?}", i, predictions);
    }

    // Test strided pattern
    println!("\nTesting strided access pattern:");
    for i in (0..20).step_by(2) {
        let predictions = prefetcher.access(i).await;
        println!("Accessed: {}, Prefetched: {:?}", i, predictions);
    }

    // Test repeated pattern
    println!("\nTesting pattern with repetitions:");
    let pattern = vec![1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3];
    for &i in &pattern {
        let predictions = prefetcher.access(i).await;
        println!("Accessed: {}, Prefetched: {:?}", i, predictions);
    }

    // Print final stats
    let (hits, misses, accuracy) = prefetcher.get_stats();
    println!("\nFinal Stats:");
    println!("Hits: {}", hits);
    println!("Misses: {}", misses);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Clean up
    drop(prefetcher);
    let _ = prediction_handler.await;
}