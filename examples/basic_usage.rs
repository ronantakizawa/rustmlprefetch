use ml_prefetcher::PredictivePrefetcher;

#[tokio::main]
async fn main() {
    // Create prefetcher with custom configuration
    let mut prefetcher = PredictivePrefetcher::with_config(
        4,      // history size
        0.2,    // minimum confidence
        4       // maximum window size
    );
    
    // Start async predictor
    let mut rx = prefetcher.start_async_predictor().await;
    
    // Spawn task to handle async predictions
    let prediction_handler = tokio::spawn(async move {
        while let Some(batch) = rx.recv().await {
            println!("Pattern {:?} ({:.2} confidence) predicts: {} -> {:?}", 
                    batch.pattern_type,
                    batch.confidence,
                    batch.address,
                    batch.predictions);
        }
    });

    println!("1. Testing sequential access pattern:");
    for i in 1..10 {
        let predictions = prefetcher.access(i).await;
        println!("Accessed: {}, Predicted next: {:?}", i, predictions);
    }

    println!("\n2. Testing strided access pattern (step = 2):");
    for i in (0..20).step_by(2) {
        let predictions = prefetcher.access(i).await;
        println!("Accessed: {}, Predicted next: {:?}", i, predictions);
    }

    println!("\n3. Testing repeated pattern [1,2,3]:");
    let pattern = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
    for &i in &pattern {
        let predictions = prefetcher.access(i).await;
        println!("Accessed: {}, Predicted next: {:?}", i, predictions);
    }

    // Get final statistics
    let (hits, misses, accuracy) = prefetcher.get_stats();
    println!("\nFinal Statistics:");
    println!("Hits: {}", hits);
    println!("Misses: {}", misses);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Clean up
    drop(prefetcher);
    let _ = prediction_handler.await;
}