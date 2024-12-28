use ml_prefetcher::PredictivePrefetcher;
use tokio::time::sleep;
use std::time::Duration;

#[tokio::main]
async fn main() {
    // Create prefetcher with optimized configuration
    let mut prefetcher = PredictivePrefetcher::with_config(
        8,      // larger history for better pattern detection
        0.2,    // start predicting early
        4       // max window size
    );
    
    // Start async predictor
    let mut rx = prefetcher.start_async_predictor().await;
    
    // Spawn prediction handler
    let prediction_handler = tokio::spawn(async move {
        while let Some(batch) = rx.recv().await {
            if !batch.predictions.is_empty() {
                println!("Pattern {:?} ({:.2} confidence): {} -> {:?}", 
                    batch.pattern_type,
                    batch.confidence,
                    batch.address,
                    batch.predictions);
            }
            sleep(Duration::from_millis(50)).await;
        }
    });

    // Test sequential pattern
    println!("\n=== Testing Sequential Pattern ===");
    for i in 1..=10 {
        let predictions = prefetcher.access(i).await;
        println!("Access: {}, Predictions: {:?}", i, predictions);
        sleep(Duration::from_millis(100)).await;
    }

    // Test strided pattern
    println!("\n=== Testing Strided Pattern (step = 3) ===");
    for i in (0..30).step_by(3) {
        let predictions = prefetcher.access(i).await;
        println!("Access: {}, Predictions: {:?}", i, predictions);
        sleep(Duration::from_millis(100)).await;
    }

    // Test repeated pattern
    println!("\n=== Testing Repeated Pattern [1,2,3] ===");
    let repeated = [1, 2, 3, 1, 2, 3, 1, 2, 3];
    for &val in &repeated {
        let predictions = prefetcher.access(val).await;
        println!("Access: {}, Predictions: {:?}", val, predictions);
        sleep(Duration::from_millis(100)).await;
    }

    // Test pattern transition
    println!("\n=== Testing Pattern Transition ===");
    // Start with sequential
    for i in 1..=5 {
        let predictions = prefetcher.access(i).await;
        println!("Sequential - Access: {}, Predictions: {:?}", i, predictions);
        sleep(Duration::from_millis(100)).await;
    }
    // Transition to strided
    for i in (10..25).step_by(2) {
        let predictions = prefetcher.access(i).await;
        println!("Strided - Access: {}, Predictions: {:?}", i, predictions);
        sleep(Duration::from_millis(100)).await;
    }

    // Test mixed patterns
    println!("\n=== Testing Mixed Patterns ===");
    let mixed = [1, 2, 3, 10, 20, 30, 1, 2, 3, 5, 6, 7];
    for &val in &mixed {
        let predictions = prefetcher.access(val).await;
        println!("Access: {}, Predictions: {:?}", val, predictions);
        sleep(Duration::from_millis(100)).await;
    }

    // Print final stats
    let (hits, misses, accuracy) = prefetcher.get_stats();
    println!("\n=== Final Statistics ===");
    println!("Hits: {}", hits);
    println!("Misses: {}", misses);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Clean up
    drop(prefetcher);
    let _ = prediction_handler.await;
}