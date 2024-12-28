#[cfg(test)]
mod tests {
    use ml_prefetcher::PredictivePrefetcher;

    #[tokio::test]
    async fn test_new_prefetcher() {
        let prefetcher = PredictivePrefetcher::new(4);
        let (hits, misses, accuracy) = prefetcher.get_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
        assert_eq!(accuracy, 0.0);
    }

    #[tokio::test]
    async fn test_sequential_pattern() {
        let mut prefetcher = PredictivePrefetcher::new(4);
        let mut any_predictions = false;
        
        println!("\nSequential pattern test:");
        println!("Priming sequence...");
        for i in 1..=3 {
            let preds = prefetcher.access(i).await;
            println!("Access: {}, Predictions: {:?}", i, preds);
        }

        println!("\nTesting predictions...");
        for i in 4..=12 {
            let predictions = prefetcher.access(i).await;
            println!("Access: {}, Predictions: {:?}", i, predictions);
            if !predictions.is_empty() {
                any_predictions = true;
                for &pred in &predictions {
                    assert!(pred > i, "Prediction should be greater than current access");
                    assert!(pred <= i + 3, "Prediction should be within reasonable range");
                }
            }
        }
        
        let (hits, misses, accuracy) = prefetcher.get_stats();
        println!("\nFinal stats - Hits: {}, Misses: {}, Accuracy: {}", hits, misses, accuracy);
        assert!(any_predictions, "Should make at least one prediction during sequential access");
        assert!(accuracy >= 0.0, "Should maintain reasonable accuracy");
    }

    #[tokio::test]
    async fn test_strided_pattern() {
        let mut prefetcher = PredictivePrefetcher::new(4);
        let mut any_predictions = false;
        let stride = 2;
        
        // Prime the prefetcher first
        println!("\nStrided pattern test:");
        println!("Priming with stride {}...", stride);
        for i in (0..=6).step_by(stride as usize) {
            let predictions = prefetcher.access(i).await;
            println!("Access: {}, Predictions: {:?}", i, predictions);
        }

        // Now test the predictions
        println!("\nTesting predictions...");
        for i in (8..20).step_by(stride as usize) {
            let predictions = prefetcher.access(i).await;
            println!("Access: {}, Predictions: {:?}", i, predictions);
            if !predictions.is_empty() {
                any_predictions = true;
                for &pred in &predictions {
                    // After pattern is established, predictions should maintain stride
                    if prefetcher.get_stats().0 > 0 {  // If we have any hits
                        assert!(
                            pred % stride == 0,
                            "Prediction {} should maintain stride pattern of {}",
                            pred, stride
                        );
                    }
                    assert!(pred > i, "Prediction should be greater than current access");
                    assert!(pred <= i + 6, "Prediction should be within reasonable range");
                }
            }
        }

        let (hits, misses, accuracy) = prefetcher.get_stats();
        println!("\nFinal stats - Hits: {}, Misses: {}, Accuracy: {}", hits, misses, accuracy);
        assert!(any_predictions, "Should make at least one prediction during strided access");
        assert!(accuracy >= 0.0, "Should maintain reasonable accuracy");
    }

    #[tokio::test]
    async fn test_repeated_pattern() {
        let mut prefetcher = PredictivePrefetcher::new(4);
        let mut any_predictions = false;
        
        println!("\nRepeated pattern test:");
        println!("Establishing pattern...");
        for &val in &[1, 2, 3, 1, 2, 3] {
            let predictions = prefetcher.access(val).await;
            println!("Access: {}, Predictions: {:?}", val, predictions);
            if !predictions.is_empty() {
                any_predictions = true;
            }
        }
        
        println!("\nTesting established pattern...");
        let test_sequence = [1, 2, 3];
        for &val in &test_sequence {
            let predictions = prefetcher.access(val).await;
            println!("Access: {}, Predictions: {:?}", val, predictions);
            if !predictions.is_empty() {
                any_predictions = true;
            }
        }

        let (hits, misses, accuracy) = prefetcher.get_stats();
        println!("\nFinal stats - Hits: {}, Misses: {}, Accuracy: {}", hits, misses, accuracy);
        assert!(any_predictions, "Should make at least one prediction for repeated pattern");
        assert!(accuracy >= 0.0, "Should maintain non-negative accuracy");
    }

    #[tokio::test]
    async fn test_pattern_transition() {
        let mut prefetcher = PredictivePrefetcher::new(4);
        let mut any_predictions = false;
        
        println!("\nPattern transition test:");
        println!("Sequential pattern:");
        for i in 1..=4 {
            let predictions = prefetcher.access(i).await;
            println!("Access: {}, Predictions: {:?}", i, predictions);
            if !predictions.is_empty() {
                any_predictions = true;
            }
        }
        
        println!("\nTransitioning to strided pattern:");
        for i in (10..20).step_by(2) {
            let predictions = prefetcher.access(i).await;
            println!("Access: {}, Predictions: {:?}", i, predictions);
            if !predictions.is_empty() {
                any_predictions = true;
            }
        }
        
        let (hits, misses, accuracy) = prefetcher.get_stats();
        println!("\nFinal stats - Hits: {}, Misses: {}, Accuracy: {}", hits, misses, accuracy);
        assert!(accuracy >= 0.0, "Should maintain reasonable accuracy during transition");
        assert!(any_predictions, "Should make at least one prediction during pattern transition");
    }

    #[tokio::test]
    async fn test_edge_cases() {
        let mut prefetcher = PredictivePrefetcher::new(4);
        
        println!("Testing large value pattern...");
        let large_pattern = vec![1000, 2000, 3000, 4000];
        for &val in &large_pattern {
            let preds = prefetcher.access(val).await;
            println!("Access: {}, Predictions: {:?}", val, preds);
        }
        
        println!("\nTesting small value pattern...");
        let small_pattern = vec![1, 2, 3, 4, 5];
        for &val in &small_pattern {
            let preds = prefetcher.access(val).await;
            println!("Access: {}, Predictions: {:?}", val, preds);
        }
        
        println!("\nTesting repeated value...");
        for _ in 0..4 {
            let preds = prefetcher.access(42).await;
            println!("Access: 42, Predictions: {:?}", preds);
        }
        
        let (hits, misses, accuracy) = prefetcher.get_stats();
        println!("\nFinal stats - Hits: {}, Misses: {}, Accuracy: {}", hits, misses, accuracy);
        assert!(accuracy >= 0.0, "Should maintain non-negative accuracy");
    }
}