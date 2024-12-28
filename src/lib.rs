//! ML Prefetcher: A ready-to-use prefetcher for predicting memory access patterns
//!
//! This crate provides a smart prefetcher that can predict various types of access patterns:
//! - Sequential patterns (1, 2, 3, 4...)
//! - Strided patterns (2, 4, 6, 8...)
//! - Repeated patterns (1, 2, 3, 1, 2, 3...)
//!
//! # Example
//!
//! ```no_run
//! use ml_prefetcher::PredictivePrefetcher;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create a new prefetcher
//!     let mut prefetcher = PredictivePrefetcher::new(4);
//!     
//!     // Make predictions
//!     println!("Sequential pattern test:");
//!     for i in 1..=3 {
//!         let predictions = prefetcher.access(i).await;
//!         println!("Access: {}, Predicted next: {:?}", i, predictions);
//!     }
//!     
//!     // Get accuracy stats
//!     let (hits, misses, accuracy) = prefetcher.get_stats();
//!     println!("Accuracy: {:.2}%", accuracy * 100.0);
//! }
//! ```
//!
//! For advanced usage with custom settings:
//!
//! ```no_run
//! use ml_prefetcher::PredictivePrefetcher;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut prefetcher = PredictivePrefetcher::with_config(
//!         8,    // history size
//!         0.2,  // minimum confidence
//!         4     // maximum window size
//!     );
//!     
//!     // Use with async/await
//!     let predictions = prefetcher.access(1).await;
//!     println!("Access: 1, Predicted: {:?}", predictions);
//! }
//! ```

mod prefetcher;

pub use prefetcher::PredictivePrefetcher;
pub use prefetcher::PatternType;
pub use prefetcher::PredictionBatch;