//! ML Prefetcher: A machine learning based prefetcher for predicting access patterns
//!
//! This crate provides a prefetcher that can learn and predict various types of access patterns:
//! - Sequential patterns (1, 2, 3, 4...)
//! - Strided patterns (2, 4, 6, 8...)
//! - Repeated patterns (1, 2, 3, 1, 2, 3...)
//!
//! # Example
//!
//! ```rust
//! use ml_prefetcher::PredictivePrefetcher;
//!
//! // Create a new prefetcher with history size of 4
//! let mut prefetcher = PredictivePrefetcher::new(4);
//!
//! // Access some addresses and get predictions
//! let predictions = prefetcher.access(1);
//! println!("Predicted next addresses: {:?}", predictions);
//!
//! // Get prediction statistics
//! let (hits, misses, accuracy) = prefetcher.get_stats();
//! println!("Accuracy: {:.2}%", accuracy * 100.0);
//! ```
//!
//! # Features
//!
//! - Pattern detection: Automatically detects sequential, strided, and repeated patterns
//! - Adaptive learning: Adjusts predictions based on pattern stability
//! - Performance tracking: Monitors prediction accuracy
//! - No external dependencies: Pure Rust implementation
//!
//! # Use Cases
//!
//! - Memory prefetching
//! - Cache optimization
//! - I/O prediction
//! - Database query optimization
//! - Content delivery networks
//! - Storage system optimization

mod prefetcher;

pub use prefetcher::PredictivePrefetcher;