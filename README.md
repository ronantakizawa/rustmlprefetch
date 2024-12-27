# ML-Based Predictive Prefetcher

A machine learning-based memory access pattern predictor implemented in Rust. This prefetcher uses machine learning techniques to predict future memory access patterns based on observed access history.

## Features

- Pattern Recognition:
  - Sequential access patterns
  - Strided access patterns
  - Repeated patterns
  - Dynamic pattern transitions
  - Handles random access gracefully

- Learning Capabilities:
  - Online learning through perceptron
  - Adaptive confidence thresholds
  - Pattern-specific prediction strategies
  - Dynamic training phase

- Safety Features:
  - Handles large value ranges
  - Overflow protection
  - Memory-safe implementation
  - Edge case handling

## Quick Start

```bash
# Clone the repository
git clone [your-repo-url]
cd rustprefetcher

# Build the project
cargo build

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture
```

## Usage

```rust
use ml_prefetcher::PredictivePrefetcher;

// Create a new prefetcher with history size of 4
let mut prefetcher = PredictivePrefetcher::new(4);

// Access memory addresses and get predictions
let predictions = prefetcher.access(42);

// Get prediction statistics
let (hits, misses, accuracy) = prefetcher.get_stats();
```

## Pattern Types

The prefetcher recognizes several types of access patterns:

1. Sequential Patterns
   - Consecutive memory accesses (e.g., 1, 2, 3, 4)
   - Common in array traversal

2. Strided Patterns
   - Fixed-interval accesses (e.g., 0, 2, 4, 6)
   - Common in matrix operations

3. Repeated Patterns
   - Recurring sequences (e.g., 1, 2, 3, 1, 2, 3)
   - Common in loop iterations

## Performance

Based on test results, the prefetcher achieves:
- Up to 90% accuracy for sequential patterns
- 75-87.5% accuracy for strided patterns
- Adaptive learning for pattern transitions
- Graceful handling of random access

## Testing

The project includes comprehensive tests for:
- Basic functionality
- Pattern recognition
- Training phase behavior
- Edge cases
- Pattern transitions
- Random access handling

Run tests with output to see detailed behavior:
```bash
cargo test -- --nocapture
```

## Implementation Details

The prefetcher uses:
- Perceptron-based learning
- Dynamic confidence thresholds
- Pattern-specific optimizations
- History-based prediction
- Adaptive learning rates

## Safety and Edge Cases

The implementation handles:
- Large memory addresses
- Overflow conditions
- Pattern transitions
- Random access sequences
- Training phase adaptation



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.