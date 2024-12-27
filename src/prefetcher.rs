
use std::collections::{HashMap, VecDeque};
use rand::Rng;

#[derive(Debug, Clone, PartialEq)]
enum PatternType {
    Sequential,
    Strided,
    Repeated,
    Unknown
}

#[derive(Clone, Debug)]
struct AccessPattern {
    last_n_addresses: VecDeque<usize>,
    stride: i32,
    frequency: u32,
    confidence: f64,
    pattern_type: PatternType,
    repeating_length: usize,
}

#[derive(Debug)]
struct Perceptron {
    weights: Vec<f64>,
    learning_rate: f64,
    threshold: f64,
}

impl Perceptron {
    fn new(feature_count: usize) -> Self {
        let mut rng = rand::thread_rng();
        Perceptron {
            weights: (0..feature_count).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            learning_rate: 0.2,  // Increased learning rate
            threshold: -0.2,     // Start with lower threshold to encourage predictions
        }
    }

    fn predict(&self, features: &[f64]) -> (bool, f64) {
        let sum: f64 = features.iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum();
        (sum > self.threshold, sum)
    }

    fn train(&mut self, features: &[f64], label: bool) {
        let (prediction, _) = self.predict(features);
        if prediction != label {
            let update = if label { self.learning_rate } else { -self.learning_rate };
            for (w, x) in self.weights.iter_mut().zip(features.iter()) {
                *w += update * x;
            }
        }
    }
}

#[derive(Debug)]
pub struct PredictivePrefetcher {
    history: VecDeque<usize>,
    pattern_table: HashMap<usize, AccessPattern>,
    model: Perceptron,
    history_size: usize,
    prefetch_queue: VecDeque<(usize, f64)>,
    hits: u32,
    misses: u32,
    last_predictions: Vec<usize>,
    min_confidence: f64,
    training_phase: bool,
}

impl PredictivePrefetcher {
    pub fn new(history_size: usize) -> Self {
        PredictivePrefetcher {
            history: VecDeque::with_capacity(history_size),
            pattern_table: HashMap::new(),
            model: Perceptron::new(history_size + 4),
            history_size,
            prefetch_queue: VecDeque::new(),
            hits: 0,
            misses: 0,
            last_predictions: Vec::new(),
            min_confidence: -0.3,  // Lower initial confidence threshold
            training_phase: true,
        }
    }

    fn detect_repeating_pattern(&self) -> usize {
        if self.history.len() < 3 {
            return 0;
        }

        // Look at last 6 accesses maximum
        let recent: Vec<_> = self.history.iter().rev().take(6).collect();

        // Try pattern lengths from 2 to 3
        for len in 2..=3 {
            if recent.len() < len * 2 {
                continue;
            }

            // Get candidate pattern and previous sequence
            let pattern = &recent[0..len];
            let previous = &recent[len..len*2];
            
            // Check if pattern matches previous sequence
            if pattern.iter().zip(previous.iter()).all(|(a, b)| a == b) {
                // Quick check to ensure it's not just the same number repeated
                if pattern.iter().zip(pattern.iter().skip(1)).any(|(a, b)| a != b) {
                    return len;
                }
            }
        }
        
        0
    }

    fn detect_pattern_type(&self) -> PatternType {
        if self.history.len() < 3 {
            return PatternType::Unknown;
        }

        // Check for repeating patterns first
        if self.detect_repeating_pattern() > 0 {
            return PatternType::Repeated;
        }

        // Get the last few differences
        // Convert to Vec of differences
        let diffs: Vec<i64> = self.history
            .iter()
            .zip(self.history.iter().skip(1))
            .map(|(a, b)| *b as i64 - *a as i64)
            .collect();

        if diffs.len() >= 2 && diffs.windows(2).all(|w| w[0] == w[1]) {
            if diffs[0].abs() == 1 {
                PatternType::Sequential
            } else {
                PatternType::Strided
            }
        } else {
            PatternType::Unknown
        }
    }

    fn extract_features(history_size: usize, pattern: &AccessPattern) -> Vec<f64> {
        let mut features = Vec::with_capacity(history_size + 4);
        
        // Normalize addresses relative to the first address
        if let Some(&base) = pattern.last_n_addresses.front() {
            for &addr in &pattern.last_n_addresses {
                features.push((addr as i32 - base as i32) as f64 / 10.0);
            }
        }
        
        // Add pattern characteristics
        features.push(pattern.stride as f64 / 10.0);
        features.push((pattern.frequency as f64).min(10.0) / 10.0);
        features.push(pattern.confidence);
        
        // Pattern type as feature
        features.push(match pattern.pattern_type {
            PatternType::Sequential => 1.0,
            PatternType::Strided => 0.7,
            PatternType::Repeated => 0.5,
            PatternType::Unknown => 0.0,
        });
        
        features
    }

    fn create_pattern(&self, address: usize) -> AccessPattern {
        let mut pattern = AccessPattern {
            last_n_addresses: self.history.clone(),
            stride: 0,
            frequency: 1,
            confidence: 0.0,
            pattern_type: PatternType::Unknown,
            repeating_length: 0,
        };

        if self.history.len() >= 2 {
            let last_addr = self.history[self.history.len() - 2];
            pattern.stride = (address as i64 - last_addr as i64)
                .max(i32::MIN as i64)
                .min(i32::MAX as i64) as i32;
        }

        pattern.pattern_type = self.detect_pattern_type();
        pattern.repeating_length = self.detect_repeating_pattern();
        pattern
    }

    fn predict_next_addresses(&self, current: usize, pattern: &AccessPattern) -> Vec<usize> {
        let mut predictions = Vec::new();
        
        match pattern.pattern_type {
            PatternType::Repeated => {
                // For repeated patterns, look at the last occurrence of the pattern
                // and predict the next few values that followed it
                if pattern.repeating_length > 0 {
                    let pattern_len = pattern.repeating_length;
                    let history_vec: Vec<_> = self.history.iter().copied().collect();
                    
                    // Find the position in the current pattern
                    let mut pos_in_pattern = 0;
                    for i in 1..=pattern_len {
                        if i <= history_vec.len() 
                           && current == history_vec[history_vec.len() - i] {
                            pos_in_pattern = i;
                            break;
                        }
                    }
                    
                    // If we found our position, predict the next values
                    if pos_in_pattern > 0 {
                        for i in 1..=2 {  // Predict up to 2 next values
                            let predict_idx = history_vec.len() - pos_in_pattern - pattern_len + i;
                            if predict_idx < history_vec.len() {
                                predictions.push(history_vec[predict_idx]);
                            }
                        }
                    }
                }
            },
            PatternType::Sequential | PatternType::Strided => {
                if pattern.stride != 0 {
                    // First prediction
                    if let Some(next) = if pattern.stride > 0 {
                        current.checked_add(pattern.stride as usize)
                    } else {
                        current.checked_sub(pattern.stride.unsigned_abs() as usize)
                    } {
                        if next < usize::MAX / 2 {
                            predictions.push(next);
                            
                            // Second prediction
                            if let Some(next_next) = if pattern.stride > 0 {
                                next.checked_add(pattern.stride as usize)
                            } else {
                                next.checked_sub(pattern.stride.unsigned_abs() as usize)
                            } {
                                if next_next < usize::MAX / 2 {
                                    predictions.push(next_next);
                                }
                            }
                        }
                    }
                }
            },
            PatternType::Unknown => {
                // For unknown patterns, try simple stride prediction if we have consistent recent behavior
                if self.history.len() >= 2 {
                    let last = self.history[self.history.len() - 1];
                    let prev = self.history[self.history.len() - 2];
                    let stride = (last as i64 - prev as i64) as i32;
                    
                    if stride != 0 {
                        if let Some(next) = current.checked_add(stride.unsigned_abs() as usize) {
                            if next < usize::MAX / 2 {
                                predictions.push(next);
                            }
                        }
                    }
                }
            }
        }
        
        predictions
    }

    pub fn access(&mut self, address: usize) -> Vec<usize> {
        self.history.push_back(address);
        if self.history.len() > self.history_size {
            self.history.pop_front();
            self.training_phase = false;  // Exit training phase after filling history
        }

        let mut pattern = if let Some(existing) = self.pattern_table.get(&address) {
            let mut updated = existing.clone();
            updated.frequency += 1;
            updated.pattern_type = self.detect_pattern_type();
            updated.repeating_length = self.detect_repeating_pattern();
            updated
        } else {
            self.create_pattern(address)
        };

        // Adjust confidence thresholds based on pattern type and training phase
        // Adjust confidence thresholds based on pattern type and history
        let base_confidence = if self.training_phase { -0.4 } else { -0.3 };
        
        self.min_confidence = match pattern.pattern_type {
            PatternType::Sequential => base_confidence - 0.1,
            PatternType::Strided => {
                if pattern.frequency > 3 {
                    base_confidence - 0.1  // More confident after seeing pattern multiple times
                } else {
                    base_confidence
                }
            },
            PatternType::Repeated => {
                if pattern.repeating_length > 0 && pattern.frequency > 2 {
                    base_confidence - 0.2  // More confident for clear repetitions
                } else {
                    base_confidence
                }
            },
            PatternType::Unknown => base_confidence + 0.2,  // More conservative for unknown patterns
        };

        let mut prefetch_addresses = Vec::new();

        if self.history.len() >= 3 {  // Reduced minimum history requirement
            let features = Self::extract_features(self.history_size, &pattern);
            let (should_prefetch, confidence) = self.model.predict(&features);
            
            pattern.confidence = confidence;

            if should_prefetch && confidence > self.min_confidence {
                let predictions = self.predict_next_addresses(address, &pattern);
                for &next_addr in &predictions {
                    if !self.last_predictions.contains(&next_addr) {
                        prefetch_addresses.push(next_addr);
                        self.prefetch_queue.push_back((next_addr, confidence * 0.9));
                        self.last_predictions.push(next_addr);
                    }
                }

                // Keep last predictions list manageable
                while self.last_predictions.len() > self.history_size * 2 {
                    self.last_predictions.remove(0);
                }
            }
        }

        // Clean up old predictions
        while self.last_predictions.len() > self.history_size * 2 {
            self.last_predictions.remove(0);
        }

        // Update learning based on prediction accuracy
        if self.last_predictions.contains(&address) {
            self.hits += 1;
            pattern.confidence += 0.2;  // Increased confidence boost
            pattern.confidence = pattern.confidence.min(1.0);
            let features = Self::extract_features(self.history_size, &pattern);
            self.model.train(&features, true);
        } else if let Some((predicted, _)) = self.prefetch_queue.pop_front() {
            if predicted != address {
                self.misses += 1;
                if let Some(pred_pattern) = self.pattern_table.get(&predicted).cloned() {
                    let mut updated_pattern = pred_pattern;
                    updated_pattern.confidence -= 0.1;
                    updated_pattern.confidence = updated_pattern.confidence.max(-1.0);
                    let features = Self::extract_features(self.history_size, &updated_pattern);
                    self.model.train(&features, false);
                    self.pattern_table.insert(predicted, updated_pattern);
                }
            }
        }

        self.pattern_table.insert(address, pattern);
        prefetch_addresses
    }

    pub fn get_stats(&self) -> (u32, u32, f64) {
        let accuracy = if self.hits + self.misses > 0 {
            self.hits as f64 / (self.hits + self.misses) as f64
        } else {
            0.0
        };
        (self.hits, self.misses, accuracy)
    }
    
    #[cfg(test)]
    pub fn get_history(&self) -> &VecDeque<usize> {
        &self.history
    }
}
