use std::collections::{HashMap, VecDeque};
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternType {
    Sequential,
    Strided,
    Repeated,
    Unknown
}

#[derive(Clone, Debug)]
pub struct AccessPattern {
    pattern_type: PatternType,
    stride: i32,
    frequency: u32,
    confidence: f64,
    window_size: usize,
}

#[derive(Debug)]
pub struct PredictionBatch {
    pub address: i32,
    pub predictions: Vec<i32>,
    pub pattern_type: PatternType,
    pub confidence: f64,
}

pub struct PredictivePrefetcher {
    history: VecDeque<i32>,
    pattern_table: HashMap<i32, AccessPattern>,
    history_size: usize,
    hits: u32,
    misses: u32,
    prediction_tx: Option<mpsc::Sender<PredictionBatch>>,
    min_confidence: f64,
    max_window_size: usize,
}

impl AccessPattern {
    fn new(pattern_type: PatternType, stride: i32, min_confidence: f64) -> Self {
        AccessPattern {
            pattern_type,
            stride,
            frequency: 1,
            confidence: min_confidence,
            window_size: 2,
        }
    }

    fn update(&mut self, was_hit: bool, max_window_size: usize) {
        self.frequency += 1;
        if was_hit {
            self.confidence = (self.confidence * 0.8) + 0.2;
            if self.window_size < max_window_size && self.confidence > 0.5 {
                self.window_size += 1;
            }
        } else {
            self.confidence *= 0.8;
            if self.window_size > 2 {
                self.window_size -= 1;
            }
        }
    }

    fn generate_predictions(&self, address: i32, history: &VecDeque<i32>) -> Vec<i32> {
        let mut predictions = Vec::new();
        
        if self.confidence >= 0.2 {
            match self.pattern_type {
                PatternType::Sequential => {
                    for i in 1..=self.window_size {
                        predictions.push(address + i as i32);
                    }
                },
                PatternType::Strided => {
                    let mut next = address;
                    for _ in 0..self.window_size {
                        next += self.stride;
                        predictions.push(next);
                    }
                },
                PatternType::Repeated => {
                    if let Some(cycle_start) = history.len().checked_sub(self.stride as usize) {
                        for i in 0..self.window_size {
                            if let Some(&next) = history.get(cycle_start + i) {
                                predictions.push(next);
                            }
                        }
                    }
                },
                PatternType::Unknown => {
                    predictions.push(address + 1);
                }
            }
        }
        
        predictions
    }
}

impl PredictivePrefetcher {
    pub fn new(history_size: usize) -> Self {
        Self::with_config(history_size, 0.2, 4)
    }

    pub fn with_config(history_size: usize, min_confidence: f64, max_window_size: usize) -> Self {
        PredictivePrefetcher {
            history: VecDeque::with_capacity(history_size),
            pattern_table: HashMap::new(),
            history_size,
            hits: 0,
            misses: 0,
            prediction_tx: None,
            min_confidence,
            max_window_size,
        }
    }

    pub async fn start_async_predictor(&mut self) -> mpsc::Receiver<PredictionBatch> {
        let (tx, rx) = mpsc::channel(100);
        self.prediction_tx = Some(tx);
        rx
    }

    fn detect_pattern(&self) -> (PatternType, i32) {
        if self.history.len() < 2 {
            return (PatternType::Unknown, 0);
        }

        let vec: Vec<_> = self.history.iter().copied().collect();
        
        // Sequential pattern detection
        let mut sequential_matches = 0;
        for i in 1..vec.len() {
            if vec[i] == vec[i - 1] + 1 {
                sequential_matches += 1;
            }
        }
        if sequential_matches >= (vec.len() - 1) / 2 {
            return (PatternType::Sequential, 1);
        }

        // Strided pattern detection
        let mut stride_matches = HashMap::new();
        for i in 1..vec.len() {
            let stride = vec[i] - vec[i - 1];
            *stride_matches.entry(stride).or_insert(0) += 1;
        }

        if let Some((&stride, &count)) = stride_matches.iter().max_by_key(|&(_, count)| count) {
            if count >= (vec.len() - 1) / 2 && stride != 0 {
                return (PatternType::Strided, stride);
            }
        }

        // Repeated pattern detection
        if vec.len() >= 3 {
            for len in 2..=vec.len()/2 {
                let mut matches = 0;
                let mut possible = 0;
                for i in 0..vec.len() - len {
                    if vec[i] == vec[i + len] {
                        matches += 1;
                    }
                    possible += 1;
                }
                if possible > 0 && matches as f64 / possible as f64 > 0.5 {
                    return (PatternType::Repeated, len as i32);
                }
            }
        }

        (PatternType::Unknown, 0)
    }

    pub async fn access(&mut self, address: i32) -> Vec<i32> {
        // Check if current access was predicted
        let was_hit = if let Some(prev_addr) = self.history.back() {
            if let Some(pattern) = self.pattern_table.get(prev_addr) {
                let prev_predictions = pattern.generate_predictions(*prev_addr, &self.history);
                prev_predictions.contains(&address)
            } else {
                false
            }
        } else {
            false
        };

        // Update hits/misses
        if was_hit {
            self.hits += 1;
        } else if !self.pattern_table.is_empty() {
            self.misses += 1;
        }

        // Update history
        self.history.push_back(address);
        if self.history.len() > self.history_size {
            self.history.pop_front();
        }

        // Detect pattern and create new pattern
        let (pattern_type, stride) = self.detect_pattern();
        let new_pattern = AccessPattern::new(pattern_type.clone(), stride, self.min_confidence);
        let predictions = new_pattern.generate_predictions(address, &self.history);

        // Update pattern table
        if let Some(pattern) = self.pattern_table.get_mut(&address) {
            pattern.update(was_hit, self.max_window_size);
        } else {
            self.pattern_table.insert(address, new_pattern);
        }

        // Send async predictions if configured
        if let Some(tx) = &self.prediction_tx {
            let confidence = self.pattern_table.get(&address).map_or(0.0, |p| p.confidence);
            let batch = PredictionBatch {
                address,
                predictions: predictions.clone(),
                pattern_type,
                confidence,
            };
            let _ = tx.send(batch).await;
        }

        predictions
    }

    pub fn get_stats(&self) -> (u32, u32, f64) {
        let accuracy = if self.hits + self.misses > 0 {
            self.hits as f64 / (self.hits + self.misses) as f64
        } else {
            0.0
        };
        (self.hits, self.misses, accuracy)
    }
}