use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: String,
    pub prompt: String,
    pub status: TaskStatus,
    pub output_path: Option<PathBuf>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    Queued,
    Processing,
    Completed,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Queued => write!(f, "queued"),
            TaskStatus::Processing => write!(f, "processing"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
        }
    }
}

pub struct GenerationJob {
    pub task_id: String,
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
}

pub struct AppState {
    pub tasks: Arc<RwLock<HashMap<String, TaskInfo>>>,
    pub job_sender: mpsc::UnboundedSender<GenerationJob>,
    pub models_loaded: Arc<AtomicBool>,
    pub backend_name: String,
    pub output_dir: PathBuf,
    // Settings
    pub num_steps: Arc<AtomicUsize>,
    pub seed: Arc<AtomicU64>,
    pub use_seed: Arc<AtomicBool>,
    pub low_memory: Arc<AtomicBool>,
}

impl AppState {
    pub fn queue_depth(&self) -> usize {
        let tasks = self.tasks.read().unwrap();
        tasks
            .values()
            .filter(|t| t.status == TaskStatus::Queued || t.status == TaskStatus::Processing)
            .count()
    }

    pub fn get_steps(&self) -> usize {
        self.num_steps.load(Ordering::Relaxed)
    }

    pub fn get_seed(&self) -> Option<u64> {
        if self.use_seed.load(Ordering::Relaxed) {
            Some(self.seed.load(Ordering::Relaxed))
        } else {
            None
        }
    }

    pub fn set_steps(&self, steps: usize) {
        let steps = steps.clamp(1, 50);
        self.num_steps.store(steps, Ordering::Relaxed);
    }

    pub fn set_seed(&self, seed: Option<u64>) {
        match seed {
            Some(s) => {
                self.seed.store(s, Ordering::Relaxed);
                self.use_seed.store(true, Ordering::Relaxed);
            }
            None => {
                self.use_seed.store(false, Ordering::Relaxed);
            }
        }
    }

    pub fn set_low_memory(&self, enabled: bool) {
        self.low_memory.store(enabled, Ordering::Relaxed);
    }

    pub fn get_low_memory(&self) -> bool {
        self.low_memory.load(Ordering::Relaxed)
    }
}
