use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_width")]
    pub width: usize,
    #[serde(default = "default_height")]
    pub height: usize,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
}

fn default_width() -> usize {
    512
}

fn default_height() -> usize {
    512
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerateResponse {
    pub task_id: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskResponse {
    pub task_id: String,
    pub status: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusResponse {
    pub models_loaded: bool,
    pub backend: String,
    pub queue_depth: usize,
}

#[derive(Debug, Deserialize)]
pub struct SettingsRequest {
    pub steps: Option<usize>,
    pub seed: Option<u64>,
    pub low_memory: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct SettingsResponse {
    pub steps: usize,
    pub seed: Option<u64>,
    pub low_memory: bool,
}
