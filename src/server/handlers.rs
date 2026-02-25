use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use super::models::*;
use super::state::*;

/// POST /generate - Submit a new image generation job
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let task_id = uuid::Uuid::new_v4().to_string();

    // Use per-request overrides or fall back to server defaults
    let steps = req.steps.or_else(|| Some(state.get_steps()));
    let seed = req.seed.or_else(|| state.get_seed());

    let job = GenerationJob {
        task_id: task_id.clone(),
        prompt: req.prompt.clone(),
        width: req.width,
        height: req.height,
        steps,
        seed,
    };

    // Register task
    {
        let mut tasks = state.tasks.write().unwrap();
        tasks.insert(
            task_id.clone(),
            TaskInfo {
                task_id: task_id.clone(),
                prompt: req.prompt,
                status: TaskStatus::Queued,
                output_path: None,
                error: None,
            },
        );
    }

    // Send to GPU worker
    if let Err(e) = state.job_sender.send(job) {
        // Worker is down — mark task as failed
        let mut tasks = state.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(&task_id) {
            task.status = TaskStatus::Failed;
            task.error = Some(format!("Failed to queue job: {}", e));
        }
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(GenerateResponse {
                task_id,
                status: "failed".to_string(),
            }),
        );
    }

    (
        StatusCode::ACCEPTED,
        Json(GenerateResponse {
            task_id,
            status: "queued".to_string(),
        }),
    )
}

/// GET /tasks/:id - Poll task status
pub async fn get_task(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> impl IntoResponse {
    let tasks = state.tasks.read().unwrap();
    match tasks.get(&task_id) {
        Some(task) => {
            let image_url = if task.status == TaskStatus::Completed {
                Some(format!("/images/{}", task.task_id))
            } else {
                None
            };

            (
                StatusCode::OK,
                Json(TaskResponse {
                    task_id: task.task_id.clone(),
                    status: task.status.to_string(),
                    prompt: task.prompt.clone(),
                    image_url,
                    error: task.error.clone(),
                }),
            )
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(TaskResponse {
                task_id,
                status: "not_found".to_string(),
                prompt: String::new(),
                image_url: None,
                error: Some("Task not found".to_string()),
            }),
        ),
    }
}

/// GET /images/:id - Serve generated image as PNG
pub async fn get_image(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> impl IntoResponse {
    let tasks = state.tasks.read().unwrap();
    match tasks.get(&task_id) {
        Some(task) if task.status == TaskStatus::Completed => {
            if let Some(ref path) = task.output_path {
                match std::fs::read(path) {
                    Ok(data) => {
                        return (
                            StatusCode::OK,
                            [("content-type", "image/png")],
                            data,
                        )
                            .into_response();
                    }
                    Err(e) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read image: {}", e),
                        )
                            .into_response();
                    }
                }
            }
            (StatusCode::NOT_FOUND, "Image file not found").into_response()
        }
        Some(task) => (
            StatusCode::BAD_REQUEST,
            format!("Task is not completed yet (status: {})", task.status),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "Task not found").into_response(),
    }
}

/// GET /status - Health check / server status
pub async fn status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(StatusResponse {
        models_loaded: state.models_loaded.load(Ordering::Acquire),
        backend: state.backend_name.clone(),
        queue_depth: state.queue_depth(),
    })
}

/// POST /settings - Update generation settings
pub async fn update_settings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SettingsRequest>,
) -> impl IntoResponse {
    if let Some(steps) = req.steps {
        state.set_steps(steps);
    }
    if let Some(seed) = req.seed {
        if seed == 0 {
            state.set_seed(None);
        } else {
            state.set_seed(Some(seed));
        }
    }
    if let Some(low_memory) = req.low_memory {
        state.set_low_memory(low_memory);
    }

    Json(SettingsResponse {
        steps: state.get_steps(),
        seed: state.get_seed(),
        low_memory: state.get_low_memory(),
    })
}
