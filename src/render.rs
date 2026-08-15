//! Offline rendering of the visualizer to a video file.
//!
//! The same shader and audio analysis that drive the live view are run on a private
//! headless wgpu device (so the GUI keeps its own device to itself), each frame is
//! read back to the CPU, and the raw RGBA bytes are piped into ffmpeg, which muxes
//! them with the *original* audio file into an H.264 MP4. Frame N is rendered for
//! audio time N / fps, so the result is deterministic and stays in sync regardless
//! of how fast the machine can actually render.
//!
//! Throughput notes (measured on a GTX 1070 at 1080p60): the shader plus readback
//! costs ~6 ms/frame, while x264 `-preset medium` swallows ~15 ms/frame. So the loop
//! keeps `FRAMES_IN_FLIGHT` frames queued on the GPU instead of rendering one and
//! stalling on it, and defaults to a hardware encoder when one is available — the
//! encoder is otherwise the wall everything else waits on.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock};

use eframe::egui;
use eframe::egui_wgpu::wgpu;

// Non-sRGB, to match the format egui_wgpu picks for the window surface: the shader
// output is written to the target verbatim in both paths, so the video's pixels are
// the same bytes the live view shows.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// How many frames may be queued on the GPU before the CPU waits for the oldest one.
// Deeper queues stop paying off once the encoder is the bottleneck, and each slot
// costs a full readback buffer (33 MB at 4K).
const FRAMES_IN_FLIGHT: usize = 3;

/// Which ffmpeg encoder to hand the frames to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Encoder {
    /// Hardware if one is available and actually works, else x264.
    Auto,
    Gpu,
    X264,
}

/// x264 speed/quality tradeoff. Ignored by hardware encoders.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Speed {
    Fast,
    Balanced,
    Best,
}

impl Speed {
    fn preset(self) -> &'static str {
        match self {
            Self::Fast => "veryfast",
            Self::Balanced => "medium",
            Self::Best => "slow",
        }
    }
}

pub struct RenderRequest {
    pub audio_path: PathBuf,
    pub output_path: PathBuf,
    pub samples: Arc<Vec<f32>>,
    pub sample_rate: u32,
    pub channels: u16,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub crf: u32,
    pub encoder: Encoder,
    pub speed: Speed,
    // Snapshot of the visual settings taken when the render was started, so that
    // fiddling with the live controls mid-render doesn't change the video.
    pub fft_enabled: bool,
    pub gain: f32,
    pub wave_window_ms: f32,
    pub trigger_mode: bool,
}

pub enum RenderMsg {
    Progress { frame: u32, total: u32 },
    Finished(PathBuf),
    Canceled(PathBuf),
    Failed(String),
}

/// Kick off a render on a background thread. Progress arrives on `tx`; setting
/// `cancel` stops the render at the next frame boundary.
pub fn spawn(req: RenderRequest, tx: Sender<RenderMsg>, cancel: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let msg = match render(&req, &tx, &cancel) {
            Ok(true) => RenderMsg::Finished(req.output_path.clone()),
            Ok(false) => RenderMsg::Canceled(req.output_path.clone()),
            Err(e) => RenderMsg::Failed(e.to_string()),
        };
        let _ = tx.send(msg);
    });
}

/// `<audio dir>/<audio stem>_visualizer.mp4`
pub fn default_output_path(audio_path: &Path) -> PathBuf {
    let stem = audio_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "visualizer".to_string());
    let dir = audio_path.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!("{stem}_visualizer.mp4"))
}

/// Whether an `ffmpeg` binary can be found on PATH.
pub fn ffmpeg_available() -> bool {
    command("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

struct HwEncoder {
    /// ffmpeg encoder name.
    name: &'static str,
    /// Added to the requested CRF to get this encoder's quality parameter. The
    /// hardware scales run "hotter" than x264's CRF: measured against a lossless
    /// render of this shader, NVENC at cq 28 lands on the same SSIM *and* the same
    /// file size as x264 at crf 18, while cq 18 is 4x the size for no visible gain.
    /// Only NVENC has been measured here; the others stay at parity.
    quality_offset: u32,
    /// Rate-control flags, with `{q}` replaced by the adjusted quality value.
    flags: &'static [&'static str],
}

// Vendor hardware encoders, in the order we'd like to use them.
const HW_ENCODERS: &[HwEncoder] = &[
    HwEncoder {
        name: "h264_nvenc",
        quality_offset: 10,
        flags: &[
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "{q}",
            "-b:v",
            "0",
            "-profile:v",
            "high",
        ],
    },
    HwEncoder {
        name: "h264_qsv",
        quality_offset: 0,
        flags: &["-preset", "medium", "-global_quality", "{q}"],
    },
    HwEncoder {
        name: "h264_amf",
        quality_offset: 0,
        flags: &[
            "-quality", "balanced", "-rc", "cqp", "-qp_i", "{q}", "-qp_p", "{q}",
        ],
    },
];

/// The hardware encoder to use, if any. Probed once by actually encoding a few
/// frames: ffmpeg builds list encoders for hardware the machine may not have.
pub fn hw_encoder() -> Option<&'static str> {
    static DETECTED: OnceLock<Option<&'static str>> = OnceLock::new();
    *DETECTED.get_or_init(|| {
        HW_ENCODERS.iter().map(|e| e.name).find(|name| {
            command("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error"])
                .args(["-f", "lavfi", "-i", "color=black:s=256x256:d=0.1:r=25"])
                .args(["-c:v", name, "-f", "null", "-"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        })
    })
}

// The `-c:v ...` portion of the ffmpeg command line for this request.
fn video_encoder_args(req: &RenderRequest) -> Vec<String> {
    let hw = match req.encoder {
        Encoder::X264 => None,
        Encoder::Auto | Encoder::Gpu => hw_encoder(),
    };

    if let Some(enc) = hw.and_then(|name| HW_ENCODERS.iter().find(|e| e.name == name)) {
        let quality = (req.crf + enc.quality_offset).min(51).to_string();
        let mut args = vec!["-c:v".to_string(), enc.name.to_string()];
        args.extend(enc.flags.iter().map(|f| f.replace("{q}", &quality)));
        return args;
    }

    vec![
        "-c:v".to_string(),
        "libx264".to_string(),
        "-preset".to_string(),
        req.speed.preset().to_string(),
        "-crf".to_string(),
        req.crf.to_string(),
    ]
}

/// Number of video frames a render will produce, for time estimates in the UI.
pub fn frame_count(total_samples: usize, sample_rate: u32, channels: u16, fps: u32) -> u32 {
    let frames = total_samples / (channels.max(1) as usize);
    let duration = frames as f64 / sample_rate.max(1) as f64;
    ((duration * fps as f64).ceil() as u32).max(1)
}

// Child processes inherit the GUI's (absent) console on Windows; without this flag a
// console window flashes up for ffmpeg on every render.
fn command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

type BoxError = Box<dyn std::error::Error>;

// Returns Ok(true) when the video was written, Ok(false) when it was canceled.
fn render(
    req: &RenderRequest,
    tx: &Sender<RenderMsg>,
    cancel: &AtomicBool,
) -> Result<bool, BoxError> {
    let channels = req.channels.max(1) as usize;
    let total = frame_count(req.samples.len(), req.sample_rate, req.channels, req.fps);

    let (device, queue) = create_device()?;
    let gpu = crate::create_visualizer_gpu(&device, TARGET_FORMAT);

    // One render target per in-flight slot: a single one would be overwritten by the
    // next frame while an older frame's copy is still pending.
    let targets: Vec<wgpu::Texture> = (0..FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("offline_render_target_{i}")),
                size: wgpu::Extent3d {
                    width: req.width,
                    height: req.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TARGET_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })
        .collect();
    let target_views: Vec<wgpu::TextureView> = targets
        .iter()
        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();

    // copy_texture_to_buffer requires rows padded to 256 bytes; the padding is
    // stripped again before the frame goes to ffmpeg.
    let unpadded_row = req.width as usize * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_row = unpadded_row.div_ceil(align) * align;
    // One readback buffer per in-flight frame, so the GPU can be working on the
    // next frames while the CPU is blocked feeding the encoder an older one.
    let readback: Vec<wgpu::Buffer> = (0..FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("offline_readback_{i}")),
                size: (padded_row * req.height as usize) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();

    let mut child = command("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        // Video: raw frames on stdin.
        .args(["-f", "rawvideo", "-pixel_format", "rgba"])
        .args(["-video_size", &format!("{}x{}", req.width, req.height)])
        .args(["-framerate", &req.fps.to_string(), "-i", "pipe:0"])
        // Audio: the original file, so it is copied/encoded from the source, not
        // from the samples we decoded for the visualizer.
        .arg("-i")
        .arg(&req.audio_path)
        .args(["-map", "0:v:0", "-map", "1:a:0"])
        .args(video_encoder_args(req))
        .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        .args(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        .arg(&req.output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Couldn't start ffmpeg ({e}). Is it on your PATH?"))?;

    // Drain stderr on its own thread; a full pipe would otherwise deadlock ffmpeg.
    let ffmpeg_log = Arc::new(Mutex::new(String::new()));
    if let Some(mut stderr) = child.stderr.take() {
        let log = ffmpeg_log.clone();
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = stderr.read_to_string(&mut buf);
            if let Ok(mut log) = log.lock() {
                *log = buf;
            }
        });
    }
    let mut stdin = child.stdin.take().ok_or("ffmpeg stdin unavailable")?;

    let mut history = crate::new_history_buffer();
    // Only needed when rows come back padded; otherwise frames go straight from the
    // mapped buffer into the pipe.
    let rows_padded = padded_row != unpadded_row;
    let mut frame_bytes = if rows_padded {
        vec![0u8; unpadded_row * req.height as usize]
    } else {
        Vec::new()
    };
    let resolution = egui::vec2(req.width as f32, req.height as f32);
    let mut canceled = false;
    let mut pipe_error: Option<std::io::Error> = None;

    // Frames handed to the GPU but not yet written out, oldest first.
    struct InFlight {
        slot: usize,
        submission: wgpu::SubmissionIndex,
        mapped: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    }
    let mut in_flight: std::collections::VecDeque<InFlight> = std::collections::VecDeque::new();
    let mut written = 0u32;

    // Retire the oldest queued frame: wait for just that submission (newer ones stay
    // in flight on the GPU), then push its pixels into ffmpeg.
    macro_rules! drain_one {
        () => {{
            let f: InFlight = in_flight.pop_front().expect("drained an empty queue");
            device.poll(wgpu::PollType::Wait {
                submission_index: Some(f.submission),
                timeout: None,
            })?;
            f.mapped.recv()??;
            let buffer = &readback[f.slot];
            let result = {
                let view = buffer.get_mapped_range(..);
                if rows_padded {
                    for row in 0..req.height as usize {
                        let src = row * padded_row;
                        let dst = row * unpadded_row;
                        frame_bytes[dst..dst + unpadded_row]
                            .copy_from_slice(&view[src..src + unpadded_row]);
                    }
                    stdin.write_all(&frame_bytes)
                } else {
                    stdin.write_all(&view)
                }
            };
            buffer.unmap();
            written += 1;
            let _ = tx.send(RenderMsg::Progress {
                frame: written,
                total,
            });
            result
        }};
    }

    for frame in 0..total {
        if cancel.load(Ordering::Relaxed) {
            canceled = true;
            break;
        }

        // Wait for a free slot before recording more work.
        if in_flight.len() == FRAMES_IN_FLIGHT
            && let Err(e) = drain_one!()
        {
            pipe_error = Some(e);
            break;
        }
        let slot = frame as usize % FRAMES_IN_FLIGHT;

        let t = frame as f64 / req.fps as f64;
        let audio_frame = (t * req.sample_rate as f64) as usize;
        let window = crate::compute_visualizer_window(
            &req.samples,
            audio_frame,
            channels,
            req.sample_rate,
            req.wave_window_ms,
            req.trigger_mode,
        );
        crate::advance_history(&mut history, &window, req.fft_enabled, req.gain);
        crate::upload_frame(&queue, &gpu, &history, t as f32, resolution);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offline_frame"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("offline_visualizer_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_views[slot],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            crate::draw_visualizer(&mut pass, &gpu);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets[slot],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback[slot],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row as u32),
                    rows_per_image: Some(req.height),
                },
            },
            wgpu::Extent3d {
                width: req.width,
                height: req.height,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit(Some(encoder.finish()));

        // Queued now, resolved once this frame reaches the front of the queue.
        let (map_tx, map_rx) = std::sync::mpsc::channel();
        readback[slot].map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = map_tx.send(r);
        });
        in_flight.push_back(InFlight {
            slot,
            submission,
            mapped: map_rx,
        });
    }

    // Flush whatever is still queued (unless we are bailing out).
    while !in_flight.is_empty() && !canceled && pipe_error.is_none() {
        if let Err(e) = drain_one!() {
            pipe_error = Some(e);
        }
    }

    if canceled {
        // Closing stdin would make ffmpeg finalize the truncated video; kill it
        // instead and leave the partial file for the user to delete.
        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        return Ok(false);
    }

    // Closing stdin signals end-of-stream, then ffmpeg finalizes the container.
    drop(stdin);
    let status = child.wait()?;
    let log = ffmpeg_log
        .lock()
        .map(|l| l.trim().to_string())
        .unwrap_or_default();

    if let Some(e) = pipe_error {
        return Err(if log.is_empty() {
            format!("ffmpeg stopped reading frames: {e}").into()
        } else {
            format!("ffmpeg failed: {log}").into()
        });
    }
    if !status.success() {
        return Err(if log.is_empty() {
            format!("ffmpeg exited with {status}").into()
        } else {
            format!("ffmpeg failed: {log}").into()
        });
    }

    Ok(true)
}

fn create_device() -> Result<(wgpu::Device, wgpu::Queue), BoxError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: crate::get_backend_from_env(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("offline_visualizer_device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))?;
    Ok((device, queue))
}
