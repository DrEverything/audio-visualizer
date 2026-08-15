#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod render;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::num::{NonZero, NonZeroU64};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use eframe::egui;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use eframe::egui_wgpu::{self, wgpu};
use rodio::{Decoder, MixerDeviceSink, Player, Source};
use rustfft::{FftPlanner, num_complex::Complex};

pub(crate) fn get_backend_from_env() -> wgpu::Backends {
    if let Ok(backend_str) = std::env::var("WGPU_BACKEND") {
        match backend_str.to_ascii_lowercase().as_str() {
            "vulkan" => wgpu::Backends::VULKAN,
            "dx12" => wgpu::Backends::DX12,
            "metal" => wgpu::Backends::METAL,
            "gl" => wgpu::Backends::GL,
            "webgpu" => wgpu::Backends::BROWSER_WEBGPU,
            _ => wgpu::Backends::PRIMARY,
        }
    } else {
        wgpu::Backends::PRIMARY
    }
}

// Dimensions of the RGBA8 waterfall history texture the shader reads.
pub(crate) const HISTORY_WIDTH: usize = 1024;
pub(crate) const HISTORY_HEIGHT: usize = 256;
pub(crate) const HISTORY_ROW_BYTES: usize = HISTORY_WIDTH * 4;

// GPU objects needed to draw one visualizer frame. Built once per device, so the
// live (egui swapchain) and offline (headless video render) paths stay identical.
pub(crate) struct VisualizerGpu {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    audio_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

// Struct to store persistent WGPU resources used by the shader callback
struct VisualizerRenderResources {
    gpu: VisualizerGpu,
    history_buffer: Vec<u8>, // RGBA8 waterfall history buffer (1024 * 256 * 4 bytes)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VisualizerUniforms {
    u_time: f32,
    u_resolution_x: f32,
    u_resolution_y: f32,
    u_pad: f32,
}

// The 2048-point plan is reused per thread; planning it every frame (as the naive
// version did) reallocates twiddle tables thousands of times during a video render.
thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

// Scroll the waterfall history down one row and write the current frame's analysis
// into row 0. `samples` is the 2048-sample visualizer window.
pub(crate) fn advance_history(history: &mut [u8], samples: &[f32], fft_enabled: bool, gain: f32) {
    let row_size = HISTORY_ROW_BYTES;
    // Shift history down by 1 row (row 0 moves to row 1, etc.)
    history.copy_within(0..row_size * (HISTORY_HEIGHT - 1), row_size);

    let mut new_row = vec![0u8; row_size];

    // Temporal decay factor for the R (terrain) channel, mimicking Web Audio
    // AnalyserNode smoothing (Shadertoy's audio source). Values rise instantly but
    // decay over ~15 frames. Without this, a one-frame transient occupies a single
    // history row: a paper-thin tall wall in the 3D terrain that the raymarcher
    // steps over stochastically, rendering as speckled "ghost" streaks receding
    // into the distance (worst on the left, where the loud bass bins live).
    const DECAY: f32 = (0.82 / 15.) * 17.;

    if fft_enabled {
        // 1. Run 2048-point Forward FFT with a Hann window. Web Audio applies a
        //    window (Blackman) before its FFT; a raw rectangular window leaks
        //    -13 dB sidelobes around loud bass bins that flicker frame-to-frame
        //    as shimmering noise on the left side of the spectrum.
        let fft = FFT_PLANNER.with(|planner| planner.borrow_mut().plan_fft_forward(2048));

        let n = samples.len() as f32;
        let mut fft_buffer: Vec<Complex<f32>> = samples
            .iter()
            .enumerate()
            .map(|(idx, &s)| {
                let w = 0.5 - 0.5 * (std::f32::consts::TAU * idx as f32 / (n - 1.0)).cos();
                Complex { re: s * w, im: 0.0 }
            })
            .collect();

        fft.process(&mut fft_buffer);

        // 2. Map FFT bins to the new row
        for i in 0..HISTORY_WIDTH {
            let c = fft_buffer[i];
            let magnitude = c.norm();

            // High-frequency pre-emphasis: boost higher frequency bins for visualization
            let freq_boost = 1.0 + (i as f32 / 128.0);

            // Apply normalized gain and scale. Divide by 1024 (not 2048) to
            // compensate the Hann window's 0.5 coherent gain. tanh soft-limits
            // instead of a hard clamp so loud bins don't flat-top into cliffs.
            let val = (magnitude / 1024.0 * gain * freq_boost).sqrt().tanh();
            // Temporal smoothing: rise instantly, decay slowly (peak-hold style),
            // reading the previous frame's row (now shifted to row 1).
            let prev = history[row_size + i * 4] as f32 / 255.0;
            let val = val.max(prev * DECAY);
            new_row[i * 4] = (val * 255.0) as u8; // R: FFT magnitude, 0..1
            let s = samples[i * 2];
            new_row[i * 4 + 1] = ((s * 0.5 + 0.5) * 255.0) as u8; // G: waveform centered at 0.5
            let raw = (magnitude / 1024.0 * gain).sqrt().clamp(0.0, 1.0);
            new_row[i * 4 + 2] = (raw * 255.0) as u8; // B: raw magnitude
            new_row[i * 4 + 3] = 255;
        }
    } else {
        // Waveform Only Mode: Red contains the waveform too so the terrain reacts to it!
        for i in 0..HISTORY_WIDTH {
            let s = samples[i * 2];
            // Gain is applied here too, but the slider caps it low in this mode
            // (see UI) since the raw wave clips into noise past ~1.5.
            let v = (s.abs() * gain).clamp(0.0, 1.0);
            // Same temporal smoothing as FFT mode (see DECAY above) so transient
            // peaks form sloped ridges in the history instead of thin walls.
            let prev = history[row_size + i * 4] as f32 / 255.0;
            let v = v.max(prev * DECAY);
            let byte = (v * 255.0) as u8;
            new_row[i * 4] = byte; // R drives terrain (rectified-ish via 0.5 center)
            new_row[i * 4 + 1] = byte;
            new_row[i * 4 + 2] = byte;
            new_row[i * 4 + 3] = 255;
        }
    }

    // Copy new row to the start of the history buffer
    history[0..row_size].copy_from_slice(&new_row);
}

// A history buffer primed with the "silence" baseline the shader expects.
pub(crate) fn new_history_buffer() -> Vec<u8> {
    let mut history_buffer = vec![0u8; HISTORY_WIDTH * HISTORY_HEIGHT * 4];
    for y in 0..HISTORY_HEIGHT {
        for x in 0..HISTORY_WIDTH {
            let idx = (y * HISTORY_WIDTH + x) * 4;
            history_buffer[idx] = 0; // Red (silent baseline)
            history_buffer[idx + 1] = 128; // Green (silent baseline)
            history_buffer[idx + 2] = 0; // Blue (silent baseline)
            history_buffer[idx + 3] = 255; // Alpha (opaque Snorm)
        }
    }
    history_buffer
}

// Upload this frame's history texture and uniforms.
pub(crate) fn upload_frame(
    queue: &wgpu::Queue,
    gpu: &VisualizerGpu,
    history: &[u8],
    time: f32,
    resolution: egui::Vec2,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu.audio_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        history,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(HISTORY_ROW_BYTES as u32),
            rows_per_image: Some(HISTORY_HEIGHT as u32),
        },
        wgpu::Extent3d {
            width: HISTORY_WIDTH as u32,
            height: HISTORY_HEIGHT as u32,
            depth_or_array_layers: 1,
        },
    );

    let uniforms = VisualizerUniforms {
        u_time: time,
        u_resolution_x: resolution.x,
        u_resolution_y: resolution.y,
        u_pad: 0.0,
    };
    queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
}

// Build the shader pipeline, audio texture and bind group for a given device and
// render target format.
pub(crate) fn create_visualizer_gpu(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> VisualizerGpu {
    // Compile Shader Module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("visualizer_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("visualizer.wgsl").into()),
    });

    // Create uniform buffer (16 bytes aligned)
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("visualizer_uniforms"),
        contents: bytemuck::cast_slice(&[0.0_f32; 4]),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    });

    // Create audio texture (1024x256, Rgba8Unorm format)
    let audio_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("audio_texture"),
        size: wgpu::Extent3d {
            width: HISTORY_WIDTH as u32,
            height: HISTORY_HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let audio_texture_view = audio_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Create linear clamped sampler matching Shadertoy audio channel settings
    // (music channels default to filter=linear, wrap=clamp). Repeat would bleed the
    // right edge of the texture into the left edge via linear filtering.
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("audio_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    // Bind Group Layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("visualizer_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // Pipeline Layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("visualizer_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    // Render Pipeline
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("visualizer_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(target_format.into())],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // Bind Group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("visualizer_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&audio_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    VisualizerGpu {
        pipeline,
        uniform_buffer,
        audio_texture,
        bind_group,
    }
}

// Record the fullscreen visualizer draw into an existing render pass.
pub(crate) fn draw_visualizer(render_pass: &mut wgpu::RenderPass<'_>, gpu: &VisualizerGpu) {
    render_pass.set_pipeline(&gpu.pipeline);
    render_pass.set_bind_group(0, &gpu.bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

enum AudioMessage {
    Decoded {
        file_name: String,
        file_path: PathBuf,
        samples: Arc<Vec<f32>>,
        sample_rate: u32,
        channels: u16,
        playback_pos: Arc<AtomicUsize>,
    },
    Error(String),
}

// Custom Rodio Source that tracks the current playback position
struct VisualizerSource {
    samples: Arc<Vec<f32>>,
    pos: usize,
    sample_rate: u32,
    channels: u16,
    shared_pos: Arc<AtomicUsize>,
}

impl Iterator for VisualizerSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.samples.len() {
            let sample = self.samples[self.pos];
            self.pos += 1;
            // Update current play index
            self.shared_pos.store(self.pos, Ordering::Relaxed);
            Some(sample)
        } else {
            None
        }
    }
}

impl Source for VisualizerSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> NonZero<u16> {
        NonZero::new(self.channels).unwrap()
    }

    fn sample_rate(&self) -> NonZero<u32> {
        NonZero::new(self.sample_rate).unwrap()
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        let total_samples = self.samples.len() as f64;
        let secs = total_samples / (self.sample_rate as f64 * self.channels as f64);
        Some(std::time::Duration::from_secs_f64(secs))
    }
}

struct VisualizerCallback {
    time: f32,
    resolution: egui::Vec2,
    samples: Vec<f32>,
    fft_enabled: bool,
    gain: f32,
}

impl egui_wgpu::CallbackTrait for VisualizerCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get_mut::<VisualizerRenderResources>() {
            advance_history(
                &mut res.history_buffer,
                &self.samples,
                self.fft_enabled,
                self.gain,
            );
            upload_frame(
                queue,
                &res.gpu,
                &res.history_buffer,
                self.time,
                self.resolution,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(res) = resources.get::<VisualizerRenderResources>() {
            // Set dynamic viewport to match paint area exactly to prevent scaling issues
            let rect = info.viewport;
            render_pass.set_viewport(
                rect.min.x,
                rect.min.y,
                rect.width(),
                rect.height(),
                0.0,
                1.0,
            );
            draw_visualizer(render_pass, &res.gpu);
        }
    }
}

pub struct LeApp {
    // Audio stream & Player management
    _stream: Option<MixerDeviceSink>,
    player: Option<Player>,

    // Live system audio (WASAPI loopback) capture, active only while enabled
    system_audio_enabled: bool,
    system_capture: Option<SystemAudioCapture>,

    // Current playing buffer
    current_file_name: Option<String>,
    current_file_path: Option<PathBuf>,
    samples: Option<Arc<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    playback_pos: Option<Arc<AtomicUsize>>,

    // Playback settings
    volume: f32,
    gain: f32,           // visual gain for FFT mode (0.1..=8.0)
    wave_gain: f32,      // visual gain for raw waveform mode, capped low (0.1..=1.5)
    trigger_mode: bool,  // true = Lock Phase (zero-crossing search), false = continuous raw buffer
    wave_window_ms: f32, // size of window to display in ms (e.g. 5ms to 150ms)
    fft_enabled: bool,   // true = standard fourier transform landscape, false = waveform only

    // Channels to communicate with background decoder thread
    rx: Receiver<AudioMessage>,
    tx: Sender<AudioMessage>,

    // Video export settings & job state
    export: ExportState,

    // UI state
    status_msg: String,
    time_start: Instant,
    is_loading: bool,
    controls_alpha: f32,               // for fading controls in/out on hover
    controls_rect: Option<egui::Rect>, // last-rendered panel rect, for hover detection
}

// Everything the "Export video" window needs: the settings it edits, plus the
// handle to a render running on a background thread.
struct ExportState {
    window_open: bool,
    width: u32,
    height: u32,
    fps: u32,
    crf: u32,
    encoder: render::Encoder,
    speed: render::Speed,
    output_path: String,
    // Cached `ffmpeg -version` probe, refreshed each time the window is opened.
    ffmpeg_ok: Option<bool>,
    // Name of the working hardware encoder, if the probe found one.
    hw_encoder: Option<&'static str>,

    // Active job (None when idle)
    job: Option<ExportJob>,
    // Result of the last finished job, shown until another one starts.
    last_result: Option<String>,
}

struct ExportJob {
    rx: Receiver<render::RenderMsg>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    frame: u32,
    total: u32,
    started: Instant,
}

impl LeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
        let device = &wgpu_render_state.device;

        let gpu = create_visualizer_gpu(device, wgpu_render_state.target_format);

        // Insert resources so the callback can retrieve them later
        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(VisualizerRenderResources {
                gpu,
                history_buffer: new_history_buffer(),
            });

        // Try to setup Audio device using new rodio 0.22 API
        let stream = match rodio::stream::DeviceSinkBuilder::open_default_sink() {
            Ok(s) => Some(s),
            Err(e) => {
                log::warn!("Audio device not found/ready: {}", e);
                None
            }
        };

        let player = if let Some(ref s) = stream {
            let mixer = s.mixer();
            Some(Player::connect_new(&mixer))
        } else {
            None
        };

        let (tx, rx) = channel();

        // Load persisted settings if available
        let volume = cc
            .storage
            .and_then(|s| eframe::get_value(s, "volume"))
            .unwrap_or(0.5);
        let gain = cc
            .storage
            .and_then(|s| eframe::get_value(s, "gain"))
            .unwrap_or(4.0);
        let wave_gain = cc
            .storage
            .and_then(|s| eframe::get_value(s, "wave_gain"))
            .unwrap_or(1.0);
        let trigger_mode = cc
            .storage
            .and_then(|s| eframe::get_value(s, "trigger_mode"))
            .unwrap_or(true);
        let wave_window_ms = cc
            .storage
            .and_then(|s| eframe::get_value(s, "wave_window_ms"))
            .unwrap_or(70.0);
        let fft_enabled = cc
            .storage
            .and_then(|s| eframe::get_value(s, "fft_enabled"))
            .unwrap_or(false);
        let system_audio_enabled = cc
            .storage
            .and_then(|s| eframe::get_value(s, "system_audio_enabled"))
            .unwrap_or(true);
        let export = ExportState {
            window_open: false,
            width: cc
                .storage
                .and_then(|s| eframe::get_value(s, "export_width"))
                .unwrap_or(1920),
            height: cc
                .storage
                .and_then(|s| eframe::get_value(s, "export_height"))
                .unwrap_or(1080),
            fps: cc
                .storage
                .and_then(|s| eframe::get_value(s, "export_fps"))
                .unwrap_or(60),
            crf: cc
                .storage
                .and_then(|s| eframe::get_value(s, "export_crf"))
                .unwrap_or(18),
            // Stored as small ints: the render enums aren't serde types.
            encoder: match cc
                .storage
                .and_then(|s| eframe::get_value::<u8>(s, "export_encoder"))
            {
                Some(1) => render::Encoder::Gpu,
                Some(2) => render::Encoder::X264,
                _ => render::Encoder::Auto,
            },
            speed: match cc
                .storage
                .and_then(|s| eframe::get_value::<u8>(s, "export_speed"))
            {
                Some(0) => render::Speed::Fast,
                Some(2) => render::Speed::Best,
                _ => render::Speed::Balanced,
            },
            output_path: String::new(),
            ffmpeg_ok: None,
            hw_encoder: None,
            job: None,
            last_result: None,
        };

        let system_capture;
        let status_msg;
        if system_audio_enabled {
            match SystemAudioCapture::new() {
                Ok(cap) => {
                    // Pause file playback so the two sources don't
                    // fight over the visualizer (and the speakers).
                    if let Some(ref player) = player {
                        player.pause();
                    }
                    status_msg = "Visualizing system audio (whatever is \
                                                     playing on your PC)"
                        .to_string();
                    system_capture = Some(cap);
                }
                Err(e) => {
                    system_capture = None;
                    status_msg = format!("Couldn't capture system audio: {}", e);
                }
            }
        } else {
            system_capture = None;
            status_msg = "Stopped system audio capture.".to_string();
        }

        Some(Self {
            _stream: stream,
            player,
            system_audio_enabled,
            system_capture,
            current_file_name: None,
            current_file_path: None,
            samples: None,
            sample_rate: 44100,
            channels: 2,
            playback_pos: None,
            volume,
            gain,
            wave_gain,
            trigger_mode,
            wave_window_ms,
            fft_enabled,
            export,
            rx,
            tx,
            status_msg,
            time_start: Instant::now(),
            is_loading: false,
            controls_alpha: 0.0,
            controls_rect: None,
        })
    }
}

fn decode_audio_file(
    path: &std::path::Path,
) -> Result<(Vec<f32>, u32, u16), Box<dyn std::error::Error + Send + Sync>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let source = Decoder::new(reader)?;

    let sample_rate: u32 = source.sample_rate().get();
    let channels: u16 = source.channels().get();

    // In rodio 0.22, Decoder yields f32 directly
    let samples: Vec<f32> = source.collect();

    Ok((samples, sample_rate, channels))
}

// Live capture of whatever is playing on the system's default output device.
// On Windows this uses WASAPI loopback: cpal opens the *output* device as an
// *input* stream, so we receive a copy of the audio the speakers are playing
// without any virtual audio cable. The captured interleaved f32 samples are kept
// in a bounded ring buffer that the visualizer reads the tail of each frame.
struct SystemAudioCapture {
    ring: Arc<Mutex<VecDeque<f32>>>,
    sample_rate: u32,
    channels: u16,
    _stream: cpal::Stream,
}

impl SystemAudioCapture {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("No default output device found")?;

        // Loopback capture uses the output device's own (output) config.
        let config = device.default_output_config()?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let sample_format = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();

        // ~2 seconds of headroom; the visualizer only reads the most recent window.
        let capacity = (sample_rate as usize) * (channels as usize) * 2;
        let ring = Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));
        let ring_cb = ring.clone();

        let err_fn = |err| log::warn!("System audio loopback stream error: {}", err);

        // WASAPI shared mode almost always hands us f32, but be tolerant of i16/u16.
        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &_| push_samples(&ring_cb, data, capacity),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &_| {
                    let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    push_samples(&ring_cb, &f, capacity);
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::U16 => device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &_| {
                    let f: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    push_samples(&ring_cb, &f, capacity);
                },
                err_fn,
                None,
            )?,
            other => return Err(format!("Unsupported sample format: {:?}", other).into()),
        };

        stream.play()?;

        Ok(Self {
            ring,
            sample_rate,
            channels,
            _stream: stream,
        })
    }

    // Snapshot the current ring buffer contents (oldest -> newest, interleaved).
    fn snapshot(&self) -> Vec<f32> {
        match self.ring.lock() {
            Ok(buf) => buf.iter().copied().collect(),
            Err(_) => Vec::new(),
        }
    }
}

// Push interleaved samples into the ring buffer, dropping the oldest to stay bounded.
fn push_samples(ring: &Arc<Mutex<VecDeque<f32>>>, data: &[f32], capacity: usize) {
    if let Ok(mut buf) = ring.lock() {
        for &s in data {
            if buf.len() >= capacity {
                buf.pop_front();
            }
            buf.push_back(s);
        }
    }
}

// Build the 2048-sample visualizer window from an interleaved sample buffer.
// Shared by file playback (window follows the play head) and live system audio
// (window pinned to the most recent samples, positioned by the caller).
pub(crate) fn compute_visualizer_window(
    samples: &[f32],
    current_frame: usize,
    channels: usize,
    sample_rate: u32,
    wave_window_ms: f32,
    trigger_mode: bool,
) -> Vec<f32> {
    let mut visualizer_samples = vec![0.0f32; 2048];
    let channels = channels.max(1);
    let total_samples = samples.len();
    let total_frames = total_samples / channels;

    // Compute total frames in the zoom window
    let window_frames = ((wave_window_ms / 1000.0) * sample_rate as f32) as usize;
    let window_frames = window_frames.max(32); // at least 32 frames for 2048 window

    let mut start_frame = current_frame;

    if trigger_mode {
        // Stabilize wave phase via Oscilloscope Rising-Edge Zero-Crossing Triggering.
        // We search ahead for a zero-crossing based on mono mixed values.
        let search_len = 1024.min(total_frames.saturating_sub(current_frame));
        for f in 0..search_len {
            let f_idx = current_frame + f;

            let mut v1 = 0.0;
            let mut count = 0;
            for c in 0..channels {
                let idx = f_idx * channels + c;
                if idx < total_samples {
                    v1 += samples[idx];
                    count += 1;
                }
            }
            let mono1 = if count > 0 { v1 / count as f32 } else { 0.0 };

            let mut v2 = 0.0;
            let mut count2 = 0;
            for c in 0..channels {
                let idx = (f_idx + 1) * channels + c;
                if idx < total_samples {
                    v2 += samples[idx];
                    count2 += 1;
                }
            }
            let mono2 = if count2 > 0 { v2 / count2 as f32 } else { 0.0 };

            if mono1 < 0.0 && mono2 >= 0.0 {
                start_frame = f_idx;
                break;
            }
        }
    }

    // Downsample/interpolate the window down to 2048 samples
    for i in 0..2048 {
        let frame_offset = (i * window_frames) / 2048;
        let target_frame = start_frame + frame_offset;

        let mut sum = 0.0;
        let mut count = 0;
        for c in 0..channels {
            let idx = target_frame * channels + c;
            if idx < total_samples {
                sum += samples[idx];
                count += 1;
            }
        }
        visualizer_samples[i] = if count > 0 { sum / count as f32 } else { 0.0 };
    }

    visualizer_samples
}

impl LeApp {
    // Drain progress from a running export and fold the result back into the UI.
    fn poll_export(&mut self) {
        let mut result = None;
        if let Some(job) = &mut self.export.job {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    render::RenderMsg::Progress { frame, total } => {
                        job.frame = frame;
                        job.total = total;
                    }
                    render::RenderMsg::Finished(path) => {
                        result = Some(format!("Exported video to {}", path.display()));
                        break;
                    }
                    render::RenderMsg::Canceled(path) => {
                        result = Some(format!(
                            "Export canceled (incomplete file left at {})",
                            path.display()
                        ));
                        break;
                    }
                    render::RenderMsg::Failed(e) => {
                        result = Some(format!("Export failed: {e}"));
                        break;
                    }
                }
            }
        }
        if let Some(text) = result {
            self.export.job = None;
            self.status_msg = text.clone();
            self.export.last_result = Some(text);
        }
    }

    // Snapshot the current audio + visual settings and hand them to a background render.
    fn start_export(&mut self) {
        let (Some(samples), Some(audio_path)) =
            (self.samples.clone(), self.current_file_path.clone())
        else {
            return;
        };

        let output_path = PathBuf::from(self.export.output_path.trim());
        if output_path.as_os_str().is_empty() {
            self.export.last_result = Some("Set an output file path first.".to_string());
            return;
        }

        // yuv420p needs even dimensions.
        let width = self.export.width & !1;
        let height = self.export.height & !1;
        self.export.width = width;
        self.export.height = height;

        let total = render::frame_count(
            samples.len(),
            self.sample_rate,
            self.channels,
            self.export.fps,
        );
        let (tx, rx) = channel();
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));

        render::spawn(
            render::RenderRequest {
                audio_path,
                output_path: output_path.clone(),
                samples,
                sample_rate: self.sample_rate,
                channels: self.channels,
                width,
                height,
                fps: self.export.fps,
                crf: self.export.crf,
                encoder: self.export.encoder,
                speed: self.export.speed,
                fft_enabled: self.fft_enabled,
                gain: if self.fft_enabled {
                    self.gain
                } else {
                    self.wave_gain
                },
                wave_window_ms: self.wave_window_ms,
                trigger_mode: self.trigger_mode,
            },
            tx,
            cancel.clone(),
        );

        self.export.last_result = None;
        self.export.job = Some(ExportJob {
            rx,
            cancel,
            frame: 0,
            total,
            started: Instant::now(),
        });
        self.status_msg = format!("Rendering video to {}...", output_path.display());
    }

    // Settings + progress window. Lives outside the fading control panel so it
    // stays put while the pointer is anywhere in the window.
    fn export_window(&mut self, ctx: &egui::Context) {
        if !self.export.window_open {
            return;
        }

        let mut open = true;
        egui::Window::new("🎬 Export video")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(470.0)
            .default_pos(ctx.content_rect().center() - egui::vec2(235.0, 170.0))
            .show(ctx, |ui| {
                // Fixed width so long paths/status lines wrap instead of stretching
                // the (auto-sizing) window across the screen.
                ui.set_width(454.0);
                let rendering = self.export.job.is_some();

                if self.export.ffmpeg_ok == Some(false) {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 120, 120),
                        "ffmpeg was not found on your PATH — install it to export video.",
                    );
                    ui.add_space(6.0);
                }

                if let Some(name) = &self.current_file_name {
                    ui.label(
                        egui::RichText::new(format!("Source: {name}"))
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                }
                ui.add_space(6.0);

                ui.add_enabled_ui(!rendering, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Resolution:");
                        ui.add(
                            egui::DragValue::new(&mut self.export.width)
                                .speed(2)
                                .range(320..=3840),
                        );
                        ui.label("×");
                        ui.add(
                            egui::DragValue::new(&mut self.export.height)
                                .speed(2)
                                .range(240..=2160),
                        );
                        for (label, w, h) in [
                            ("720p", 1280, 720),
                            ("1080p", 1920, 1080),
                            ("1440p", 2560, 1440),
                            ("4K", 3840, 2160),
                        ] {
                            if ui.small_button(label).clicked() {
                                self.export.width = w;
                                self.export.height = h;
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Frame rate:");
                        for fps in [24u32, 30, 60] {
                            ui.selectable_value(&mut self.export.fps, fps, format!("{fps} fps"));
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Quality:");
                        ui.style_mut().spacing.slider_width = 160.0;
                        ui.add(
                            egui::Slider::new(&mut self.export.crf, 14..=28)
                                .show_value(true)
                                .text("CRF"),
                        )
                        .on_hover_text("Lower = better quality and bigger file");
                    });

                    // Encoder choice. The GPU encoder is typically 4x faster than
                    // x264 and is what keeps the render GPU-bound rather than
                    // encoder-bound.
                    ui.horizontal(|ui| {
                        ui.label("Encoder:");
                        let hw = self.export.hw_encoder;
                        ui.selectable_value(
                            &mut self.export.encoder,
                            render::Encoder::Auto,
                            "Auto",
                        )
                        .on_hover_text(match hw {
                            Some(name) => format!("Uses {name} (hardware) on this machine"),
                            None => "No hardware encoder found — uses x264".to_string(),
                        });
                        ui.add_enabled_ui(hw.is_some(), |ui| {
                            ui.selectable_value(
                                &mut self.export.encoder,
                                render::Encoder::Gpu,
                                "GPU",
                            )
                            .on_disabled_hover_text("No working hardware encoder was detected");
                        });
                        ui.selectable_value(
                            &mut self.export.encoder,
                            render::Encoder::X264,
                            "CPU (x264)",
                        )
                        .on_hover_text("Slowest, best compression");
                    });

                    // Only x264 uses the preset; hardware encoders ignore it.
                    let cpu_encoding = self.export.encoder == render::Encoder::X264
                        || (self.export.encoder == render::Encoder::Auto
                            && self.export.hw_encoder.is_none());
                    ui.add_enabled_ui(cpu_encoding, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("x264 speed:");
                            for (speed, label) in [
                                (render::Speed::Fast, "Fast"),
                                (render::Speed::Balanced, "Balanced"),
                                (render::Speed::Best, "Best"),
                            ] {
                                ui.selectable_value(&mut self.export.speed, speed, label);
                            }
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label("Save to:");
                        let width = ui.available_width();
                        ui.add(
                            egui::TextEdit::singleline(&mut self.export.output_path)
                                .desired_width(width)
                                .hint_text("C:\\path\\to\\output.mp4"),
                        );
                    });
                });

                let out = PathBuf::from(self.export.output_path.trim());
                if !rendering && !out.as_os_str().is_empty() && out.exists() {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 200, 120),
                        "This file already exists and will be overwritten.",
                    );
                }

                // Length / frame-count estimate.
                if let Some(samples) = &self.samples {
                    let total = render::frame_count(
                        samples.len(),
                        self.sample_rate,
                        self.channels,
                        self.export.fps,
                    );
                    let secs = total as f32 / self.export.fps as f32;
                    ui.label(
                        egui::RichText::new(format!(
                            "{} frames · {:02}:{:02} of video · settings match the live view \
                             ({}, gain {:.1}, {:.0}ms window)",
                            total,
                            (secs / 60.0) as i32,
                            (secs % 60.0) as i32,
                            if self.fft_enabled {
                                "FFT + wave"
                            } else {
                                "waveform"
                            },
                            if self.fft_enabled {
                                self.gain
                            } else {
                                self.wave_gain
                            },
                            self.wave_window_ms,
                        ))
                        .color(egui::Color32::GRAY)
                        .font(egui::FontId::proportional(11.0)),
                    );
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                if let Some(job) = &self.export.job {
                    let frac = if job.total > 0 {
                        job.frame as f32 / job.total as f32
                    } else {
                        0.0
                    };
                    let elapsed = job.started.elapsed().as_secs_f32();
                    let eta = if frac > 0.01 {
                        format!(" · ~{:.0}s left", elapsed / frac - elapsed)
                    } else {
                        String::new()
                    };
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .text(format!("frame {}/{}{}", job.frame, job.total, eta)),
                    );
                    ui.add_space(6.0);
                    if ui.button("Cancel").clicked() {
                        job.cancel.store(true, Ordering::Relaxed);
                    }
                } else {
                    let ready = self.samples.is_some()
                        && self.current_file_path.is_some()
                        && self.export.ffmpeg_ok != Some(false);
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(ready, egui::Button::new("Render MP4"))
                            .clicked()
                        {
                            self.start_export();
                        }
                        if let Some(result) = &self.export.last_result {
                            ui.label(
                                egui::RichText::new(result)
                                    .color(egui::Color32::LIGHT_GRAY)
                                    .font(egui::FontId::proportional(11.0)),
                            );
                        }
                    });
                }
            });

        self.export.window_open = open;
    }
}

impl eframe::App for LeApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx();
        ctx.set_visuals(egui::Visuals::dark());

        // Pick up progress from a video export running in the background
        self.poll_export();

        // Process message from audio decoder thread
        if let Ok(msg) = self.rx.try_recv() {
            self.is_loading = false;
            match msg {
                AudioMessage::Decoded {
                    file_name,
                    file_path,
                    samples,
                    sample_rate,
                    channels,
                    playback_pos,
                } => {
                    self.current_file_name = Some(file_name.clone());
                    // Keep the source path: the video exporter hands it to ffmpeg
                    // as the audio track.
                    self.export.output_path = render::default_output_path(&file_path)
                        .display()
                        .to_string();
                    self.current_file_path = Some(file_path);
                    self.samples = Some(samples.clone());
                    self.sample_rate = sample_rate;
                    self.channels = channels;
                    self.playback_pos = Some(playback_pos.clone());
                    self.status_msg = format!(
                        "Playing: {} ({}Hz, {} channels)",
                        file_name, sample_rate, channels
                    );

                    // Stop previous sound and close stream/sink
                    if let Some(ref player) = self.player {
                        player.stop();
                    }
                    self.player = None;
                    self._stream = None;

                    // Re-create the stream and player with the new sample rate and channels
                    let new_stream = if let (Some(rate), Some(ch)) =
                        (NonZero::new(sample_rate), NonZero::new(channels))
                    {
                        if let Ok(builder) = rodio::stream::DeviceSinkBuilder::from_default_device()
                        {
                            let builder = builder.with_sample_rate(rate).with_channels(ch);
                            builder.open_sink_or_fallback().ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Fallback to default sink if custom configuration failed
                    let new_stream = new_stream
                        .or_else(|| rodio::stream::DeviceSinkBuilder::open_default_sink().ok());

                    if let Some(s) = new_stream {
                        let mixer = s.mixer();
                        let new_player = Player::connect_new(&mixer);
                        new_player.set_volume(self.volume);

                        let source = VisualizerSource {
                            samples: samples.clone(),
                            pos: 0,
                            sample_rate,
                            channels,
                            shared_pos: playback_pos.clone(),
                        };
                        new_player.append(source);
                        new_player.play();

                        self.player = Some(new_player);
                        self._stream = Some(s);
                    } else {
                        self.status_msg =
                            "No audio output device found (Visualizer only mode).".to_string();
                    }
                }
                AudioMessage::Error(err) => {
                    self.status_msg = format!("Decoding error: {}", err);
                }
            }
        }

        // Process dropped file events
        if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
            let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
            for file in dropped_files {
                if let Some(path) = file.path {
                    self.is_loading = true;
                    self.status_msg = format!(
                        "Decoding {}...",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    );

                    let tx = self.tx.clone();
                    std::thread::spawn(move || match decode_audio_file(&path) {
                        Ok((samples, sample_rate, channels)) => {
                            let playback_pos = Arc::new(AtomicUsize::new(0));
                            let _ = tx.send(AudioMessage::Decoded {
                                file_name: path
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .into_owned(),
                                file_path: path.clone(),
                                samples: Arc::new(samples),
                                sample_rate,
                                channels,
                                playback_pos,
                            });
                        }
                        Err(e) => {
                            let _ = tx.send(AudioMessage::Error(e.to_string()));
                        }
                    });
                }
            }
        }

        // Fallback simulation mode if no audio output device is present
        if self.player.is_none() && self.samples.is_some() {
            if let (Some(samples), Some(playback_pos)) = (&self.samples, &self.playback_pos) {
                let dt = ctx.input(|i| i.stable_dt);
                let samples_to_advance =
                    (dt * self.sample_rate as f32 * self.channels as f32) as usize;
                let current = playback_pos.load(Ordering::Relaxed);
                let new_pos = (current + samples_to_advance).min(samples.len());
                playback_pos.store(new_pos, Ordering::Relaxed);
            }
        }

        // Get full UI rect
        let rect = ui.max_rect();
        let time = self.time_start.elapsed().as_secs_f32();

        // Extract and trigger audio samples to write to visualizer (2048 samples window)
        let mut visualizer_samples = vec![0.0f32; 2048];
        if self.system_audio_enabled {
            // Live system audio: window is pinned to the most recent captured samples.
            if let Some(cap) = &self.system_capture {
                let snapshot = cap.snapshot();
                let channels = cap.channels as usize;
                let total_frames = snapshot.len() / channels.max(1);
                let window_frames =
                    ((self.wave_window_ms / 1000.0) * cap.sample_rate as f32) as usize;
                // Leave 1024 frames of headroom at the tail so the trigger search and
                // windowing don't run past the newest captured samples.
                let current_frame = total_frames.saturating_sub(window_frames + 1024);
                visualizer_samples = compute_visualizer_window(
                    &snapshot,
                    current_frame,
                    channels,
                    cap.sample_rate,
                    self.wave_window_ms,
                    self.trigger_mode,
                );
            }
        } else if let (Some(samples), Some(playback_pos)) = (&self.samples, &self.playback_pos) {
            let current_frame =
                playback_pos.load(Ordering::Relaxed) / (self.channels as usize).max(1);
            visualizer_samples = compute_visualizer_window(
                samples,
                current_frame,
                self.channels as usize,
                self.sample_rate,
                self.wave_window_ms,
                self.trigger_mode,
            );
        }

        // Draw visualizer shader. Pass accurate painting rectangle size to ensure pixel scaling matching.
        let paint_rect = rect;
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            paint_rect,
            VisualizerCallback {
                time,
                resolution: paint_rect.size(),
                samples: visualizer_samples,
                fft_enabled: self.fft_enabled,
                gain: if self.fft_enabled {
                    self.gain
                } else {
                    self.wave_gain
                },
            },
        ));

        // Force repaint to animate shader
        ctx.request_repaint();

        // Calculate floating overlay parameters. Width tracks the window so the panel
        // never overflows a narrow window (leaving a 40px margin), capped at 1050px on
        // wide screens. The panel is anchored bottom-center and sizes its own height,
        // so its contents can wrap onto extra rows without being clipped.
        let screen_w = rect.width();
        let control_w = (screen_w - 40.0).clamp(280.0, 1050.0);

        // Show controls when the pointer is near the bottom of the window, or hovering
        // the panel itself (using last frame's measured rect so tall/wrapped panels stay
        // visible while being used).
        let show_controls = if let Some(hover_pos) = ctx.input(|i| i.pointer.hover_pos()) {
            let in_band = hover_pos.y > rect.bottom() - 150.0;
            let in_panel = self
                .controls_rect
                .is_some_and(|r| r.expand(8.0).contains(hover_pos));
            in_band || in_panel
        } else {
            false
        };

        // Smooth fade transition
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        let target_alpha = if show_controls { 1.0 } else { 0.0 };
        self.controls_alpha += (target_alpha - self.controls_alpha) * 8.0 * dt;
        self.controls_alpha = self.controls_alpha.clamp(0.0, 1.0);

        // Render controls panel only if it's visible
        if self.controls_alpha > 0.001 {
            let area_response = egui::Area::new(egui::Id::new("controls"))
                .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -20.0))
                .show(ctx, |ui| {
                    ui.set_width(control_w);
                    ui.set_opacity(self.controls_alpha);

                    egui::Frame::new()
                        .fill(egui::Color32::from_black_alpha(190))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)))
                        .corner_radius(egui::CornerRadius::same(16))
                        .inner_margin(16.0)
                        .show(ui, |ui| {
                            // Status Row
                            ui.horizontal(|ui| {
                                if self.is_loading {
                                    ui.add(egui::Spinner::new().size(14.0));
                                    ui.add_space(6.0);
                                }
                                ui.label(
                                    egui::RichText::new(&self.status_msg)
                                        .color(egui::Color32::WHITE)
                                        .font(egui::FontId::proportional(13.0)),
                                );
                            });

                            ui.add_space(8.0);

                            // Progress / Seeking Row
                            let mut total_duration_secs = 0.0;
                            let mut current_secs = 0.0;

                            if let (Some(samples), Some(playback_pos)) =
                                (&self.samples, &self.playback_pos)
                            {
                                let current_idx = playback_pos.load(Ordering::Relaxed);
                                let total_samples = samples.len();
                                total_duration_secs = total_samples as f32
                                    / (self.sample_rate as f32 * self.channels as f32);
                                current_secs = current_idx as f32
                                    / (self.sample_rate as f32 * self.channels as f32);
                            }

                            let mut seek_to_secs = current_secs;
                            ui.horizontal(|ui| {
                                ui.style_mut().spacing.slider_width = control_w - 180.0;
                                if total_duration_secs > 0.0 {
                                    let current_time_str = format!(
                                        "{:02}:{:02}",
                                        (current_secs / 60.0) as i32,
                                        (current_secs % 60.0) as i32
                                    );
                                    let total_time_str = format!(
                                        "{:02}:{:02}",
                                        (total_duration_secs / 60.0) as i32,
                                        (total_duration_secs % 60.0) as i32
                                    );

                                    ui.label(
                                        egui::RichText::new(current_time_str)
                                            .color(egui::Color32::LIGHT_GRAY),
                                    );
                                    let slider = egui::Slider::new(
                                        &mut seek_to_secs,
                                        0.0..=total_duration_secs,
                                    )
                                    .show_value(false);
                                    let response = ui.add(slider);
                                    ui.label(
                                        egui::RichText::new(total_time_str)
                                            .color(egui::Color32::LIGHT_GRAY),
                                    );

                                    if response.changed() {
                                        let target_sample_idx = (seek_to_secs
                                            * self.sample_rate as f32
                                            * self.channels as f32)
                                            as usize;
                                        if let Some(ref samples) = self.samples {
                                            let target_sample_idx =
                                                target_sample_idx.min(samples.len());
                                            if let Some(ref player) = self.player {
                                                player.stop();
                                                let new_playback_pos =
                                                    Arc::new(AtomicUsize::new(target_sample_idx));
                                                self.playback_pos = Some(new_playback_pos.clone());
                                                let source = VisualizerSource {
                                                    samples: samples.clone(),
                                                    pos: target_sample_idx,
                                                    sample_rate: self.sample_rate,
                                                    channels: self.channels,
                                                    shared_pos: new_playback_pos,
                                                };
                                                player.append(source);
                                                player.play();
                                            }
                                        }
                                    }
                                } else {
                                    ui.add_enabled(
                                        false,
                                        egui::Slider::new(&mut seek_to_secs, 0.0..=1.0)
                                            .show_value(false),
                                    );
                                }
                            });

                            ui.add_space(8.0);

                            // Media Control Row. Wrapped so the buttons/sliders flow onto
                            // additional rows instead of overflowing on narrow windows.
                            ui.horizontal_wrapped(|ui| {
                                let is_playing = self.player.as_ref().map_or(false, |p| {
                                    !p.is_paused()
                                        && self.playback_pos.as_ref().map_or(false, |pos| {
                                            pos.load(Ordering::Relaxed)
                                                < self.samples.as_ref().map_or(0, |s| s.len())
                                        })
                                });

                                // Play/Pause Button
                                if is_playing {
                                    if ui.button(egui::RichText::new("⏸").size(16.0)).clicked() {
                                        if let Some(ref player) = self.player {
                                            player.pause();
                                        }
                                    }
                                } else {
                                    let can_play = self.samples.is_some();
                                    if ui
                                        .add_enabled(
                                            can_play,
                                            egui::Button::new(egui::RichText::new("▶").size(16.0)),
                                        )
                                        .clicked()
                                    {
                                        if let Some(ref player) = self.player {
                                            let current = self
                                                .playback_pos
                                                .as_ref()
                                                .map_or(0, |pos| pos.load(Ordering::Relaxed));
                                            let total =
                                                self.samples.as_ref().map_or(0, |s| s.len());
                                            if current >= total {
                                                player.stop();
                                                let new_playback_pos =
                                                    Arc::new(AtomicUsize::new(0));
                                                self.playback_pos = Some(new_playback_pos.clone());
                                                let source = VisualizerSource {
                                                    samples: self.samples.as_ref().unwrap().clone(),
                                                    pos: 0,
                                                    sample_rate: self.sample_rate,
                                                    channels: self.channels,
                                                    shared_pos: new_playback_pos,
                                                };
                                                player.append(source);
                                            }
                                            player.play();
                                        }
                                    }
                                }

                                // Stop Button
                                if ui
                                    .add_enabled(
                                        self.samples.is_some(),
                                        egui::Button::new(egui::RichText::new("⏹").size(16.0)),
                                    )
                                    .clicked()
                                {
                                    if let Some(ref player) = self.player {
                                        player.stop();
                                    }
                                    if let Some(ref p) = self.playback_pos {
                                        p.store(0, Ordering::Relaxed);
                                    }
                                }

                                ui.separator();

                                // Export the loaded track to a video file.
                                let can_export =
                                    self.samples.is_some() && self.current_file_path.is_some();
                                let export_btn =
                                    ui.add_enabled(can_export, egui::Button::new("🎬 Export"));
                                let export_btn = if can_export {
                                    export_btn.on_hover_text(
                                        "Render the visualizer and this track into an MP4",
                                    )
                                } else {
                                    export_btn.on_disabled_hover_text(
                                        "Drop an audio file in first — exporting renders that file",
                                    )
                                };
                                if export_btn.clicked() {
                                    self.export.window_open = true;
                                    self.export.ffmpeg_ok = Some(render::ffmpeg_available());
                                    // Probes by running a tiny real encode; cached
                                    // after the first call.
                                    self.export.hw_encoder = render::hw_encoder();
                                }

                                ui.separator();

                                // Volume Slider
                                ui.label("🔊");
                                let mut vol = self.volume;
                                ui.style_mut().spacing.slider_width = 80.0;
                                if ui
                                    .add(egui::Slider::new(&mut vol, 0.0..=1.0).show_value(false))
                                    .changed()
                                {
                                    self.volume = vol;
                                    if let Some(ref player) = self.player {
                                        player.set_volume(vol);
                                    }
                                }

                                ui.separator();

                                // Visual Gain Slider. Raw waveform mode gets a much lower
                                // ceiling (it clips into noise past ~1.5), and keeps its own
                                // value so switching modes doesn't clobber the FFT gain.
                                ui.label("Gain:");
                                ui.style_mut().spacing.slider_width = 80.0;
                                if self.fft_enabled {
                                    ui.add(
                                        egui::Slider::new(&mut self.gain, 0.1..=8.0)
                                            .show_value(true),
                                    );
                                } else {
                                    ui.add(
                                        egui::Slider::new(&mut self.wave_gain, 0.1..=1.5)
                                            .show_value(true),
                                    );
                                }

                                ui.separator();

                                // Visual Mode Toggle (FFT vs Waveform)
                                ui.checkbox(&mut self.fft_enabled, "Standard (FFT + Wave)");

                                ui.separator();

                                // System Audio (WASAPI loopback) Toggle: visualize whatever
                                // is currently playing on the machine's default output device.
                                let mut sys = self.system_audio_enabled;
                                if ui
                                    .checkbox(&mut sys, "System Audio")
                                    .on_hover_text(
                                        "Visualize whatever is currently playing on your PC",
                                    )
                                    .changed()
                                {
                                    if sys {
                                        match SystemAudioCapture::new() {
                                            Ok(cap) => {
                                                // Pause file playback so the two sources don't
                                                // fight over the visualizer (and the speakers).
                                                if let Some(ref player) = self.player {
                                                    player.pause();
                                                }
                                                self.status_msg =
                                                    "Visualizing system audio (whatever is \
                                                     playing on your PC)"
                                                        .to_string();
                                                self.system_capture = Some(cap);
                                                self.system_audio_enabled = true;
                                            }
                                            Err(e) => {
                                                self.system_audio_enabled = false;
                                                self.system_capture = None;
                                                self.status_msg =
                                                    format!("Couldn't capture system audio: {}", e);
                                            }
                                        }
                                    } else {
                                        self.system_capture = None;
                                        self.system_audio_enabled = false;
                                        self.status_msg =
                                            "Stopped system audio capture.".to_string();
                                    }
                                }

                                ui.separator();

                                // Phase Lock Checkbox
                                ui.checkbox(&mut self.trigger_mode, "Lock Phase");

                                ui.separator();

                                // Window Zoom Slider
                                ui.label("Frames shown:");
                                ui.style_mut().spacing.slider_width = 80.0;
                                ui.add(
                                    egui::Slider::new(&mut self.wave_window_ms, 5.0..=150.0)
                                        .suffix("ms"),
                                );
                            });
                        });
                });
            // Remember where the panel landed so next frame's hover test keeps it
            // visible while the pointer is over it (even if it wrapped taller).
            self.controls_rect = Some(area_response.response.rect);
        } else {
            self.controls_rect = None;
        }

        // Export settings window (independent of the fading control panel)
        self.export_window(ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "volume", &self.volume);
        eframe::set_value(storage, "gain", &self.gain);
        eframe::set_value(storage, "wave_gain", &self.wave_gain);
        eframe::set_value(storage, "trigger_mode", &self.trigger_mode);
        eframe::set_value(storage, "wave_window_ms", &self.wave_window_ms);
        eframe::set_value(storage, "fft_enabled", &self.fft_enabled);
        eframe::set_value(storage, "system_audio_enabled", &self.system_audio_enabled);
        eframe::set_value(storage, "export_width", &self.export.width);
        eframe::set_value(storage, "export_height", &self.export.height);
        eframe::set_value(storage, "export_fps", &self.export.fps);
        eframe::set_value(storage, "export_crf", &self.export.crf);
        eframe::set_value(
            storage,
            "export_encoder",
            &match self.export.encoder {
                render::Encoder::Auto => 0u8,
                render::Encoder::Gpu => 1,
                render::Encoder::X264 => 2,
            },
        );
        eframe::set_value(
            storage,
            "export_speed",
            &match self.export.speed {
                render::Speed::Fast => 0u8,
                render::Speed::Balanced => 1,
                render::Speed::Best => 2,
            },
        );
    }
}

fn load_icon() -> Option<egui::IconData> {
    let image_bytes = include_bytes!("../logo.png");
    let image = image::load_from_memory(image_bytes).ok()?;
    let rgba_image = image.to_rgba8();
    let (width, height) = rgba_image.dimensions();
    Some(egui::IconData {
        rgba: rgba_image.into_raw(),
        width,
        height,
    })
}

const CLI_USAGE: &str = "\
Usage: audio-visualizer [--render <audio file> [options]]

With no arguments the GUI starts. With --render the visualizer is rendered
offline (no window) and muxed with the audio into an MP4 via ffmpeg.

Options:
  -o, --out <file.mp4>   Output file (default: <audio>_visualizer.mp4)
      --size <WxH>       Video size (default: 1920x1080)
      --fps <n>          Frame rate (default: 60)
      --crf <n>          Quality, lower is better (default: 18)
      --encoder <e>      auto | gpu | cpu  (default: auto — hardware if available)
      --speed <s>        x264 preset: fast | balanced | best (default: balanced)
      --fft              Use the FFT + wave mode instead of waveform only
      --gain <f>         Visual gain (default: 1.0 waveform / 4.0 with --fft)
      --window <ms>      Waveform window in ms (default: 70)
      --no-trigger       Disable zero-crossing phase lock
";

// A GUI-subsystem binary has no console of its own; borrow the one it was
// launched from so --render can report progress.
#[cfg(windows)]
fn attach_console() {
    unsafe extern "system" {
        fn AttachConsole(process_id: u32) -> i32;
    }
    const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
    unsafe {
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

#[cfg(not(windows))]
fn attach_console() {}

// Headless render driven from the command line. Returns a process exit code.
fn cli_render(args: &[String]) -> i32 {
    use std::io::Write;

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut width = 1920u32;
    let mut height = 1080u32;
    let mut fps = 60u32;
    let mut crf = 18u32;
    let mut fft_enabled = false;
    let mut gain: Option<f32> = None;
    let mut wave_window_ms = 70.0f32;
    let mut trigger_mode = true;
    let mut encoder = render::Encoder::Auto;
    let mut speed = render::Speed::Balanced;

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let next = |i: &mut usize| -> Option<String> {
            *i += 1;
            args.get(*i).cloned()
        };
        match arg {
            "--render" | "-r" => input = next(&mut i).map(PathBuf::from),
            "--out" | "-o" => output = next(&mut i).map(PathBuf::from),
            "--size" => {
                if let Some(size) = next(&mut i) {
                    let (w, h) = match size.split_once(['x', 'X']) {
                        Some(parts) => parts,
                        None => {
                            eprintln!("Invalid --size '{size}', expected e.g. 1920x1080");
                            return 2;
                        }
                    };
                    match (w.parse(), h.parse()) {
                        (Ok(w), Ok(h)) => {
                            width = w;
                            height = h;
                        }
                        _ => {
                            eprintln!("Invalid --size '{size}', expected e.g. 1920x1080");
                            return 2;
                        }
                    }
                }
            }
            "--fps" => fps = next(&mut i).and_then(|v| v.parse().ok()).unwrap_or(fps),
            "--crf" => crf = next(&mut i).and_then(|v| v.parse().ok()).unwrap_or(crf),
            "--encoder" => {
                encoder = match next(&mut i).as_deref() {
                    Some("auto") => render::Encoder::Auto,
                    Some("gpu") => render::Encoder::Gpu,
                    Some("cpu") | Some("x264") => render::Encoder::X264,
                    other => {
                        eprintln!(
                            "Invalid --encoder '{}', expected auto|gpu|cpu",
                            other.unwrap_or("")
                        );
                        return 2;
                    }
                }
            }
            "--speed" => {
                speed = match next(&mut i).as_deref() {
                    Some("fast") => render::Speed::Fast,
                    Some("balanced") => render::Speed::Balanced,
                    Some("best") => render::Speed::Best,
                    other => {
                        eprintln!(
                            "Invalid --speed '{}', expected fast|balanced|best",
                            other.unwrap_or("")
                        );
                        return 2;
                    }
                }
            }
            "--fft" => fft_enabled = true,
            "--gain" => gain = next(&mut i).and_then(|v| v.parse().ok()),
            "--window" => {
                wave_window_ms = next(&mut i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(wave_window_ms)
            }
            "--no-trigger" => trigger_mode = false,
            "--help" | "-h" => {
                println!("{CLI_USAGE}");
                return 0;
            }
            other => {
                eprintln!("Unknown argument '{other}'\n\n{CLI_USAGE}");
                return 2;
            }
        }
        i += 1;
    }

    let Some(input) = input else {
        eprintln!("--render needs an audio file\n\n{CLI_USAGE}");
        return 2;
    };
    let output = output.unwrap_or_else(|| render::default_output_path(&input));

    if !render::ffmpeg_available() {
        eprintln!("ffmpeg was not found on your PATH.");
        return 1;
    }

    println!("Decoding {}...", input.display());
    let (samples, sample_rate, channels) = match decode_audio_file(&input) {
        Ok(decoded) => decoded,
        Err(e) => {
            eprintln!("Couldn't decode {}: {e}", input.display());
            return 1;
        }
    };

    let (tx, rx) = channel();
    render::spawn(
        render::RenderRequest {
            audio_path: input,
            output_path: output,
            samples: Arc::new(samples),
            sample_rate,
            channels,
            width: width & !1, // yuv420p needs even dimensions
            height: height & !1,
            fps,
            crf,
            encoder,
            speed,
            fft_enabled,
            gain: gain.unwrap_or(if fft_enabled { 4.0 } else { 1.0 }),
            wave_window_ms,
            trigger_mode,
        },
        tx,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
    );

    for msg in rx {
        match msg {
            render::RenderMsg::Progress { frame, total } => {
                print!("\rRendering frame {frame}/{total}");
                let _ = std::io::stdout().flush();
            }
            render::RenderMsg::Finished(path) => {
                println!("\nWrote {}", path.display());
                return 0;
            }
            render::RenderMsg::Canceled(_) => {
                println!("\nCanceled");
                return 1;
            }
            render::RenderMsg::Failed(e) => {
                eprintln!("\n{e}");
                return 1;
            }
        }
    }
    1
}

fn main() -> eframe::Result {
    env_logger::init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args
        .iter()
        .any(|a| a == "--render" || a == "-r" || a == "--help" || a == "-h")
    {
        attach_console();
        std::process::exit(cli_render(&args));
    }

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor {
                    backends: get_backend_from_env(),
                    flags: wgpu::InstanceFlags::default(),
                    memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                    backend_options: wgpu::BackendOptions::default(),
                    display: None,
                },
                display_handle: None,
                power_preference: wgpu::PowerPreference::HighPerformance,
                native_adapter_selector: None,
                device_descriptor: std::sync::Arc::new(|_adapter| wgpu::DeviceDescriptor {
                    label: Some("egui wgpu device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                }),
            }),
            ..Default::default()
        },
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_resizable(true)
            .with_title("Audio Visualizer")
            .with_icon(load_icon().unwrap_or_default()),
        ..Default::default()
    };

    eframe::run_native(
        "Audio Visualizer",
        native_options,
        Box::new(|cc| match LeApp::new(cc) {
            Some(app) => Ok(Box::new(app)),
            None => {
                log::error!("Failed to initialize WGPU renderer.");
                Err("WGPU renderer initialization failed".into())
            }
        }),
    )
}
