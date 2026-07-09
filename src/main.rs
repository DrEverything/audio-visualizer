#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::num::{NonZero, NonZeroU64};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use eframe::egui;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use eframe::egui_wgpu::{self, wgpu};
use rodio::{Decoder, MixerDeviceSink, Player, Source};
use rustfft::{FftPlanner, num_complex::Complex};

fn get_backend_from_env() -> wgpu::Backends {
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

// Struct to store persistent WGPU resources used by the shader callback
struct VisualizerRenderResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    audio_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
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

enum AudioMessage {
    Decoded {
        file_name: String,
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
            let row_size = 1024 * 4;
            // Shift history down by 1 row (row 0 moves to row 1, etc.)
            res.history_buffer.copy_within(0..row_size * 255, row_size);

            let mut new_row = vec![0u8; row_size];

            if self.fft_enabled {
                // 1. Run 2048-point Forward FFT
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(2048);

                let mut fft_buffer: Vec<Complex<f32>> = self
                    .samples
                    .iter()
                    .map(|&s| Complex { re: s, im: 0.0 })
                    .collect();

                fft.process(&mut fft_buffer);

                // 2. Map FFT bins to the new row
                for i in 0..1024 {
                    let c = fft_buffer[i];
                    let magnitude = c.norm();

                    // High-frequency pre-emphasis: boost higher frequency bins for visualization
                    let freq_boost = 1.0 + (i as f32 / 128.0);

                    // Apply normalized gain and scale (divide by FFT window 2048.0).
                    // Use a smooth tanh saturation instead of a hard clamp(0,1): a hard clamp
                    // flat-tops loud bins into vertical-edged mesas (the loud bass/left bins hit
                    // the ceiling first), and the raymarcher renders those cliffs as bright streak
                    // artifacts on the left "when peaks are high enough". tanh asymptotes to 1.0
                    // smoothly so the terrain stays continuous, matching Shadertoy's normalized FFT.
                    let val = (magnitude / 2048.0 * self.gain * freq_boost).sqrt().tanh();
                    new_row[i * 4] = (val * 255.0) as u8; // R: FFT magnitude, 0..1
                    let s = self.samples[i * 2];
                    new_row[i * 4 + 1] = ((s * 0.5 + 0.5) * 255.0) as u8; // G: waveform centered at 0.5
                    let raw = (magnitude / 2048.0 * self.gain).sqrt().clamp(0.0, 1.0);
                    new_row[i * 4 + 2] = (raw * 255.0) as u8; // B: raw magnitude
                    new_row[i * 4 + 3] = 255;
                }
            } else {
                // Waveform Only Mode: Red contains the waveform too so the terrain reacts to it!
                for i in 0..1024 {
                    let s = self.samples[i * 2];
                    // let v = ((s * 0.5 + 0.5)).clamp(0.0, 1.0);   // map -1..1 to 0..1
                    // Soft-limit (tanh) instead of a hard clamp so loud peaks don't flat-top the
                    // terrain into vertical-edged cliffs that raymarch into bright streak artifacts.
                    let v = (s.abs() * self.gain).tanh();
                    let byte = (v * 255.0) as u8;
                    new_row[i * 4] = byte; // R drives terrain (rectified-ish via 0.5 center)
                    new_row[i * 4 + 1] = byte;
                    new_row[i * 4 + 2] = byte;
                    new_row[i * 4 + 3] = 255;
                }
            }

            // Copy new row to the start of history_buffer
            res.history_buffer[0..row_size].copy_from_slice(&new_row);

            // Upload the entire 1024x256 texture to the GPU
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &res.audio_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &res.history_buffer,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_size as u32),
                    rows_per_image: Some(256),
                },
                wgpu::Extent3d {
                    width: 1024,
                    height: 256,
                    depth_or_array_layers: 1,
                },
            );

            // Upload uniforms
            let uniforms = VisualizerUniforms {
                u_time: self.time,
                u_resolution_x: self.resolution.x,
                u_resolution_y: self.resolution.y,
                u_pad: 0.0,
            };
            queue.write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
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
            render_pass.set_pipeline(&res.pipeline);
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
            render_pass.set_bind_group(0, &res.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

pub struct LeApp {
    // Audio stream & Player management
    _stream: Option<MixerDeviceSink>,
    player: Option<Player>,

    // Current playing buffer
    current_file_name: Option<String>,
    samples: Option<Arc<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    playback_pos: Option<Arc<AtomicUsize>>,

    // Playback settings
    volume: f32,
    gain: f32,
    trigger_mode: bool, // true = Lock Phase (zero-crossing search), false = continuous raw buffer
    wave_window_ms: f32,     // size of window to display in ms (e.g. 5ms to 150ms)
    fft_enabled: bool,  // true = standard fourier transform landscape, false = waveform only

    // Channels to communicate with background decoder thread
    rx: Receiver<AudioMessage>,
    tx: Sender<AudioMessage>,

    // UI state
    status_msg: String,
    time_start: Instant,
    is_loading: bool,
    controls_alpha: f32, // for fading controls in/out on hover
}

impl LeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
        let device = &wgpu_render_state.device;

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
                width: 1024,
                height: 256,
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

        // Create linear sampler matching Shadertoy audio channel settings.
        // Shadertoy's iChannel audio texture wraps (Repeat) by default. The horizontal
        // coordinate (p.x + 4.5) / 30.0 goes negative on the left side of the screen; with
        // ClampToEdge that pins to column 0 and extrudes the loudest (bass/first) bin into a
        // tall flat wall on high peaks, which raymarches into bright streak artifacts. Repeat
        // wraps those out-of-range samples to the quiet high end instead, matching the original.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("audio_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
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
                targets: &[Some(wgpu_render_state.target_format.into())],
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

        // Initialize history_buffer with silent values
        let mut history_buffer = vec![0u8; 1024 * 256 * 4];
        for y in 0..256 {
            for x in 0..1024 {
                let idx = (y * 1024 + x) * 4;
                history_buffer[idx] = 0; // Red (silent baseline)
                history_buffer[idx + 1] = 128; // Green (silent baseline)
                history_buffer[idx + 2] = 0; // Blue (silent baseline)
                history_buffer[idx + 3] = 255; // Alpha (opaque Snorm)
            }
        }

        // Insert resources so the callback can retrieve them later
        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(VisualizerRenderResources {
                pipeline,
                uniform_buffer,
                audio_texture,
                bind_group,
                history_buffer,
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
        let volume = cc.storage
            .and_then(|s| eframe::get_value(s, "volume"))
            .unwrap_or(0.5);
        let gain = cc.storage
            .and_then(|s| eframe::get_value(s, "gain"))
            .unwrap_or(8.0);
        let trigger_mode = cc.storage
            .and_then(|s| eframe::get_value(s, "trigger_mode"))
            .unwrap_or(true);
        let wave_window_ms = cc.storage
            .and_then(|s| eframe::get_value(s, "wave_window_ms"))
            .unwrap_or(150.0);
        let fft_enabled = cc.storage
            .and_then(|s| eframe::get_value(s, "fft_enabled"))
            .unwrap_or(true);

        Some(Self {
            _stream: stream,
            player,
            current_file_name: None,
            samples: None,
            sample_rate: 44100,
            channels: 2,
            playback_pos: None,
            volume,
            gain,
            trigger_mode,
            wave_window_ms,
            fft_enabled,
            rx,
            tx,
            status_msg: "Drag & Drop an audio file (MP3, WAV, FLAC, OGG) here to play!".to_string(),
            time_start: Instant::now(),
            is_loading: false,
            controls_alpha: 0.0,
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

impl eframe::App for LeApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx();
        ctx.set_visuals(egui::Visuals::dark());



        // Process message from audio decoder thread
        if let Ok(msg) = self.rx.try_recv() {
            self.is_loading = false;
            match msg {
                AudioMessage::Decoded {
                    file_name,
                    samples,
                    sample_rate,
                    channels,
                    playback_pos,
                } => {
                    self.current_file_name = Some(file_name.clone());
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
                    let new_stream = if let (Some(rate), Some(ch)) = (NonZero::new(sample_rate), NonZero::new(channels)) {
                        if let Ok(builder) = rodio::stream::DeviceSinkBuilder::from_default_device() {
                            let builder = builder.with_sample_rate(rate).with_channels(ch);
                            builder.open_sink_or_fallback().ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Fallback to default sink if custom configuration failed
                    let new_stream = new_stream.or_else(|| {
                        rodio::stream::DeviceSinkBuilder::open_default_sink().ok()
                    });

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
        if let (Some(samples), Some(playback_pos)) = (&self.samples, &self.playback_pos) {
            let current_idx = playback_pos.load(Ordering::Relaxed);
            let total_samples = samples.len();
            let total_frames = total_samples / (self.channels as usize);
            let current_frame = current_idx / (self.channels as usize);

            // Compute total frames in the zoom window
            let window_frames = ((self.wave_window_ms / 1000.0) * self.sample_rate as f32) as usize;
            let window_frames = window_frames.max(32); // at least 32 frames for 2048 window

            let mut start_frame = current_frame;

            if self.trigger_mode {
                // Stabilize wave phase via Oscilloscope Rising-Edge Zero-Crossing Triggering.
                // We search ahead for a zero-crossing based on mono mixed values.
                let search_len = 1024.min(total_frames.saturating_sub(current_frame));
                for f in 0..search_len {
                    let f_idx = current_frame + f;

                    let mut v1 = 0.0;
                    let mut count = 0;
                    for c in 0..(self.channels as usize) {
                        let idx = f_idx * (self.channels as usize) + c;
                        if idx < total_samples {
                            v1 += samples[idx];
                            count += 1;
                        }
                    }
                    let mono1 = if count > 0 { v1 / count as f32 } else { 0.0 };

                    let mut v2 = 0.0;
                    let mut count2 = 0;
                    for c in 0..(self.channels as usize) {
                        let idx = (f_idx + 1) * (self.channels as usize) + c;
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
                for c in 0..(self.channels as usize) {
                    let idx = target_frame * (self.channels as usize) + c;
                    if idx < total_samples {
                        sum += samples[idx];
                        count += 1;
                    }
                }
                visualizer_samples[i] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
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
                gain: self.gain,
            },
        ));

        // Force repaint to animate shader
        ctx.request_repaint();

        // Calculate floating overlay parameters
        let screen_w = rect.width();
        let control_w = (screen_w * 0.82).clamp(500.0, 1050.0);
        let control_h = 130.0;

        let overlay_pos = egui::pos2(
            rect.left() + (screen_w - control_w) * 0.5,
            rect.bottom() - control_h - 20.0,
        );

        let overlay_rect = egui::Rect::from_min_size(overlay_pos, egui::vec2(control_w, control_h));

        // Show controls only if the pointer is inside the bottom control panel region
        let show_controls = if let Some(hover_pos) = ctx.input(|i| i.pointer.hover_pos()) {
            overlay_rect.contains(hover_pos) || hover_pos.y > rect.bottom() - 150.0
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
            egui::Area::new(egui::Id::new("controls"))
                .fixed_pos(overlay_pos)
                .show(ctx, |ui| {
                    ui.set_width(control_w);
                    ui.set_height(control_h);
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

                            // Media Control Row
                            ui.horizontal(|ui| {
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

                                // Waveform Visual Gain Slider
                                ui.label("Gain:");
                                ui.style_mut().spacing.slider_width = 80.0;
                                ui.add(
                                    egui::Slider::new(&mut self.gain, 0.1..=8.0).show_value(true),
                                );

                                ui.separator();

                                // Visual Mode Toggle (FFT vs Waveform)
                                ui.checkbox(&mut self.fft_enabled, "Standard (FFT + Wave)");

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
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "volume", &self.volume);
        eframe::set_value(storage, "gain", &self.gain);
        eframe::set_value(storage, "trigger_mode", &self.trigger_mode);
        eframe::set_value(storage, "wave_window_ms", &self.wave_window_ms);
        eframe::set_value(storage, "fft_enabled", &self.fft_enabled);
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

fn main() -> eframe::Result {
    env_logger::init();

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
