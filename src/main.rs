mod types;

use std::num::NonZeroU64;
use std::time::Instant;

use eframe::egui;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use eframe::egui_wgpu::{self, wgpu};

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

fn main() -> eframe::Result {
    // Initialize logger for debugging
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
                device_descriptor: std::sync::Arc::new(|_adapter| {
                    wgpu::DeviceDescriptor {
                        label: Some("egui wgpu device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        experimental_features: wgpu::ExperimentalFeatures::disabled(),
                        memory_hints: wgpu::MemoryHints::default(),
                        trace: wgpu::Trace::Off,
                    }
                }),
            }),
            ..Default::default()
        },
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 750.0])
            .with_resizable(true)
            .with_title("Interactive WGPU Raymarching"),
        ..Default::default()
    };

    eframe::run_native(
        "Interactive WGPU Raymarching",
        native_options,
        Box::new(|cc| {
            match RaymarchApp::new(cc) {
                Some(app) => Ok(Box::new(app)),
                None => {
                    log::error!("Failed to initialize WGPU renderer. Falling back to simple UI.");
                    Err("WGPU renderer initialization failed".into())
                }
            }
        }),
    )
}

struct RaymarchApp {
    // UI controls / Uniform fields
    speed: f32,
    morph_factor: f32,
    camera_rot: [f32; 2], // yaw, pitch
    camera_zoom: f32,
    light_dir: [f32; 3],
    steps: u32,
    color_palette: u32,
    glow_intensity: f32,

    // Time tracking
    time: f32,
    last_time: Instant,
    
    // Status flag
    wgpu_initialized: bool,

    // Shader hot reloading
    shader_path: std::path::PathBuf,
    last_shader_modified: Option<std::time::SystemTime>,
    shader_error: Option<String>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RaymarchUniforms {
    resolution: [f32; 2],
    time: f32,
    speed: f32,
    camera_rot: [f32; 2],
    camera_zoom: f32,
    morph_factor: f32,
    light_dir: [f32; 3],
    steps: u32,
    color_palette: u32,
    glow_intensity: f32,
    _padding1: u32,
    _padding2: u32,
}

impl RaymarchApp {
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
        let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
        let device = &wgpu_render_state.device;

        // Compile our WGSL shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shader.wgsl").into()),
        });

        // Create bind group layout for uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raymarch_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raymarch_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raymarch_pipeline"),
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

        // Initialize uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raymarch_uniform_buffer"),
            contents: bytemuck::cast_slice(&[RaymarchUniforms {
                resolution: [800.0, 600.0],
                time: 0.0,
                speed: 1.0,
                camera_rot: [0.0, 0.0],
                camera_zoom: 5.0,
                morph_factor: 0.0,
                light_dir: [1.0, 1.0, -1.0],
                steps: 64,
                color_palette: 0,
                glow_intensity: 1.0,
                _padding1: 0,
                _padding2: 0,
            }]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raymarch_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Store persistent WGPU resources in eframe's paint_callback_resources
        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(RaymarchRenderResources {
                pipeline,
                bind_group,
                uniform_buffer,
            });

        let shader_path = std::path::PathBuf::from("src/shader.wgsl");
        let last_shader_modified = std::fs::metadata(&shader_path)
            .and_then(|m| m.modified())
            .ok();

        Some(Self {
            speed: 1.0,
            morph_factor: 0.5,
            camera_rot: [0.0, 0.3],
            camera_zoom: 5.5,
            light_dir: [1.5, 2.0, -1.0],
            steps: 80,
            color_palette: 0,
            glow_intensity: 1.2,
            time: 0.0,
            last_time: Instant::now(),
            wgpu_initialized: true,
            shader_path,
            last_shader_modified,
            shader_error: None,
        })
    }

    fn reload_shader(&mut self, device: &wgpu::Device, render_state: &eframe::egui_wgpu::RenderState) {
        let shader_source = match std::fs::read_to_string(&self.shader_path) {
            Ok(src) => src,
            Err(e) => {
                self.shader_error = Some(format!("Failed to read shader file: {}", e));
                return;
            }
        };

        // Push validation error scope
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch_shader_hot"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Recreate bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raymarch_bind_group_layout_hot"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raymarch_pipeline_layout_hot"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create pipeline
        let _pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raymarch_pipeline_hot"),
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
                targets: &[Some(render_state.target_format.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Pop validation error scope
        let error = pollster::block_on(error_scope.pop());

        if let Some(err) = error {
            self.shader_error = Some(format!("Shader validation error:\n{}", err));
            log::error!("Shader validation error: {}", err);
        } else {
            // Re-fetch pipeline creation inside safety boundary
            let mut renderer_lock = render_state.renderer.write();
            if let Some(raymarch_res) = renderer_lock.callback_resources.get_mut::<RaymarchRenderResources>() {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("raymarch_bind_group_hot"),
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: raymarch_res.uniform_buffer.as_entire_binding(),
                    }],
                });

                raymarch_res.pipeline = _pipeline;
                raymarch_res.bind_group = bind_group;
                
                self.shader_error = None;
                log::info!("Shader hot-reloaded successfully!");
            }
        }
    }
}

impl eframe::App for RaymarchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx();
        if !self.wgpu_initialized {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.heading("WGPU renderer failed to initialize. Please check support.");
                });
            });
            return;
        }

        // Check for shader file modifications to support hot reloading
        if let Some(render_state) = _frame.wgpu_render_state() {
            if let Ok(metadata) = std::fs::metadata(&self.shader_path) {
                if let Ok(modified) = metadata.modified() {
                    if self.last_shader_modified.map_or(true, |last| modified > last) {
                        self.last_shader_modified = Some(modified);
                        self.reload_shader(&render_state.device, render_state);
                    }
                }
            }
        }

        // Calculate delta time
        let now = Instant::now();
        let dt = now.duration_since(self.last_time).as_secs_f32();
        self.last_time = now;
        self.time += dt;

        // Repaint continuously to support animation
        ctx.request_repaint();

        // Layout sidebar panels and main content
        egui::Panel::left("control_panel")
            .resizable(true)
            .default_size(320.0)
            .show(ui, |ui| {
                ui.add_space(10.0);
                ui.heading("Raymarching Parameters");
                ui.add_space(15.0);

                // Show shader errors in the sidebar if any compile failed
                if let Some(ref err) = self.shader_error {
                    ui.group(|ui| {
                        ui.colored_label(egui::Color32::LIGHT_RED, "❌ Shader Error:");
                        egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| {
                            ui.weak(err);
                        });
                    });
                    ui.add_space(10.0);
                }

                ui.group(|ui| {
                    ui.label("Animation");
                    ui.add(egui::Slider::new(&mut self.speed, 0.0..=2.0).text("Speed"));
                    if ui.button("Pause").clicked() {
                        self.speed = 0.0;
                    }
                    if ui.button("Play").clicked() {
                        self.speed = 1.0;
                    }
                });
                
                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.label("Morph Factor (SDF Blend)");
                    ui.add(egui::Slider::new(&mut self.morph_factor, 0.0..=2.0)
                        .text("Sphere ➔ Torus ➔ Box"));
                    ui.label("Controls the smooth minimum blending of the primary geometry.");
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.label("Rendering Quality");
                    ui.add(egui::Slider::new(&mut self.steps, 16..=128).text("Max Steps"));
                    ui.add(egui::Slider::new(&mut self.glow_intensity, 0.0..=3.0).text("Glow Intensity"));
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.label("Color Scheme");
                    egui::ComboBox::from_label("Palette")
                        .selected_text(match self.color_palette {
                            0 => "Neon Rainbow",
                            1 => "Sunset Warmth",
                            2 => "Forest Gold",
                            3 => "Cyberpunk",
                            _ => "Unknown",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.color_palette, 0, "Neon Rainbow");
                            ui.selectable_value(&mut self.color_palette, 1, "Sunset Warmth");
                            ui.selectable_value(&mut self.color_palette, 2, "Forest Gold");
                            ui.selectable_value(&mut self.color_palette, 3, "Cyberpunk");
                        });
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.label("Lighting Direction");
                    ui.add(egui::Slider::new(&mut self.light_dir[0], -3.0..=3.0).text("Light X"));
                    ui.add(egui::Slider::new(&mut self.light_dir[1], 0.1..=4.0).text("Light Y"));
                    ui.add(egui::Slider::new(&mut self.light_dir[2], -3.0..=3.0).text("Light Z"));
                });

                ui.add_space(20.0);
                
                if ui.button("Reset Defaults").clicked() {
                    self.speed = 1.0;
                    self.morph_factor = 0.5;
                    self.camera_rot = [0.0, 0.3];
                    self.camera_zoom = 5.5;
                    self.light_dir = [1.5, 2.0, -1.0];
                    self.steps = 80;
                    self.color_palette = 0;
                    self.glow_intensity = 1.2;
                }
                
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);
                ui.vertical_centered(|ui| {
                    ui.weak("Drag the viewport to rotate camera");
                    ui.weak("Use mouse wheel to zoom");
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.render_canvas(ui);
            });
        });
    }
}

impl RaymarchApp {
    fn render_canvas(&mut self, ui: &mut egui::Ui) {
        // Allocate all available space for the shader canvas
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        // Handle camera rotation by mouse drag
        if response.dragged() {
            self.camera_rot[0] += response.drag_delta().x * 0.005; // Yaw
            self.camera_rot[1] = (self.camera_rot[1] + response.drag_delta().y * 0.005)
                .clamp(-1.4, 1.4); // Pitch clamped to avoid flipping upside down
        }

        // Handle camera zoom via mouse scroll
        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta != 0.0 {
            self.camera_zoom = (self.camera_zoom - scroll_delta * 0.005).clamp(2.0, 12.0);
        }

        // Prepare uniforms structure
        let uniforms = RaymarchUniforms {
            resolution: [rect.width(), rect.height()],
            time: self.time,
            speed: self.speed,
            camera_rot: self.camera_rot,
            camera_zoom: self.camera_zoom,
            morph_factor: self.morph_factor,
            light_dir: self.light_dir,
            steps: self.steps,
            color_palette: self.color_palette,
            glow_intensity: self.glow_intensity,
            _padding1: 0,
            _padding2: 0,
        };

        // Add custom WGPU callback to paint
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            RaymarchCallback { uniforms },
        ));
    }
}

// Custom paint callback data containing state for a single frame
struct RaymarchCallback {
    uniforms: RaymarchUniforms,
}

impl egui_wgpu::CallbackTrait for RaymarchCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(raymarch_res) = resources.get::<RaymarchRenderResources>() {
            raymarch_res.prepare(queue, self.uniforms);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(raymarch_res) = resources.get::<RaymarchRenderResources>() {
            raymarch_res.paint(render_pass);
        }
    }
}

// Persistent resources stored in eframe's paint_callback_resources
struct RaymarchRenderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

impl RaymarchRenderResources {
    fn prepare(&self, queue: &wgpu::Queue, uniforms: RaymarchUniforms) {
        // Write the uniforms structure into the GPU buffer
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        // Draw 3 vertices for a single screen-covering triangle
        render_pass.draw(0..3, 0..1);
    }
}
