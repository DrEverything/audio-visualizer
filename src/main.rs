mod bvh;
mod types;
mod path_tracer;
mod rasterizer;
mod raymarcher;

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use eframe::egui;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use eframe::egui_wgpu::{self, wgpu};

use bvh::{GpuMaterial, build_scene};

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
            .with_inner_size([1200.0, 800.0])
            .with_resizable(true)
            .with_title("Physically Accurate GPU Path Tracer"),
        ..Default::default()
    };

    eframe::run_native(
        "Physically Accurate GPU Path Tracer",
        native_options,
        Box::new(|cc| {
            match RaymarchApp::new(cc) {
                Some(app) => Ok(Box::new(app)),
                None => {
                    log::error!("Failed to initialize WGPU renderer.");
                    Err("WGPU renderer initialization failed".into())
                }
            }
        }),
    )
}

struct MaterialParams {
    name: String,
    base_color: [f32; 3],
    metallic: f32,
    roughness: f32,
    ior: f32,
    transmission: f32,
    emissive: [f32; 3],
}

impl MaterialParams {
    fn to_gpu(&self) -> GpuMaterial {
        GpuMaterial {
            base_color: [self.base_color[0], self.base_color[1], self.base_color[2], self.metallic],
            properties: [self.roughness, self.ior, self.transmission, 0.0],
            emissive: [self.emissive[0], self.emissive[1], self.emissive[2], 0.0],
        }
    }
}

pub trait Renderer: Send + Sync {
    fn name(&self) -> &'static str;
    
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &PathTraceUniforms,
        shared: &SharedRenderResources,
        target_view: &wgpu::TextureView,
        bind_group_pt: &wgpu::BindGroup,
    ) -> Vec<wgpu::CommandBuffer>;

    fn check_reload(
        &mut self,
        device: &wgpu::Device,
        render_state: &eframe::egui_wgpu::RenderState,
    ) -> bool;

    fn error(&self) -> Option<&str>;

    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool;

    fn update_uniforms(&self, uniforms: &mut PathTraceUniforms);

    fn supports_accumulation(&self) -> bool {
        false
    }

    fn max_accumulation_frames(&self) -> u32 {
        1
    }

    fn reset_defaults(&mut self) {}
}

struct RendererVector {
    renderers: Arc<Mutex<Vec<Box<dyn Renderer>>>>,
}

#[derive(Clone)]
pub struct SharedRenderResources {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub display_bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub tri_material_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub bvh_buffer: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
    pub textures: Arc<Mutex<Option<AccumTextures>>>,
    pub display_pipeline: wgpu::RenderPipeline,
}

struct RaymarchApp {
    renderers: Arc<Mutex<Vec<Box<dyn Renderer>>>>,
    selected_renderer_idx: usize,

    // Animation control parameters
    animate: bool,
    auto_rotate_camera: bool,
    animation_speed: f32,
    time: f32,

    // Materials list
    materials: Vec<MaterialParams>,
    selected_material_idx: usize,

    // Camera settings
    camera_rot: [f32; 2], // yaw, pitch
    camera_zoom: f32,
    prev_camera_rot: [f32; 2],
    prev_camera_zoom: f32,

    // Accumulation stats
    frame_index: u32,
    accum_frame: u32,
    wgpu_initialized: bool,

    // Scene geometry offsets
    bvh_offsets: [u32; 4],
    tri_offsets: [u32; 4],

    // Frame-by-frame convergence
    render_frame_by_frame: bool,
    fps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PathTraceUniforms {
    pub resolution: [f32; 2],
    pub camera_rot: [f32; 2],
    pub camera_zoom: f32,
    pub frame_index: u32,
    pub max_depth: u32,
    pub samples_per_frame: u32,
    pub aperture: f32,
    pub focal_distance: f32,
    pub env_light_intensity: f32,
    pub prev_camera_zoom: f32,
    pub prev_camera_rot: [f32; 2],
    pub time: f32,
    pub accum_frame: u32,
    pub bvh_offsets: [u32; 4],
    pub tri_offsets: [u32; 4],
    pub dt: f32,
    pub render_mode: u32,  // 0 = PathTracer, 1 = Deferred, 2 = Raymarcher
    pub tonemap_mode: u32, // 0 = Reinhard, 1 = ACES
    pub _pad_align2: u32,
}

impl RaymarchApp {
    pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
        let wgpu_render_state = cc.wgpu_render_state.as_ref()?;
        let device = &wgpu_render_state.device;

        // 1. Build Scene (Vertices, Indices, Material mapping, and BVH)
        let (
            vertices,
            indices,
            tri_materials,
            bvh_nodes,
            bvh_offsets,
            tri_offsets,
        ) = build_scene();
        log::info!("Scene built with {} vertices, {} triangles, {} BVH nodes.", 
                  vertices.len(), indices.len() / 3, bvh_nodes.len());

        // 2. Initialize Material definitions
        let materials = vec![
            MaterialParams {
                name: "Left Wall (Red)".to_string(),
                base_color: [0.75, 0.15, 0.15],
                metallic: 0.0,
                roughness: 0.8,
                ior: 1.5,
                transmission: 0.0,
                emissive: [0.0; 3],
            },
            MaterialParams {
                name: "Right Wall (Green)".to_string(),
                base_color: [0.15, 0.75, 0.15],
                metallic: 0.0,
                roughness: 0.8,
                ior: 1.5,
                transmission: 0.0,
                emissive: [0.0; 3],
            },
            MaterialParams {
                name: "Walls/Floor (White)".to_string(),
                base_color: [0.75, 0.75, 0.75],
                metallic: 0.0,
                roughness: 0.8,
                ior: 1.5,
                transmission: 0.0,
                emissive: [0.0; 3],
            },
            MaterialParams {
                name: "Ceiling Light".to_string(),
                base_color: [0.75, 0.75, 0.75],
                metallic: 0.0,
                roughness: 0.8,
                ior: 1.5,
                transmission: 0.0,
                emissive: [12.0, 12.0, 12.0],
            },
            MaterialParams {
                name: "Mechanical Part / Torus".to_string(),
                base_color: [0.91, 0.92, 0.92],
                metallic: 1.0,
                roughness: 0.15,
                ior: 1.5,
                transmission: 0.0,
                emissive: [0.0; 3],
            },
            MaterialParams {
                name: "Gold Sphere".to_string(),
                base_color: [1.0, 0.78, 0.34],
                metallic: 1.0,
                roughness: 0.05,
                ior: 1.5,
                transmission: 0.0,
                emissive: [0.0; 3],
            },
            MaterialParams {
                name: "Glass Sphere".to_string(),
                base_color: [0.95, 0.95, 0.95],
                metallic: 0.0,
                roughness: 0.0,
                ior: 1.5,
                transmission: 1.0,
                emissive: [0.0; 3],
            },
        ];

        let gpu_materials: Vec<GpuMaterial> = materials.iter().map(|m| m.to_gpu()).collect();

        // Compile display shader
        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./display.wgsl").into()),
        });

        // Create GPU storage and uniform buffers
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_buffer"),
            size: std::mem::size_of::<PathTraceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tri_material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tri_material_buffer"),
            contents: bytemuck::cast_slice(&tri_materials),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material_buffer"),
            contents: bytemuck::cast_slice(&gpu_materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_buffer"),
            contents: bytemuck::cast_slice(&bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // 3. Create Bind Group Layouts
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let display_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let pt_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let disp_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("disp_pipeline_layout"),
            bind_group_layouts: &[Some(&display_bind_group_layout)],
            immediate_size: 0,
        });

        // Create Display Render Pipeline
        // Target format matches egui's viewport format
        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display_pipeline"),
            layout: Some(&disp_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &display_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: Some("fs_display"),
                targets: &[Some(wgpu_render_state.target_format.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let shared_resources = SharedRenderResources {
            bind_group_layout,
            display_bind_group_layout,
            uniform_buffer,
            vertex_buffer,
            index_buffer,
            tri_material_buffer,
            material_buffer,
            bvh_buffer,
            sampler,
            textures: Arc::new(Mutex::new(None)),
            display_pipeline,
        };

        // Create modular renderers
        let renderers: Vec<Box<dyn Renderer>> = vec![
            Box::new(path_tracer::PathTracer::new(device, &pt_pipeline_layout)),
            Box::new(rasterizer::Rasterizer::new(device, &pt_pipeline_layout)),
            Box::new(raymarcher::Raymarcher::new(device, &pt_pipeline_layout)),
        ];

        let renderers_arc = Arc::new(Mutex::new(renderers));

        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(shared_resources.clone());

        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(RendererVector {
                renderers: renderers_arc.clone(),
            });

        Some(Self {
            renderers: renderers_arc,
            selected_renderer_idx: 2,
            animate: false,
            auto_rotate_camera: false,
            animation_speed: 1.0,
            time: 0.0,
            materials,
            selected_material_idx: 4, // default to Mechanical Part
            camera_rot: [0.0, 0.3],
            camera_zoom: 5.5,
            prev_camera_rot: [0.0, 0.3],
            prev_camera_zoom: 5.5,
            frame_index: 0,
            accum_frame: 0,
            wgpu_initialized: true,
            bvh_offsets,
            tri_offsets,
            render_frame_by_frame: false,
            fps: 30.0,
        })
    }
}

impl eframe::App for RaymarchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx();
        if !self.wgpu_initialized {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.heading("WGPU renderer failed to initialize.");
                });
            });
            return;
        }

        // Check for shader file changes for hot reloading
        if let Some(render_state) = _frame.wgpu_render_state() {
            let mut renderers = self.renderers.lock().unwrap();
            if let Some(renderer) = renderers.get_mut(self.selected_renderer_idx) {
                if renderer.check_reload(&render_state.device, render_state) {
                    self.accum_frame = 0;
                }
            }
        }

        // Continually request repaint if animating or if we have not reached max convergence
        let need_repaint = {
            let renderers = self.renderers.lock().unwrap();
            let active_renderer = &renderers[self.selected_renderer_idx];
            self.animate
                || self.auto_rotate_camera
                || (active_renderer.supports_accumulation() && self.accum_frame < active_renderer.max_accumulation_frames())
        };
        if need_repaint {
            ctx.request_repaint();
        }

        // Control Panel Sidebar
        egui::Panel::left("pt_control_panel")
            .resizable(true)
            .default_size(340.0)
            .show(ui, |ui| {
                ui.add_space(10.0);
                ui.heading("Render Mode Controls");
                ui.add_space(15.0);

                let active_error = {
                    let renderers = self.renderers.lock().unwrap();
                    renderers[self.selected_renderer_idx].error().map(|e| e.to_string())
                };

                if let Some(ref err) = active_error {
                    ui.group(|ui| {
                        ui.colored_label(egui::Color32::LIGHT_RED, "❌ Shader Compilation Error:");
                        egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| {
                            ui.weak(err);
                        });
                    });
                    ui.add_space(10.0);
                }

                // Render Mode Selection
                ui.group(|ui| {
                    ui.label("Rendering Approach:");
                    let prev_idx = self.selected_renderer_idx;
                    ui.horizontal(|ui| {
                        let renderers = self.renderers.lock().unwrap();
                        for (idx, r) in renderers.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_renderer_idx, idx, r.name());
                        }
                    });
                    if self.selected_renderer_idx != prev_idx {
                        self.accum_frame = 0;
                    }
                });
                ui.add_space(10.0);

                // Ray stats
                let supports_accum = {
                    let renderers = self.renderers.lock().unwrap();
                    renderers[self.selected_renderer_idx].supports_accumulation()
                };
                if supports_accum {
                    let max_accum = {
                        let renderers = self.renderers.lock().unwrap();
                        renderers[self.selected_renderer_idx].max_accumulation_frames()
                    };
                    ui.group(|ui| {
                        ui.label(format!("Accumulated Frames: {} / {}", self.accum_frame, max_accum));
                        if ui.button("Reset Accumulation").clicked() {
                            self.accum_frame = 0;
                        }
                    });
                    ui.add_space(10.0);
                }

                // Active Renderer Specific controls
                let mut renderer_changed = false;
                {
                    let mut renderers = self.renderers.lock().unwrap();
                    if let Some(r) = renderers.get_mut(self.selected_renderer_idx) {
                        renderer_changed = r.draw_ui(ui);
                    }
                }
                if renderer_changed {
                    self.accum_frame = 0;
                }
                ui.add_space(10.0);

                // Animation Controls
                ui.group(|ui| {
                    ui.label("Animation Controls");
                    let mut anim_changed = false;
                    if ui.checkbox(&mut self.animate, "Animate Objects").changed() {
                        anim_changed = true;
                    }
                    if ui.checkbox(&mut self.auto_rotate_camera, "Auto-rotate Camera").changed() {
                        anim_changed = true;
                    }
                    ui.add(egui::Slider::new(&mut self.animation_speed, 0.1..=4.0).text("Speed Multiplier"));
                    if ui.checkbox(&mut self.render_frame_by_frame, "Frame-by-Frame Convergence").changed() {
                        anim_changed = true;
                    }
                    if self.render_frame_by_frame {
                        if ui.add(egui::Slider::new(&mut self.fps, 10.0..=60.0).text("FPS")).changed() {
                            anim_changed = true;
                        }
                    }
                    if ui.button("Reset Time").clicked() {
                        self.time = 0.0;
                        self.accum_frame = 0;
                    }
                    if anim_changed {
                        self.accum_frame = 0;
                    }
                });
                ui.add_space(10.0);

                // Interactive Material Editor
                ui.group(|ui| {
                    ui.label("Interactive Material Editor");
                    
                    let prev_idx = self.selected_material_idx;
                    egui::ComboBox::from_label("Selected Material")
                        .selected_text(&self.materials[self.selected_material_idx].name)
                        .show_ui(ui, |ui| {
                            for (idx, mat) in self.materials.iter().enumerate() {
                                ui.selectable_value(&mut self.selected_material_idx, idx, &mat.name);
                            }
                        });

                    if self.selected_material_idx != prev_idx {
                        // Reset selection focus but don't need to rebuild
                    }

                    ui.add_space(8.0);
                    let mut mat_changed = false;
                    let mat = &mut self.materials[self.selected_material_idx];

                    // Base Color Picker
                    ui.horizontal(|ui| {
                        ui.label("Base Color:");
                        if ui.color_edit_button_rgb(&mut mat.base_color).changed() {
                            mat_changed = true;
                        }
                    });

                    // Roughness & Metallic
                    if ui.add(egui::Slider::new(&mut mat.roughness, 0.0..=1.0).text("Roughness")).changed() {
                        mat_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut mat.metallic, 0.0..=1.0).text("Metallic")).changed() {
                        mat_changed = true;
                    }

                    // Transmission & Refraction (IOR)
                    if ui.add(egui::Slider::new(&mut mat.transmission, 0.0..=1.0).text("Transmission (Glass)")).changed() {
                        mat_changed = true;
                    }
                    if ui.add(egui::Slider::new(&mut mat.ior, 1.0..=2.5).text("Index of Refraction")).changed() {
                        mat_changed = true;
                    }

                    // Emissive
                    ui.horizontal(|ui| {
                        ui.label("Emissive:");
                        if ui.color_edit_button_rgb(&mut mat.emissive).changed() {
                            mat_changed = true;
                        }
                    });
                    
                    if mat_changed {
                        self.accum_frame = 0;
                        // Write updated materials list to GPU buffer
                        if let Some(render_state) = _frame.wgpu_render_state() {
                            let gpu_mats: Vec<GpuMaterial> = self.materials.iter().map(|m| m.to_gpu()).collect();
                            let renderer_lock = render_state.renderer.read();
                            if let Some(res) = renderer_lock.callback_resources.get::<SharedRenderResources>() {
                                render_state.queue.write_buffer(&res.material_buffer, 0, bytemuck::cast_slice(&gpu_mats));
                            }
                        }
                    }
                });

                ui.add_space(20.0);
                if ui.button("Reset Scene Defaults").clicked() {
                    self.camera_rot = [0.0, 0.3];
                    self.camera_zoom = 5.5;
                    self.animate = false;
                    self.auto_rotate_camera = false;
                    self.animation_speed = 1.0;
                    self.time = 0.0;
                    self.accum_frame = 0;
                    
                    // Reset materials
                    self.materials[0].base_color = [0.75, 0.15, 0.15]; // Red
                    self.materials[1].base_color = [0.15, 0.75, 0.15]; // Green
                    self.materials[2].base_color = [0.75, 0.75, 0.75]; // White
                    self.materials[3].emissive = [12.0, 12.0, 12.0];
                    self.materials[4].roughness = 0.15;
                    self.materials[4].metallic = 1.0;
                    self.materials[5].roughness = 0.05;
                    self.materials[5].metallic = 1.0;
                    self.materials[6].transmission = 1.0;
                    self.materials[6].roughness = 0.0;

                    let mut renderers = self.renderers.lock().unwrap();
                    for r in renderers.iter_mut() {
                        r.reset_defaults();
                    }

                    if let Some(render_state) = _frame.wgpu_render_state() {
                        let gpu_mats: Vec<GpuMaterial> = self.materials.iter().map(|m| m.to_gpu()).collect();
                        let renderer_lock = render_state.renderer.read();
                        if let Some(res) = renderer_lock.callback_resources.get::<SharedRenderResources>() {
                            render_state.queue.write_buffer(&res.material_buffer, 0, bytemuck::cast_slice(&gpu_mats));
                        }
                    }
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);
                ui.vertical_centered(|ui| {
                    ui.weak("Drag on the canvas to rotate camera");
                    ui.weak("Scroll wheel to zoom camera");
                });
            });

        // Viewport canvas
        egui::CentralPanel::default().show(ui, |ui| {
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.render_canvas(ui);
            });
        });
    }
}

impl RaymarchApp {
    fn render_canvas(&mut self, ui: &mut egui::Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let mut anim_active = false;

        let dt_step = if self.animate {
            if self.render_frame_by_frame {
                let max_accum = {
                    let renderers = self.renderers.lock().unwrap();
                    renderers[self.selected_renderer_idx].max_accumulation_frames()
                };
                if self.accum_frame >= max_accum {
                    (1.0 / self.fps) * self.animation_speed
                } else {
                    0.0
                }
            } else {
                dt * self.animation_speed
            }
        } else {
            0.0
        };

        if dt_step > 0.0 {
            self.time += dt_step;
            self.accum_frame = 0;
            anim_active = true;
        }

        if self.auto_rotate_camera {
            self.camera_rot[0] += dt * 0.15 * self.animation_speed;
            anim_active = true;
        }

        // Handle camera navigation
        let mut cam_changed = false;
        if response.dragged() {
            self.camera_rot[0] += response.drag_delta().x * 0.005; // Yaw
            self.camera_rot[1] = (self.camera_rot[1] + response.drag_delta().y * 0.005)
                .clamp(0.05, 1.56); // Pitch
            cam_changed = true;
        }

        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta != 0.0 {
            self.camera_zoom = (self.camera_zoom - scroll_delta * 0.005).clamp(2.0, 12.0);
            cam_changed = true;
        }

        if cam_changed || anim_active {
            self.accum_frame = 0;
        }

        // Prepare uniforms struct
        let mut uniforms = PathTraceUniforms {
            resolution: [rect.width(), rect.height()],
            camera_rot: self.camera_rot,
            camera_zoom: self.camera_zoom,
            frame_index: self.frame_index,
            max_depth: 4,
            samples_per_frame: 1,
            aperture: 0.02,
            focal_distance: 5.5,
            env_light_intensity: 0.4,
            prev_camera_zoom: self.prev_camera_zoom,
            prev_camera_rot: self.prev_camera_rot,
            time: self.time,
            accum_frame: self.accum_frame,
            bvh_offsets: self.bvh_offsets,
            tri_offsets: self.tri_offsets,
            dt: if self.render_frame_by_frame { (1.0 / self.fps) * self.animation_speed } else { dt * self.animation_speed },
            render_mode: 0,
            tonemap_mode: 0,
            _pad_align2: 0,
        };

        {
            let renderers = self.renderers.lock().unwrap();
            if let Some(r) = renderers.get(self.selected_renderer_idx) {
                r.update_uniforms(&mut uniforms);
            }
        }

        // Update previous camera parameters for the next frame
        self.prev_camera_rot = self.camera_rot;
        self.prev_camera_zoom = self.camera_zoom;

        // Increment accumulation frame if static and below limit
        let supports_accum = {
            let renderers = self.renderers.lock().unwrap();
            renderers[self.selected_renderer_idx].supports_accumulation()
        };
        let max_accum = {
            let renderers = self.renderers.lock().unwrap();
            renderers[self.selected_renderer_idx].max_accumulation_frames()
        };
        let is_static = supports_accum && !self.auto_rotate_camera && (!self.animate || self.render_frame_by_frame);
        if is_static {
            if self.accum_frame < max_accum {
                self.accum_frame += 1;
            }
        } else {
            self.accum_frame = 0;
        }

        // Always increment frame_index for ping-ponging and noise seeding
        self.frame_index = self.frame_index.wrapping_add(1);

        // Custom WGPU callback
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            RaymarchCallback {
                uniforms,
                renderer_idx: self.selected_renderer_idx,
            },
        ));
    }
}

struct RaymarchCallback {
    uniforms: PathTraceUniforms,
    renderer_idx: usize,
}

impl egui_wgpu::CallbackTrait for RaymarchCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut cmd_buffers = Vec::new();
        if let (Some(shared), Some(renderers_res)) = (
            resources.get::<SharedRenderResources>(),
            resources.get::<RendererVector>(),
        ) {
            // Write uniforms to the GPU buffer
            queue.write_buffer(&shared.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));

            // Ensure accumulation textures are allocated at the correct size
            let width = self.uniforms.resolution[0].max(1.0) as u32;
            let height = self.uniforms.resolution[1].max(1.0) as u32;

            let mut tex_lock = shared.textures.lock().unwrap();
            let need_recreate = tex_lock.as_ref().map_or(true, |tex| tex.width != width || tex.height != height);

            if need_recreate {
                *tex_lock = Some(AccumTextures::new(device, width, height, shared));
            }

            if let Some(ref textures) = *tex_lock {
                let use_a = self.uniforms.frame_index % 2 == 0;
                let target_view = if use_a { &textures.view_a } else { &textures.view_b };
                let bind_group_pt = if use_a { &textures.bind_group_pt_a } else { &textures.bind_group_pt_b };

                let renderers = renderers_res.renderers.lock().unwrap();
                if let Some(renderer) = renderers.get(self.renderer_idx) {
                    cmd_buffers = renderer.prepare(device, queue, &self.uniforms, shared, target_view, bind_group_pt);
                }
            }
        }
        cmd_buffers
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(shared) = resources.get::<SharedRenderResources>() {
            let tex_lock = shared.textures.lock().unwrap();
            if let Some(ref textures) = *tex_lock {
                let use_a = self.uniforms.frame_index % 2 == 0;
                let bind_group_disp = if use_a { &textures.bind_group_disp_a } else { &textures.bind_group_disp_b };

                render_pass.set_pipeline(&shared.display_pipeline);
                render_pass.set_bind_group(0, bind_group_disp, &[]);
                render_pass.draw(0..6, 0..1);
            }
        }
    }
}

pub struct AccumTextures {
    width: u32,
    height: u32,
    #[allow(dead_code)]
    texture_a: wgpu::Texture,
    view_a: wgpu::TextureView,
    #[allow(dead_code)]
    texture_b: wgpu::Texture,
    view_b: wgpu::TextureView,
    bind_group_pt_a: wgpu::BindGroup,
    bind_group_pt_b: wgpu::BindGroup,
    bind_group_disp_a: wgpu::BindGroup,
    bind_group_disp_b: wgpu::BindGroup,
}

impl AccumTextures {
    fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        res: &SharedRenderResources,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture_desc = wgpu::TextureDescriptor {
            label: Some("accum_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let texture_a = device.create_texture(&texture_desc);
        let view_a = texture_a.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_b = device.create_texture(&texture_desc);
        let view_b = texture_b.create_view(&wgpu::TextureViewDescriptor::default());

        // Bind Group PT A: reads B, writes A
        let bind_group_pt_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_pt_a"),
            layout: &res.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&res.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: res.bvh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: res.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: res.index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: res.tri_material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: res.material_buffer.as_entire_binding(),
                },
            ],
        });

        // Bind Group PT B: reads A, writes B
        let bind_group_pt_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_pt_b"),
            layout: &res.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&res.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: res.bvh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: res.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: res.index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: res.tri_material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: res.material_buffer.as_entire_binding(),
                },
            ],
        });

        // Display Bind Group A: reads A
        let bind_group_disp_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_disp_a"),
            layout: &res.display_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&res.sampler),
                },
            ],
        });

        // Display Bind Group B: reads B
        let bind_group_disp_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_disp_b"),
            layout: &res.display_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&res.sampler),
                },
            ],
        });

        Self {
            width,
            height,
            texture_a,
            view_a,
            texture_b,
            view_b,
            bind_group_pt_a,
            bind_group_pt_b,
            bind_group_disp_a,
            bind_group_disp_b,
        }
    }
}
