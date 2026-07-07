use crate::{Renderer, SharedRenderResources, PathTraceUniforms};
use eframe::egui;
use eframe::egui_wgpu::wgpu;

pub struct PathTracer {
    path_trace_pipeline: wgpu::RenderPipeline,
    
    // Shader hot reloading
    path: std::path::PathBuf,
    last_modified: Option<std::time::SystemTime>,
    error: Option<String>,

    // Path tracer control parameters
    max_depth: u32,
    samples_per_frame: u32,
    aperture: f32,
    focal_distance: f32,
    env_light_intensity: f32,
    max_accumulation_frames: u32,
}

impl PathTracer {
    pub fn new(device: &wgpu::Device, pt_pipeline_layout: &wgpu::PipelineLayout) -> Self {
        // Compile path trace shader
        let path_trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("path_trace_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./path_trace.wgsl").into()),
        });

        let path_trace_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("path_trace_pipeline"),
            layout: Some(pt_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &path_trace_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &path_trace_shader,
                entry_point: Some("fs_path_trace"),
                targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let path = std::path::PathBuf::from("src/path_trace.wgsl");
        let last_modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();

        Self {
            path_trace_pipeline,
            path,
            last_modified,
            error: None,
            max_depth: 4,
            samples_per_frame: 1,
            aperture: 0.02,
            focal_distance: 5.5,
            env_light_intensity: 0.4,
            max_accumulation_frames: 1024,
        }
    }
}

impl Renderer for PathTracer {
    fn name(&self) -> &'static str {
        "Path Tracer"
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _uniforms: &PathTraceUniforms,
        _shared: &SharedRenderResources,
        target_view: &wgpu::TextureView,
        bind_group_pt: &wgpu::BindGroup,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut pt_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pt_pass_encoder"),
        });

        {
            let mut render_pass = pt_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_trace_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.path_trace_pipeline);
            render_pass.set_bind_group(0, bind_group_pt, &[]);
            render_pass.draw(0..6, 0..1);
        }

        vec![pt_encoder.finish()]
    }

    fn check_reload(&mut self, device: &wgpu::Device, render_state: &eframe::egui_wgpu::RenderState) -> bool {
        if let Ok(metadata) = std::fs::metadata(&self.path) {
            if let Ok(modified) = metadata.modified() {
                if self.last_modified.map_or(true, |last| modified > last) {
                    self.last_modified = Some(modified);
                    
                    let shader_source = match std::fs::read_to_string(&self.path) {
                        Ok(src) => src,
                        Err(e) => {
                            self.error = Some(format!("Failed to read shader file: {}", e));
                            return false;
                        }
                    };

                    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("path_trace_shader_hot"),
                        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                    });

                    let validation_error = pollster::block_on(error_scope.pop());

                    if let Some(err) = validation_error {
                        self.error = Some(format!("Shader validation error:\n{}", err));
                        log::error!("Path trace shader validation error: {}", err);
                        return false;
                    } else {
                        // Recreate pipeline using layouts from render state callback resources
                        let shared = render_state.renderer.read().callback_resources.get::<SharedRenderResources>().cloned();
                        if let Some(shared_res) = shared {
                            let pt_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("pt_pipeline_layout_hot"),
                                bind_group_layouts: &[Some(&shared_res.bind_group_layout)],
                                immediate_size: 0,
                            });

                            let new_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("path_trace_pipeline_hot"),
                                layout: Some(&pt_pipeline_layout),
                                vertex: wgpu::VertexState {
                                    module: &shader,
                                    entry_point: Some("vs_main"),
                                    buffers: &[],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                },
                                fragment: Some(wgpu::FragmentState {
                                    module: &shader,
                                    entry_point: Some("fs_path_trace"),
                                    targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                }),
                                primitive: wgpu::PrimitiveState::default(),
                                depth_stencil: None,
                                multisample: wgpu::MultisampleState::default(),
                                multiview_mask: None,
                                cache: None,
                            });

                            self.path_trace_pipeline = new_pipeline;
                            self.error = None;
                            log::info!("Path trace shader hot-reloaded successfully!");
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.group(|ui| {
            ui.label("Path Tracer Controls");
            ui.add_space(5.0);

            let prev_depth = self.max_depth;
            ui.add(egui::Slider::new(&mut self.max_depth, 1..=8).text("Max Bounces"));
            if self.max_depth != prev_depth {
                changed = true;
            }

            let prev_samples = self.samples_per_frame;
            ui.add(egui::Slider::new(&mut self.samples_per_frame, 1..=4).text("Samples per Frame"));
            if self.samples_per_frame != prev_samples {
                changed = true;
            }

            let prev_aperture = self.aperture;
            ui.add(egui::Slider::new(&mut self.aperture, 0.0..=0.2).text("Aperture (DoF)"));
            if self.aperture != prev_aperture {
                changed = true;
            }

            let prev_focal = self.focal_distance;
            ui.add(egui::Slider::new(&mut self.focal_distance, 1.0..=15.0).text("Focus Distance"));
            if self.focal_distance != prev_focal {
                changed = true;
            }

            let prev_env = self.env_light_intensity;
            ui.add(egui::Slider::new(&mut self.env_light_intensity, 0.0..=2.0).text("Sky Intensity"));
            if self.env_light_intensity != prev_env {
                changed = true;
            }

            let prev_max_accum = self.max_accumulation_frames;
            ui.add(egui::Slider::new(&mut self.max_accumulation_frames, 1..=2048).text("Max Accum Frames"));
            if self.max_accumulation_frames != prev_max_accum {
                changed = true;
            }
        });

        changed
    }

    fn update_uniforms(&self, uniforms: &mut PathTraceUniforms) {
        uniforms.render_mode = 0;
        uniforms.tonemap_mode = 0;
        uniforms.max_depth = self.max_depth;
        uniforms.samples_per_frame = self.samples_per_frame;
        uniforms.aperture = self.aperture;
        uniforms.focal_distance = self.focal_distance;
        uniforms.env_light_intensity = self.env_light_intensity;
    }

    fn supports_accumulation(&self) -> bool {
        true
    }

    fn max_accumulation_frames(&self) -> u32 {
        self.max_accumulation_frames
    }
}
