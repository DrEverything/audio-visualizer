use crate::{Renderer, SharedRenderResources, PathTraceUniforms};
use eframe::egui;
use eframe::egui_wgpu::wgpu;

pub struct Raymarcher {
    raymarch_pipeline: wgpu::RenderPipeline,

    // Shader hot reloading
    path: std::path::PathBuf,
    last_modified: Option<std::time::SystemTime>,
    error: Option<String>,

    aa_level: u32,
}

impl Raymarcher {
    pub fn new(device: &wgpu::Device, pt_pipeline_layout: &wgpu::PipelineLayout) -> Self {
        // Compile raymarch shader
        let raymarch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./raymarch.wgsl").into()),
        });

        // Create Raymarching Render Pipeline
        let raymarch_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raymarch_pipeline"),
            layout: Some(pt_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &raymarch_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &raymarch_shader,
                entry_point: Some("fs_raymarch"),
                targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let path = std::path::PathBuf::from("src/raymarch.wgsl");
        let last_modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();

        Self {
            raymarch_pipeline,
            path,
            last_modified,
            error: None,
            aa_level: 2,
        }
    }
}

impl Renderer for Raymarcher {
    fn name(&self) -> &'static str {
        "Raymarching"
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
        let mut rm_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rm_pass_encoder"),
        });

        {
            let mut render_pass = rm_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raymarch_pass"),
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

            render_pass.set_pipeline(&self.raymarch_pipeline);
            render_pass.set_bind_group(0, bind_group_pt, &[]);
            render_pass.draw(0..6, 0..1);
        }

        vec![rm_encoder.finish()]
    }

    fn check_reload(&mut self, device: &wgpu::Device, render_state: &eframe::egui_wgpu::RenderState) -> bool {
        if let Ok(metadata) = std::fs::metadata(&self.path) {
            if let Ok(modified) = metadata.modified() {
                if self.last_modified.map_or(true, |last| modified > last) {
                    self.last_modified = Some(modified);

                    let shader_source = match std::fs::read_to_string(&self.path) {
                        Ok(src) => src,
                        Err(e) => {
                            self.error = Some(format!("Failed to read raymarch shader file: {}", e));
                            return false;
                        }
                    };

                    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("raymarch_shader_hot"),
                        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                    });

                    let validation_error = pollster::block_on(error_scope.pop());

                    if let Some(err) = validation_error {
                        self.error = Some(format!("Shader validation error:\n{}", err));
                        log::error!("Raymarch shader validation error: {}", err);
                        return false;
                    } else {
                        let shared = render_state.renderer.read().callback_resources.get::<SharedRenderResources>().cloned();
                        if let Some(shared_res) = shared {
                            let pt_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("pt_pipeline_layout_hot"),
                                bind_group_layouts: &[Some(&shared_res.bind_group_layout)],
                                immediate_size: 0,
                            });

                            let new_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("raymarch_pipeline_hot"),
                                layout: Some(&pt_pipeline_layout),
                                vertex: wgpu::VertexState {
                                    module: &shader,
                                    entry_point: Some("vs_main"),
                                    buffers: &[],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                },
                                fragment: Some(wgpu::FragmentState {
                                    module: &shader,
                                    entry_point: Some("fs_raymarch"),
                                    targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                }),
                                primitive: wgpu::PrimitiveState::default(),
                                depth_stencil: None,
                                multisample: wgpu::MultisampleState::default(),
                                multiview_mask: None,
                                cache: None,
                            });

                            self.raymarch_pipeline = new_pipeline;
                            self.error = None;
                            log::info!("Raymarch shader hot-reloaded successfully!");
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
            ui.label("Raymarching Controls");
            ui.weak("Rendering procedural SDFs...");
            ui.add_space(4.0);
            
            ui.horizontal(|ui| {
                ui.label("Anti-aliasing (AA):");
                let prev_aa = self.aa_level;
                egui::ComboBox::from_id_salt("aa_level_select")
                    .selected_text(format!("{}x", self.aa_level))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.aa_level, 1, "1x (None)");
                        ui.selectable_value(&mut self.aa_level, 2, "2x");
                        ui.selectable_value(&mut self.aa_level, 3, "3x");
                        ui.selectable_value(&mut self.aa_level, 4, "4x");
                    });
                if self.aa_level != prev_aa {
                    changed = true;
                }
            });
        });
        changed
    }

    fn update_uniforms(&self, uniforms: &mut PathTraceUniforms) {
        uniforms.render_mode = 2;
        uniforms.tonemap_mode = 0;
        uniforms.samples_per_frame = self.aa_level;
    }
}
