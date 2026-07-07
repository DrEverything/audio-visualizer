use crate::{Renderer, SharedRenderResources, PathTraceUniforms};
use eframe::egui;
use eframe::egui_wgpu::wgpu;

pub struct Rasterizer {
    gbuffer_pipeline: wgpu::RenderPipeline,
    deferred_lighting_pipeline: wgpu::RenderPipeline,
    deferred_lighting_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer: std::sync::Mutex<Option<GBufferTextures>>,

    // Shader hot reloading
    path: std::path::PathBuf,
    last_modified: Option<std::time::SystemTime>,
    error: Option<String>,
}

impl Rasterizer {
    pub fn new(device: &wgpu::Device, pt_pipeline_layout: &wgpu::PipelineLayout) -> Self {
        // Compile deferred rasterization shader
        let rasterize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rasterize_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./rasterize.wgsl").into()),
        });

        // Create G-buffer Rasterization Render Pipeline
        let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gbuffer_pipeline"),
            layout: Some(pt_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rasterize_shader,
                entry_point: Some("vs_gbuffer"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &rasterize_shader,
                entry_point: Some("fs_gbuffer"),
                targets: &[
                    Some(wgpu::TextureFormat::Rgba8Unorm.into()),
                    Some(wgpu::TextureFormat::Rgba16Float.into()),
                    Some(wgpu::TextureFormat::Rgba8Unorm.into()),
                    Some(wgpu::TextureFormat::Rg16Float.into()),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Create Bind Group Layout for Deferred Lighting pass
        let deferred_lighting_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("deferred_lighting_bind_group_layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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

        let deferred_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("deferred_pipeline_layout"),
            bind_group_layouts: &[Some(&deferred_lighting_bind_group_layout)],
            immediate_size: 0,
        });

        // Create Deferred Lighting Render Pipeline
        let deferred_lighting_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("deferred_lighting_pipeline"),
            layout: Some(&deferred_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rasterize_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &rasterize_shader,
                entry_point: Some("fs_deferred_lighting"),
                targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let path = std::path::PathBuf::from("src/rasterize.wgsl");
        let last_modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();

        Self {
            gbuffer_pipeline,
            deferred_lighting_pipeline,
            deferred_lighting_bind_group_layout,
            gbuffer: std::sync::Mutex::new(None),
            path,
            last_modified,
            error: None,
        }
    }
}

impl Renderer for Rasterizer {
    fn name(&self) -> &'static str {
        "Deferred PBR"
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        uniforms: &PathTraceUniforms,
        shared: &SharedRenderResources,
        target_view: &wgpu::TextureView,
        bind_group_pt: &wgpu::BindGroup,
    ) -> Vec<wgpu::CommandBuffer> {
        let width = uniforms.resolution[0].max(1.0) as u32;
        let height = uniforms.resolution[1].max(1.0) as u32;

        let mut gb_lock = self.gbuffer.lock().unwrap();
        let need_recreate_gb = gb_lock.as_ref().map_or(true, |gb| gb.width != width || gb.height != height);
        if need_recreate_gb {
            *gb_lock = Some(GBufferTextures::new(device, width, height, shared, &self.deferred_lighting_bind_group_layout));
        }

        let mut cmd_buffers = Vec::new();

        if let Some(ref gb_textures) = *gb_lock {
            // 1. G-buffer Pass
            let mut gb_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gbuffer_pass_encoder"),
            });

            {
                let mut rpass = gb_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gbuffer_raster_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gb_textures.albedo_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gb_textures.normal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gb_textures.material_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gb_textures.motion_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gb_textures.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                rpass.set_pipeline(&self.gbuffer_pipeline);
                rpass.set_bind_group(0, bind_group_pt, &[]);

                let total_indices = (shared.index_buffer.size() / 4) as u32;
                let tri_offsets = &uniforms.tri_offsets;

                for inst_id in 0..4 {
                    let start_idx = tri_offsets[inst_id as usize] * 3;
                    let end_idx = if inst_id < 3 {
                        tri_offsets[inst_id as usize + 1] * 3
                    } else {
                        total_indices
                    };
                    let count = end_idx - start_idx;

                    rpass.draw(0..count, inst_id..inst_id + 1);
                }
            }
            cmd_buffers.push(gb_encoder.finish());

            // 2. Deferred Lighting Pass
            let mut def_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deferred_lighting_encoder"),
            });

            {
                let mut rpass = def_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("deferred_shading_pass"),
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

                rpass.set_pipeline(&self.deferred_lighting_pipeline);
                rpass.set_bind_group(0, &gb_textures.bind_group_lighting, &[]);
                rpass.draw(0..6, 0..1);
            }
            cmd_buffers.push(def_encoder.finish());
        }

        cmd_buffers
    }

    fn check_reload(&mut self, device: &wgpu::Device, render_state: &eframe::egui_wgpu::RenderState) -> bool {
        if let Ok(metadata) = std::fs::metadata(&self.path) {
            if let Ok(modified) = metadata.modified() {
                if self.last_modified.map_or(true, |last| modified > last) {
                    self.last_modified = Some(modified);

                    let shader_source = match std::fs::read_to_string(&self.path) {
                        Ok(src) => src,
                        Err(e) => {
                            self.error = Some(format!("Failed to read rasterize shader file: {}", e));
                            return false;
                        }
                    };

                    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("rasterize_shader_hot"),
                        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                    });

                    let validation_error = pollster::block_on(error_scope.pop());

                    if let Some(err) = validation_error {
                        self.error = Some(format!("Shader validation error:\n{}", err));
                        log::error!("Rasterize shader validation error: {}", err);
                        return false;
                    } else {
                        let shared = render_state.renderer.read().callback_resources.get::<SharedRenderResources>().cloned();
                        if let Some(shared_res) = shared {
                            let pt_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("pt_pipeline_layout_hot"),
                                bind_group_layouts: &[Some(&shared_res.bind_group_layout)],
                                immediate_size: 0,
                            });

                            let deferred_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("deferred_pipeline_layout_hot"),
                                bind_group_layouts: &[Some(&self.deferred_lighting_bind_group_layout)],
                                immediate_size: 0,
                            });

                            let new_gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("gbuffer_pipeline_hot"),
                                layout: Some(&pt_pipeline_layout),
                                vertex: wgpu::VertexState {
                                    module: &shader,
                                    entry_point: Some("vs_gbuffer"),
                                    buffers: &[],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                },
                                fragment: Some(wgpu::FragmentState {
                                    module: &shader,
                                    entry_point: Some("fs_gbuffer"),
                                    targets: &[
                                        Some(wgpu::TextureFormat::Rgba8Unorm.into()),
                                        Some(wgpu::TextureFormat::Rgba16Float.into()),
                                        Some(wgpu::TextureFormat::Rgba8Unorm.into()),
                                        Some(wgpu::TextureFormat::Rg16Float.into()),
                                    ],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                }),
                                primitive: wgpu::PrimitiveState {
                                    topology: wgpu::PrimitiveTopology::TriangleList,
                                    strip_index_format: None,
                                    front_face: wgpu::FrontFace::Ccw,
                                    cull_mode: None,
                                    unclipped_depth: false,
                                    polygon_mode: wgpu::PolygonMode::Fill,
                                    conservative: false,
                                },
                                depth_stencil: Some(wgpu::DepthStencilState {
                                    format: wgpu::TextureFormat::Depth32Float,
                                    depth_write_enabled: Some(true),
                                    depth_compare: Some(wgpu::CompareFunction::Less),
                                    stencil: wgpu::StencilState::default(),
                                    bias: wgpu::DepthBiasState::default(),
                                }),
                                multisample: wgpu::MultisampleState::default(),
                                multiview_mask: None,
                                cache: None,
                            });

                            let new_deferred_lighting_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("deferred_lighting_pipeline_hot"),
                                layout: Some(&deferred_pipeline_layout),
                                vertex: wgpu::VertexState {
                                    module: &shader,
                                    entry_point: Some("vs_main"),
                                    buffers: &[],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                },
                                fragment: Some(wgpu::FragmentState {
                                    module: &shader,
                                    entry_point: Some("fs_deferred_lighting"),
                                    targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                }),
                                primitive: wgpu::PrimitiveState::default(),
                                depth_stencil: None,
                                multisample: wgpu::MultisampleState::default(),
                                multiview_mask: None,
                                cache: None,
                            });

                            self.gbuffer_pipeline = new_gbuffer_pipeline;
                            self.deferred_lighting_pipeline = new_deferred_lighting_pipeline;

                            // Force G-buffer textures recreate to bind correctly
                            if let Ok(mut gb_guard) = self.gbuffer.lock() {
                                *gb_guard = None;
                            }

                            self.error = None;
                            log::info!("Rasterize shader hot-reloaded successfully!");
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
        ui.group(|ui| {
            ui.label("Rasterizer Controls");
            ui.weak("Rendering Cornell Box with Deferred PBR...");
        });
        false
    }

    fn update_uniforms(&self, uniforms: &mut PathTraceUniforms) {
        uniforms.render_mode = 1;
        uniforms.tonemap_mode = 1;
    }
}

pub struct GBufferTextures {
    width: u32,
    height: u32,
    #[allow(dead_code)]
    albedo_texture: wgpu::Texture,
    albedo_view: wgpu::TextureView,
    #[allow(dead_code)]
    normal_texture: wgpu::Texture,
    normal_view: wgpu::TextureView,
    #[allow(dead_code)]
    material_texture: wgpu::Texture,
    material_view: wgpu::TextureView,
    #[allow(dead_code)]
    motion_texture: wgpu::Texture,
    motion_view: wgpu::TextureView,
    #[allow(dead_code)]
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    bind_group_lighting: wgpu::BindGroup,
}

impl GBufferTextures {
    fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        shared: &SharedRenderResources,
        deferred_lighting_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_albedo"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let albedo_view = albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_normal"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let material_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_material"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let material_view = material_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let motion_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_motion"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let motion_view = motion_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_lighting = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group_lighting"),
            layout: deferred_lighting_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shared.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&motion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&shared.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: shared.material_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            width,
            height,
            albedo_texture,
            albedo_view,
            normal_texture,
            normal_view,
            material_texture,
            material_view,
            motion_texture,
            motion_view,
            depth_texture,
            depth_view,
            bind_group_lighting,
        }
    }
}
