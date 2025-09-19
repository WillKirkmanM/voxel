use ab_glyph::FontArc;
use cgmath::{Matrix4, Vector3};
use crossbeam_channel::{Receiver, Sender};
use player::Frustum;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_text::glyph_brush::{Section, Text};
use wgpu_text::{BrushBuilder, TextBrush};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod block;
mod chunk;
mod player;
mod texture;
mod world;

const MAX_MESH_JOBS_PER_FRAME: usize = 4;

struct ChunkMesh {
    pos: (i32, i32),
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
    texture_layer: u32,
    ambient_occlusion: f32,
}

impl Vertex {
    fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as BufferAddress,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as BufferAddress,
                    shader_location: 3,
                    format: VertexFormat::Uint32,
                },
                VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as BufferAddress
                        + mem::size_of::<u32>() as BufferAddress,
                    shader_location: 4,
                    format: VertexFormat::Float32,
                },
            ],
        }
    }
}

struct ChunkRenderData {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    num_indices: u32,
    model_slot: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelUniform {
    model: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    screen_dims_and_flags: [f32; 4],
}

struct GpuState {
    window: Arc<Window>,
    device: Device,
    queue: Queue,
    surface: Surface<'static>,
    surface_config: SurfaceConfiguration,
    render_pipeline: RenderPipeline,
    world: Arc<Mutex<world::World>>,
    chunk_render_data: HashMap<(i32, i32), ChunkRenderData>,
    diffuse_bind_group: BindGroup,
    player: player::Player,
    player_controller: player::PlayerController,
    camera_uniform: player::CameraUniform,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    frustum: Frustum,
    scene_uniforms: SceneUniforms,
    scene_buffer: Buffer,
    scene_bind_group: BindGroup,
    scene_bind_group_layout: BindGroupLayout,
    scene_bind_group_dummy: BindGroup,
    dummy_depth_texture: texture::Texture,
    model_buffer: Buffer,
    model_bind_group: BindGroup,
    model_uniform_align: u64,
    max_model_slots: u32,
    next_model_slot: u32,
    free_model_slots: Vec<u32>,
    depth_texture: texture::Texture,
    text_brush: TextBrush,
    last_fps_update: Instant,
    frame_count: u32,
    fps: u32,
    frame_time: f32,
    mesh_sender: Sender<ChunkMesh>,
    mesh_receiver: Receiver<ChunkMesh>,
    chunks_in_flight: HashSet<(i32, i32)>,
    last_player_chunk_pos: (i32, i32),
}

impl GpuState {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = Instance::new(&InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2Array,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        let diffuse_texture_array = texture::Texture::load_texture_array(&device, &queue).unwrap();
        let diffuse_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&diffuse_texture_array.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&diffuse_texture_array.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let player = player::Player::new(cgmath::Point3::new(8.0, 80.0, 8.0));
        let player_controller = player::PlayerController::new();
        let camera_uniform = player::CameraUniform::new();
        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let model_uniform_size = std::mem::size_of::<ModelUniform>() as u64;
        let model_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(model_uniform_size),
                },
                count: None,
            }],
            label: Some("model_bind_group_layout"),
        });

        let render_diam = (world::RENDER_DISTANCE * 2 + 1) as u32;
        let max_chunks = render_diam * render_diam;
        let align = 256u64;
        let model_uniform_align = ((model_uniform_size + align - 1) / align) * align;
        let model_buffer_size = model_uniform_align * max_chunks as u64;

        let model_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Model Buffer (dynamic)"),
            size: model_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &model_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &model_buffer,
                    offset: 0,
                    size: NonZeroU64::new(model_uniform_align),
                }),
            }],
            label: Some("model_bind_group"),
        });

        let depth_texture = texture::Texture::create_depth_texture(
            &device,
            &surface_config,
            "depth_texture",
            1,
            true,
        );
        let dummy_depth_texture = texture::Texture::create_sampled_depth_texture(
            &device,
            &surface_config,
            "dummy_sampled_depth",
        );
        let scene_uniforms = SceneUniforms {
            screen_dims_and_flags: [size.width as f32, size.height as f32, 0.0, 0.0],
        };
        let scene_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Scene Uniform Buffer"),
            contents: bytemuck::cast_slice(&[scene_uniforms]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let scene_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
            label: Some("scene_bind_group_layout"),
        });
        let scene_bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &scene_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&depth_texture.view),
                },
            ],
            label: Some("scene_bind_group"),
        });
        let scene_bind_group_dummy = device.create_bind_group(&BindGroupDescriptor {
            layout: &scene_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: scene_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&dummy_depth_texture.view),
                },
            ],
            label: Some("scene_bind_group_dummy"),
        });

        let world = Arc::new(Mutex::new(world::World::new()));
        let chunk_render_data = HashMap::new();
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &camera_bind_group_layout,
                &model_bind_group_layout,
                &scene_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        let font = FontArc::try_from_slice(include_bytes!("../fonts/Roboto-Regular.ttf")).unwrap();
        let text_brush =
            BrushBuilder::using_font(font).build(&device, size.width, size.height, surface_format);
        let (mesh_sender, mesh_receiver) = crossbeam_channel::unbounded();

        let projection = cgmath::perspective(
            cgmath::Deg(player.fov_deg),
            surface_config.width as f32 / surface_config.height as f32,
            0.1,
            1000.0,
        );
        let view = Matrix4::look_to_rh(
            player.get_eye_position(),
            player.direction,
            Vector3::unit_y(),
        );
        let frustum = Frustum::from_matrix(player::OPENGL_TO_WGPU_MATRIX * projection * view);

        Self {
            window,
            device,
            queue,
            surface,
            surface_config,
            render_pipeline,
            world,
            chunk_render_data,
            diffuse_bind_group,
            player,
            player_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            frustum,
            scene_uniforms,
            scene_buffer,
            scene_bind_group,
            scene_bind_group_layout,
            scene_bind_group_dummy,
            dummy_depth_texture,
            model_buffer,
            model_bind_group,
            model_uniform_align,
            max_model_slots: max_chunks,
            next_model_slot: 0,
            free_model_slots: Vec::new(),
            depth_texture,
            text_brush,
            last_fps_update: Instant::now(),
            frame_count: 0,
            fps: 0,
            frame_time: 0.0,
            mesh_sender,
            mesh_receiver,
            chunks_in_flight: HashSet::new(),
            last_player_chunk_pos: (-9999, -9999),
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);

            self.depth_texture = texture::Texture::create_depth_texture(
                &self.device,
                &self.surface_config,
                "depth_texture",
                1,
                true,
            );
            self.dummy_depth_texture = texture::Texture::create_sampled_depth_texture(
                &self.device,
                &self.surface_config,
                "dummy_sampled_depth",
            );

            self.text_brush
                .resize_view(new_size.width as f32, new_size.height as f32, &self.queue);

            self.scene_uniforms.screen_dims_and_flags[0] = new_size.width as f32;
            self.scene_uniforms.screen_dims_and_flags[1] = new_size.height as f32;

            self.scene_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                layout: &self.scene_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.scene_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.depth_texture.view),
                    },
                ],
                label: Some("scene_bind_group"),
            });

            self.scene_bind_group_dummy = self.device.create_bind_group(&BindGroupDescriptor {
                layout: &self.scene_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.scene_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.dummy_depth_texture.view),
                    },
                ],
                label: Some("scene_bind_group_dummy"),
            });
        }
    }

    fn input_keyboard(&mut self, key: KeyCode, key_state: ElementState) {
        self.player_controller.process_keyboard(key, key_state);
        if key_state == ElementState::Pressed {
            match key {
                KeyCode::KeyR => {
                    let flag = self.scene_uniforms.screen_dims_and_flags[2];
                    self.scene_uniforms.screen_dims_and_flags[2] = 1.0 - flag;
                }
                _ => {}
            }
        }
    }

    fn input_mouse(&mut self, delta: (f64, f64)) {
        self.player_controller
            .process_mouse(delta, &mut self.player);
    }

    fn update(&mut self, dt: Duration) {
        self.frame_time = dt.as_secs_f32() * 1000.0;
        self.frame_count = self.frame_count.saturating_add(1);
        if self.last_fps_update.elapsed() >= Duration::from_secs(1) {
            self.fps = self.frame_count;
            self.frame_count = 0;
            self.last_fps_update = Instant::now();
        }

        self.player.update(dt, &self.player_controller, &self.world);
        self.update_chunks();

        let projection = cgmath::perspective(
            cgmath::Deg(self.player.fov_deg),
            self.surface_config.width as f32 / self.surface_config.height as f32,
            0.1,
            1000.0,
        );

        let view = Matrix4::look_to_rh(
            self.player.get_eye_position(),
            self.player.direction,
            Vector3::unit_y(),
        );

        self.camera_uniform.view_proj = (player::OPENGL_TO_WGPU_MATRIX * projection * view).into();
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.queue.write_buffer(
            &self.scene_buffer,
            0,
            bytemuck::cast_slice(&[self.scene_uniforms]),
        );

        self.frustum = Frustum::from_matrix(projection * view);
    }

    fn update_chunks(&mut self) {
        while let Ok(mesh) = self.mesh_receiver.try_recv() {
            if !self.chunks_in_flight.remove(&mesh.pos) {
                continue;
            }
            if mesh.indices.is_empty() {
                continue;
            }

            let vertex_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("Chunk VB {:?}", mesh.pos)),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: BufferUsages::VERTEX,
            });
            let index_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("Chunk IB {:?}", mesh.pos)),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: BufferUsages::INDEX,
            });

            let model_matrix = cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                (mesh.pos.0 * chunk::CHUNK_WIDTH as i32) as f32,
                0.0,
                (mesh.pos.1 * chunk::CHUNK_DEPTH as i32) as f32,
            ));
            let model_uniform = ModelUniform {
                model: model_matrix.into(),
            };

            let slot = if let Some(free) = self.free_model_slots.pop() {
                free
            } else {
                let s = self.next_model_slot;
                if s >= self.max_model_slots {
                    continue;
                }
                self.next_model_slot += 1;
                s
            };

            let offset_bytes = (slot as u64) * self.model_uniform_align;
            self.queue.write_buffer(
                &self.model_buffer,
                offset_bytes,
                bytemuck::cast_slice(&[model_uniform]),
            );

            self.chunk_render_data.insert(
                mesh.pos,
                ChunkRenderData {
                    vertex_buffer,
                    index_buffer,
                    num_indices: mesh.indices.len() as u32,
                    model_slot: slot,
                },
            );
        }

        let (player_cx, player_cz) = self.player.get_chunk_pos();
        if (player_cx, player_cz) == self.last_player_chunk_pos {
            return;
        }
        self.last_player_chunk_pos = (player_cx, player_cz);

        let mut required_chunks_vec = Vec::new();
        for x in -world::RENDER_DISTANCE..=world::RENDER_DISTANCE {
            for z in -world::RENDER_DISTANCE..=world::RENDER_DISTANCE {
                required_chunks_vec.push((player_cx + x, player_cz + z));
            }
        }
        required_chunks_vec.sort_by_key(|(cx, cz)| {
            let dx = cx - player_cx;
            let dz = cz - player_cz;
            dx * dx + dz * dz
        });

        let mut jobs_dispatched = 0;
        for pos in &required_chunks_vec {
            if jobs_dispatched >= MAX_MESH_JOBS_PER_FRAME {
                break;
            }
            if !self.chunk_render_data.contains_key(pos) && !self.chunks_in_flight.contains(pos) {
                self.chunks_in_flight.insert(*pos);
                let world_clone = Arc::clone(&self.world);
                let sender_clone = self.mesh_sender.clone();
                let job_pos = *pos;
                rayon::spawn(move || {
                    let mut world = world_clone.lock().unwrap();
                    world.ensure_chunk(job_pos.0, job_pos.1);
                    for dx in -1..=1 {
                        for dz in -1..=1 {
                            world.ensure_chunk(job_pos.0 + dx, job_pos.1 + dz);
                        }
                    }
                    if let Some(chunk) = world.get_chunk(job_pos.0, job_pos.1) {
                        let (vertices, indices) = chunk.build_mesh(&world);
                        let _ = sender_clone.send(ChunkMesh {
                            pos: job_pos,
                            vertices,
                            indices,
                        });
                    }
                });
                jobs_dispatched += 1;
            }
        }

        let required_set: HashSet<(i32, i32)> = required_chunks_vec.into_iter().collect();
        let mut to_remove = Vec::new();
        for (pos, _data) in &self.chunk_render_data {
            if !required_set.contains(pos) {
                to_remove.push(*pos);
            }
        }
        for pos in to_remove {
            if let Some(data) = self.chunk_render_data.remove(&pos) {
                self.free_model_slots.push(data.model_slot);
            }
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        let output_texture = self.surface.get_current_texture()?;
        let view = output_texture
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("3D Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.53,
                            g: 0.81,
                            b: 0.92,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(3, &self.scene_bind_group_dummy, &[]);

            for (pos, data) in &self.chunk_render_data {
                if !self.frustum.is_box_visible(*pos) {
                    continue;
                }
                let dynamic_offset = (data.model_slot as u64 * self.model_uniform_align) as u32;
                render_pass.set_bind_group(2, &self.model_bind_group, &[dynamic_offset]);
                render_pass.set_vertex_buffer(0, data.vertex_buffer.slice(..));
                render_pass.set_index_buffer(data.index_buffer.slice(..), IndexFormat::Uint32);
                render_pass.draw_indexed(0..data.num_indices, 0, 0..1);
            }
        }

        let fps_text = format!("FPS: {} ({:.2} ms)", self.fps, self.frame_time);
        let pos_text = format!(
            "\nPos: {:.2}, {:.2}, {:.2}",
            self.player.position.x, self.player.position.y, self.player.position.z
        );
        let ssr_text = if self.scene_uniforms.screen_dims_and_flags[2] > 0.5 {
            "\nSSR: ON"
        } else {
            "\nSSR: OFF"
        };
        let stats_section = Section::default()
            .with_screen_position((10.0, 10.0))
            .with_bounds((
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ))
            .with_text(vec![
                Text::new(&fps_text)
                    .with_color([1.0, 1.0, 1.0, 1.0])
                    .with_scale(30.0),
                Text::new(&pos_text)
                    .with_color([1.0, 1.0, 1.0, 1.0])
                    .with_scale(20.0),
                Text::new(ssr_text)
                    .with_color([1.0, 1.0, 1.0, 1.0])
                    .with_scale(20.0),
            ]);
        self.text_brush
            .queue(&self.device, &self.queue, vec![&stats_section])
            .unwrap();

        let mut text_encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut text_pass = text_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            self.text_brush.draw(&mut text_pass);
        }

        self.queue
            .submit(vec![encoder.finish(), text_encoder.finish()]);
        output_texture.present();
        Ok(())
    }

    fn window(&self) -> &Window {
        &self.window
    }
}

#[derive(Default)]
struct App {
    gpu_state: Option<GpuState>,
    last_frame: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu_state.is_none() {
            let attributes = Window::default_attributes().with_title("Minecraft Clone");
            let window = Arc::new(event_loop.create_window(attributes).unwrap());
            let gpu_state = pollster::block_on(GpuState::new(window));
            self.gpu_state = Some(gpu_state);

            if let Some(state) = self.gpu_state.as_mut() {
                state
                    .window()
                    .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                    .or_else(|_| {
                        state
                            .window()
                            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                    })
                    .ok();
                state.window().set_cursor_visible(false);
            }
            self.last_frame = Some(Instant::now());
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Some(state) = self.gpu_state.as_mut() {
            match event {
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(key),
                            state: key_state,
                            ..
                        },
                    ..
                } => {
                    state.input_keyboard(key, key_state);
                }
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(new_size) => {
                    state.resize(new_size);
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(self.last_frame.unwrap_or(now));
                    self.last_frame = Some(now);

                    state.update(dt);
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                _ => (),
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.gpu_state.as_mut() {
            if let DeviceEvent::MouseMotion { delta } = event {
                state.input_mouse(delta);
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.gpu_state.as_mut() {
            state.window().request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
