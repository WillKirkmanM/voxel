use anyhow::*;
use image::GenericImageView;
use wgpu::{TexelCopyBufferLayout, TexelCopyTextureInfoBase};

pub struct Texture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn load_texture_array(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self> {
        let texture_files = [
            (
                "grass_top.png",
                &include_bytes!("../textures/grass_top.png")[..],
            ),
            (
                "grass_side.png",
                &include_bytes!("../textures/grass_side.png")[..],
            ),
            ("dirt.png", &include_bytes!("../textures/dirt.png")[..]),
            ("stone.png", &include_bytes!("../textures/stone.png")[..]),
            ("wood.png", &include_bytes!("../textures/wood.png")[..]),
            ("leaves.png", &include_bytes!("../textures/leaves.png")[..]),
            ("sand.png", &include_bytes!("../textures/sand.png")[..]),
            ("water.png", &include_bytes!("../textures/water.png")[..]),
            ("cloud.png", &include_bytes!("../textures/cloud.png")[..]),
        ];

        let mut images = Vec::new();
        let mut target_dims: Option<(u32, u32)> = None;

        for (name, bytes) in texture_files {
            let img = image::load_from_memory(bytes)
                .with_context(|| format!("Failed to load texture {}", name))?;

            // For the first image record target dims. For subsequent images,
            // resize to match the first image if necessary (nearest filter keeps texels sharp).
            let rgba = if let Some((tw, th)) = target_dims {
                if img.dimensions() != (tw, th) {
                    image::imageops::resize(&img, tw, th, image::imageops::FilterType::Nearest)
                } else {
                    img.to_rgba8()
                }
            } else {
                let first = img.to_rgba8();
                target_dims = Some(first.dimensions());
                first
            };

            images.push(rgba);
        }

        let dimensions = images[0].dimensions();
        let layer_count = images.len() as u32;

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: layer_count,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture_array"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (i, image_data) in images.iter().enumerate() {
            let copy_size = wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            };

            let origin = wgpu::Origin3d {
                x: 0,
                y: 0,
                z: i as u32,
            };

            // Ensure bytes_per_row is properly aligned. If not, copy rows into a padded buffer.
            let bytes_per_pixel: u32 = 4;
            let unpadded_bytes_per_row_u32 = bytes_per_pixel * dimensions.0;
            let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row_u32 =
                ((unpadded_bytes_per_row_u32 + align - 1) / align) * align;

            let unpadded_bytes_per_row = unpadded_bytes_per_row_u32 as usize;
            let padded_bytes_per_row = padded_bytes_per_row_u32 as usize;
            let height = dimensions.1 as usize;

            let raw = image_data.as_raw();
            let expected_raw_len = unpadded_bytes_per_row
                .checked_mul(height)
                .expect("image row/height multiplication overflow");
            if raw.len() != expected_raw_len {
                panic!(
                    "Unexpected image data size for layer {}: raw.len()={} expected={}. \
                     (width={}, height={}, bpp={})",
                    i,
                    raw.len(),
                    expected_raw_len,
                    dimensions.0,
                    dimensions.1,
                    bytes_per_pixel
                );
            }

            if padded_bytes_per_row == unpadded_bytes_per_row {
                queue.write_texture(
                    TexelCopyTextureInfoBase {
                        texture: &texture,
                        mip_level: 0,
                        origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    raw,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(unpadded_bytes_per_row_u32),
                        rows_per_image: Some(dimensions.1),
                    },
                    copy_size,
                );
            } else {
                // Build padded buffer: each row is padded to padded_bytes_per_row.
                let mut padded = vec![0u8; padded_bytes_per_row.checked_mul(height).unwrap()];
                for row in 0..height {
                    let src_start = row * unpadded_bytes_per_row;
                    let src_end = src_start + unpadded_bytes_per_row;
                    let dst_start = row * padded_bytes_per_row;
                    padded[dst_start..dst_start + unpadded_bytes_per_row]
                        .copy_from_slice(&raw[src_start..src_end]);
                }
                queue.write_texture(
                    TexelCopyTextureInfoBase {
                        texture: &texture,
                        mip_level: 0,
                        origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &padded,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row_u32),
                        rows_per_image: Some(dimensions.1),
                    },
                    copy_size,
                );
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self { view, sampler })
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
        sample_count: u32,
        can_be_sampled: bool,
    ) -> Self {
        use wgpu::Extent3d;

        let size = Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        // Depth texture must be created with RENDER_ATTACHMENT to be used as a depth attachment.
        // If we also want to sample from it in a shader, include TEXTURE_BINDING.
        let usage = if can_be_sampled {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };

        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage,
            view_formats: &[],
        };

        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self { view, sampler }
    }

    pub fn create_sampled_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        Self { view, sampler }
    }
}
