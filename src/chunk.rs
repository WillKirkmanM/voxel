use crate::Vertex;
use crate::block::{Block, BlockType};
use crate::world::World;
use noise::{NoiseFn, Perlin};
use rand::prelude::*;

/// Provide a helper on BlockType so mesh code can query transparency.
impl BlockType {
    pub fn is_transparent(&self) -> bool {
        match self {
            // Transparent or non-solid blocks that should let faces show through
            BlockType::Air | BlockType::Water | BlockType::Leaves | BlockType::Cloud => true,
            // All other block types are considered opaque for meshing purposes
            _ => false,
        }
    }
}

pub const CHUNK_WIDTH: usize = 16;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_DEPTH: usize = 16;
const SEA_LEVEL: isize = 62;
const CLOUD_LEVEL: usize = 128;

fn fbm2(noise: &Perlin, x: f64, z: f64) -> f64 {
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut sum = 0.0;
    for _ in 0..4 {
        sum += amp * noise.get([x * 0.003 * freq, z * 0.003 * freq]);
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

pub struct Chunk {
    blocks: Vec<Block>,
    position: (i32, i32),
}

impl Chunk {
    pub fn new(position: (i32, i32), noise: &Perlin) -> Self {
        let mut blocks = vec![Block::new(BlockType::Air); CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH];
        let mut rng = StdRng::seed_from_u64((position.0 as u64) << 32 | (position.1 as u64));

        for x in 0..CHUNK_WIDTH {
            for z in 0..CHUNK_DEPTH {
                let world_x = (position.0 * CHUNK_WIDTH as i32 + x as i32) as f64;
                let world_z = (position.1 * CHUNK_DEPTH as i32 + z as i32) as f64;

                let height_val = fbm2(noise, world_x, world_z);
                let terrain_height = (SEA_LEVEL as f64 + height_val * 35.0)
                    .clamp(0.0, (CHUNK_HEIGHT - 1) as f64)
                    as isize;

                for y_isize in 0..CHUNK_HEIGHT as isize {
                    let y = y_isize as usize;
                    let index = Self::get_index(x, y, z);

                    if y_isize > terrain_height {
                        if y_isize <= SEA_LEVEL {
                            blocks[index] = Block::new(BlockType::Water);
                        }
                    } else if y_isize == terrain_height {
                        if y_isize < SEA_LEVEL + 2 {
                            blocks[index] = Block::new(BlockType::Sand);
                        } else {
                            blocks[index] = Block::new(BlockType::Grass);
                        }
                    } else if y_isize > terrain_height - 4 {
                        if y_isize < SEA_LEVEL + 2 {
                            blocks[index] = Block::new(BlockType::Sand);
                        } else {
                            blocks[index] = Block::new(BlockType::Dirt);
                        }
                    } else {
                        blocks[index] = Block::new(BlockType::Stone);
                    }
                }

                let surface_y = terrain_height as usize;
                if surface_y + 6 < CHUNK_HEIGHT && terrain_height > (SEA_LEVEL + 1) as isize {
                    let grass_idx = Self::get_index(x, surface_y, z);
                    if blocks[grass_idx].block_type == BlockType::Grass
                        && rng.random::<f32>() < 0.02
                    {
                        for dy in 1..=5 {
                            blocks[Self::get_index(x, surface_y + dy, z)] =
                                Block::new(BlockType::Wood);
                        }
                        let top_y = surface_y + 5;
                        for ly in top_y.saturating_sub(2)..=top_y {
                            for lx in x.saturating_sub(2)..=(x + 2).min(CHUNK_WIDTH - 1) {
                                for lz in z.saturating_sub(2)..=(z + 2).min(CHUNK_DEPTH - 1) {
                                    let dx = lx as isize - x as isize;
                                    let dy = ly as isize - top_y as isize;
                                    let dz = lz as isize - z as isize;
                                    let dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                                    if dist < 3.5 {
                                        let li = Self::get_index(lx, ly, lz);
                                        if blocks[li].block_type == BlockType::Air {
                                            blocks[li] = Block::new(BlockType::Leaves);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for x in 0..CHUNK_WIDTH {
            for z in 0..CHUNK_DEPTH {
                let world_x = (position.0 * CHUNK_WIDTH as i32 + x as i32) as f64;
                let world_z = (position.1 * CHUNK_DEPTH as i32 + z as i32) as f64;
                let cloud_noise = noise.get([world_x * 0.01, world_z * 0.01]);

                if cloud_noise > 0.55 {
                    let index = Self::get_index(x, CLOUD_LEVEL, z);
                    if blocks[index].block_type == BlockType::Air {
                        blocks[index] = Block::new(BlockType::Cloud);
                    }
                }
            }
        }

        Self { blocks, position }
    }

    fn get_index(x: usize, y: usize, z: usize) -> usize {
        y + z * CHUNK_HEIGHT + x * CHUNK_HEIGHT * CHUNK_DEPTH
    }

    pub fn get_block(&self, x: usize, y: usize, z: usize) -> Option<&Block> {
        if x < CHUNK_WIDTH && y < CHUNK_HEIGHT && z < CHUNK_DEPTH {
            self.blocks.get(Self::get_index(x, y, z))
        } else {
            None
        }
    }

    fn is_solid(block_type: BlockType) -> bool {
        !matches!(
            block_type,
            BlockType::Air | BlockType::Water | BlockType::Leaves | BlockType::Cloud
        )
    }

    pub fn build_mesh(&self, world: &World) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                for x in 0..CHUNK_WIDTH {
                    let block = &self.blocks[Self::get_index(x, y, z)];
                    if block.block_type == BlockType::Air {
                        continue;
                    }

                    let is_occluded =
                    // Check if we are not at a chunk boundary
                    (x > 0 && z > 0 && y > 0 && x < CHUNK_WIDTH - 1 && z < CHUNK_DEPTH - 1 && y < CHUNK_HEIGHT - 1) &&
                    // Check if all 6 neighbors are solid (not transparent)
                    !self.blocks[Self::get_index(x + 1, y, z)].block_type.is_transparent() &&
                    !self.blocks[Self::get_index(x - 1, y, z)].block_type.is_transparent() &&
                    !self.blocks[Self::get_index(x, y + 1, z)].block_type.is_transparent() &&
                    !self.blocks[Self::get_index(x, y - 1, z)].block_type.is_transparent() &&
                    !self.blocks[Self::get_index(x, y, z + 1)].block_type.is_transparent() &&
                    !self.blocks[Self::get_index(x, y, z - 1)].block_type.is_transparent();

                    if is_occluded {
                        continue; // Skip this block, it's not visible
                    }

                    // World position is only for neighbor checking
                    let wx = self.position.0 as isize * CHUNK_WIDTH as isize + x as isize;
                    let wy = y as isize;
                    let wz = self.position.1 as isize * CHUNK_DEPTH as isize + z as isize;

                    // Check neighbors in all 6 directions
                    // [dx, dy, dz, face_type, axis]
                    let neighbors: &[(isize, isize, isize, usize)] = &[
                        (1, 0, 0, 0),  // Right (+X)
                        (-1, 0, 0, 0), // Left (-X)
                        (0, 1, 0, 1),  // Top (+Y)
                        (0, -1, 0, 1), // Bottom (-Y)
                        (0, 0, 1, 2),  // Front (+Z)
                        (0, 0, -1, 2), // Back (-Z)
                    ];

                    for &(dx, dy, dz, axis) in neighbors {
                        let neighbor_block_type = world
                            .get_block(wx + dx, wy + dy, wz + dz)
                            .map_or(BlockType::Air, |b| b.block_type);

                        if neighbor_block_type.is_transparent() {
                            let fx = x as f32;
                            let fy = y as f32;
                            let fz = z as f32;

                            // Vertices are now relative to the chunk's origin
                            let (v0, v1, v2, v3, normal) = match (dx, dy, dz) {
                                (1, 0, 0) => (
                                    // Right
                                    [fx + 1.0, fy, fz],
                                    [fx + 1.0, fy, fz + 1.0],
                                    [fx + 1.0, fy + 1.0, fz + 1.0],
                                    [fx + 1.0, fy + 1.0, fz],
                                    [1.0, 0.0, 0.0],
                                ),
                                (-1, 0, 0) => (
                                    // Left
                                    [fx, fy, fz + 1.0],
                                    [fx, fy, fz],
                                    [fx, fy + 1.0, fz],
                                    [fx, fy + 1.0, fz + 1.0],
                                    [-1.0, 0.0, 0.0],
                                ),
                                (0, 1, 0) => (
                                    // Top
                                    [fx, fy + 1.0, fz],
                                    [fx + 1.0, fy + 1.0, fz],
                                    [fx + 1.0, fy + 1.0, fz + 1.0],
                                    [fx, fy + 1.0, fz + 1.0],
                                    [0.0, 1.0, 0.0],
                                ),
                                (0, -1, 0) => (
                                    // Bottom
                                    [fx, fy, fz + 1.0],
                                    [fx + 1.0, fy, fz + 1.0],
                                    [fx + 1.0, fy, fz],
                                    [fx, fy, fz],
                                    [0.0, -1.0, 0.0],
                                ),
                                (0, 0, 1) => (
                                    // Front
                                    [fx + 1.0, fy, fz + 1.0],
                                    [fx, fy, fz + 1.0],
                                    [fx, fy + 1.0, fz + 1.0],
                                    [fx + 1.0, fy + 1.0, fz + 1.0],
                                    [0.0, 0.0, 1.0],
                                ),
                                (0, 0, -1) => (
                                    // Back
                                    [fx, fy, fz],
                                    [fx + 1.0, fy, fz],
                                    [fx + 1.0, fy + 1.0, fz],
                                    [fx, fy + 1.0, fz],
                                    [0.0, 0.0, -1.0],
                                ),
                                _ => continue,
                            };

                            let ao = Self::calculate_ao(wx, wy, wz, (dx, dy, dz), world);

                            let base_index = vertices.len() as u32;
                            let layer = get_texture_layer(block.block_type, axis);

                            vertices.push(Vertex {
                                position: v0,
                                tex_coords: [0.0, 1.0],
                                normal,
                                texture_layer: layer,
                                ambient_occlusion: ao[0],
                            });
                            vertices.push(Vertex {
                                position: v1,
                                tex_coords: [1.0, 1.0],
                                normal,
                                texture_layer: layer,
                                ambient_occlusion: ao[1],
                            });
                            vertices.push(Vertex {
                                position: v2,
                                tex_coords: [1.0, 0.0],
                                normal,
                                texture_layer: layer,
                                ambient_occlusion: ao[2],
                            });
                            vertices.push(Vertex {
                                position: v3,
                                tex_coords: [0.0, 0.0],
                                normal,
                                texture_layer: layer,
                                ambient_occlusion: ao[3],
                            });

                            indices.extend_from_slice(&[
                                base_index,
                                base_index + 1,
                                base_index + 2,
                                base_index,
                                base_index + 2,
                                base_index + 3,
                            ]);
                        }
                    }
                }
            }
        }
        (vertices, indices)
    }

    fn calculate_ao(
        x: isize,
        y: isize,
        z: isize,
        face_normal: (isize, isize, isize),
        world: &World,
    ) -> [f32; 4] {
        let mut ao = [1.0; 4];

        // Define corner checks relative to the block's corner
        // s1 and s2 are side neighbors, c is the corner neighbor
        let get_occlusion = |s1: bool, s2: bool, c: bool| -> f32 {
            match (s1, s2, c) {
                (true, true, _) => 0.4, // Darkest if both sides are blocked
                (true, false, true) | (false, true, true) => 0.6, // Darker if a side and corner are blocked
                (true, false, false) | (false, true, false) => 0.8, // Slightly dark if only one side is blocked
                (false, false, true) => 0.8, // Slightly dark if only corner is blocked
                (false, false, false) => 1.0, // Fully lit
            }
        };

        // Iterate over the 4 vertices of the face
        for i in 0..4 {
            // Determine the neighbors to check for each vertex
            let (p1, p2, p3) = Self::get_ao_neighbors(i, face_normal);

            let n1_solid = Self::is_solid(
                world
                    .get_block(x + p1.0, y + p1.1, z + p1.2)
                    .map_or(BlockType::Air, |b| b.block_type),
            );
            let n2_solid = Self::is_solid(
                world
                    .get_block(x + p2.0, y + p2.1, z + p2.2)
                    .map_or(BlockType::Air, |b| b.block_type),
            );
            let n3_solid = Self::is_solid(
                world
                    .get_block(x + p3.0, y + p3.1, z + p3.2)
                    .map_or(BlockType::Air, |b| b.block_type),
            );

            ao[i] = get_occlusion(n1_solid, n2_solid, n3_solid);
        }

        // Flip AO values for faces with specific winding orders to match vertices
        if face_normal == (-1, 0, 0) || face_normal == (0, -1, 0) || face_normal == (0, 0, -1) {
            ao.swap(1, 3);
        }

        ao
    }

    fn get_ao_neighbors(
        vertex_index: usize,
        face: (isize, isize, isize),
    ) -> (
        (isize, isize, isize),
        (isize, isize, isize),
        (isize, isize, isize),
    ) {
        let up = (0, 1, 0);
        let down = (0, -1, 0);
        let right = (1, 0, 0);
        let left = (-1, 0, 0);
        let front = (0, 0, 1);
        let back = (0, 0, -1);

        // This defines the 3 blocks that meet at each corner of a face's vertex
        let neighbors = match face {
            (1, 0, 0) => [
                (back, down, (1, -1, -1)),
                (front, down, (1, -1, 1)),
                (front, up, (1, 1, 1)),
                (back, up, (1, 1, -1)),
            ], // Right
            (-1, 0, 0) => [
                (front, down, (-1, -1, 1)),
                (back, down, (-1, -1, -1)),
                (back, up, (-1, 1, -1)),
                (front, up, (-1, 1, 1)),
            ], // Left
            (0, 1, 0) => [
                (left, back, (-1, 1, -1)),
                (right, back, (1, 1, -1)),
                (right, front, (1, 1, 1)),
                (left, front, (-1, 1, 1)),
            ], // Top
            (0, -1, 0) => [
                (left, front, (-1, -1, 1)),
                (right, front, (1, -1, 1)),
                (right, back, (1, -1, -1)),
                (left, back, (-1, -1, -1)),
            ], // Bottom
            (0, 0, 1) => [
                (right, down, (1, -1, 1)),
                (left, down, (-1, -1, 1)),
                (left, up, (-1, 1, 1)),
                (right, up, (1, 1, 1)),
            ], // Front
            (0, 0, -1) => [
                (left, down, (-1, -1, -1)),
                (right, down, (1, -1, -1)),
                (right, up, (1, -1, -1)),
                (left, up, (-1, 1, -1)),
            ], // Back
            _ => unreachable!(),
        };
        neighbors[vertex_index]
    }
}

fn get_texture_layer(block_type: BlockType, axis: usize) -> u32 {
    match block_type {
        BlockType::Grass => {
            if axis == 1 {
                0
            } else {
                1
            }
        }
        BlockType::Dirt => 2,
        BlockType::Stone => 3,
        BlockType::Wood => 4,
        BlockType::Leaves => 5,
        BlockType::Sand => 6,
        BlockType::Water => 7,
        BlockType::Cloud => 8,
        BlockType::Air => 9,
    }
}
