use crate::block::Block;
use crate::chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use noise::Perlin;
use rustc_hash::FxHashMap;

pub const RENDER_DISTANCE: i32 = 8;

pub struct World {
    chunks: FxHashMap<(i32, i32), Chunk>,
    noise: Perlin,
}

impl World {
    pub fn new() -> Self {
        let noise = Perlin::new(42);
        let chunks = FxHashMap::default();

        Self { chunks, noise }
    }

    pub fn ensure_chunk(&mut self, cx: i32, cz: i32) {
        if !self.chunks.contains_key(&(cx, cz)) {
            let chunk = Chunk::new((cx, cz), &self.noise);
            self.chunks.insert((cx, cz), chunk);
        }
    }

    pub fn get_chunk(&self, cx: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&(cx, cz))
    }

    pub fn get_block(&self, wx: isize, wy: isize, wz: isize) -> Option<&Block> {
        if wy < 0 || wy >= CHUNK_HEIGHT as isize {
            return None;
        }

        let cx = div_floor(wx, CHUNK_WIDTH as isize) as i32;
        let cz = div_floor(wz, CHUNK_DEPTH as isize) as i32;
        let lx = (wx.rem_euclid(CHUNK_WIDTH as isize)) as usize;
        let lz = (wz.rem_euclid(CHUNK_DEPTH as isize)) as usize;
        let ly = wy as usize;

        self.chunks.get(&(cx, cz))?.get_block(lx, ly, lz)
    }
}

fn div_floor(a: isize, b: isize) -> isize {
    let d = a / b;
    let r = a % b;
    if (r > 0 && b < 0) || (r < 0 && b > 0) {
        d - 1
    } else {
        d
    }
}
