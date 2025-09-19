#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Air,
    Dirt,
    Grass,
    Stone,
    Wood,
    Leaves,
    Sand,
    Water,
    Cloud,
}

#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub block_type: BlockType,
}

impl Block {
    pub fn new(block_type: BlockType) -> Self {
        Self { block_type }
    }

    pub fn is_solid(&self) -> bool {
        // A block is solid if it's not air. Used for collision.
        // Clouds are not solid.
        matches!(
            self.block_type,
            BlockType::Dirt
                | BlockType::Grass
                | BlockType::Stone
                | BlockType::Wood
                | BlockType::Leaves
                | BlockType::Sand
        )
    }
}
