use crate::chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH};
use crate::world::World;
use cgmath::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use winit::event::ElementState;
use winit::keyboard::KeyCode;

const MAX_SPEED: f32 = 4.3;
const PLAYER_SPRINT_MULTIPLIER: f32 = 5.0;
const GRAVITY: f32 = -28.0;
const JUMP_VELOCITY: f32 = 9.0;
const MOUSE_SENSITIVITY: f32 = 0.002;
const PLAYER_HEIGHT: f32 = 1.8;
const PLAYER_WIDTH: f32 = 0.6;
const PLAYER_EYE_OFFSET: f32 = 0.2;

const GROUND_ACCEL: f32 = 30.0;
const AIR_ACCEL: f32 = 5.0;
const GROUND_DRAG: f32 = 10.0;
const AIR_DRAG: f32 = 1.0;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }
}

pub struct Player {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
    pub direction: Vector3<f32>,
    pub on_ground: bool,
    pub height: f32,
    pub fov_deg: f32,
}

impl Player {
    pub fn new(position: Point3<f32>) -> Self {
        Self {
            position,
            velocity: Vector3::zero(),
            yaw: Rad(-1.57),
            pitch: Rad(0.0),
            direction: -Vector3::unit_z(),
            on_ground: true,
            height: PLAYER_HEIGHT,
            fov_deg: 70.0,
        }
    }

    pub fn get_eye_position(&self) -> Point3<f32> {
        // Use current height so crouching lowers the eye position.
        self.position + Vector3::new(0.0, self.height - PLAYER_EYE_OFFSET, 0.0)
    }

    pub fn get_chunk_pos(&self) -> (i32, i32) {
        let cx = (self.position.x / crate::chunk::CHUNK_WIDTH as f32).floor() as i32;
        let cz = (self.position.z / crate::chunk::CHUNK_DEPTH as f32).floor() as i32;
        (cx, cz)
    }

    pub fn update(
        &mut self,
        dt: Duration,
        controller: &PlayerController,
        world: &Arc<Mutex<World>>,
    ) {
        let dt_secs = dt.as_secs_f32();

        self.direction = Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();

        if !controller.is_flying {
            self.velocity.y += GRAVITY * dt_secs;
        }

        let forward = Vector3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize();
        let right = Vector3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize();

        let mut desired_dir = Vector3::zero();
        if controller.is_forward_pressed {
            desired_dir += forward;
        }
        if controller.is_backward_pressed {
            desired_dir -= forward;
        }
        if controller.is_right_pressed {
            desired_dir += right;
        }
        if controller.is_left_pressed {
            desired_dir -= right;
        }

        if desired_dir.magnitude2() > 0.0 {
            desired_dir = desired_dir.normalize();
        }

        if controller.is_flying {
            if controller.is_jump_pressed {
                desired_dir.y += 1.0;
            }
            if controller.is_crouch_pressed {
                desired_dir.y -= 1.0;
            }
        }

        // Determine sprinting early so we can affect acceleration
        // allow sprint while flying or on ground, but disable while crouching
        let sprinting = controller.is_sprint_pressed && !controller.is_crouch_pressed;

        // Apply sprinting to acceleration when on ground for a very noticeable effect
        // When flying use ground accel for responsive movement; when airborne (not flying) keep small air accel.
        let accel_base = if controller.is_flying {
            GROUND_ACCEL
        } else if self.on_ground {
            GROUND_ACCEL
        } else {
            AIR_ACCEL
        };
        let accel = if sprinting {
            accel_base * PLAYER_SPRINT_MULTIPLIER
        } else {
            accel_base
        };

        // Conserve horizontal speed when airborne by removing drag when not on ground and not flying.
        let drag = if self.on_ground {
            GROUND_DRAG
        } else if controller.is_flying {
            AIR_DRAG
        } else {
            0.0
        };

        let mut horizontal_vel = Vector3::new(self.velocity.x, 0.0, self.velocity.z);
        let current_speed = horizontal_vel.magnitude();

        let drag_force = drag * current_speed;
        if current_speed > 0.0 {
            let drag_vec = -horizontal_vel.normalize() * drag_force;
            horizontal_vel += drag_vec * dt_secs;
            self.velocity.x = horizontal_vel.x;
            self.velocity.z = horizontal_vel.z;
        }

        // If flying, apply 3D accel; otherwise only horizontal accel (y handled by gravity/jump)
        if controller.is_flying {
            let dir_norm = if desired_dir.magnitude2() > 0.0 {
                desired_dir.normalize()
            } else {
                Vector3::zero()
            };
            self.velocity += dir_norm * accel * dt_secs;
        } else {
            // horizontal component only
            let horiz = Vector3::new(desired_dir.x, 0.0, desired_dir.z);
            self.velocity += horiz * accel * dt_secs;
        }

        // Handle crouch: update current height immediately (could be smoothed later)
        let target_height = if controller.is_crouch_pressed {
            PLAYER_HEIGHT * 0.5
        } else {
            PLAYER_HEIGHT
        };
        self.height = target_height;

        // Sprint affects max speed too (still disabled while crouching)
        let mut max_speed = if sprinting {
            MAX_SPEED * PLAYER_SPRINT_MULTIPLIER
        } else {
            MAX_SPEED
        };
        if controller.is_crouch_pressed {
            max_speed *= 0.5;
        }
        // If flying allow a larger vertical speed cap by scaling the max for vertical component
        let max_vertical_speed = if controller.is_flying {
            max_speed
        } else {
            MAX_SPEED
        };

        let horizontal_speed_after_accel =
            Vector2::new(self.velocity.x, self.velocity.z).magnitude();
        if horizontal_speed_after_accel > max_speed {
            let scale = max_speed / horizontal_speed_after_accel;
            self.velocity.x *= scale;
            self.velocity.z *= scale;
        }

        // Cap vertical speed when flying (or in general)
        if controller.is_flying {
            if self.velocity.y > max_vertical_speed {
                self.velocity.y = max_vertical_speed;
            }
            if self.velocity.y < -max_vertical_speed {
                self.velocity.y = -max_vertical_speed;
            }
        }

        if controller.is_jump_pressed && self.on_ground && !controller.is_flying {
            self.velocity.y = JUMP_VELOCITY;
        }

        let world_lock = world.lock().unwrap();
        let mut next_pos = self.position;
        let delta_p = self.velocity * dt_secs;
        // flying means we are not grounded
        self.on_ground = !controller.is_flying && false;

        if controller.is_flying {
            // simple flying integration: apply full delta (no ground collision for Y), collisions still prevent entering solid blocks
            next_pos += delta_p;
            // but still check collisions axis-aligned individually to avoid clipping
            let player_bounds = self.get_bounding_box(next_pos);
            if self.collides_with_world(&world_lock, &player_bounds) {
                // try zeroing vertical and/or horizontal components if collision
                // revert vertical first
                next_pos.y = self.position.y;
                let player_bounds = self.get_bounding_box(next_pos);
                if self.collides_with_world(&world_lock, &player_bounds) {
                    // revert horizontal too
                    next_pos.x = self.position.x;
                    next_pos.z = self.position.z;
                }
            }
            self.position = next_pos;
            return;
        }

        self.on_ground = false;

        next_pos.y += delta_p.y;
        let player_bounds = self.get_bounding_box(next_pos);
        if self.collides_with_world(&world_lock, &player_bounds) {
            next_pos.y = self.position.y;
            if self.velocity.y < 0.0 {
                self.on_ground = true;
            }
            self.velocity.y = 0.0;
        }

        next_pos.x += delta_p.x;
        let player_bounds = self.get_bounding_box(next_pos);
        if self.collides_with_world(&world_lock, &player_bounds) {
            next_pos.x = self.position.x;
            self.velocity.x = 0.0;
        }

        next_pos.z += delta_p.z;
        let player_bounds = self.get_bounding_box(next_pos);
        if self.collides_with_world(&world_lock, &player_bounds) {
            next_pos.z = self.position.z;
            self.velocity.z = 0.0;
        }

        self.position = next_pos;
    }

    fn get_bounding_box(&self, position: Point3<f32>) -> Aabb {
        // Use current height (accounts for crouch)
        let half_width = PLAYER_WIDTH / 2.0;
        Aabb::new(
            position - Vector3::new(half_width, 0.0, half_width),
            position + Vector3::new(half_width, self.height, half_width),
        )
    }

    fn collides_with_world(&self, world: &World, bounds: &Aabb) -> bool {
        let min_x = bounds.min.x.floor() as isize;
        let max_x = bounds.max.x.ceil() as isize;
        let min_y = bounds.min.y.floor() as isize;
        let max_y = bounds.max.y.ceil() as isize;
        let min_z = bounds.min.z.floor() as isize;
        let max_z = bounds.max.z.ceil() as isize;

        for y in min_y..max_y {
            for z in min_z..max_z {
                for x in min_x..max_x {
                    if let Some(block) = world.get_block(x, y, z) {
                        if block.is_solid() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

#[derive(Default)]
pub struct PlayerController {
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_jump_pressed: bool,
    is_crouch_pressed: bool,
    is_sprint_pressed: bool,
    is_flying: bool,
}

impl PlayerController {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) -> bool {
        let is_pressed = state == ElementState::Pressed;
        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            KeyCode::Space => {
                self.is_jump_pressed = is_pressed;
                true
            }
            KeyCode::ShiftLeft | KeyCode::ShiftRight | KeyCode::KeyL => {
                self.is_sprint_pressed = is_pressed;
                true
            }
            KeyCode::KeyF => {
                if is_pressed {
                    self.is_flying = !self.is_flying;
                }
                true
            }
            KeyCode::ControlLeft => {
                self.is_crouch_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&self, delta: (f64, f64), player: &mut Player) {
        player.yaw += Rad(delta.0 as f32 * MOUSE_SENSITIVITY);
        player.pitch -= Rad(delta.1 as f32 * MOUSE_SENSITIVITY);
        let pi_2 = std::f32::consts::FRAC_PI_2;
        player.pitch = Rad(player.pitch.0.clamp(-pi_2 + 0.01, pi_2 - 0.01));
    }
}

pub struct Aabb {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}
impl Aabb {
    fn new(p1: Point3<f32>, p2: Point3<f32>) -> Self {
        Self {
            min: Point3::new(p1.x.min(p2.x), p1.y.min(p2.y), p1.z.min(p2.z)),
            max: Point3::new(p1.x.max(p2.x), p1.y.max(p2.y), p1.z.max(p2.z)),
        }
    }
}

pub struct Frustum {
    planes: [Vector4<f32>; 6],
}

impl Frustum {
    pub fn from_matrix(mat: Matrix4<f32>) -> Self {
        let mut planes = [Vector4::zero(); 6];
        planes[0] = mat.row(3) + mat.row(0); // Left
        planes[1] = mat.row(3) - mat.row(0); // Right
        planes[2] = mat.row(3) + mat.row(1); // Bottom
        planes[3] = mat.row(3) - mat.row(1); // Top
        planes[4] = mat.row(3) + mat.row(2); // Near
        planes[5] = mat.row(3) - mat.row(2); // Far

        for p in planes.iter_mut() {
            *p /= p.truncate().magnitude();
        }
        Self { planes }
    }

    pub fn is_box_visible(&self, pos: (i32, i32)) -> bool {
        let min = Point3::new(
            (pos.0 * CHUNK_WIDTH as i32) as f32,
            0.0,
            (pos.1 * CHUNK_DEPTH as i32) as f32,
        );
        let max = Point3::new(
            min.x + CHUNK_WIDTH as f32,
            CHUNK_HEIGHT as f32,
            min.z + CHUNK_DEPTH as f32,
        );

        for plane in &self.planes {
            let mut p_vertex = min;
            if plane.x > 0.0 {
                p_vertex.x = max.x;
            }
            if plane.y > 0.0 {
                p_vertex.y = max.y;
            }
            if plane.z > 0.0 {
                p_vertex.z = max.z;
            }

            if plane.dot(p_vertex.to_homogeneous()) < 0.0 {
                return false;
            }
        }
        true
    }
}
