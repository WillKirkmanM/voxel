struct SceneUniforms {
    screen_dims_and_flags: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: mat4x4<f32>;

struct Model {
    model: mat4x4<f32>,
};
@group(2) @binding(0)
var<storage, read> model_storage: Model;


@group(3) @binding(0)
var<uniform> scene: SceneUniforms;

@group(3) @binding(1)
var depth_texture: texture_depth_2d;

@group(0) @binding(0)
var t_diffuse: texture_2d_array<f32>;
@group(0) @binding(1)
var s_sampler: sampler;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) texture_layer: u32,
    @location(4) ambient_occlusion: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) @interpolate(flat) texture_layer: u32,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) ambient_occlusion: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos4 = model_storage.model * vec4<f32>(input.position, 1.0);
    out.world_position = world_pos4.xyz;
    out.clip_position = camera * world_pos4;
    out.tex_coords = input.tex_coords;
    out.texture_layer = input.texture_layer;

    out.world_normal = (model_storage.model * vec4<f32>(input.normal, 0.0)).xyz;
    out.ambient_occlusion = input.ambient_occlusion;
    return out;
}


fn get_light(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, -0.6));
    let ambient = 0.4;
    let diffuse = max(dot(world_normal, light_dir), 0.0) * 0.6;
    return ambient + diffuse;
}


fn screen_space_reflect(clip_pos: vec4<f32>, world_pos: vec3<f32>, normal: vec3<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    let view_vec = normalize(world_pos - camera_pos);
    let reflect_vec = reflect(view_vec, normal);

    let step = 0.2;
    let max_steps = 60;
    let thickness = 0.1;

    for (var i: i32 = 0; i < max_steps; i = i + 1) {
        let sample_world_pos = world_pos + reflect_vec * step * f32(i);
        let sample_clip_pos = camera * vec4(sample_world_pos, 1.0);


        var sample_uv: vec2<f32> = sample_clip_pos.xy / sample_clip_pos.w;
        sample_uv = sample_uv * 0.5 + 0.5;


        sample_uv.y = 1.0 - sample_uv.y;


        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            break;
        }


        let pixel_coords_f = floor(sample_uv * scene.screen_dims_and_flags.xy);
        let pixel_coords = vec2<i32>(pixel_coords_f);
        let depth = textureLoad(depth_texture, pixel_coords, 0);


        let scene_z = (camera[2][3]) / (depth * 2.0 - 1.0 - camera[2][2]);

        if scene_z < sample_clip_pos.z && scene_z > sample_clip_pos.z - thickness {

            return vec3<f32>(sample_uv.x, sample_uv.y, 1.0);
        }
    }


    return vec3<f32>(-1.0, -1.0, -1.0);
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(t_diffuse, s_sampler, in.tex_coords, in.texture_layer);


    if (in.texture_layer == 5u || in.texture_layer == 8u) && color.a < 0.1 {
        discard;
    }

    let light = get_light(in.world_position, normalize(in.world_normal));
    var final_color = color * light * in.ambient_occlusion;


    if in.texture_layer == 7u {
        let view_dir = normalize(in.world_position - camera[3].xyz);
        let fresnel = pow(1.0 - abs(dot(view_dir, normalize(in.world_normal))), 4.0);
        let sky_color = vec4<f32>(0.53, 0.81, 0.92, 1.0);

        var reflection_color = sky_color;


        if scene.screen_dims_and_flags.z > 0.5 {
            let reflect_uv = screen_space_reflect(in.clip_position, in.world_position, normalize(in.world_normal), camera[3].xyz);
            if reflect_uv.z > 0.0 {

                reflection_color = mix(sky_color, vec4<f32>(0.2, 0.2, 0.3, 1.0), 0.5);
            }
        }

        final_color = mix(final_color, reflection_color, fresnel * 0.8);
    }

    return final_color;
}