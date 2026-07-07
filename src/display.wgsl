struct Uniforms {
    resolution: vec2<f32>,
    camera_rot: vec2<f32>,
    camera_zoom: f32,
    frame_index: u32,
    max_depth: u32,
    samples_per_frame: u32,
    aperture: f32,
    focal_distance: f32,
    env_light_intensity: f32,
    prev_camera_zoom: f32,
    prev_camera_rot: vec2<f32>,
    time: f32,
    accum_frame: u32,
    bvh_offsets: vec4<u32>,
    tri_offsets: vec4<u32>,
    dt: f32,
    render_mode: u32,
    tonemap_mode: u32,
    _pad_align2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var prev_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

fn fxaa(tex: texture_2d<f32>, sam: sampler, uv: vec2<f32>, resolution: vec2<f32>) -> vec3<f32> {
    let dx = 1.0 / resolution.x;
    let dy = 1.0 / resolution.y;
    
    let color_center = textureSampleLevel(tex, sam, uv, 0.0).rgb;
    
    let luma_coeffs = vec3<f32>(0.299, 0.587, 0.114);
    let luma_center = dot(color_center, luma_coeffs);
    
    let luma_n = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(0.0, -dy), 0.0).rgb, luma_coeffs);
    let luma_s = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(0.0, dy), 0.0).rgb, luma_coeffs);
    let luma_e = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(dx, 0.0), 0.0).rgb, luma_coeffs);
    let luma_w = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(-dx, 0.0), 0.0).rgb, luma_coeffs);
    
    let luma_min = min(luma_center, min(min(luma_n, luma_s), min(luma_e, luma_w)));
    let luma_max = max(luma_center, max(max(luma_n, luma_s), max(luma_e, luma_w)));
    
    let contrast = luma_max - luma_min;
    
    // FXAA threshold: skip if contrast is too low
    if (contrast < max(0.0312, luma_max * 0.125)) {
        return color_center;
    }
    
    let luma_nw = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(-dx, -dy), 0.0).rgb, luma_coeffs);
    let luma_ne = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(dx, -dy), 0.0).rgb, luma_coeffs);
    let luma_sw = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(-dx, dy), 0.0).rgb, luma_coeffs);
    let luma_se = dot(textureSampleLevel(tex, sam, uv + vec2<f32>(dx, dy), 0.0).rgb, luma_coeffs);
    
    let luma_ns = luma_n + luma_s;
    let luma_we = luma_w + luma_e;
    
    let subpixel_blend1 = luma_ns + luma_we;
    let subpixel_blend2 = (luma_nw + luma_ne) + (luma_sw + luma_se);
    let subpixel_blend = (2.0 * subpixel_blend1 + subpixel_blend2) / 12.0;
    
    let subpixel_contrast = abs(subpixel_blend - luma_center);
    let subpixel_blend_factor = clamp(subpixel_contrast / contrast, 0.0, 1.0);
    
    let edge_horiz = abs((luma_nw + 2.0 * luma_n + luma_ne) - (luma_sw + 2.0 * luma_s + luma_se));
    let edge_vert = abs((luma_nw + 2.0 * luma_w + luma_sw) - (luma_ne + 2.0 * luma_e + luma_se));
    
    let is_horizontal = edge_horiz >= edge_vert;
    
    let step_length = select(dx, dy, is_horizontal);
    let offset_dir = select(vec2<f32>(dx, 0.0), vec2<f32>(0.0, dy), is_horizontal);
    
    let luma1 = select(luma_w, luma_n, is_horizontal);
    let luma2 = select(luma_e, luma_s, is_horizontal);
    
    let gradient1 = abs(luma1 - luma_center);
    let gradient2 = abs(luma2 - luma_center);
    
    let is_1_dominant = gradient1 >= gradient2;
    let final_offset = step_length * 0.5 * subpixel_blend_factor;
    
    let sample_offset = select(offset_dir * final_offset, -offset_dir * final_offset, is_1_dominant);
    
    return textureSampleLevel(tex, sam, uv + sample_offset, 0.0).rgb;
}

@fragment
fn fs_display(in: VertexOutput) -> @location(0) vec4<f32> {
    // Flip Y during sampling to match display target coordinates
    let sample_uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    
    var color = vec3<f32>(0.0);
    if (uniforms.render_mode != 0u) {
        // Apply FXAA for Rasterizer (1u) and Raymarcher (2u) to get smooth shapes with zero overhead
        color = fxaa(prev_texture, texture_sampler, sample_uv, uniforms.resolution);
    } else {
        color = textureSampleLevel(prev_texture, texture_sampler, sample_uv, 0.0).rgb;
    }

    if (uniforms.render_mode == 0u) {
        // Reinhard tone mapping (only for Path Tracer)
        color = color / (color + vec3<f32>(1.0));
    }
    
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
