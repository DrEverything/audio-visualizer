// Upscales the reduced-resolution visualizer pass onto the egui surface. Only
// used when the render scale is below 100%; at 100% the visualizer draws
// straight into the egui pass and this never runs.

@group(0) @binding(0) var u_src: texture_2d<f32>;
@group(0) @binding(1) var u_src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index & 1u) * 4 - 1);
    let y = f32(i32(in_vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

// Catmull-Rom bicubic rather than plain bilinear: bilinear smears the thin glow
// lines of the wave noticeably, where Catmull-Rom keeps them crisp. The 4x4
// kernel is folded into 9 taps by letting the linear sampler blend the two
// middle texels of each row/column.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(u_src));
    let inv_size = 1.0 / size;

    let sample_pos = in.uv * size;
    let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos1;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;

    let p0 = (tex_pos1 - 1.0) * inv_size;
    let p12 = (tex_pos1 + w2 / w12) * inv_size;
    let p3 = (tex_pos1 + 2.0) * inv_size;

    var c = vec4<f32>(0.0);
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p0.x, p0.y), 0.0) * w0.x * w0.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p12.x, p0.y), 0.0) * w12.x * w0.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p3.x, p0.y), 0.0) * w3.x * w0.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p0.x, p12.y), 0.0) * w0.x * w12.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p12.x, p12.y), 0.0) * w12.x * w12.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p3.x, p12.y), 0.0) * w3.x * w12.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p0.x, p3.y), 0.0) * w0.x * w3.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p12.x, p3.y), 0.0) * w12.x * w3.y;
    c += textureSampleLevel(u_src, u_src_sampler, vec2(p3.x, p3.y), 0.0) * w3.x * w3.y;

    // The negative lobes can overshoot next to bright lines.
    return clamp(c, vec4(0.0), vec4(1.0));
}
