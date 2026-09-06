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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(u_src, u_src_sampler, in.uv, 0.0);
}
