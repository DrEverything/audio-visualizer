struct Uniforms {
    u_time: f32,
    u_resolution_x: f32,
    u_resolution_y: f32,
    u_pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var u_audio: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;

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
    let u_resolution = vec2<f32>(uniforms.u_resolution_x, uniforms.u_resolution_y);
    
    // Map normalized UV to GLSL-style gl_FragCoord where bottom-left is (0,0) and top-right is (width, height).
    // This places the direct landscape in the sky (top half) and its reflection on the floor (bottom half).
    let I = vec2<f32>(in.uv.x * u_resolution.x, (1.0 - in.uv.y) * u_resolution.y);
    let u_time = uniforms.u_time;

    var fragColor = vec4<f32>(0.0);
    var i: f32 = 0.0;
    var d: f32 = 0.0;
    var z: f32 = 0.0;
    var r: f32 = 0.0;

    for (var step: f32 = 0; step < 9e1; step = step + 1) {
        let R = vec3<f32>(u_resolution.x, u_resolution.y, u_resolution.x);
        
        // Raymarch sample point
        var p = z * normalize(vec3<f32>(2.0 * I.x, 2.0 * I.y, 0.0) - R);
        
        // Shift camera and get reflection coordinates (GLSL: r = max(-++p, 0.0).y)
        p = p + vec3<f32>(1.0);
        r = max(-p.y, 0.0);
        
        // Mirror and music
        let tex_x = (p.x + 4.5) / 30.0;
        let tex_y = (-p.z - 3.0) * 70.0 / R.y;
        
        // Sample audio texture using textureSampleLevel (clamp sampler mode handles edge values)
        let audio_val = textureSampleLevel(u_audio, u_sampler, vec2<f32>(tex_x, tex_y), 0.0).r;
        p.y += (r + r) - 4.0 * audio_val;
        
        // Step forward (reflections are softer)
        let d_inner = p.z + 3.0;
        let max_val = max(d_inner, -d_inner * 0.1);
        d = 0.1 * (0.1 * r + abs(p.y) / (1.0 + r + r + r * r) + max_val);
        z += d;
        
        // Loop update for color
        let color_offset = vec4<f32>(0.0, 2.0, 4.0, 3.0);
        let cos_val = cos(vec4<f32>(z * 0.5 + u_time) + color_offset);
        fragColor += (cos_val + 1.3) / d / z;
    }
    
    fragColor = tanh(fragColor / 9e2);
    return vec4<f32>(fragColor);
}
