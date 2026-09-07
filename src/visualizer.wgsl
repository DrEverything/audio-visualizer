struct Uniforms {
    u_time: f32,
    u_resolution_x: f32,
    u_resolution_y: f32,
    // Stop marching once a single step's weight 1/(d*z) drops below this; 0
    // disables the test. Every term added to the accumulator is non-negative and
    // z only grows, so this bounds the error at (steps left) * 2.3 * u_cutoff
    // / 9e2 — at 0.05 that is well under what an 8-bit target can represent.
    u_cutoff: f32,
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

// A literal rather than a uniform, so the compiler can still unroll. Lowering it
// is not a useful quality knob in any case: this shader accumulates a glow over
// the whole march rather than stopping at a surface, so truncating it dims the
// image measurably — 90 -> 80 steps already costs ~36 dB PSNR, where the render
// scale and the cutoff below both stay above 50 dB.
const STEPS: i32 = 90;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let u_resolution = vec2(uniforms.u_resolution_x, uniforms.u_resolution_y);

    // Map normalized UV to GLSL-style gl_FragCoord where bottom-left is (0,0) and top-right is (width, height).
    // This places the direct landscape in the sky (top half) and its reflection on the floor (bottom half).
    let I = vec2(in.uv.x * u_resolution.x, (1.0 - in.uv.y) * u_resolution.y);
    let u_time = uniforms.u_time;
    let cutoff = uniforms.u_cutoff;

    // Loop invariants. The ray direction in particular used to be re-normalized
    // on all 90 iterations. Note that scaling u_resolution scales the
    // pre-normalized vector uniformly, so the direction — and hence the whole
    // image — is independent of the pixel count this pass is rasterized at,
    // which is what makes the render-scale knob a pure sampling-density change.
    let R = vec3(u_resolution.x, u_resolution.y, u_resolution.x);
    let rd = normalize(vec3(2.0 * I.x, 2.0 * I.y, 0.0) - R);
    let inv_ry = 1.0 / R.y;

    var color = vec4(0.0);
    var d: f32 = 0.0;
    var z: f32 = 0.0;
    var r: f32 = 0.0;

    for (var step: i32 = 0; step < STEPS; step = step + 1) {
        // Raymarch sample point, with the camera shift folded in
        // (GLSL: r = max(-++p, 0.0).y).
        var p = z * rd + vec3(1.0);
        r = max(-p.y, 0.0);

        // Mirror and music
        let tex_x = (p.x + 4.5) / 30.0;
        let tex_y = (-p.z - 3.0) * 70.0 * inv_ry;

        // Sample audio texture using textureSampleLevel (clamp sampler mode handles edge values)
        let audio_val = textureSampleLevel(u_audio, u_sampler, vec2(tex_x, tex_y), 0.0).r;
        p.y += (r + r) - 4.0 * audio_val;

        // Step forward (reflections are softer). 1 + 2r + r*r is (1 + r) squared.
        let d_inner = p.z + 3.0;
        let one_r = 1.0 + r;
        d = 0.1 * (0.1 * r + abs(p.y) / (one_r * one_r) + max(d_inner, -d_inner * 0.1));
        z += d;

        // Loop update for color. One reciprocal instead of two divides. The
        // four-wide cos is left as it was: folding it into a single cos/sin plus
        // a rotation measured *slower* than the hardware's vector version.
        let w = 1.0 / (d*2.0 * z);
        let cos_val = cos(vec4(z * 0.5 + u_time) + vec4(0.0, 2.0, 4.0, 3.0));
        color += (cos_val + 1.3) * w;

        // This step is too faint to reach the target, and every later one is
        // fainter still. Deliberately the *only* test in the loop: an additional
        // "accumulator has saturated the tanh" check was exact, but cost 13%.
        if (w < cutoff) { break; }
    }

    color = tanh(color / 9e2);
    return vec4(color);
}
