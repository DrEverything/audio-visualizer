struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    speed: f32,
    camera_rot: vec2<f32>,
    camera_zoom: f32,
    morph_factor: f32,
    light_dir: vec3<f32>,
    steps: u32,
    color_palette: u32,
    glow_intensity: f32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a single triangle covering the screen in clip space
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// Rotation utils
fn rotate_x(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(
        v.x,
        v.y * c - v.z * s,
        v.y * s + v.z * c
    );
}

fn rotate_y(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(
        v.x * c + v.z * s,
        v.y,
        -v.x * s + v.z * c
    );
}

// SDF Functions
fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sd_torus(p: vec3<f32>, t: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Smooth minimum for blending shapes
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Evaluate scene distance field
fn map(p: vec3<f32>) -> vec2<f32> {
    let t = uniforms.time * uniforms.speed;
    
    // Main morphing shape at origin
    let p_center = p;
    let d_sphere = sd_sphere(p_center, 1.2);
    
    let p_torus = rotate_x(rotate_y(p_center, t * 0.5), t * 0.3);
    let d_torus = sd_torus(p_torus, vec2<f32>(1.1, 0.35));
    
    let p_box = rotate_y(p_center, t * 0.6);
    let d_box = sd_box(p_box, vec3<f32>(0.85));
    
    // Morph blend
    let m = uniforms.morph_factor;
    var d = 0.0;
    if (m < 1.0) {
        d = smin(d_sphere, d_torus, 0.3);
        d = mix(d_sphere, d, m);
    } else {
        let m2 = m - 1.0;
        d = smin(d_torus, d_box, 0.3);
        d = mix(d_torus, d, m2);
    }
    
    // Orbiting glowing satellite
    let orbit_r = 2.4 + 0.3 * sin(t * 1.2);
    let sat_pos = vec3<f32>(orbit_r * cos(t * 1.5), 0.7 * sin(t * 2.0), orbit_r * sin(t * 1.5));
    let d_sat = sd_sphere(p - sat_pos, 0.3);
    
    // Combine primary shape and satellite
    let d_scene = smin(d, d_sat, 0.4);
    
    var mat_id = 1.0;
    if (d_sat < d) {
        mat_id = 2.0;
    }
    
    // Floor plane
    let d_floor = p.y + 2.0;
    
    if (d_floor < d_scene) {
        return vec2<f32>(d_floor, 3.0); // 3.0 is floor material
    }
    
    return vec2<f32>(d_scene, mat_id);
}

// Calculate normal vector
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    return normalize(vec3<f32>(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
    ));
}

// Estimate Ambient Occlusion
fn calc_ao(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        let hr = 0.01 + 0.12 * f32(i) / 4.0;
        let aopos = n * hr + p;
        let dd = map(aopos).x;
        occ += -(dd - hr) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

// Calculate soft shadow
fn shadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, maxt: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    for (var i = 0; i < 24; i = i + 1) {
        let h = map(ro + rd * t).x;
        if (h < 0.001) {
            return 0.0;
        }
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.02, 0.2);
        if (t > maxt) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

// Palette generator
fn palette(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318 * (c * t + d));
}

fn get_color(t_val: f32, palette_idx: u32) -> vec3<f32> {
    if (palette_idx == 0u) { // Neon / Rainbow
        return palette(t_val, vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.33, 0.67));
    } else if (palette_idx == 1u) { // Sunset Warmth
        return palette(t_val, vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(2.0, 1.0, 0.0), vec3<f32>(0.5, 0.20, 0.25));
    } else if (palette_idx == 2u) { // Forest Green/Gold
        return palette(t_val, vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.3, 0.20, 0.20));
    } else { // Cyberpunk Blue/Magenta
        return palette(t_val, vec3<f32>(0.8, 0.5, 0.4), vec3<f32>(0.2, 0.4, 0.2), vec3<f32>(2.0, 1.0, 1.0), vec3<f32>(0.0, 0.25, 0.25));
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    let uv = vec2<f32>(in.uv.x * aspect, in.uv.y);
    
    // Ray origin & direction
    var ro = vec3<f32>(0.0, 0.0, -uniforms.camera_zoom);
    var rd = normalize(vec3<f32>(uv, 1.5));
    
    // Apply camera rotation
    ro = rotate_x(rotate_y(ro, uniforms.camera_rot.x), uniforms.camera_rot.y);
    rd = rotate_x(rotate_y(rd, uniforms.camera_rot.x), uniforms.camera_rot.y);
    
    var t = 0.1;
    let max_dist = 40.0;
    var hit = false;
    var d_res = vec2<f32>(0.0);
    var glow = 0.0;
    
    for (var i = 0u; i < uniforms.steps; i = i + 1u) {
        let p = ro + rd * t;
        d_res = map(p);
        let d = d_res.x;
        
        // Accumulate glow around satellite (material 2)
        if (d_res.y == 2.0) {
            glow += 0.008 / (0.005 + d * d);
        }
        
        if (d < 0.001) {
            hit = true;
            break;
        }
        t += d;
        if (t > max_dist) {
            break;
        }
    }
    
    // Default background
    let bg_grad = 0.5 * (rd.y + 1.0);
    let bg_color = mix(vec3<f32>(0.02, 0.015, 0.05), vec3<f32>(0.1, 0.08, 0.16), bg_grad);
    var col = bg_color;
    
    if (hit) {
        let p = ro + rd * t;
        let n = calc_normal(p);
        let r = reflect(rd, n);
        
        let light_dir = normalize(uniforms.light_dir);
        let occ = calc_ao(p, n);
        let shad = shadow(p, light_dir, 0.01, 8.0);
        
        let dif = clamp(dot(n, light_dir), 0.0, 1.0) * shad;
        let spe = pow(clamp(dot(r, light_dir), 0.0, 1.0), 16.0) * dif;
        let amb = 0.5 + 0.5 * n.y;
        
        var mat_col = vec3<f32>(0.0);
        let mat_id = d_res.y;
        
        if (mat_id == 1.0) { // Primary morphing shape
            let p_len = length(p.xyz);
            mat_col = get_color(p_len * 0.15 + sin(uniforms.time * 0.25) * 0.1, uniforms.color_palette);
        } else if (mat_id == 2.0) { // Satellite
            mat_col = get_color(uniforms.time * 0.2, uniforms.color_palette) * 1.5;
        } else if (mat_id == 3.0) { // Floor Grid
            let grid_size = 1.0;
            let f = fract(p.xz / grid_size);
            let grid = step(0.96, f.x) + step(0.96, f.y);
            let floor_base = vec3<f32>(0.03, 0.03, 0.05);
            let grid_col = get_color(0.3, uniforms.color_palette) * 0.7;
            mat_col = mix(floor_base, grid_col, clamp(grid, 0.0, 1.0));
            // Floor reflection
            mat_col += vec3<f32>(0.05) * max(0.0, r.y);
        }
        
        var lighting = vec3<f32>(0.0);
        lighting += dif * vec3<f32>(1.0, 0.95, 0.85);
        lighting += spe * vec3<f32>(1.0, 0.9, 0.7) * 1.5;
        lighting += amb * vec3<f32>(0.15, 0.2, 0.3) * occ;
        lighting += vec3<f32>(0.02); // Ambient base
        
        col = mat_col * lighting;
        
        // Fog
        col = mix(col, bg_color, 1.0 - exp(-0.015 * t * t));
    }
    
    // Add glow effect
    col += get_color(uniforms.time * 0.3, uniforms.color_palette) * glow * uniforms.glow_intensity * 0.1;
    
    // Reinhard tone mapping & gamma correction
    col = col / (col + vec3<f32>(1.0));
    col = pow(col, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(col, 1.0);
}
