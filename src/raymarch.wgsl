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
@group(0) @binding(1) var prev_texture_unused: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler_unused: sampler;

// We declare the storage buffers for bind group layout compatibility
struct BvhNode {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    left_child: u32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    tri_count: u32,
}

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
}

struct Material {
    base_color: vec4<f32>, // r, g, b, metallic
    properties: vec4<f32>, // roughness, ior, transmission, padding
    emissive: vec4<f32>,   // r, g, b, padding
}

@group(0) @binding(3) var<storage, read> bvh_nodes_unused: array<BvhNode>;
@group(0) @binding(4) var<storage, read> vertices_unused: array<Vertex>;
@group(0) @binding(5) var<storage, read> indices_unused: array<u32>;
@group(0) @binding(6) var<storage, read> tri_materials_unused: array<u32>;
@group(0) @binding(7) var<storage, read> materials_unused: array<Material>;

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

// Helper utilities
fn dot2_v2(v: vec2<f32>) -> f32 { return dot(v, v); }
fn dot2_v3(v: vec3<f32>) -> f32 { return dot(v, v); }
fn ndot(a: vec2<f32>, b: vec2<f32>) -> f32 { return a.x * b.x - a.y * b.y; }

// Rotation utils
fn rotate_x(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

fn rotate_y(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

fn setCamera(ro: vec3<f32>, ta: vec3<f32>, cr: f32) -> mat3x3<f32> {
    let cw = normalize(ta - ro);
    let cp = vec3<f32>(sin(cr), cos(cr), 0.0);
    let cu = normalize(cross(cw, cp));
    let cv = cross(cu, cw);
    return mat3x3<f32>(cu, cv, cw);
}


// ------------------------------------------------------------------
// SDF Distance Functions
// ------------------------------------------------------------------
fn sdPlane(p: vec3<f32>) -> f32 {
    return p.y;
}

fn sdSphere(p: vec3<f32>, s: f32) -> f32 {
    return length(p) - s;
}

fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3<f32>(0.0)));
}

fn sdBoxFrame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32 {
    let p_abs = abs(p) - b;
    let q = abs(p_abs + vec3<f32>(e)) - vec3<f32>(e);
    return min(min(
        length(max(vec3<f32>(p_abs.x, q.y, q.z), vec3<f32>(0.0))) + min(max(p_abs.x, max(q.y, q.z)), 0.0),
        length(max(vec3<f32>(q.x, p_abs.y, q.z), vec3<f32>(0.0))) + min(max(q.x, max(p_abs.y, q.z)), 0.0)),
        length(max(vec3<f32>(q.x, q.y, p_abs.z), vec3<f32>(0.0))) + min(max(q.x, max(q.y, p_abs.z)), 0.0)
    );
}

fn sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

fn sdCappedTorus(p: vec3<f32>, sc: vec2<f32>, ra: f32, rb: f32) -> f32 {
    var p_mod = p;
    p_mod.x = abs(p_mod.x);
    var k = length(p_mod.xy);
    if (sc.y * p_mod.x > sc.x * p_mod.y) {
        k = dot(p_mod.xy, sc);
    }
    return sqrt(dot(p_mod, p_mod) + ra * ra - 2.0 * ra * k) - rb;
}

fn sdHexPrism(p: vec3<f32>, h: vec2<f32>) -> f32 {
    var p_mod = p;
    let k = vec3<f32>(-0.8660254, 0.5, 0.57735);
    p_mod = abs(p_mod);
    let dot_val = min(dot(k.xy, p_mod.xy), 0.0);
    p_mod.x = p_mod.x - 2.0 * dot_val * k.x;
    p_mod.y = p_mod.y - 2.0 * dot_val * k.y;
    let d = vec2<f32>(
        length(p_mod.xy - vec2<f32>(clamp(p_mod.x, -k.z * h.x, k.z * h.x), h.x)) * sign(p_mod.y - h.x),
        p_mod.z - h.y
    );
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdOctogonPrism(p: vec3<f32>, r: f32, h: f32) -> f32 {
    var p_mod = p;
    let k = vec3<f32>(-0.9238795325, 0.3826834323, 0.4142135623);
    p_mod = abs(p_mod);
    var dot_val = min(dot(k.xy, p_mod.xy), 0.0);
    p_mod.x = p_mod.x - 2.0 * dot_val * k.x;
    p_mod.y = p_mod.y - 2.0 * dot_val * k.y;
    dot_val = min(dot(vec2<f32>(-k.x, k.y), p_mod.xy), 0.0);
    p_mod.x = p_mod.x + 2.0 * dot_val * k.x;
    p_mod.y = p_mod.y - 2.0 * dot_val * k.y;
    p_mod.x = p_mod.x - clamp(p_mod.x, -k.z * r, k.z * r);
    p_mod.y = p_mod.y - r;
    let d = vec2<f32>(length(p_mod.xy) * sign(p_mod.y), p_mod.z - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdCapsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

fn sdRoundCone1(p: vec3<f32>, r1: f32, r2: f32, h: f32) -> f32 {
    let q = vec2<f32>(length(p.xz), p.y);
    let b = (r1 - r2) / h;
    let a = sqrt(1.0 - b * b);
    let k = dot(q, vec2<f32>(-b, a));
    if (k < 0.0) { return length(q) - r1; }
    if (k > a * h) { return length(q - vec2<f32>(0.0, h)) - r2; }
    return dot(q, vec2<f32>(a, b)) - r1;
}

fn sdRoundCone2(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r1: f32, r2: f32) -> f32 {
    let ba = b - a;
    let l2 = dot(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = dot(pa, ba);
    let z = y - l2;
    let x2 = dot2_v3(pa * l2 - ba * y);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) { return sqrt(x2 + z2) * il2 - r2; }
    if (sign(y) * a2 * y2 < k) { return sqrt(x2 + y2) * il2 - r1; }
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

fn sdTriPrism(p: vec3<f32>, h: vec2<f32>) -> f32 {
    let k = sqrt(3.0);
    let h_x = h.x * 0.5 * k;
    var p_xy = p.xy / h_x;
    p_xy.x = abs(p_xy.x) - 1.0;
    p_xy.y = p_xy.y + 1.0 / k;
    if (p_xy.x + k * p_xy.y > 0.0) {
        p_xy = vec2<f32>(p_xy.x - k * p_xy.y, -k * p_xy.x - p_xy.y) / 2.0;
    }
    p_xy.x = p_xy.x - clamp(p_xy.x, -2.0, 0.0);
    let d1 = length(p_xy) * sign(-p_xy.y) * h_x;
    let d2 = abs(p.z) - h.y;
    return length(max(vec2<f32>(d1, d2), vec2<f32>(0.0))) + min(max(d1, d2), 0.0);
}

fn sdCylinder1(p: vec3<f32>, h: vec2<f32>) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdCylinder2(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let baba = dot(ba, ba);
    let paba = dot(pa, ba);
    let x = length(pa * baba - ba * paba) - r * baba;
    let y = abs(paba - baba * 0.5) - baba * 0.5;
    let x2 = x * x;
    let y2 = y * y * baba;
    var d = 0.0;
    if (max(x, y) < 0.0) {
        d = -min(x2, y2);
    } else {
        var val_x = 0.0;
        if (x > 0.0) { val_x = x2; }
        var val_y = 0.0;
        if (y > 0.0) { val_y = y2; }
        d = val_x + val_y;
    }
    return sign(d) * sqrt(abs(d)) / baba;
}

fn sdCone1(p: vec3<f32>, c: vec2<f32>, h: f32) -> f32 {
    let q = h * vec2<f32>(c.x, -c.y) / c.y;
    let w = vec2<f32>(length(p.xz), p.y);
    let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    let b = w - q * vec2<f32>(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    let k = sign(q.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}

fn sdCappedCone1(p: vec3<f32>, h: f32, r1: f32, r2: f32) -> f32 {
    let q = vec2<f32>(length(p.xz), p.y);
    let k1 = vec2<f32>(r2, h);
    let k2 = vec2<f32>(r2 - r1, 2.0 * h);
    var min_val = r2;
    if (q.y < 0.0) {
        min_val = r1;
    }
    let ca = vec2<f32>(q.x - min(q.x, min_val), abs(q.y) - h);
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot2_v2(k2), 0.0, 1.0);
    var s = 1.0;
    if (cb.x < 0.0 && ca.y < 0.0) {
        s = -1.0;
    }
    return s * sqrt(min(dot2_v2(ca), dot2_v2(cb)));
}

fn sdCappedCone2(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32, rb: f32) -> f32 {
    let rba = rb - ra;
    let baba = dot(b - a, b - a);
    let papa = dot(p - a, p - a);
    let paba = dot(p - a, b - a) / baba;
    let x = sqrt(max(0.0, papa - paba * paba * baba));
    var val_r = rb;
    if (paba < 0.5) {
        val_r = ra;
    }
    let cax = max(0.0, x - val_r);
    let cay = abs(paba - 0.5) - 0.5;
    let k = rba * rba + baba;
    let f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);
    let cbx = x - ra - f * rba;
    let cby = paba - f;
    var s = 1.0;
    if (cbx < 0.0 && cay < 0.0) {
        s = -1.0;
    }
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

fn sdSolidAngle(pos: vec3<f32>, c: vec2<f32>, ra: f32) -> f32 {
    let p = vec2<f32>(length(pos.xz), pos.y);
    let l = length(p) - ra;
    let m = length(p - c * clamp(dot(p, c), 0.0, ra));
    return max(l, m * sign(c.y * p.x - c.x * p.y));
}

fn sdOctahedron(p: vec3<f32>, s: f32) -> f32 {
    let p_abs = abs(p);
    let m = p_abs.x + p_abs.y + p_abs.z - s;
    var q: vec3<f32>;
    if (3.0 * p_abs.x < m) {
        q = p_abs.xyz;
    } else if (3.0 * p_abs.y < m) {
        q = p_abs.yzx;
    } else if (3.0 * p_abs.z < m) {
        q = p_abs.zxy;
    } else {
        return m * 0.57735027;
    }
    let k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3<f32>(q.x, q.y - s + k, q.z - k));
}

fn sdPyramid(p: vec3<f32>, h: f32) -> f32 {
    let m2 = h * h + 0.25;
    var p_mod = p;
    p_mod.x = abs(p_mod.x);
    p_mod.z = abs(p_mod.z);
    if (p_mod.z > p_mod.x) {
        let tmp = p_mod.x;
        p_mod.x = p_mod.z;
        p_mod.z = tmp;
    }
    p_mod.x = p_mod.x - 0.5;
    p_mod.z = p_mod.z - 0.5;
    var q = vec3<f32>(p_mod.z, h * p_mod.y - 0.5 * p_mod.x, h * p_mod.x + 0.5 * p_mod.y);
    let s = max(-q.x, 0.0);
    let t = clamp((q.y - 0.5 * p_mod.z) / (m2 + 0.25), 0.0, 1.0);
    let a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    let b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    var d2 = 0.0;
    if (min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) {
        d2 = 0.0;
    } else {
        d2 = min(a, b);
    }
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p_mod.y));
}

fn sdRhombus(p: vec3<f32>, la: f32, lb: f32, h: f32, ra: f32) -> f32 {
    let p_abs = abs(p);
    let b = vec2<f32>(la, lb);
    let f = clamp(ndot(b, b - 2.0 * p_abs.xz) / dot(b, b), -1.0, 1.0);
    let q = vec2<f32>(length(p_abs.xz - 0.5 * b * vec2<f32>(1.0 - f, 1.0 + f)) * sign(p_abs.x * b.y + p_abs.z * b.x - b.x * b.y) - ra, p_abs.y - h);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));
}

fn sdHorseshoe(p: vec3<f32>, c: vec2<f32>, r: f32, le: f32, w: vec2<f32>) -> f32 {
    var p_mod = p;
    p_mod.x = abs(p_mod.x);
    let l = length(p_mod.xy);
    let mat = mat2x2<f32>(-c.x, c.y, c.y, c.x);
    let rotated = mat * p_mod.xy;
    p_mod.x = rotated.x;
    p_mod.y = rotated.y;
    var new_x = l * sign(-c.x);
    if (p_mod.y > 0.0 || p_mod.x > 0.0) {
        new_x = p_mod.x;
    }
    var new_y = l;
    if (p_mod.x > 0.0) {
        new_y = p_mod.y;
    }
    p_mod.x = new_x;
    p_mod.y = new_y;
    let p_xy = vec2<f32>(p_mod.x, abs(p_mod.y - r)) - vec2<f32>(le, 0.0);
    let q = vec2<f32>(length(max(p_xy, vec2<f32>(0.0))) + min(0.0, max(p_xy.x, p_xy.y)), p_mod.z);
    let d = abs(q) - w;
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdU(p: vec3<f32>, r: f32, le: f32, w: vec2<f32>) -> f32 {
    var p_mod = p;
    var px = length(p_mod.xy);
    if (p_mod.y > 0.0) {
        px = abs(p_mod.x);
    }
    p_mod.x = px;
    p_mod.x = abs(p_mod.x - r);
    p_mod.y = p_mod.y - le;
    let k = max(p_mod.x, p_mod.y);
    let q = vec2<f32>(select(length(max(p_mod.xy, vec2<f32>(0.0))), -k, k < 0.0), abs(p_mod.z)) - w;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sdMandelbulb(p: vec3<f32>) -> vec2<f32> {
    // 1. Y-axis Spin (Rotation over time)
    let angle = uniforms.time * 0.2;
    let s = sin(angle);
    let c = cos(angle);
    var rotated_p = vec3<f32>(
        p.x * c + p.z * s,
        p.y,
        -p.x * s + p.z * c
    );

    var w = rotated_p;
    var dr = 1.0;
    var r = 0.0;
    
    // 2. Pulsing Power (Morphs power dynamically between 3.0 and 8.0 over time)
    let power = 3.0 + 5.0 * (0.5 + 0.5 * sin(uniforms.time * 0.3));
    var trap = 1e20; // Orbit trap for coloring
    
    for (var i = 0; i < 10; i = i + 1) {
        r = length(w);
        if (r > 2.0) { break; }
        
        // Orbit trap (minimum squared distance to trace fractal color)
        trap = min(trap, dot(w, w));
        
        // Convert to polar coordinates with safety clamping
        var theta = 0.0;
        var phi = 0.0;
        if (r > 0.0001) {
            theta = acos(clamp(w.y / r, -1.0, 1.0));
            phi = atan2(w.x, w.z);
        }
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        
        // Scale and rotate potential
        let zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;
        
        // Convert back to cartesian coordinates
        w = zr * vec3<f32>(sin(theta) * sin(phi), cos(theta), sin(theta) * cos(phi));
        
        // 3. Structural Distortion (Morphs geometry offsets slightly over time)
        let dist_offset = vec3<f32>(
            sin(uniforms.time * 0.5),
            cos(uniforms.time * 0.7),
            sin(uniforms.time * 0.2)
        ) * 0.03;
        
        w = w + rotated_p + dist_offset;
    }
    
    let d = 0.5 * log(r) * r / dr;
    
    // 4. Color Cycling (Shifts and cycles orbit trap material indexes over time)
    let mat_index = 3.0 + trap * (15.0 + 10.0 * sin(uniforms.time * 0.5)) + uniforms.time * 1.5;
    return vec2<f32>(d, mat_index);
}

// ------------------------------------------------------------------
// Combination Operator
// ------------------------------------------------------------------
fn opU(d1: vec2<f32>, d2: vec2<f32>) -> vec2<f32> {
    if (d1.x < d2.x) {
        return d1;
    } else {
        return d2;
    }
}

// ------------------------------------------------------------------
// Map Scene
// ------------------------------------------------------------------
fn map(pos: vec3<f32>) -> vec2<f32> {
    var res = vec2<f32>(pos.y, 0.0);

    // 5. Hover Bobbing (Floats the Mandelbulb up and down over time)
    let bobbing = 0.15 * sin(uniforms.time * 0.8);
    let mb = sdMandelbulb(pos - vec3<f32>(0.0, 1.2 + bobbing, 0.0));
    res = opU(res, mb);

    /* Commented out all other shapes
    // bounding box 1
    if (sdBox(pos - vec3<f32>(-2.0, 0.3, 0.25), vec3<f32>(0.3, 0.3, 1.0)) < res.x) {
        res = opU(res, vec2<f32>(sdSphere(pos - vec3<f32>(-2.0, 0.25, 0.0), 0.25), 26.9));
        res = opU(res, vec2<f32>(sdRhombus((pos - vec3<f32>(-2.0, 0.25, 1.0)).xzy, 0.15, 0.25, 0.04, 0.08), 17.0));
    }

    // bounding box 2
    if (sdBox(pos - vec3<f32>(0.0, 0.3, -1.0), vec3<f32>(0.35, 0.3, 2.5)) < res.x) {
        res = opU(res, vec2<f32>(sdCappedTorus((pos - vec3<f32>(0.0, 0.30, 1.0)) * vec3<f32>(1.0, -1.0, 1.0), vec2<f32>(0.866025, -0.5), 0.25, 0.05), 25.0));
        res = opU(res, vec2<f32>(sdBoxFrame(pos - vec3<f32>(0.0, 0.25, 0.0), vec3<f32>(0.3, 0.25, 0.2), 0.025), 16.9));
        res = opU(res, vec2<f32>(sdCone1(pos - vec3<f32>(0.0, 0.45, -1.0), vec2<f32>(0.6, 0.8), 0.45), 55.0));
        res = opU(res, vec2<f32>(sdCappedCone1(pos - vec3<f32>(0.0, 0.25, -2.0), 0.25, 0.25, 0.1), 13.67));
        res = opU(res, vec2<f32>(sdSolidAngle(pos - vec3<f32>(0.0, 0.00, -3.0), vec2<f32>(3.0, 4.0) / 5.0, 0.4), 49.13));
    }

    // bounding box 3
    if (sdBox(pos - vec3<f32>(1.0, 0.3, -1.0), vec3<f32>(0.35, 0.3, 2.5)) < res.x) {
        res = opU(res, vec2<f32>(sdTorus((pos - vec3<f32>(1.0, 0.30, 1.0)).xzy, vec2<f32>(0.25, 0.05)), 7.1));
        res = opU(res, vec2<f32>(sdBox(pos - vec3<f32>(1.0, 0.25, 0.0), vec3<f32>(0.3, 0.25, 0.1)), 3.0));
        res = opU(res, vec2<f32>(sdCapsule(pos - vec3<f32>(1.0, 0.00, -1.0), vec3<f32>(-0.1, 0.1, -0.1), vec3<f32>(0.2, 0.4, 0.2), 0.1), 31.9));
        res = opU(res, vec2<f32>(sdCylinder1(pos - vec3<f32>(1.0, 0.25, -2.0), vec2<f32>(0.15, 0.25)), 8.0));
        res = opU(res, vec2<f32>(sdHexPrism(pos - vec3<f32>(1.0, 0.2, -3.0), vec2<f32>(0.2, 0.05)), 18.4));
    }

    // bounding box 4
    if (sdBox(pos - vec3<f32>(-1.0, 0.35, -1.0), vec3<f32>(0.35, 0.35, 2.5)) < res.x) {
        res = opU(res, vec2<f32>(sdPyramid(pos - vec3<f32>(-1.0, -0.6, -3.0), 1.0), 13.56));
        res = opU(res, vec2<f32>(sdOctahedron(pos - vec3<f32>(-1.0, 0.15, -2.0), 0.35), 23.56));
        res = opU(res, vec2<f32>(sdTriPrism(pos - vec3<f32>(-1.0, 0.15, -1.0), vec2<f32>(0.3, 0.05)), 43.5));
        res = opU(res, vec2<f32>(sdEllipsoid(pos - vec3<f32>(-1.0, 0.25, 0.0), vec3<f32>(0.2, 0.25, 0.05)), 43.17));
        res = opU(res, vec2<f32>(sdHorseshoe(pos - vec3<f32>(-1.0, 0.25, 1.0), vec2<f32>(cos(1.3), sin(1.3)), 0.2, 0.3, vec2<f32>(0.03, 0.08)), 11.5));
    }

    // bounding box 5
    if (sdBox(pos - vec3<f32>(2.0, 0.3, -1.0), vec3<f32>(0.35, 0.3, 2.5)) < res.x) {
        res = opU(res, vec2<f32>(sdOctogonPrism(pos - vec3<f32>(2.0, 0.2, -3.0), 0.2, 0.05), 51.8));
        res = opU(res, vec2<f32>(sdCylinder2(pos - vec3<f32>(2.0, 0.14, -2.0), vec3<f32>(0.1, -0.1, 0.0), vec3<f32>(-0.2, 0.35, 0.1), 0.08), 31.2));
        res = opU(res, vec2<f32>(sdCappedCone2(pos - vec3<f32>(2.0, 0.09, -1.0), vec3<f32>(0.1, 0.0, 0.0), vec3<f32>(-0.2, 0.40, 0.1), 0.15, 0.05), 46.1));
        res = opU(res, vec2<f32>(sdRoundCone2(pos - vec3<f32>(2.0, 0.15, 0.0), vec3<f32>(0.1, 0.0, 0.0), vec3<f32>(-0.1, 0.35, 0.1), 0.15, 0.05), 51.7));
        res = opU(res, vec2<f32>(sdRoundCone1(pos - vec3<f32>(2.0, 0.20, 1.0), 0.2, 0.1, 0.3), 37.0));
    }
    */

    return res;
}

// Ray-box intersection (Slab method)
fn iBox(ro: vec3<f32>, rd: vec3<f32>, rad: vec3<f32>) -> vec2<f32> {
    let m = vec3<f32>(1.0) / rd;
    let n = m * ro;
    let k = abs(m) * rad;
    let t1 = -n - k;
    let t2 = -n + k;
    return vec2<f32>(
        max(max(t1.x, t1.y), t1.z),
        min(min(t2.x, t2.y), t2.z)
    );
}

fn raycast(ro: vec3<f32>, rd: vec3<f32>, tmax_in: f32) -> vec2<f32> {
    var res = vec2<f32>(-1.0, -1.0);
    var tmin = 1.0;
    var tmax = tmax_in;

    // raytrace floor plane
    let tp1 = (0.0 - ro.y) / rd.y;
    if (tp1 > 0.0) {
        tmax = min(tmax, tp1);
        res = vec2<f32>(tp1, 1.0);
    }

    // raymarch primitives (updated bounding box for Mandelbulb)
    let tb = iBox(ro - vec3<f32>(0.0, 1.2, 0.0), rd, vec3<f32>(1.5, 1.5, 1.5));
    if (tb.x < tb.y && tb.y > 0.0 && tb.x < tmax) {
        tmin = max(tb.x, tmin);
        tmax = min(tb.y, tmax);

        var t = tmin;
        for (var i = 0; i < 70; i = i + 1) {
            if (t >= tmax) { break; }
            let h = map(ro + rd * t);
            if (abs(h.x) < (0.0002 * t)) { 
                res = vec2<f32>(t, h.y); 
                break;
            }
            t = t + h.x;
        }
    }

    return res;
}

fn calcSoftshadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, tmax_in: f32) -> f32 {
    var tmax = tmax_in;
    // bounding volume
    let tp = (2.5 - ro.y) / rd.y;
    if (tp > 0.0) {
        tmax = min(tmax, tp);
    }

    var res = 1.0;
    var t = mint;
    for (var i = 0; i < 24; i = i + 1) {
        let h = map(ro + rd * t).x;
        let s = clamp(8.0 * h / t, 0.0, 1.0);
        res = min(res, s);
        t = t + clamp(h, 0.01, 0.2);
        if (res < 0.004 || t > tmax) { break; }
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

// Normal calculation (prevent compiler from unrolling/inlining)
fn calcNormal(pos: vec3<f32>) -> vec3<f32> {
    var n = vec3<f32>(0.0);
    let zero = min(uniforms.frame_index, 0u);
    for (var i = zero; i < zero + 4u; i = i + 1u) {
        let e = 0.5773 * (2.0 * vec3<f32>(
            f32(((i + 3u) >> 1u) & 1u),
            f32((i >> 1u) & 1u),
            f32(i & 1u)
        ) - vec3<f32>(1.0));
        n = n + e * map(pos + 0.0005 * e).x;
    }
    return normalize(n);
}

// Ambient Occlusion
fn calcAO(pos: vec3<f32>, nor: vec3<f32>) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        let h = 0.01 + 0.12 * f32(i) / 4.0;
        let d = map(pos + h * nor).x;
        occ = occ + (h - d) * sca;
        sca = sca * 0.95;
        if (occ > 0.35) { break; }
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * nor.y);
}

// Checkerboard floor filter
fn checkersGradBox(p: vec2<f32>, dpdx: vec2<f32>, dpdy: vec2<f32>) -> f32 {
    let w = abs(dpdx) + abs(dpdy) + vec2<f32>(0.001);
    let i = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5) - abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;
    return 0.5 - 0.5 * i.x * i.y;                  
}

fn render(ro: vec3<f32>, rd: vec3<f32>, rdx: vec3<f32>, rdy: vec3<f32>) -> vec3<f32> { 
    // background
    var col = vec3<f32>(0.7, 0.7, 0.9) - max(rd.y, 0.0) * 0.3;
    
    // raycast scene
    let res = raycast(ro, rd, 20.0);
    let t = res.x;
    let m = res.y;
    if (m > -0.5) {
        let pos = ro + t * rd;
        var nor = vec3<f32>(0.0, 1.0, 0.0);
        if (m >= 1.5) {
            nor = calcNormal(pos);
        }
        let refl = reflect(rd, nor);
        
        // material        
        col = 0.2 + 0.2 * sin(m * 2.0 + vec3<f32>(0.0, 1.0, 2.0));
        var ks = 1.0;
        
        if (m < 1.5) {
            // project pixel footprint into the plane
            let dpdx = ro.y * (rd / rd.y - rdx / rdx.y);
            let dpdy = ro.y * (rd / rd.y - rdy / rdy.y);

            let f = checkersGradBox(3.0 * pos.xz, 3.0 * dpdx.xz, 3.0 * dpdy.xz);
            col = 0.15 + f * vec3<f32>(0.05);
            ks = 0.4;
        }

        // lighting
        let occ = calcAO(pos, nor);
        var lin = vec3<f32>(0.0);

        // sun
        {
            let lig = normalize(vec3<f32>(-0.5, 0.4, -0.6));
            let hal = normalize(lig - rd);
            var dif = clamp(dot(nor, lig), 0.0, 1.0);
            if (dif > 0.0001) {
                dif = dif * calcSoftshadow(pos, lig, 0.02, 2.5);
            }
            var spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 16.0);
            spe = spe * dif;
            spe = spe * (0.04 + 0.96 * pow(clamp(1.0 - dot(hal, lig), 0.0, 1.0), 5.0));
            lin = lin + col * 2.20 * dif * vec3<f32>(1.30, 1.00, 0.70);
            lin = lin + 5.00 * spe * vec3<f32>(1.30, 1.00, 0.70) * ks;
        }
        // sky
        {
            var dif = sqrt(clamp(0.5 + 0.5 * nor.y, 0.0, 1.0));
            dif = dif * occ;
            var spe = smoothstep(-0.2, 0.2, refl.y);
            spe = spe * dif;
            spe = spe * (0.04 + 0.96 * pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 5.0));
            if (spe > 0.001) {
                spe = spe * calcSoftshadow(pos, refl, 0.02, 2.5);
            }
            lin = lin + col * 0.60 * dif * vec3<f32>(0.40, 0.60, 1.15);
            lin = lin + 2.00 * spe * vec3<f32>(0.40, 0.60, 1.30) * ks;
        }
        // back
        {
            let dif = clamp(dot(nor, normalize(vec3<f32>(0.5, 0.0, 0.6))), 0.0, 1.0);
            let dif_occ = dif * occ;
            lin = lin + col * 0.55 * dif_occ * vec3<f32>(0.25, 0.25, 0.25);
        }
        // sss
        {
            let dif = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 2.0);
            let dif_occ = dif * occ;
            lin = lin + col * 0.25 * dif_occ * vec3<f32>(1.00, 1.00, 1.00);
        }
        
        col = lin;
        col = mix(col, vec3<f32>(0.7, 0.7, 0.9), 1.0 - exp(-0.0001 * t * t * t));
    }

    return vec3<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)));
}

@fragment
fn fs_raymarch(in: VertexOutput) -> @location(0) vec4<f32> {
    let frag_coord = in.position.xy;
    
    // Set camera target (centered at the Mandelbulb's center of mass)
    let ta = vec3<f32>(0.0, 1.2, 0.0);
    
    // Position camera dynamically based on interactive controls: yaw, pitch, and zoom
    let yaw = uniforms.camera_rot.x;
    let pitch = uniforms.camera_rot.y;
    let r = uniforms.camera_zoom;
    
    let offset = vec3<f32>(
        r * cos(pitch) * sin(yaw),
        r * sin(pitch),
        r * cos(pitch) * cos(yaw)
    );
    let ro = ta + offset;
    
    // Camera-to-world transformation
    let ca = setCamera(ro, ta, 0.0);
    
    // Focal length matching the original GLSL
    let fl: f32 = 2.5;
    
    // Anti-aliasing level retrieved dynamically from uniforms
    let AA = i32(uniforms.samples_per_frame);
    
    var tot = vec3<f32>(0.0);
    
    if (AA > 1) {
        for (var m = 0; m < AA; m = m + 1) {
            for (var n = 0; n < AA; n = n + 1) {
                // Pixel coordinates offset
                let o = vec2<f32>(f32(m), f32(n)) / f32(AA) - 0.5;
                let coord = frag_coord + o;
                
                // Flip Y coordinate mapping so that positive Y is up (sky) and negative Y is down (ground)
                let p = vec2<f32>(2.0 * coord.x - uniforms.resolution.x, uniforms.resolution.y - 2.0 * coord.y) / uniforms.resolution.y;
                let px = vec2<f32>(2.0 * (coord.x + 1.0) - uniforms.resolution.x, uniforms.resolution.y - 2.0 * coord.y) / uniforms.resolution.y;
                let py = vec2<f32>(2.0 * coord.x - uniforms.resolution.x, uniforms.resolution.y - 2.0 * (coord.y + 1.0)) / uniforms.resolution.y;
                
                // Point rays using look-at transform
                let rd = ca * normalize(vec3<f32>(p, fl));
                let rdx = ca * normalize(vec3<f32>(px, fl));
                let rdy = ca * normalize(vec3<f32>(py, fl));
                
                tot = tot + render(ro, rd, rdx, rdy);
            }
        }
        tot = tot / f32(AA * AA);
    } else {
        // Flip Y coordinate mapping so that positive Y is up (sky) and negative Y is down (ground)
        let p = vec2<f32>(2.0 * frag_coord.x - uniforms.resolution.x, uniforms.resolution.y - 2.0 * frag_coord.y) / uniforms.resolution.y;
        let px = vec2<f32>(2.0 * (frag_coord.x + 1.0) - uniforms.resolution.x, uniforms.resolution.y - 2.0 * frag_coord.y) / uniforms.resolution.y;
        let py = vec2<f32>(2.0 * frag_coord.x - uniforms.resolution.x, uniforms.resolution.y - 2.0 * (frag_coord.y + 1.0)) / uniforms.resolution.y;
        
        let rd = ca * normalize(vec3<f32>(p, fl));
        let rdx = ca * normalize(vec3<f32>(px, fl));
        let rdy = ca * normalize(vec3<f32>(py, fl));
        
        tot = render(ro, rd, rdx, rdy);
    }
    
    // We return linear color. The fs_display shader will perform gamma correction!
    return vec4<f32>(tot, 1.0);
}
