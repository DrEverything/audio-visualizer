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

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
}

struct Material {
    base_color: vec4<f32>, // r, g, b, metallic
    properties: vec4<f32>, // roughness, ior, transmission, padding
    emissive: vec4<f32>,   // r, g, b, padding
}

// ==========================================
// G-BUFFER RASTERIZATION PASS BINDINGS
// ==========================================
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var prev_texture_unused: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler_unused: sampler;
@group(0) @binding(3) var<storage, read> bvh_nodes_unused: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var<storage, read> tri_materials: array<u32>;
@group(0) @binding(7) var<storage, read> materials: array<Material>;

struct GBufferVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) prev_clip_position: vec4<f32>,
    @location(4) @interpolate(flat) mat_idx: u32,
}

// Procedural instance transform based on time
fn get_instance_transform(inst_id: u32, time: f32) -> mat4x4<f32> {
    if (inst_id == 1u) {
        // Rotating Torus
        let theta = time * 1.2;
        let c = cos(theta);
        let s = sin(theta);
        return mat4x4<f32>(
            vec4<f32>(c, 0.0, s, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(-s, 0.0, c, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        );
    } else if (inst_id == 2u) {
        // Bouncing Gold Sphere
        let dy = sin(time * 2.5) * 0.4;
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, dy, 0.0, 1.0)
        );
    } else if (inst_id == 3u) {
        // Bouncing Glass Sphere (out of phase)
        let dy = sin(time * 2.5 + 3.14159265359) * 0.4;
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, dy, 0.0, 1.0)
        );
    }
    // Identity for Cornell Box / default
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
}

// Rotation utilities
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

fn world_to_clip(p_world: vec3<f32>, rot: vec2<f32>, zoom: f32) -> vec4<f32> {
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    
    var eye = vec3<f32>(0.0, 0.0, -zoom);
    eye = rotate_x(rotate_y(eye, rot.x), rot.y);

    var right = vec3<f32>(1.0, 0.0, 0.0);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    right = rotate_x(rotate_y(right, rot.x), rot.y);
    up = rotate_x(rotate_y(up, rot.x), rot.y);
    
    let forward = normalize(cross(right, up));
    
    let p_rel = p_world - eye;
    let x_cam = dot(p_rel, right);
    let y_cam = dot(p_rel, up);
    let z_cam = dot(p_rel, forward);
    
    let near = 0.1;
    let far = 100.0;
    let w_clip = z_cam;
    let x_clip = x_cam * (1.5 / aspect);
    let y_clip = y_cam * 1.5;
    let z_clip = (z_cam * far / (far - near)) - (far * near / (far - near));
    
    return vec4<f32>(x_clip, y_clip, z_clip, w_clip);
}

@vertex
fn vs_gbuffer(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) inst_id: u32,
) -> GBufferVertexOutput {
    // Dynamic instance geometry offsets
    var start_index = 0u;
    if (inst_id == 0u) {
        start_index = uniforms.tri_offsets.x * 3u;
    } else if (inst_id == 1u) {
        start_index = uniforms.tri_offsets.y * 3u;
    } else if (inst_id == 2u) {
        start_index = uniforms.tri_offsets.z * 3u;
    } else if (inst_id == 3u) {
        start_index = uniforms.tri_offsets.w * 3u;
    }

    let global_index_idx = start_index + v_idx;
    let vertex_idx = indices[global_index_idx];
    let vertex = vertices[vertex_idx];
    
    let material_idx = tri_materials[global_index_idx / 3u];

    var out: GBufferVertexOutput;
    
    // Apply current transformation
    let model_matrix = get_instance_transform(inst_id, uniforms.time);
    let world_pos = (model_matrix * vec4<f32>(vertex.position.xyz, 1.0)).xyz;
    out.clip_position = world_to_clip(world_pos, uniforms.camera_rot, uniforms.camera_zoom);
    out.world_position = world_pos;
    
    let normal_matrix = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    out.world_normal = normalize(normal_matrix * vertex.normal.xyz);
    out.uv = vec2<f32>(0.0); // Procedural meshes do not currently have UVs in this setup
    out.mat_idx = material_idx;

    // Apply previous transformation (for motion vectors)
    let prev_model_matrix = get_instance_transform(inst_id, uniforms.time - uniforms.dt);
    let prev_world_pos = (prev_model_matrix * vec4<f32>(vertex.position.xyz, 1.0)).xyz;
    out.prev_clip_position = world_to_clip(prev_world_pos, uniforms.prev_camera_rot, uniforms.prev_camera_zoom);

    return out;
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
    @location(3) motion: vec2<f32>,
}

@fragment
fn fs_gbuffer(in: GBufferVertexOutput) -> GBufferOutput {
    if (in.mat_idx == 7u) {
        discard;
    }

    var out: GBufferOutput;
    
    let mat = materials[in.mat_idx];
    let albedo = mat.base_color.rgb;
    let metallic = mat.base_color.w;
    let roughness = mat.properties.x;
    let emissive = mat.emissive.rgb;
    
    out.albedo = vec4<f32>(albedo, 1.0);
    // XYZ = World Normal, W = Material index (packed for easy deferred fetch)
    out.normal = vec4<f32>(normalize(in.world_normal), f32(in.mat_idx));
    
    // Average emissive component
    let emissive_avg = (emissive.r + emissive.g + emissive.b) / 3.0;
    out.material = vec4<f32>(roughness, metallic, emissive_avg, 1.0);
    
    // Motion vectors: NDC difference
    let current_ndc = in.clip_position.xy / in.clip_position.w;
    let prev_ndc = in.prev_clip_position.xy / in.prev_clip_position.w;
    out.motion = current_ndc - prev_ndc;
    
    return out;
}

// ==========================================
// DEFERRED LIGHTING PASS SHADERS
// ==========================================
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

@group(0) @binding(0) var<uniform> uniforms_def: Uniforms;
@group(0) @binding(1) var g_albedo: texture_2d<f32>;
@group(0) @binding(2) var g_normal: texture_2d<f32>;
@group(0) @binding(3) var g_material: texture_2d<f32>;
@group(0) @binding(4) var g_depth: texture_depth_2d;
@group(0) @binding(5) var g_motion: texture_2d<f32>;
@group(0) @binding(6) var g_sampler: sampler;
@group(0) @binding(7) var<storage, read> materials_def: array<Material>;

// PBR Helper: Trowbridge-Reitz GGX normal distribution function (NDF)
fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let nom = a2;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return nom / (3.14159265359 * denom * denom);
}

// PBR Helper: Geometry Smith height-correlated masking-shadowing function
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    let nom = NdotV;
    let denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = geometry_schlick_ggx(NdotV, roughness);
    let ggx1 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}

// PBR Helper: Fresnel Schlick approximation
fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ACES Filmic Tonemapping Curve Approximation
fn aces_approx(v: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_deferred_lighting(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = uniforms_def.resolution.x / uniforms_def.resolution.y;

    // 1. Sample G-buffer textures (flipping Y to match display pass coordinates)
    let sample_uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    let albedo_texel = textureSample(g_albedo, g_sampler, sample_uv);
    let normal_texel = textureSample(g_normal, g_sampler, sample_uv);
    let material_texel = textureSample(g_material, g_sampler, sample_uv);
    let depth = textureSample(g_depth, g_sampler, sample_uv);

    // If depth is 1.0, this is the background / sky
    if (depth >= 0.99999) {
        let uv_centered = in.uv * 2.0 - vec2<f32>(1.0);
        var rd = normalize(vec3<f32>(uv_centered.x * aspect, uv_centered.y, 1.5));
        rd = rotate_x(rotate_y(rd, uniforms_def.camera_rot.x), uniforms_def.camera_rot.y);
        
        let sky_grad = 0.5 * (rd.y + 1.0);
        let sky_color = mix(vec3<f32>(0.02, 0.015, 0.05), vec3<f32>(0.1, 0.15, 0.25), sky_grad) * uniforms_def.env_light_intensity;
        return vec4<f32>(sky_color, 1.0);
    }

    // 2. Extract material parameters and normal
    let world_normal = normalize(normal_texel.xyz);
    let mat_idx = u32(round(normal_texel.w));
    
    // Fetch exact properties directly from the materials buffer (guarantees perfect alignment with PT materials)
    let mat = materials_def[mat_idx];
    let albedo = mat.base_color.rgb;
    let metallic = mat.base_color.w;
    let roughness = max(mat.properties.x, 0.05); // Clamp roughness to avoid division by zero
    let emissive = mat.emissive.rgb;

    // 3. Reconstruct world position using depth and ray direction
    var eye = vec3<f32>(0.0, 0.0, -uniforms_def.camera_zoom);
    eye = rotate_x(rotate_y(eye, uniforms_def.camera_rot.x), uniforms_def.camera_rot.y);

    var right = vec3<f32>(1.0, 0.0, 0.0);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    right = rotate_x(rotate_y(right, uniforms_def.camera_rot.x), uniforms_def.camera_rot.y);
    up = rotate_x(rotate_y(up, uniforms_def.camera_rot.x), uniforms_def.camera_rot.y);
    
    let forward = normalize(cross(right, up));

    let uv_centered = in.uv * 2.0 - vec2<f32>(1.0);
    var rd = normalize(vec3<f32>(uv_centered.x * aspect, uv_centered.y, 1.5));
    rd = rotate_x(rotate_y(rd, uniforms_def.camera_rot.x), uniforms_def.camera_rot.y);

    // Compute eye-space depth (z_cam)
    let near = 0.1;
    let far = 100.0;
    let z_cam = (far * near) / (far - depth * (far - near));
    let t = z_cam / dot(rd, forward);
    let world_pos = eye + rd * t;

    // 4. Shading calculations (direct lighting)
    let V = normalize(eye - world_pos);
    let N = world_normal;

    // Base reflectiveness
    var F0 = vec3<f32>(0.04);
    F0 = mix(F0, albedo, metallic);

    var total_radiance = vec3<f32>(0.0);

    // Light 1: Directional Light (Sun)
    {
        let L = normalize(vec3<f32>(0.5, 1.0, 0.3)); // Light direction
        let H = normalize(V + L);
        let radiance = vec3<f32>(1.0, 0.95, 0.9) * 1.5; // Light color & intensity

        // Cook-Torrance BRDF
        let NDF = distribution_ggx(N, H, roughness);
        let G = geometry_smith(N, V, L, roughness);
        let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        let numerator = NDF * G * F;
        let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        let specular = numerator / denominator;

        let kS = F;
        var kD = vec3<f32>(1.0) - kS;
        kD *= 1.0 - metallic;

        let NdotL = max(dot(N, L), 0.0);
        total_radiance += (kD * albedo / 3.14159265359 + specular) * radiance * NdotL;
    }

    // Light 2: Point Light (Ceiling Light)
    {
        let light_pos = vec3<f32>(0.0, 1.95, 0.0);
        let L = normalize(light_pos - world_pos);
        let H = normalize(V + L);

        let distance = length(light_pos - world_pos);
        let attenuation = 1.0 / (1.0 + 0.22 * distance + 0.20 * distance * distance);
        let radiance = vec3<f32>(12.0, 12.0, 12.0) * attenuation; // Ceiling light emissive is around 12.0

        // Cook-Torrance BRDF
        let NDF = distribution_ggx(N, H, roughness);
        let G = geometry_smith(N, V, L, roughness);
        let F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        let numerator = NDF * G * F;
        let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        let specular = numerator / denominator;

        let kS = F;
        var kD = vec3<f32>(1.0) - kS;
        kD *= 1.0 - metallic;

        let NdotL = max(dot(N, L), 0.0);
        total_radiance += (kD * albedo / 3.14159265359 + specular) * radiance * NdotL;
    }

    // 5. Add Emissive and Ambient term
    let ambient = vec3<f32>(0.03) * albedo; // Simple constant ambient term for direct-only pass
    var color = total_radiance + emissive + ambient;

    // 6. ACES Tonemapping
    color = aces_approx(color);

    return vec4<f32>(color, 1.0);
}
