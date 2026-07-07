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

@group(0) @binding(3) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(4) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var<storage, read> tri_materials: array<u32>;
@group(0) @binding(7) var<storage, read> materials: array<Material>;

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

// PCG Random Number Generator
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn init_rng(pixel: vec2<i32>, frame_index: u32) -> u32 {
    let p_index = u32(pixel.x) + u32(pixel.y) * 4096u;
    return pcg_hash(p_index + frame_index * 1234567u);
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}

fn rand_unit_vector(seed: ptr<function, u32>) -> vec3<f32> {
    let z = rand_f32(seed) * 2.0 - 1.0;
    let a = rand_f32(seed) * 6.283185307;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

fn rand_cosine_hemisphere(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let u1 = rand_f32(seed);
    let u2 = rand_f32(seed);
    let r = sqrt(u1);
    let theta = 6.283185307 * u2;
    
    let local_dir = vec3<f32>(r * cos(theta), r * sin(theta), sqrt(max(0.0, 1.0 - u1)));
    
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    
    return tangent * local_dir.x + bitangent * local_dir.y + n * local_dir.z;
}

// Ray-AABB intersection (Slab method) returning hit distance
fn intersect_aabb_dist(ro: vec3<f32>, inv_d: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>, max_t: f32, t_out: ptr<function, f32>) -> bool {
    let t0 = (box_min - ro) * inv_d;
    let t1 = (box_max - ro) * inv_d;
    
    let tmin_v = min(t0, t1);
    let tmax_v = max(t0, t1);
    
    let tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    let tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    
    if (tmax >= max(0.0, tmin) && tmin < max_t) {
        *t_out = tmin;
        return true;
    }
    return false;
}

struct TriIntersection {
    hit: bool,
    t: f32,
    u: f32,
    v: f32,
}

// Ray-Triangle intersection (Möller-Trumbore)
fn intersect_triangle(ro: vec3<f32>, rd: vec3<f32>, tri_idx: u32) -> TriIntersection {
    var result: TriIntersection;
    result.hit = false;
 
    let idx0 = indices[tri_idx * 3u + 0u];
    let idx1 = indices[tri_idx * 3u + 1u];
    let idx2 = indices[tri_idx * 3u + 2u];

    let v0 = vertices[idx0].position.xyz;
    let v1 = vertices[idx1].position.xyz;
    let v2 = vertices[idx2].position.xyz;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(rd, edge2);
    let a = dot(edge1, h);

    if (a > -0.0000001 && a < 0.0000001) {
        return result;
    }

    let f = 1.0 / a;
    let s = ro - v0;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(rd, q);

    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(edge2, q);
    if (t > 0.0001) {
        result.hit = true;
        result.t = t;
        result.u = u;
        result.v = v;
    }

    return result;
}

struct HitRecord {
    hit: bool,
    t: f32,
    tri_idx: i32,
    u: f32,
    v: f32,
    instance_idx: i32,
}

// Traverse a specific BLAS instance
fn intersect_blas(ro: vec3<f32>, rd: vec3<f32>, inv_d: vec3<f32>, bvh_offset: u32, tri_offset: u32, is_primary: bool, max_t: f32, instance_idx: i32) -> HitRecord {
    var rec: HitRecord;
    rec.hit = false;
    rec.t = max_t;
    rec.tri_idx = -1;
    rec.instance_idx = instance_idx;

    var stack: array<u32, 24>;
    var stack_ptr = 0u;

    // Test root node of this BLAS
    let root = bvh_nodes[bvh_offset];
    let root_min = vec3<f32>(root.min_x, root.min_y, root.min_z);
    let root_max = vec3<f32>(root.max_x, root.max_y, root.max_z);
    var t_root = 0.0;
    if (!intersect_aabb_dist(ro, inv_d, root_min, root_max, rec.t, &t_root)) {
        return rec;
    }

    stack[stack_ptr] = 0u;
    stack_ptr = stack_ptr + 1u;

    while (stack_ptr > 0u) {
        stack_ptr = stack_ptr - 1u;
        let local_node_idx = stack[stack_ptr];
        let node = bvh_nodes[bvh_offset + local_node_idx];

        let tri_start = node.left_child;
        let tri_count = node.tri_count;

        if (tri_count > 0u) {
            // Leaf Node: test triangles
            for (var i = 0u; i < tri_count; i = i + 1u) {
                let global_tri_idx = tri_offset + tri_start + i;
                let tri_inter = intersect_triangle(ro, rd, global_tri_idx);
                if (tri_inter.hit && tri_inter.t < rec.t && tri_inter.t > 0.0002) {
                    let mat_idx = tri_materials[global_tri_idx];
                    if (is_primary && mat_idx == 7u) {
                        // Ignore front wall for primary camera rays so we can see inside
                        continue;
                    }
                    rec.hit = true;
                    rec.t = tri_inter.t;
                    rec.tri_idx = i32(global_tri_idx);
                    rec.u = tri_inter.u;
                    rec.v = tri_inter.v;
                }
            }
        } else {
            // Inner Node: Test left and right children (local offsets)
            let left_child = tri_start;
            let right_child = left_child + 1u;
            
            let left_node = bvh_nodes[bvh_offset + left_child];
            let right_node = bvh_nodes[bvh_offset + right_child];
            
            let left_min = vec3<f32>(left_node.min_x, left_node.min_y, left_node.min_z);
            let left_max = vec3<f32>(left_node.max_x, left_node.max_y, left_node.max_z);
            let right_min = vec3<f32>(right_node.min_x, right_node.min_y, right_node.min_z);
            let right_max = vec3<f32>(right_node.max_x, right_node.max_y, right_node.max_z);
            
            var t_left = 1e9;
            var t_right = 1e9;
            let hit_left = intersect_aabb_dist(ro, inv_d, left_min, left_max, rec.t, &t_left);
            let hit_right = intersect_aabb_dist(ro, inv_d, right_min, right_max, rec.t, &t_right);
            
            if (hit_left && hit_right) {
                if (t_left < t_right) {
                    if (stack_ptr < 22u) {
                        stack[stack_ptr] = right_child;
                        stack_ptr = stack_ptr + 1u;
                        stack[stack_ptr] = left_child;
                        stack_ptr = stack_ptr + 1u;
                    }
                } else {
                    if (stack_ptr < 22u) {
                        stack[stack_ptr] = left_child;
                        stack_ptr = stack_ptr + 1u;
                        stack[stack_ptr] = right_child;
                        stack_ptr = stack_ptr + 1u;
                    }
                }
            } else if (hit_left) {
                if (stack_ptr < 23u) {
                    stack[stack_ptr] = left_child;
                    stack_ptr = stack_ptr + 1u;
                }
            } else if (hit_right) {
                if (stack_ptr < 23u) {
                    stack[stack_ptr] = right_child;
                    stack_ptr = stack_ptr + 1u;
                }
            }
        }
    }

    return rec;
}

// Two-Level Acceleration structure ray-scene intersection
fn intersect_scene(ro: vec3<f32>, rd: vec3<f32>, is_primary: bool) -> HitRecord {
    var closest_rec: HitRecord;
    closest_rec.hit = false;
    closest_rec.t = 1e9;
    closest_rec.tri_idx = -1;
    closest_rec.instance_idx = -1;

    let inv_d = 1.0 / (rd + vec3<f32>(
        select(-1e-9, 1e-9, rd.x >= 0.0),
        select(-1e-9, 1e-9, rd.y >= 0.0),
        select(-1e-9, 1e-9, rd.z >= 0.0)
    ));

    // Instance 0: Cornell Box (Static, identity transform)
    {
        let bvh_off = uniforms.bvh_offsets[0];
        let tri_off = uniforms.tri_offsets[0];
        let rec = intersect_blas(ro, rd, inv_d, bvh_off, tri_off, is_primary, closest_rec.t, 0);
        if (rec.hit && rec.t < closest_rec.t) {
            closest_rec = rec;
        }
    }

    // Instance 1: Mechanical Part / Torus (Rotation around Y-axis by theta)
    {
        let theta = uniforms.time * 1.2;
        let cos_t = cos(-theta);
        let sin_t = sin(-theta);
        
        let center_x = 0.0;
        let center_z = 0.0;
        
        var ro_local = ro;
        ro_local.x = (ro.x - center_x) * cos_t + (ro.z - center_z) * sin_t + center_x;
        ro_local.z = -(ro.x - center_x) * sin_t + (ro.z - center_z) * cos_t + center_z;
        
        var rd_local = rd;
        rd_local.x = rd.x * cos_t + rd.z * sin_t;
        rd_local.z = -rd.x * sin_t + rd.z * cos_t;
        
        let bvh_off = uniforms.bvh_offsets[1];
        let tri_off = uniforms.tri_offsets[1];
        let inv_d_local = 1.0 / (rd_local + vec3<f32>(
            select(-1e-9, 1e-9, rd_local.x >= 0.0),
            select(-1e-9, 1e-9, rd_local.y >= 0.0),
            select(-1e-9, 1e-9, rd_local.z >= 0.0)
        ));
        
        let rec = intersect_blas(ro_local, rd_local, inv_d_local, bvh_off, tri_off, is_primary, closest_rec.t, 1);
        if (rec.hit && rec.t < closest_rec.t) {
            closest_rec = rec;
        }
    }

    // Instance 2: Gold Sphere (Bounces up and down)
    {
        let gold_dy = sin(uniforms.time * 2.5) * 0.4;
        let ro_local = ro - vec3<f32>(0.0, gold_dy, 0.0);
        
        let bvh_off = uniforms.bvh_offsets[2];
        let tri_off = uniforms.tri_offsets[2];
        
        let rec = intersect_blas(ro_local, rd, inv_d, bvh_off, tri_off, is_primary, closest_rec.t, 2);
        if (rec.hit && rec.t < closest_rec.t) {
            closest_rec = rec;
        }
    }

    // Instance 3: Glass Sphere (Bounces up and down out of phase)
    {
        let glass_dy = sin(uniforms.time * 2.5 + 3.14159265) * 0.4;
        let ro_local = ro - vec3<f32>(0.0, glass_dy, 0.0);
        
        let bvh_off = uniforms.bvh_offsets[3];
        let tri_off = uniforms.tri_offsets[3];
        
        let rec = intersect_blas(ro_local, rd, inv_d, bvh_off, tri_off, is_primary, closest_rec.t, 3);
        if (rec.hit && rec.t < closest_rec.t) {
            closest_rec = rec;
        }
    }

    return closest_rec;
}

// Path Tracer ray trace pass
fn trace_path(origin: vec3<f32>, direction: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    var ro = origin;
    var rd = direction;

    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);

    for (var depth = 0u; depth < uniforms.max_depth; depth = depth + 1u) {
        let is_primary = (depth == 0u);
        let rec = intersect_scene(ro, rd, is_primary);
        if (!rec.hit) {
            // Hit environment (gradient sky light)
            let sky_grad = 0.5 * (rd.y + 1.0);
            let sky_color = mix(vec3<f32>(0.02, 0.015, 0.05), vec3<f32>(0.1, 0.15, 0.25), sky_grad) * uniforms.env_light_intensity;
            radiance += throughput * sky_color;
            break;
        }

        let mat_idx = tri_materials[rec.tri_idx];
        let mat = materials[mat_idx];

        // Emissive light contribution (Clamped to prevent bright white noise fireflies)
        radiance += clamp(throughput * mat.emissive.rgb, vec3<f32>(0.0), vec3<f32>(20.0));

        let p = ro + rd * rec.t;

        // Smooth normal interpolation
        let idx0 = indices[u32(rec.tri_idx) * 3u + 0u];
        let idx1 = indices[u32(rec.tri_idx) * 3u + 1u];
        let idx2 = indices[u32(rec.tri_idx) * 3u + 2u];
        let n0 = vertices[idx0].normal.xyz;
        let n1 = vertices[idx1].normal.xyz;
        let n2 = vertices[idx2].normal.xyz;
        var n = normalize((1.0 - rec.u - rec.v) * n0 + rec.u * n1 + rec.v * n2);

        // Apply rotation transform to the normal if it belongs to the rotating mechanical part
        if (rec.instance_idx == 1) {
            let theta = uniforms.time * 1.2;
            let cos_t = cos(theta);
            let sin_t = sin(theta);
            let nx = n.x;
            let nz = n.z;
            n.x = nx * cos_t + nz * sin_t;
            n.z = -nx * sin_t + nz * cos_t;
        }

        // Flip normal if ray hits back-face (needed for glass refraction)
        let into = dot(n, rd) < 0.0;
        let outward_normal = select(-n, n, into);

        let roughness = mat.properties.x;
        let ior = mat.properties.y;
        let transmission = mat.properties.z;
        let metallic = mat.base_color.w;
        let albedo = mat.base_color.rgb;

        if (transmission > 0.0) {
            // GLASS REFRACTION BSDF
            let nc = 1.0;
            let nt = ior;
            let n_ratio = select(nt / nc, nc / nt, into);
            let ddn = dot(rd, outward_normal);
            let cos2t = 1.0 - n_ratio * n_ratio * (1.0 - ddn * ddn);

            if (cos2t < 0.0) {
                // Total Internal Reflection (TIR)
                rd = reflect(rd, n);
                ro = p + outward_normal * 0.001;
                throughput *= albedo;
            } else {
                // Snell Refraction
                let tdir = normalize(rd * n_ratio - outward_normal * (ddn * n_ratio + sqrt(cos2t)));
                
                // Fresnel term Schlick approximation
                let a = nt - nc;
                let b = nt + nc;
                let r0 = (a * a) / (b * b);
                let c = 1.0 - select(-ddn, dot(tdir, n), into);
                let fresnel = r0 + (1.0 - r0) * pow(c, 5.0);

                if (rand_f32(seed) < fresnel) {
                    // Fresnel reflection
                    rd = reflect(rd, n);
                    ro = p + outward_normal * 0.001;
                    throughput *= albedo;
                } else {
                    // Refraction transmission
                    rd = tdir;
                    ro = p - outward_normal * 0.001;
                    throughput *= albedo;
                }
            }
        } else {
            // OPAQUE SURFACE (METAL / DIELECTRIC)
            let spec_prob = mix(0.04, 1.0, metallic);
            
            if (rand_f32(seed) < spec_prob) {
                // Specular Lobe: Rough mirror approximation
                let H = normalize(outward_normal + roughness * rand_unit_vector(seed));
                rd = reflect(rd, H);
                ro = p + outward_normal * 0.001;

                let f0 = mix(vec3<f32>(0.04), albedo, metallic);
                let c = 1.0 - max(0.0, dot(rd, outward_normal));
                let F = f0 + (vec3<f32>(1.0) - f0) * pow(c, 5.0);
                
                throughput *= F / spec_prob;
            } else {
                // Diffuse Lobe: Cosine-weighted Lambertian scattering
                rd = rand_cosine_hemisphere(outward_normal, seed);
                ro = p + outward_normal * 0.001;
                
                throughput *= albedo / (1.0 - spec_prob);
            }
        }

        // Russian Roulette termination
        if (depth > 2u) {
            let p_rr = max(throughput.x, max(throughput.y, throughput.z));
            if (rand_f32(seed) > p_rr) {
                break;
            }
            throughput *= 1.0 / p_rr;
        }
    }

    return radiance;
}

@fragment
fn fs_path_trace(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    
    // Hardware-defined integer pixel coordinates
    let pixel_coords = vec2<i32>(in.position.xy);

    var prev_color = vec3<f32>(0.0);
    var has_history = false;

    var seed = init_rng(pixel_coords, uniforms.frame_index);

    // Compute camera position and ray direction
    var cam_pos = vec3<f32>(0.0, 0.0, -uniforms.camera_zoom);
    cam_pos = rotate_x(rotate_y(cam_pos, uniforms.camera_rot.x), uniforms.camera_rot.y);

    var cam_right = vec3<f32>(1.0, 0.0, 0.0);
    var cam_up = vec3<f32>(0.0, 1.0, 0.0);
    cam_right = rotate_x(rotate_y(cam_right, uniforms.camera_rot.x), uniforms.camera_rot.y);
    cam_up = rotate_x(rotate_y(cam_up, uniforms.camera_rot.x), uniforms.camera_rot.y);

    // Primary sample trace
    var color_sum = vec3<f32>(0.0);
    let s_count = uniforms.samples_per_frame;

    for (var s = 0u; s < s_count; s = s + 1u) {
        let jitter = vec2<f32>(rand_f32(&seed) - 0.5, rand_f32(&seed) - 0.5) / uniforms.resolution;
        let uv = in.uv * 2.0 - vec2<f32>(1.0);
        let uv_jittered = uv + jitter;

        var rd = normalize(vec3<f32>(uv_jittered.x * aspect, uv_jittered.y, 1.5));
        rd = rotate_x(rotate_y(rd, uniforms.camera_rot.x), uniforms.camera_rot.y);

        let focal_point = cam_pos + rd * uniforms.focal_distance;
        
        let r_lens = uniforms.aperture * sqrt(rand_f32(&seed));
        let theta_lens = rand_f32(&seed) * 6.283185307;
        let lens_offset = cam_right * (r_lens * cos(theta_lens)) + cam_up * (r_lens * sin(theta_lens));
        
        let ro_dof = cam_pos + lens_offset;
        let rd_dof = normalize(focal_point - ro_dof);

        color_sum += trace_path(ro_dof, rd_dof, &seed);
    }

    let current_avg = color_sum / f32(s_count);
    var final_color = current_avg;

    if (uniforms.accum_frame > 0u) {
        // Static Progressive Accumulation
        prev_color = textureLoad(prev_texture, pixel_coords, 0).rgb;
        let weight = 1.0 / f32(uniforms.accum_frame + 1u);
        final_color = mix(prev_color, current_avg, weight);
    } else {
        // Temporal Reprojection (Denoises in real time during animation)
        let uv = in.uv * 2.0 - vec2<f32>(1.0);
        var rd = normalize(vec3<f32>(uv.x * aspect, uv.y, 1.5));
        rd = rotate_x(rotate_y(rd, uniforms.camera_rot.x), uniforms.camera_rot.y);

        let rec = intersect_scene(cam_pos, rd, true);
        var uv_prev = vec2<f32>(-1.0);
        
        var prev_cam_pos = vec3<f32>(0.0, 0.0, -uniforms.prev_camera_zoom);
        prev_cam_pos = rotate_x(rotate_y(prev_cam_pos, uniforms.prev_camera_rot.x), uniforms.prev_camera_rot.y);

        if (rec.hit) {
            var P = cam_pos + rd * rec.t;
            
            // Apply motion vectors / dynamic reprojection for moving objects
            if (rec.instance_idx == 1) {
                // Torus / Mechanical Part
                let center_x = 0.0;
                let center_z = 0.0;
                
                // 1. Current local position (un-rotate by current time)
                let theta_curr = uniforms.time * 1.2;
                let cos_curr = cos(-theta_curr);
                let sin_curr = sin(-theta_curr);
                
                var P_local = P;
                P_local.x = (P.x - center_x) * cos_curr + (P.z - center_z) * sin_curr + center_x;
                P_local.z = -(P.x - center_x) * sin_curr + (P.z - center_z) * cos_curr + center_z;
                
                // 2. Previous world position (rotate by previous time: time - dt)
                let theta_prev = (uniforms.time - uniforms.dt) * 1.2;
                let cos_prev = cos(theta_prev);
                let sin_prev = sin(theta_prev);
                
                P.x = (P_local.x - center_x) * cos_prev - (P_local.z - center_z) * sin_prev + center_x;
                P.z = (P_local.x - center_x) * sin_prev + (P_local.z - center_z) * cos_prev + center_z;
            } else if (rec.instance_idx == 2) {
                // Gold Sphere
                let gold_dy_curr = sin(uniforms.time * 2.5) * 0.4;
                let gold_dy_prev = sin((uniforms.time - uniforms.dt) * 2.5) * 0.4;
                P.y = P.y - gold_dy_curr + gold_dy_prev;
            } else if (rec.instance_idx == 3) {
                // Glass Sphere
                let glass_dy_curr = sin(uniforms.time * 2.5 + 3.14159265) * 0.4;
                let glass_dy_prev = sin((uniforms.time - uniforms.dt) * 2.5 + 3.14159265) * 0.4;
                P.y = P.y - glass_dy_curr + glass_dy_prev;
            }

            let v_cam = rotate_y(rotate_x(P - prev_cam_pos, -uniforms.prev_camera_rot.y), -uniforms.prev_camera_rot.x);
            if (v_cam.z > 0.001) {
                let x_screen = (v_cam.x / v_cam.z) * 1.5 / aspect;
                let y_screen = (v_cam.y / v_cam.z) * 1.5;
                uv_prev = vec2<f32>(x_screen, y_screen) * 0.5 + vec2<f32>(0.5);
            }
        } else {
            // Reproject sky ray direction
            let rd_cam = rotate_y(rotate_x(rd, -uniforms.prev_camera_rot.y), -uniforms.prev_camera_rot.x);
            if (rd_cam.z > 0.001) {
                let x_screen = (rd_cam.x / rd_cam.z) * 1.5 / aspect;
                let y_screen = (rd_cam.y / rd_cam.z) * 1.5;
                uv_prev = vec2<f32>(x_screen, y_screen) * 0.5 + vec2<f32>(0.5);
            }
        }

        if (uv_prev.x >= 0.0 && uv_prev.x <= 1.0 && uv_prev.y >= 0.0 && uv_prev.y <= 1.0) {
            let prev_pixel = vec2<i32>(i32(uv_prev.x * uniforms.resolution.x), i32((1.0 - uv_prev.y) * uniforms.resolution.y));
            let prev_val = textureLoad(prev_texture, prev_pixel, 0).rgb;
            final_color = mix(prev_val, current_avg, 0.15); // EMA blend: 15% new frame, 85% history
        } else {
            final_color = current_avg;
        }
    }

    return vec4<f32>(final_color, 1.0);
}
