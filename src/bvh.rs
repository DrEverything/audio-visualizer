use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 4], // x, y, z, 1.0
    pub normal: [f32; 4],   // x, y, z, 0.0
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub left_child: u32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
    pub tri_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],  // r, g, b, metallic
    pub properties: [f32; 4],  // roughness, ior, transmission, padding
    pub emissive: [f32; 4],    // r, g, b, padding
}

pub struct BvhBuildTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub centroid: [f32; 3],
    pub tri_idx: u32,
}

// Midpoint split BVH builder
pub fn build_bvh(
    triangles: &mut [BvhBuildTriangle],
    tri_start: usize,
    tri_count: usize,
    node_idx: usize,
    nodes: &mut Vec<GpuBvhNode>,
) {
    // Compute bounding box of triangles
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for tri in &triangles[tri_start..tri_start + tri_count] {
        for v in &[tri.v0, tri.v1, tri.v2] {
            min[0] = min[0].min(v[0]);
            min[1] = min[1].min(v[1]);
            min[2] = min[2].min(v[2]);
            max[0] = max[0].max(v[0]);
            max[1] = max[1].max(v[1]);
            max[2] = max[2].max(v[2]);
        }
    }

    // Leaf node condition: <= 4 triangles
    if tri_count <= 4 {
        nodes[node_idx] = GpuBvhNode {
            min_x: min[0],
            min_y: min[1],
            min_z: min[2],
            left_child: tri_start as u32,
            max_x: max[0],
            max_y: max[1],
            max_z: max[2],
            tri_count: tri_count as u32,
        };
        return;
    }

    // Find the longest axis of the centroid bounding box
    let mut c_min = [f32::INFINITY; 3];
    let mut c_max = [f32::NEG_INFINITY; 3];
    for tri in &triangles[tri_start..tri_start + tri_count] {
        c_min[0] = c_min[0].min(tri.centroid[0]);
        c_min[1] = c_min[1].min(tri.centroid[1]);
        c_min[2] = c_min[2].min(tri.centroid[2]);
        c_max[0] = c_max[0].max(tri.centroid[0]);
        c_max[1] = c_max[1].max(tri.centroid[1]);
        c_max[2] = c_max[2].max(tri.centroid[2]);
    }

    let extents = [
        c_max[0] - c_min[0],
        c_max[1] - c_min[1],
        c_max[2] - c_min[2],
    ];
    let mut axis = 0;
    if extents[1] > extents[0] {
        axis = 1;
    }
    if extents[2] > extents[axis] {
        axis = 2;
    }

    // Split in the middle of the centroid extents
    let mid_val = 0.5 * (c_min[axis] + c_max[axis]);

    // Partition triangles based on centroid along the chosen axis
    let mut i = tri_start;
    let mut j = tri_start + tri_count - 1;
    while i <= j {
        if triangles[i].centroid[axis] < mid_val {
            i += 1;
        } else {
            triangles.swap(i, j);
            if j == 0 {
                break;
            }
            j -= 1;
        }
    }

    let mut left_count = i - tri_start;
    if left_count == 0 || left_count == tri_count {
        // Fallback: split in half if partition failed to separate
        left_count = tri_count / 2;
    }

    // Allocate children contiguously in the vector
    let left_child_idx = nodes.len();
    nodes.push(GpuBvhNode { min_x: 0.0, min_y: 0.0, min_z: 0.0, left_child: 0, max_x: 0.0, max_y: 0.0, max_z: 0.0, tri_count: 0 });
    nodes.push(GpuBvhNode { min_x: 0.0, min_y: 0.0, min_z: 0.0, left_child: 0, max_x: 0.0, max_y: 0.0, max_z: 0.0, tri_count: 0 });

    // Pack left_child index and tri_count = 0 (marking inner node)
    nodes[node_idx] = GpuBvhNode {
        min_x: min[0],
        min_y: min[1],
        min_z: min[2],
        left_child: left_child_idx as u32,
        max_x: max[0],
        max_y: max[1],
        max_z: max[2],
        tri_count: 0,
    };

    // Recursively build children in their allocated contiguous slots
    build_bvh(triangles, tri_start, left_count, left_child_idx, nodes);
    build_bvh(triangles, tri_start + left_count, tri_count - left_count, left_child_idx + 1, nodes);
}

// Procedural Cornell Box Generator
pub fn generate_cornell_box(
    vertices: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    tri_materials: &mut Vec<u32>,
) {
    // We define walls as quads (2 triangles each)
    // Cornell Box size: x in [-2, 2], y in [-2, 2], z in [-2, 2]
    
    // Helper to add a quad
    let mut add_quad = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], n: [f32; 3], mat_id: u32| {
        let base_idx = vertices.len() as u32;
        vertices.push(GpuVertex { position: [p0[0], p0[1], p0[2], 1.0], normal: [n[0], n[1], n[2], 0.0] });
        vertices.push(GpuVertex { position: [p1[0], p1[1], p1[2], 1.0], normal: [n[0], n[1], n[2], 0.0] });
        vertices.push(GpuVertex { position: [p2[0], p2[1], p2[2], 1.0], normal: [n[0], n[1], n[2], 0.0] });
        vertices.push(GpuVertex { position: [p3[0], p3[1], p3[2], 1.0], normal: [n[0], n[1], n[2], 0.0] });

        indices.push(base_idx);
        indices.push(base_idx + 1);
        indices.push(base_idx + 2);
        tri_materials.push(mat_id);

        indices.push(base_idx);
        indices.push(base_idx + 2);
        indices.push(base_idx + 3);
        tri_materials.push(mat_id);
    };

    // Left wall (Red) - x = -2
    add_quad(
        [-2.0, -2.0, -2.0],
        [-2.0, -2.0,  2.0],
        [-2.0,  2.0,  2.0],
        [-2.0,  2.0, -2.0],
        [1.0, 0.0, 0.0],
        0, // red material
    );

    // Right wall (Green) - x = 2
    add_quad(
        [ 2.0, -2.0,  2.0],
        [ 2.0, -2.0, -2.0],
        [ 2.0,  2.0, -2.0],
        [ 2.0,  2.0,  2.0],
        [-1.0, 0.0, 0.0],
        1, // green material
    );

    // Floor (White) - y = -2
    add_quad(
        [-2.0, -2.0,  2.0],
        [-2.0, -2.0, -2.0],
        [ 2.0, -2.0, -2.0],
        [ 2.0, -2.0,  2.0],
        [0.0, 1.0, 0.0],
        2, // white material
    );

    // Ceiling (White) - y = 2
    add_quad(
        [-2.0,  2.0, -2.0],
        [-2.0,  2.0,  2.0],
        [ 2.0,  2.0,  2.0],
        [ 2.0,  2.0, -2.0],
        [0.0, -1.0, 0.0],
        2, // white material
    );

    // Back wall (White) - z = 2
    add_quad(
        [ 2.0, -2.0,  2.0],
        [-2.0, -2.0,  2.0],
        [-2.0,  2.0,  2.0],
        [ 2.0,  2.0,  2.0],
        [0.0, 0.0, -1.0],
        2, // white material
    );

    // Front wall (White, unseen by camera primary rays) - z = -2
    add_quad(
        [-2.0, -2.0, -2.0],
        [ 2.0, -2.0, -2.0],
        [ 2.0,  2.0, -2.0],
        [-2.0,  2.0, -2.0],
        [0.0, 0.0, 1.0],
        7, // front wall material
    );

    // Ceiling light (White Emissive) - y = 1.98 (Larger light for faster path trace convergence)
    add_quad(
        [-0.9, 1.98, -0.9],
        [-0.9, 1.98,  0.9],
        [ 0.9, 1.98,  0.9],
        [ 0.9, 1.98, -0.9],
        [0.0, -1.0, 0.0],
        3, // ceiling light material
    );
}

// Procedural UV Sphere Generator
pub fn generate_sphere(
    center: [f32; 3],
    radius: f32,
    material_id: u32,
    stacks: usize,
    slices: usize,
    vertices: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    tri_materials: &mut Vec<u32>,
) {
    let base_idx = vertices.len() as u32;

    for i in 0..=stacks {
        let theta = (i as f32) * std::f32::consts::PI / (stacks as f32);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for j in 0..=slices {
            let phi = (j as f32) * 2.0 * std::f32::consts::PI / (slices as f32);
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let nx = sin_theta * cos_phi;
            let ny = cos_theta;
            let nz = sin_theta * sin_phi;

            let px = center[0] + radius * nx;
            let py = center[1] + radius * ny;
            let pz = center[2] + radius * nz;

            vertices.push(GpuVertex {
                position: [px, py, pz, 1.0],
                normal: [nx, ny, nz, 0.0],
            });
        }
    }

    for i in 0..stacks {
        for j in 0..slices {
            let row1 = (i * (slices + 1)) as u32;
            let row2 = ((i + 1) * (slices + 1)) as u32;

            let v0 = base_idx + row1 + j as u32;
            let v1 = base_idx + row1 + (j + 1) as u32;
            let v2 = base_idx + row2 + j as u32;
            let v3 = base_idx + row2 + (j + 1) as u32;

            // Triangle 1
            indices.push(v0);
            indices.push(v1);
            indices.push(v3);
            tri_materials.push(material_id);

            // Triangle 2
            indices.push(v0);
            indices.push(v3);
            indices.push(v2);
            tri_materials.push(material_id);
        }
    }
}

// glTF Mesh Loader
pub fn load_glb(
    path: &Path,
    material_id: u32,
    vertices: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    tri_materials: &mut Vec<u32>,
) -> Result<(), String> {
    let (gltf, buffers, _) = gltf::import(path).map_err(|e| e.to_string())?;
    let base_vertex_offset = vertices.len() as u32;

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            
            let mut prim_positions = Vec::new();
            if let Some(pos_iter) = reader.read_positions() {
                for pos in pos_iter {
                    prim_positions.push(pos);
                }
            } else {
                continue;
            }

            let mut prim_normals = Vec::new();
            if let Some(norm_iter) = reader.read_normals() {
                for norm in norm_iter {
                    prim_normals.push(norm);
                }
            } else {
                for _ in 0..prim_positions.len() {
                    prim_normals.push([0.0, 1.0, 0.0]);
                }
            }

            let prim_vertex_offset = vertices.len() as u32;
            for i in 0..prim_positions.len() {
                vertices.push(GpuVertex {
                    position: [prim_positions[i][0], prim_positions[i][1], prim_positions[i][2], 1.0],
                    normal: [prim_normals[i][0], prim_normals[i][1], prim_normals[i][2], 0.0],
                });
            }

            let mut prim_indices = Vec::new();
            if let Some(idx_iter) = reader.read_indices() {
                for idx in idx_iter.into_u32() {
                    prim_indices.push(idx + prim_vertex_offset);
                }
            } else {
                for i in 0..prim_positions.len() as u32 {
                    prim_indices.push(i + prim_vertex_offset);
                }
            }

            let tri_count = prim_indices.len() / 3;
            for t in 0..tri_count {
                indices.push(prim_indices[t * 3 + 0]);
                indices.push(prim_indices[t * 3 + 1]);
                indices.push(prim_indices[t * 3 + 2]);
                tri_materials.push(material_id);
            }
        }
    }

    // Rotate 90 degrees around Y axis to align the long axis sideways (along X) instead of depth-wise (along Z)
    let start_idx = base_vertex_offset as usize;
    if vertices.len() > start_idx {
        for v in &mut vertices[start_idx..] {
            let px = v.position[0];
            let pz = v.position[2];
            v.position[0] = pz;
            v.position[2] = -px;

            let nx = v.normal[0];
            let nz = v.normal[2];
            v.normal[0] = nz;
            v.normal[2] = -nx;
        }
    }

    // Auto-scale and center the loaded mesh to fit nicely inside the Cornell Box
    let mut bbox_min = [f32::INFINITY; 3];
    let mut bbox_max = [f32::NEG_INFINITY; 3];
    
    if vertices.len() > start_idx {
        for v in &vertices[start_idx..] {
            bbox_min[0] = bbox_min[0].min(v.position[0]);
            bbox_min[1] = bbox_min[1].min(v.position[1]);
            bbox_min[2] = bbox_min[2].min(v.position[2]);
            bbox_max[0] = bbox_max[0].max(v.position[0]);
            bbox_max[1] = bbox_max[1].max(v.position[1]);
            bbox_max[2] = bbox_max[2].max(v.position[2]);
        }
        
        let size = [
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2],
        ];
        let max_dim = size[0].max(size[1]).max(size[2]);
        
        // Scale to a max dimension of 1.0 unit (fits perfectly between spheres without overlap)
        let scale = if max_dim > 0.0 { 1.0 / max_dim } else { 1.0 };
        
        // Center horizontally (X, Z) and sit bottom perfectly on floor (y = -2.0)
        let offset_x = -0.5 * (bbox_min[0] + bbox_max[0]) * scale;
        let offset_y = -2.0 - bbox_min[1] * scale;
        let offset_z = -0.5 * (bbox_min[2] + bbox_max[2]) * scale;
        
        for v in &mut vertices[start_idx..] {
            v.position[0] = v.position[0] * scale + offset_x;
            v.position[1] = v.position[1] * scale + offset_y;
            v.position[2] = v.position[2] * scale + offset_z;
        }

        log::info!(
            "Loaded glTF model from {:?}, vertices: {}, original bounds: {:?} to {:?}, size: {:?}, scale: {}, offsets: [{}, {}, {}]",
            path, vertices.len() - start_idx, bbox_min, bbox_max, size, scale, offset_x, offset_y, offset_z
        );
    }

    Ok(())
}

// Master Scene Builder
pub fn build_scene() -> (
    Vec<GpuVertex>,
    Vec<u32>,
    Vec<u32>,
    Vec<GpuBvhNode>,
    [u32; 4], // bvh_offsets
    [u32; 4], // tri_offsets
) {
    let mut global_vertices = Vec::new();
    let mut global_indices = Vec::new();
    let mut global_tri_materials = Vec::new();
    let mut global_bvh_nodes = Vec::new();

    let mut bvh_offsets = [0u32; 4];
    let mut tri_offsets = [0u32; 4];

    // Helper to add a BLAS
    let mut add_blas = |idx: usize, vertices: Vec<GpuVertex>, mut indices: Vec<u32>, tri_materials: Vec<u32>, bvh_nodes: Vec<GpuBvhNode>| {
        bvh_offsets[idx] = global_bvh_nodes.len() as u32;
        tri_offsets[idx] = (global_indices.len() / 3) as u32;

        let vertex_offset = global_vertices.len() as u32;
        for i in &mut indices {
            *i += vertex_offset;
        }

        global_vertices.extend(vertices);
        global_indices.extend(indices);
        global_tri_materials.extend(tri_materials);
        global_bvh_nodes.extend(bvh_nodes);
    };

    // BLAS 0: Cornell Box
    let (v_cb, i_cb, m_cb, n_cb) = build_blas_generic(|v, i, m| {
        generate_cornell_box(v, i, m);
    });
    add_blas(0, v_cb, i_cb, m_cb, n_cb);

    // BLAS 1: Mechanical Part / Torus
    let (v_mp, i_mp, m_mp, n_mp) = build_blas_generic(|v, i, m| {
        let glb_path = Path::new("occt-model-generation/mechanical_part.glb");
        let load_res = if glb_path.exists() {
            load_glb(glb_path, 4, v, i, m)
        } else {
            Err("GLB file not found".to_string())
        };

        if let Err(e) = load_res {
            log::warn!("Could not load glb model ({}). Generating procedural fallback shape.", e);
            
            // Fallback procedural shape: a nice big torus in the center
            let base_idx = v.len() as u32;
            let rings = 40;
            let sides = 20;
            let r_major = 0.8f32;
            let r_minor = 0.28f32;
            for r in 0..rings {
                let theta = (r as f32) * 2.0 * std::f32::consts::PI / (rings as f32);
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                for s in 0..sides {
                    let phi = (s as f32) * 2.0 * std::f32::consts::PI / (sides as f32);
                    let cos_phi = phi.cos();
                    let sin_phi = phi.sin();

                    let nx = cos_theta * cos_phi;
                    let ny = sin_phi;
                    let nz = sin_theta * cos_phi;

                    let px = (r_major + r_minor * cos_phi) * cos_theta;
                    let py = -1.2 + r_minor * sin_phi;
                    let pz = (r_major + r_minor * cos_phi) * sin_theta;

                    v.push(GpuVertex {
                        position: [px, py, pz, 1.0],
                        normal: [nx, ny, nz, 0.0],
                    });
                }
            }
            for r in 0..rings {
                for s in 0..sides {
                    let next_r = (r + 1) % rings;
                    let next_s = (s + 1) % sides;

                    let v0 = base_idx + (r * sides + s) as u32;
                    let v1 = base_idx + (r * sides + next_s) as u32;
                    let v2 = base_idx + (next_r * sides + s) as u32;
                    let v3 = base_idx + (next_r * sides + next_s) as u32;

                    i.push(v0); i.push(v1); i.push(v3); m.push(4);
                    i.push(v0); i.push(v3); i.push(v2); m.push(4);
                }
            }
        }
    });
    add_blas(1, v_mp, i_mp, m_mp, n_mp);

    // BLAS 2: Gold Sphere
    let (v_gs, i_gs, m_gs, n_gs) = build_blas_generic(|v, i, m| {
        generate_sphere([-1.1, -1.3, -0.6], 0.7, 5, 24, 24, v, i, m);
    });
    add_blas(2, v_gs, i_gs, m_gs, n_gs);

    // BLAS 3: Glass Sphere
    let (v_gl, i_gl, m_gl, n_gl) = build_blas_generic(|v, i, m| {
        generate_sphere([1.1, -1.3, 0.6], 0.7, 6, 24, 24, v, i, m);
    });
    add_blas(3, v_gl, i_gl, m_gl, n_gl);

    (
        global_vertices,
        global_indices,
        global_tri_materials,
        global_bvh_nodes,
        bvh_offsets,
        tri_offsets,
    )
}

fn build_blas_generic(
    generator: impl FnOnce(&mut Vec<GpuVertex>, &mut Vec<u32>, &mut Vec<u32>),
) -> (Vec<GpuVertex>, Vec<u32>, Vec<u32>, Vec<GpuBvhNode>) {
    let mut local_vertices = Vec::new();
    let mut local_indices = Vec::new();
    let mut local_tri_materials = Vec::new();

    generator(&mut local_vertices, &mut local_indices, &mut local_tri_materials);

    let tri_count = local_indices.len() / 3;
    let mut build_triangles = Vec::with_capacity(tri_count);
    for t in 0..tri_count {
        let i0 = local_indices[t * 3 + 0] as usize;
        let i1 = local_indices[t * 3 + 1] as usize;
        let i2 = local_indices[t * 3 + 2] as usize;

        let v0 = [local_vertices[i0].position[0], local_vertices[i0].position[1], local_vertices[i0].position[2]];
        let v1 = [local_vertices[i1].position[0], local_vertices[i1].position[1], local_vertices[i1].position[2]];
        let v2 = [local_vertices[i2].position[0], local_vertices[i2].position[1], local_vertices[i2].position[2]];

        let centroid = [
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        ];

        build_triangles.push(BvhBuildTriangle {
            v0,
            v1,
            v2,
            centroid,
            tri_idx: t as u32,
        });
    }

    let mut bvh_nodes = Vec::new();
    bvh_nodes.push(GpuBvhNode { min_x: 0.0, min_y: 0.0, min_z: 0.0, left_child: 0, max_x: 0.0, max_y: 0.0, max_z: 0.0, tri_count: 0 });
    build_bvh(&mut build_triangles, 0, tri_count, 0, &mut bvh_nodes);

    let mut final_indices = Vec::with_capacity(local_indices.len());
    let mut final_tri_materials = Vec::with_capacity(local_tri_materials.len());

    for tri in &build_triangles {
        let orig_t = tri.tri_idx as usize;
        final_indices.push(local_indices[orig_t * 3 + 0]);
        final_indices.push(local_indices[orig_t * 3 + 1]);
        final_indices.push(local_indices[orig_t * 3 + 2]);
        final_tri_materials.push(local_tri_materials[orig_t]);
    }

    (local_vertices, final_indices, final_tri_materials, bvh_nodes)
}

