import bpy
import os
import time

# Resolve the absolute path to the GLB file in the workspace
filepath = os.path.abspath("mechanical_part.glb")
last_mtime = 0

class GLBAutoReloader(bpy.types.Operator):
    bl_idname = "wm.glb_auto_reloader"
    bl_label = "GLB Auto Reloader"
    
    _timer = None
    
    def modal(self, context, event):
        global last_mtime
        if event.type == 'TIMER':
            if not os.path.exists(filepath):
                return {'PASS_THROUGH'}
                
            current_mtime = os.path.getmtime(filepath)
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                print(f"[AutoReloader] Detected GLB modification ({time.strftime('%H:%M:%S')}), reloading...")
                self.reload_glb()
                
        return {'PASS_THROUGH'}
        
    def reload_glb(self):
        col_name = "OpenCASCADE_Import"
        
        # 1. Create or clear the dedicated collection
        if col_name in bpy.data.collections:
            col = bpy.data.collections[col_name]
            # Delete all objects inside this collection to keep Blender clean
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            col = bpy.data.collections.new(col_name)
            bpy.context.scene.collection.children.link(col)
            
        # 2. Make the collection active so glTF imports into it
        layer_col = self.find_layer_collection(bpy.context.view_layer.layer_collection, col_name)
        if layer_col:
            bpy.context.view_layer.active_layer_collection = layer_col
            
        # 3. Import the new GLB
        if os.path.exists(filepath):
            bpy.ops.object.select_all(action='DESELECT')
            # Import GLB scene
            bpy.ops.import_scene.gltf(filepath=filepath)
            print("[AutoReloader] GLB reloaded successfully.")
            
            # Force redraw of the 3D viewport to show changes instantly
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    def find_layer_collection(self, layer_col, name):
        if layer_col.name == name:
            return layer_col
        for child in layer_col.children:
            found = self.find_layer_collection(child, name)
            if found:
                return found
        return None
            
    def execute(self, context):
        global last_mtime
        
        # 1. Clean up default scene objects (delete default Cube)
        if "Cube" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
            print("[AutoReloader] Removed default Cube.")
            
        # 2. Adjust default Light to an angled Sun Light
        if "Light" in bpy.data.objects:
            light_obj = bpy.data.objects["Light"]
            light_obj.data.type = 'SUN'
            light_obj.data.energy = 5.0 # Bright sunlight
            # Rotate the sun to angle it nicely (45 deg X, 45 deg Y)
            light_obj.rotation_euler = (0.785, 0.785, 0.0)
            print("[AutoReloader] Adjusted default light to an angled Sun light.")
            
        # 3. Enable rendering enhancements (Ambient Occlusion, Bloom, Screen Space Reflections)
        scene = context.scene
        scene.render.engine = 'BLENDER_EEVEE'
        try:
            scene.eevee.use_gtao = True
            scene.eevee.use_bloom = True
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True
        except AttributeError:
            pass # Safe fallback for newer Blender Eevee engine settings
            
        # 4. Set viewport shading to MATERIAL (Material Preview) and configure lighting
        # We enable scene lights (our sun) but disable scene world to use Blender's built-in HDRI reflections.
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        space.shading.use_scene_lights = True
                        space.shading.use_scene_world = False
                        print("[AutoReloader] Viewport shading set to Material Preview with Sun light shadows.")
                        
        # 5. Perform the initial import
        if os.path.exists(filepath):
            last_mtime = os.path.getmtime(filepath)
            self.reload_glb()
        else:
            last_mtime = 0
            
        # 6. Start the background monitor timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        print("[AutoReloader] Started modal background timer.")
        return {'RUNNING_MODAL'}
        
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        print("[AutoReloader] Stopped modal background timer.")

def register():
    bpy.utils.register_class(GLBAutoReloader)
    # Start the operator after a brief delay so Blender UI is fully initialized
    bpy.app.timers.register(lambda: bpy.ops.wm.glb_auto_reloader(), first_interval=0.5)

def unregister():
    bpy.utils.unregister_class(GLBAutoReloader)

if __name__ == "__main__":
    register()
