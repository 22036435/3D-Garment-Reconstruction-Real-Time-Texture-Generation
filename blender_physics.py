import bpy
import os

def load_obj_with_texture_and_physics(obj_path, texture_path, export_path, uv_map_path):
    # Ensure the output directory exists
    output_dir = os.path.dirname(uv_map_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import the .obj file
    bpy.ops.import_scene.obj(filepath=obj_path)
    
    # Get the imported object
    obj = bpy.context.selected_objects[0]
    
    # Ensure the object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Create a new material
    mat = bpy.data.materials.new(name="TextureMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Load the texture image
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)
    
    # Connect the texture to the material
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    
    # Assign the material to the object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # Apply cloth physics
    bpy.ops.object.modifier_add(type='CLOTH')
    cloth_modifier = obj.modifiers['Cloth']
    
    # Set cloth physics properties
    cloth_modifier.settings.quality = 5
    cloth_modifier.settings.mass = 0.3
    cloth_modifier.settings.tension_stiffness = 15
    cloth_modifier.settings.bending_stiffness = 0.5
    cloth_modifier.settings.air_damping = 1
    cloth_modifier.settings.use_dynamic_mesh = True
    
    # Create UV map
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Save UV map as SVG
    bpy.ops.uv.export_layout(filepath=uv_map_path, export_all=True, mode='SVG', size=(2048, 2048))
    
    # Export as FBX
    bpy.ops.export_scene.fbx(filepath=export_path, use_selection=True)

# Example usage
obj_path = "/home/aslico/Desktop/cloth2tex/template/reference/1_wy/mesh/mesh.obj"
texture_path = "/home/aslico/Desktop/cloth2tex/template/reference/1_wy/mesh/2.jpg"
export_path = "/home/aslico/Desktop/cloth2tex/output/model.fbx"
uv_map_path = "/home/aslico/Desktop/cloth2tex/output/uv_map.svg"

load_obj_with_texture_and_physics(obj_path, texture_path, export_path, uv_map_path)