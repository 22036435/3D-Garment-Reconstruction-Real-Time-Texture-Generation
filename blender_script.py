import bpy
import sys

# Get arguments passed to the script
argv = sys.argv
argv = argv[argv.index("--") + 1:]

# Paths for FBX, texture, output model, and rendered preview
fbx_path = argv[0]
texture_path = argv[1]
output_fbx_path = argv[2]
rendered_image_path = argv[3]

# Load FBX model
def load_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath)

    # Ensure the correct object is selected and made active (assuming it's the first mesh)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            break  # Use the first MESH object found

# Apply texture and UV mapping to the active object
def apply_texture(texture_path):
    obj = bpy.context.view_layer.objects.active  # Get the active object (set above)
    
    if obj and obj.type == 'MESH':
        # Ensure the object is fully unwrapped with Smart UV Project
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)  # Smart UV Projection
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create a new material and apply texture
        mat = bpy.data.materials.new(name="TextureMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]

        # Texture image node
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(texture_path)

        # Mapping node to scale the texture
        mapping = mat.node_tree.nodes.new('ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (5.0, 5.0, 5.0)  # Adjust the scale to tile the texture

        # Texture coordinate node
        tex_coord = mat.node_tree.nodes.new('ShaderNodeTexCoord')

        # Connect nodes
        mat.node_tree.links.new(mapping.inputs['Vector'], tex_coord.outputs['UV'])
        mat.node_tree.links.new(tex_image.inputs['Vector'], mapping.outputs['Vector'])
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

        # Add bump map for depth
        bump = mat.node_tree.nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.5  # Adjust bump strength as needed
        mat.node_tree.links.new(bump.inputs['Height'], tex_image.outputs['Color'])
        mat.node_tree.links.new(bsdf.inputs['Normal'], bump.outputs['Normal'])

        # Add normal map
        normal_map = mat.node_tree.nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = 1.0  # Adjust normal map strength as needed
        mat.node_tree.links.new(normal_map.inputs['Color'], tex_image.outputs['Color'])
        mat.node_tree.links.new(bsdf.inputs['Normal'], normal_map.outputs['Normal'])

        # Adjust material properties
        bsdf.inputs['Specular'].default_value = 0.1  # Lower specular value
        bsdf.inputs['Roughness'].default_value = 0.8  # Increase roughness value

        # Assign the material to the object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        
        # Ensure the material is assigned to all faces
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')

def position_camera():
    # Remove existing cameras
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    # Add a new camera
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0.001966, -3.35282, 0.622872))
    camera = bpy.context.object
    camera.rotation_euler = (1.5708, -0.0157, -0.17)  # Rotate to specified angles
    bpy.context.scene.camera = camera  # Set the camera as the active camera

    # Set camera properties
    camera.data.lens = 35  # Set focal length (adjust as needed)
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000

    # Debugging: Print camera location and rotation
    print(f"Camera location: {camera.location}")
    print(f"Camera rotation: {camera.rotation_euler}")

# Render the model as an image for preview
def render_model(output_image_path):
    bpy.context.scene.render.filepath = output_image_path
    bpy.context.scene.camera = bpy.context.scene.objects['Camera']  # Ensure there's a camera
    bpy.ops.render.render(write_still=True)  # Render and save the image

# Export the model as FBX
def export_model(output_fbx_path):
    bpy.ops.export_scene.fbx(filepath=output_fbx_path, use_selection=False)  # Export all objects

# Delete default cube (or any other unwanted objects)
def delete_default_objects():
    for obj in bpy.data.objects:
        if obj.name == "Cube":  # Delete the default Blender cube
            bpy.data.objects.remove(obj, do_unlink=True)

# Main function
delete_default_objects()  # Remove any unwanted objects (e.g., default cube)
load_fbx(fbx_path)        # Load the FBX model
apply_texture(texture_path)  # Apply the texture to the model
position_camera()  # Position the camera in front of the model
render_model(rendered_image_path)  # Render the model for preview
export_model(output_fbx_path)  # Export the model as FBX

# Optionally save the blend file
bpy.ops.wm.save_as_mainfile(filepath="processed_model.blend")