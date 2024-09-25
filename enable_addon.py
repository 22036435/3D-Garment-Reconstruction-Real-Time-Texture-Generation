import bpy

# Enable the OBJ import/export add-on
bpy.ops.wm.addon_enable(module='io_scene_obj')

# Save user preferences to ensure the add-on remains enabled
bpy.ops.wm.save_userpref()