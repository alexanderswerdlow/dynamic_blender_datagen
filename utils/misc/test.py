import bpy

# Iterate over all objects in the scene
for obj in bpy.data.objects:
    # Check if the object is a mesh
    if obj.type == 'MESH':
        # Check if the object's bounding box dimensions resemble a cube
        bbox = obj.bound_box
        dimensions = [abs(bbox[0][i] - bbox[6][i]) for i in range(3)]
        if dimensions[0] == dimensions[1] == dimensions[2]:  # It's a cube
            print(f"Removing object: {obj.name}")
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()


output_filepath = "/mnt/ssd/aswerdlo/repos/point_odyssey/output.blend"
bpy.ops.wm.save_as_mainfile(filepath=output_filepath)