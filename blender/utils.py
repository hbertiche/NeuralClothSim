import bpy


def clean_scene():
    for object in bpy.data.objects:
        object.select_set(True)
    bpy.ops.object.delete()


def load_object(obj, name, loc, rot=(0, 0, 0)):
    bpy.ops.import_scene.obj(
        filepath=obj, split_mode="OFF", axis_forward="-Y", axis_up="Z"
    )
    bpy.ops.object.shade_smooth()
    assert len(bpy.context.selected_objects) == 1, "Multiple objects in one OBJ? " + obj
    object = bpy.context.selected_objects[0]
    object.name = name
    object.location = loc
    object.rotation_euler = rot
    return object


def select(object):
    if type(object) is str:
        object = bpy.data.objects[object]
    deselect()
    object.select_set(True)
    bpy.context.view_layer.objects.active = object
    return object


def deselect():
    for object in bpy.data.objects.values():
        object.select_set(False)


def mesh_cache_modifier(object, pc2_file):
    object = select(object)
    bpy.ops.object.modifier_add(type="MESH_CACHE")
    object.modifiers["MeshCache"].cache_format = "PC2"
    object.modifiers["MeshCache"].filepath = pc2_file


def createVertexGroups(ob, W):
    for j in range(W.shape[1]):
        vg_name = "bone" + str(j)
        createVertexGroup(ob, W[:, j], vg_name)


def createVertexGroup(ob, W, name):
    vg = ob.vertex_groups.new(name=name)
    for i, w in enumerate(W):
        vg.add([i], w, "ADD")
