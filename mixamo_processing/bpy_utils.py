import os
import sys
import bpy
import math
import numpy as np


def init_scene():
    # Gets a reference to Blender Scene object
    # Deletes all objects from default Blender scene

    # Get scene
    scene = bpy.data.scenes["Scene"]
    # Delete all objects
    for key in bpy.data.objects.keys():
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects[key].select_set(True)
        bpy.ops.object.delete(use_global=True)
    return scene


def decomposeObject(ob):
    # Decomposes a Blender object into an array of vertices (N x 3) and a list of faces (may not be triangular)

    V = [ob.matrix_world @ v.co for v in ob.data.vertices]
    F = [list(p.vertices[:]) for p in ob.data.polygons]
    return np.array(V, np.float32), F


def getVertexGroups(ob, bones):
    # Gets blend weights as an array N x K for the given 'bones'

    W = np.zeros((ob.data.vertices.__len__(), bones.__len__()), np.float32)
    for i, b in enumerate(bones):
        W[:, i] = getVG(ob, b.name)
    return W


def getVG(ob, bone):
    # Gets the blend weights of a single bone as an array with shape (N,)

    w = np.zeros((ob.data.vertices.__len__(),), np.float32)
    if bone in ob.vertex_groups:
        for i, v in enumerate(ob.data.vertices):
            for g in v.groups:
                if g.group == ob.vertex_groups[bone].index:
                    w[i] = g.weight
    return w


def get_keyframes(ob):
    # Get start and end frames of an object animation data

    keyframes = []
    anim = ob.animation_data
    if anim is not None and anim.action is not None:
        for fcu in anim.action.fcurves:
            for keyframe in fcu.keyframe_points:
                x, y = keyframe.co
                if x not in keyframes:
                    keyframes.append((math.ceil(x)))
    return min(keyframes), max(keyframes)
