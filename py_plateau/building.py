#!/usr/bin/env python
# coding: utf-8

import math
import os
import pdb

import cv2
import numpy as np
import open3d as o3d
import pyproj

from .earcut import earcut


class Building:
    """bldg:Building"""

    def __init__(self, b_id, from_srid="6697", to_srid="6677"):
        # super().__init__()
        self.cityobject_id = b_id
        self.polygons = []

        self.vertices = []
        self.triangles = []
        self.triangle_meshes = []

        # pyproj.Transformer.from_crs(<変換元座標系>, <変換先座標系> [, always_xy])
        self.transformer = pyproj.Transformer.from_crs(
            f"epsg:{from_srid}", f"epsg:{to_srid}"
        )

    def get_triangle_mesh(self):
        return self.triangle_meshes

    def transform_coordinate(self, latitude, longitude, height):
        xx, yy, zz = self.transformer.transform(latitude, longitude, height)
        return np.array([xx, yy, zz])

    def create_triangle_meshes(self, filename, poly_ids, polygons, textures=None):
        # 複数のポリゴン全てのポリゴンのUV座標を保持する
        all_mesh_uvs = []

        # 面ごとに処理を行う
        for poly_id, poly in zip(poly_ids, polygons):
            # polygonのidだけ、リンクではなくリンク元なので、#がついてないので、#をつける
            poly_id = "#" + poly_id

            transformed_polygon = [self.transform_coordinate(*x) for x in poly]
            # CityGMLと法線計算時の頂点の取扱順序が異なるため、反転させる
            transformed_polygon = transformed_polygon[::-1]
            transformed_polygon = np.array(transformed_polygon)

            normal = self.get_normal(transformed_polygon)[0]
            # 三角形分割するために二次元に変換する
            # 頂点のインデックスは変わらない
            poly_2d = np.zeros((transformed_polygon.shape[0], 2))
            for i, vertex in enumerate(transformed_polygon):
                xy = self.to_2d(vertex, normal)
                poly_2d[i] = xy

            # 入力のポリゴンに対する、三角化後のインデックスが返ってくる
            # 1次元の配列で返ってくる
            triangulated_vertices_indexes = earcut(
                np.array(poly_2d, dtype=np.int).flatten(), dim=2
            )

            if len(triangulated_vertices_indexes) > 0:
                # 現在の頂点数を取得
                vertices_start_index = len(self.vertices)
                # 複数の面を1つの配列で構成するので、面の頂点をどんどん追加していく
                self.vertices.extend(transformed_polygon)

                # 三角化した結果のインデックスは1次元なので(n, 3)の形に変換
                # XYZの座標が3つ × 三角形数になるはず
                triangles = np.array(triangulated_vertices_indexes).reshape((-1, 3))
                # 面は、面ごとに先頭の頂点が0になっている
                # が、全ての頂点1つの配列にextendしていくため、現在の頂点数を足して指定するインデックスを調整する
                triangles_offset = triangles + vertices_start_index
                self.triangles.extend(triangles_offset)

                # テクスチャを探す
                # LOD2じゃない場合はテクスチャがないので無視される
                for t in textures:
                    # ポリゴンのidと一致するidを持つUV座標を取得する
                    uv_coords = t["targets"].get(poly_id)
                    # テクスチャがある場合は、UV座標を保持する
                    # ただし、UV座標はは三角化される前の多角形の頂点に合っている
                    if uv_coords is not None:
                        image_uri = t["image_uri"]

                        t_array = triangles.reshape((-1))
                        one_mesh_uvs = []
                        for x in t_array:
                            uv = uv_coords[0, x]
                            one_mesh_uvs.append(uv)
                        all_mesh_uvs.append(one_mesh_uvs)
                        # IDはユニークのはずなので、見つけたら終了
                        break
                # この段階で三角化されていない面、されている面、頂点、

        # create triangle mesh by Open3D
        triangle_meshes = o3d.geometry.TriangleMesh()
        triangle_meshes.vertices = o3d.utility.Vector3dVector(self.vertices)
        triangle_meshes.triangles = o3d.utility.Vector3iVector(self.triangles)

        if "all_mesh_uvs" in locals():
            if all_mesh_uvs:
                pdb.set_trace()

                texture_filename = os.path.dirname(filename) + "/" + image_uri

                # pngでない場合は、pngに変換する
                img = cv2.imread(texture_filename[1:])
                texture_filename = texture_filename[1:] + ".png"
                cv2.imwrite(texture_filename, img)
                # todo: テクスチャのUV座標を面ごとに入れる方法を調べる
                # 三角形なら3点のUV座標が必要なので、(頂点の数 * 面の数)行 × 2列が必要
                # all_mesh_uvsは三角分割される前の面の頂点なので、3点以上ある
                # これを三角分割された頂点に合わせる必要がある
                # あと、順番も合ってないといけない
                # テクスチャを持っていない面も存在するのでどうするか
                triangle_meshes.triangle_uvs = o3d.utility.Vector2dVector(
                    np.array(all_mesh_uvs)
                )
                triangle_meshes.triangle_material_ids = o3d.utility.IntVector(
                    [0] * len(self.triangles)
                )
                triangle_meshes.textures = [o3d.io.read_image(texture_filename)]
            else:
                triangle_meshes.triangle_uvs.extend(
                    [np.zeros((2)) for x in range(len(self.triangles) * 3)]
                )
                triangle_meshes.triangle_material_ids.extend([0] * len(self.triangles))

        # 法線の取得
        triangle_meshes.compute_vertex_normals()

        self.triangle_meshes = triangle_meshes
        self.polygons = polygons

    # 3つ以上の点を渡して、ポリゴンの法線を求める
    @staticmethod
    def get_normal(poly):
        normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        for i, _ in enumerate(poly):
            next_index = i + 1

            if next_index == len(poly):
                next_index = 0

            point_1 = poly[i]
            point_1_x = point_1[0]
            point_1_y = point_1[1]
            point_1_z = point_1[2]

            point_2 = poly[next_index]
            point_2_x = point_2[0]
            point_2_y = point_2[1]
            point_2_z = point_2[2]

            normal[0] += (point_1_y - point_2_y) * (point_1_z + point_2_z)
            normal[1] += (point_1_z - point_2_z) * (point_1_x + point_2_x)
            normal[2] += (point_1_x - point_2_x) * (point_1_y + point_2_y)

        if (normal == np.array([0.0, 0.0, 0.0])).all():
            return (normal, False)

        normal = normal / math.sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
        )
        return (normal, True)

    # 面と法線を渡して、2次元の座標に変換する
    @staticmethod
    def to_2d(p, n):
        x3 = np.array([1.1, 1.1, 1.1])

        if (n == x3).all():
            x3 += np.array([1, 2, 3])
        x3 = x3 - np.dot(x3, n) * n
        x3 /= math.sqrt((x3**2).sum())
        y3 = np.cross(n, x3)
        return (np.dot(p, x3), np.dot(p, y3))
