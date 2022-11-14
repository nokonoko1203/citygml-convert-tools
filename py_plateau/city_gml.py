#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

import numpy as np
import open3d as o3d
from lxml import etree

from .building import Building


def str2floats(x):
    """x y z -> [x, y, z]"""
    return np.array([float(i) for i in x.text.split(" ")])


class CityGml:
    """core:CityGml"""

    def __init__(self, filename, to_srid="6677"):
        # filename
        self.filename = filename
        self.basename = os.path.splitext(os.path.basename(filename))[0]

        # split from basename
        basenames = self.basename.split("_")
        # メッシュコード
        self.mesh_code = basenames[0]
        # 地物型 (bldg)
        self.object_name = basenames[1]
        # CRS 空間参照 ID (SRID)
        self.from_srid = basenames[2]
        self.to_srid = to_srid

        # xml tree
        tree = etree.parse(filename)
        root = tree.getroot()
        self.tree = tree
        self.root = root

        # buildings
        self.obj_buildings = []

    def lod0(self):
        nsmap = self.root.nsmap
        tree = self.tree

        # scan cityObjectMember
        buildings = tree.xpath(
            "/core:CityModel/core:cityObjectMember/bldg:Building", namespaces=nsmap
        )
        for building in buildings:
            obj_building = Building(self.from_srid, self.to_srid)

            # bldg:lod0RoofEdge
            faces = building.xpath(
                "bldg:lod0RoofEdge/gml:MultiSurface/gml:surfaceMember/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList",
                namespaces=nsmap,
            )
            polygons = [str2floats(face_str).reshape((-1, 3)) for face_str in faces]
            obj_building.create_triangle_meshes(polygons)
            self.obj_buildings.append(obj_building)

    def lod1(self):
        nsmap = self.root.nsmap
        tree = self.tree

        # scan cityObjectMember
        buildings = tree.xpath(
            "/core:CityModel/core:cityObjectMember/bldg:Building", namespaces=nsmap
        )
        for building in buildings:
            obj_building = Building(self.from_srid, self.to_srid)

            # bldg:lod1Solid
            faces = building.xpath(
                "bldg:lod1Solid/gml:Solid/gml:exterior/gml:CompositeSurface/gml:surfaceMember/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList",
                namespaces=nsmap,
            )
            polygons = [str2floats(face_str).reshape((-1, 3)) for face_str in faces]
            # todo: lod2に最適化しすぎて、一旦LOD1では利用できなくなっている
            obj_building.create_triangle_meshes(polygons)
            self.obj_buildings.append(obj_building)

    def lod2(self):
        nsmap = self.root.nsmap
        tree = self.tree

        # scan cityObjectMember
        buildings = tree.xpath(
            "/core:CityModel/core:cityObjectMember/bldg:Building", namespaces=nsmap
        )
        for building in buildings:
            b_id = building.attrib.values()[0]
            obj_building = Building(b_id, self.from_srid, self.to_srid)

            # bldg:GroundSurface, bldg:RoofSurface, bldg:RoofSurface
            polygon_xpaths = [
                "bldg:boundedBy/bldg:GroundSurface/bldg:lod2MultiSurface/gml:MultiSurface/gml:surfaceMember/gml:Polygon",
                "bldg:boundedBy/bldg:RoofSurface/bldg:lod2MultiSurface/gml:MultiSurface/gml:surfaceMember/gml:Polygon",
                "bldg:boundedBy/bldg:WallSurface/bldg:lod2MultiSurface/gml:MultiSurface/gml:surfaceMember/gml:Polygon",
            ]
            poly_ids = []
            vals_list = []
            for polygon_xpath in polygon_xpaths:
                poslist_xpaths = building.xpath(polygon_xpath, namespaces=nsmap)
                for poslist_xpath in poslist_xpaths:
                    vals = poslist_xpath.xpath(
                        "gml:exterior/gml:LinearRing/gml:posList", namespaces=nsmap
                    )
                    poly_ids.append(poslist_xpath.attrib.values()[0])
                    vals_list.extend(vals)

            polygons = [str2floats(v).reshape((-1, 3)) for v in vals_list]

            # テクスチャを取得
            appearance_member = tree.xpath(
                "/core:CityModel/app:appearanceMember/app:Appearance/app:surfaceDataMember/app:ParameterizedTexture",
                namespaces=nsmap,
            )
            # 画像のURIを取得
            textures = []
            for appearance in appearance_member:
                parameter = {
                    "image_uri": None,
                    "targets": {},
                }

                # 画像のURIを取得
                for image_url in appearance.xpath("app:imageURI", namespaces=nsmap):
                    parameter["image_uri"] = image_url.text

                # テクスチャの情報を取得
                # テクスチャ1枚に対して、複数の面がある
                # targetは面のID
                for target in appearance.xpath("app:target", namespaces=nsmap):
                    poly_id = target.attrib["uri"]

                    # テクスチャのUV座標を取得
                    # 文字列になっているので、floatに変換
                    texture_coordinates = [
                        str2floats(coordinates).reshape((-1, 2))
                        for coordinates in target.xpath(
                            "app:TexCoordList/app:textureCoordinates",
                            namespaces=nsmap,
                        )
                    ]
                    # 要素数の最大値を取り出す
                    maximum_of_elements = max(
                        map(lambda x: x.shape[0], texture_coordinates)
                    )

                    # 要素の数が最大値よりも多い時？に要素を追加する
                    for coordinate_index, coord in enumerate(texture_coordinates):
                        num = maximum_of_elements - coord.shape[0]

                        if num > 0:
                            texture_coordinates[coordinate_index] = np.append(
                                coord,
                                np.tile(coord[-1].reshape(-1, 2), (num, 1)),
                                axis=0,
                            )

                    # テクスチャの座標とURIのペアを格納
                    parameter["targets"][poly_id] = np.array(texture_coordinates)

                textures.append(parameter)

            obj_building.create_triangle_meshes(
                self.filename, poly_ids, polygons, textures
            )
            self.obj_buildings.append(obj_building)

    def write_ply(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        for index, obj_building in enumerate(self.obj_buildings):
            triangle_mesh = obj_building.get_triangle_mesh()
            pathname = os.path.join(
                output_path,
                f"{self.mesh_code}_{self.object_name}_{self.to_srid}_{index:02}.ply",
            )
            o3d.io.write_triangle_mesh(pathname, triangle_mesh, write_ascii=True)

    def write_obj(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        for index, obj_building in enumerate(self.obj_buildings):
            triangle_mesh = obj_building.get_triangle_mesh()
            pathname = os.path.join(
                output_path,
                f"{self.mesh_code}_{self.object_name}_{self.to_srid}_{index:02}.obj",
            )
            o3d.io.write_triangle_mesh(pathname, triangle_mesh, write_triangle_uvs=True)
