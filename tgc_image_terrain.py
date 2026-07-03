import time

import numpy as np

from GeoPointCloud import GeoPointCloud
from infill_image import infill_image_scipy
import OSMTGC
import tgc_definitions
import tgc_tools
from tgc_image_terrain import (
    get_object_item,
    get_placed_object,
    get_trees,
    get_lidar_trees,
    set_constants,
)

status_print_duration = 1.0


def get_pixel_fast(x_pos, z_pos, height, scale, brush_type=72):
    """Same brush schema as tgc_image_terrain.get_pixel(), built directly
    instead of via json.loads() on a template string per call."""
    return {
        "tool": 0,
        "position": {"x": x_pos, "y": "-Infinity", "z": z_pos},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "_orientation": 0.0,
        "scale": {"x": scale, "y": 1.0, "z": scale},
        "type": brush_type,
        "value": height,
        "holeId": -1,
        "radius": 0.0,
        "orientation": 0.0,
    }


def build_grid_from_pointcloud(pc, heightmap_shape):
    """Reconstruct the (row, col) -> (x_enu, y_enu, z) grid from pc.points(),
    in the same order generate_course()'s per-pixel loop consumes it."""
    pts = np.array(list(pc.points()), dtype=np.float64)
    expected = heightmap_shape[0] * heightmap_shape[1]
    if pts.shape[0] != expected:
        raise ValueError(
            f"pc.points() returned {pts.shape[0]} points, expected {expected} "
            f"for heightmap shape {heightmap_shape}. This optimizer needs an "
            f"uncropped 1:1 pointcloud built the same way generate_course() "
            f"builds it, before any masking/cropping."
        )
    x_grid = pts[:, 0].reshape(heightmap_shape)
    y_grid = pts[:, 1].reshape(heightmap_shape)
    z_grid = pts[:, 2].reshape(heightmap_shape)
    return x_grid, y_grid, z_grid


def quadtree_brushes(
    x_grid,
    y_grid,
    z_grid,
    pc,
    image_scale,
    max_error=0.04,
    min_block=2,
    max_block=64,
    flat_brush_type=10,
    detail_brush_type=72,
    flat_overlap=2.6,
    printf=print,
):
    """Adaptively cover the heightmap with as few brushes as possible while
    keeping elevation error under max_error (meters) almost everywhere."""
    rows, cols = z_grid.shape
    brushes = []
    stats = {"flat_brushes": 0, "detail_brushes": 0, "blocks_visited": 0}

    stack = []
    for r0 in range(0, rows, max_block):
        for c0 in range(0, cols, max_block):
            h = min(max_block, rows - r0)
            w = min(max_block, cols - c0)
            stack.append((r0, c0, h, w))

    last_print_time = time.time()

    while stack:
        r0, c0, h, w = stack.pop()
        if h <= 0 or w <= 0:
            continue
        stats["blocks_visited"] += 1

        if time.time() > last_print_time + status_print_duration:
            last_print_time = time.time()
            printf(
                f"Quadtree pass: {stats['blocks_visited']} blocks visited, "
                f"{stats['flat_brushes']} flat + {stats['detail_brushes']} detail brushes so far"
            )

        block = z_grid[r0:r0 + h, c0:c0 + w]
        block_min = float(block.min())
        block_max = float(block.max())
        spread = block_max - block_min

        is_flat_enough = spread <= max_error
        at_floor = (h <= min_block and w <= min_block)

        if is_flat_enough or at_floor:
            ri = min(rows - 1, r0 + h // 2)
            ci = min(cols - 1, c0 + w // 2)
            x_enu = float(x_grid[ri, ci])
            y_enu = float(y_grid[ri, ci])
            x, y, z_pos = pc.enuToTGC(x_enu, y_enu, 0.0)

            if is_flat_enough and (h > min_block or w > min_block):
                mean_height = float(block.mean())
                block_extent_m = max(h, w) * image_scale
                brushes.append(
                    get_pixel_fast(
                        x, z_pos, mean_height,
                        flat_overlap * block_extent_m / 2.0,
                        brush_type=flat_brush_type,
                    )
                )
                stats["flat_brushes"] += 1
            else:
                brushes.append(
                    get_pixel_fast(
                        x, z_pos, float(z_grid[ri, ci]), image_scale,
                        brush_type=detail_brush_type,
                    )
                )
                stats["detail_brushes"] += 1
        else:
            h2, w2 = max(1, h // 2), max(1, w // 2)
            stack.append((r0, c0, h2, w2))
            stack.append((r0, c0 + w2, h2, w - w2))
            stack.append((r0 + h2, c0, h - h2, w2))
            stack.append((r0 + h2, c0 + w2, h - h2, w - w2))

    total = stats["flat_brushes"] + stats["detail_brushes"]
    naive_total = rows * cols
    reduction = 100.0 * (1.0 - (total / naive_total)) if naive_total else 0.0
    printf(
        f"Quadtree brush export complete: {stats['flat_brushes']} flat + "
        f"{stats['detail_brushes']} detail = {total} brushes "
        f"(vs {naive_total} with the stock per-pixel loop, {reduction:.1f}% fewer)"
    )
    return brushes


def generate_course_optimized(course_json, heightmap_dir_path, options_dict={}, printf=print, course_version=-1):
    """Mirrors tgc_image_terrain.generate_course() in this fork exactly
    (including the per-course_version layer_json/obj_tag resolution), but
    replaces the per-pixel height-brush loop with quadtree_brushes()."""
    if course_version not in tgc_definitions.version_tags:
        printf("invalid version")
        printf(course_version)
        return None

    if course_version == 25:
        layer_json = course_json
    elif course_version == 23:
        layer_json = course_json["userLayers2"]
    else:
        layer_json = course_json["userLayers"]

    obj_tag = tgc_definitions.version_tags[course_version]['objects']

    printf("Loading data from " + heightmap_dir_path)
    try:
        read_dictionary = np.load(heightmap_dir_path + '/heightmap.npy', allow_pickle=True).item()
        im = read_dictionary['heightmap'].astype('float32')
        import cv2
        mask = cv2.imread(heightmap_dir_path + '/mask.png', cv2.IMREAD_COLOR)
        mask = np.flip(mask, 0)

        printf("Filling holes in heightmap")
        image_scale = read_dictionary['image_scale']
        printf("Map scale is: " + str(image_scale) + " meters")

        background_ratio = None
        background_scale = None
        if options_dict.get('add_background', False):
            background_scale = float(options_dict.get('background_scale', 16.0))
            background_ratio = background_scale / image_scale
            printf("Background requested with scale: " + str(background_scale) + " meters")

        heightmap, background, holeMask = infill_image_scipy(
            im, mask, background_ratio=background_ratio,
            fill_water=options_dict.get('fill_water', False),
            purge_water=options_dict.get('purge_water', False),
            printf=printf,
        )
    except FileNotFoundError:
        printf("Could not find heightmap or mask at: " + heightmap_dir_path)
        return course_json

    course_json = set_constants(
        course_json,
        options_dict.get('flatten_fairways', False),
        options_dict.get('flatten_greens', False),
        read_dictionary['origin'][0],
        printf=printf,
    )
    layer_json["height"] = []
    layer_json["terrainHeight"] = []
    course_json[obj_tag] = []

    pc = GeoPointCloud()
    pc.addFromImage(heightmap, image_scale, read_dictionary['origin'], read_dictionary['projection'])

    if background is not None:
        background_pc = GeoPointCloud()
        background_pc.addFromImage(background, background_scale, read_dictionary['origin'], read_dictionary['projection'])
        num_points = len(background_pc.points())
        last_print_time = time.time()
        for n, i in enumerate(background_pc.points()):
            if time.time() > last_print_time + status_print_duration:
                last_print_time = time.time()
                printf(str(round(100.0 * float(n) / num_points, 2)) + "% through background heightmap")
            easting, northing = background_pc.enuToProj(i[0], i[1])
            x, y, z = pc.projToTGC(easting, northing, 0.0)
            layer_json["height"].append(
                get_pixel_fast(x, z, i[2], 2.5 * background_scale, brush_type=10)
            )

    printf("Building quadtree brush layout for main heightmap")
    x_grid, y_grid, z_grid = build_grid_from_pointcloud(pc, heightmap.shape)
    brushes = quadtree_brushes(
        x_grid, y_grid, z_grid, pc, image_scale,
        max_error=float(options_dict.get('quadtree_max_error', 0.04)),
        min_block=int(options_dict.get('quadtree_min_block', 2)),
        max_block=int(options_dict.get('quadtree_max_block', 64)),
        flat_overlap=float(options_dict.get('quadtree_flat_overlap', 2.6)),
        printf=printf,
    )
    layer_json["height"].extend(brushes)

    if options_dict.get('lidar_trees', False) and len(read_dictionary.get('trees', [])) > 0:
        printf("Adding trees from lidar data")
        mask_pc = GeoPointCloud()
        mask_pc.addFromImage(im, image_scale, read_dictionary['origin'], read_dictionary['projection'])
        for o in get_lidar_trees(course_json['theme'], options_dict.get('tree_variety', False),
                                  read_dictionary['trees'], pc, mask, mask_pc, image_scale, course_version):
            course_json[obj_tag].append(o)

    if options_dict.get('use_osm', True):
        printf("Adding golf features to lidar data")
        spline_json = tgc_tools.get_spline_configuration_json(heightmap_dir_path)
        upper_left_enu = pc.ulENU()
        lower_right_enu = pc.lrENU()
        upper_left_latlon = pc.enuToLatLon(*upper_left_enu)
        lower_right_latlon = pc.enuToLatLon(*lower_right_enu)
        result = OSMTGC.getOSMData(
            lower_right_latlon[0], upper_left_latlon[1],
            upper_left_latlon[0], lower_right_latlon[1], printf=printf,
        )
        osm_trees = OSMTGC.addOSMToTGC(
            course_json, pc, result,
            x_offset=float(options_dict.get('adjust_ew', 0.0)),
            y_offset=float(options_dict.get('adjust_ns', 0.0)),
            options_dict=options_dict, spline_configuration_json=spline_json, printf=printf,
            course_version=course_version,
        )
        if len(osm_trees) > 0:
            printf("Adding trees from OpenStreetMap")
            for o in get_trees(course_json['theme'], options_dict.get('tree_variety', False), osm_trees, course_version):
                course_json[obj_tag].append(o)

    printf("Moving course to lowest valid elevation")
    course_json = tgc_tools.elevate_terrain(course_json, None, printf=printf, course_version=course_version)

    printf("Course Description Complete (optimized quadtree export)")
    return course_json


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python tgc_image_terrain_optimized.py COURSE_DIRECTORY HEIGHTMAP_DIRECTORY COURSE_VERSION")
        print("  COURSE_VERSION: 21, 23, or 25")
        sys.exit(0)
    course_dir_path = sys.argv[1]
    heightmap_dir_path = sys.argv[2]
    course_version = int(sys.argv[3])
    print("Getting course description")
    course_json = tgc_tools.get_course_json(course_dir_path)
    print("Generating course (optimized)")
    course_json = generate_course_optimized(course_json, heightmap_dir_path, course_version=course_version)
    print("Saving new course description")
    tgc_tools.write_course_json(course_dir_path, course_json)
