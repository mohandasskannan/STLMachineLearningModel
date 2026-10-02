"""Render saved STL meshes for experiment review using existing dependencies."""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def render(paths, labels, output, columns, front=False):
    width, height = 440, 380
    canvas = Image.new('RGB', (width * columns, height * ((len(paths) + columns - 1) // columns)), 'white')
    draw = ImageDraw.Draw(canvas)
    view = np.array([1.3, 1.8 if front else -1.8, 1.1])
    view /= np.linalg.norm(view)
    right = np.cross([0, 0, 1], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    checks = []
    for index, (path, label) in enumerate(zip(paths, labels)):
        mesh = trimesh.load_mesh(path)
        checks.append({'path': str(path), 'finite': bool(np.isfinite(mesh.vertices).all()),
                       'watertight': bool(mesh.is_watertight), 'components': len(mesh.split(only_watertight=False)),
                       'extents_mm': mesh.extents.tolist()})
        vertices = mesh.vertices - mesh.bounds.mean(axis=0)
        projected = np.stack((vertices @ right, -(vertices @ up)), axis=1) * 2.45
        x, y = index % columns * width, index // columns * height
        projected += [x + width / 2, y + height / 2 + 15]
        order = np.argsort(mesh.triangles_center @ view)
        shades = 0.35 + 0.65 * np.maximum(0, mesh.face_normals @ light)
        for face in order:
            color = tuple(int(c * shades[face]) for c in (85, 165, 215))
            draw.polygon([tuple(p) for p in projected[mesh.faces[face]]], fill=color)
        draw.text((x + 12, y + 12), label, fill='black')
        draw.text((x + 12, y + height - 22), f"{checks[-1]['components']} components; watertight={mesh.is_watertight}", fill='black')
    canvas.save(output)
    return checks


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--columns', type=int, default=2)
    parser.add_argument('--front', action='store_true')
    args = parser.parse_args()
    paths = sorted(args.directory.glob('*.stl'))
    output = args.directory / ('preview-front.png' if args.front else 'preview.png')
    checks = render(paths, [p.stem for p in paths], output, args.columns, args.front)
    (args.directory / 'reopened-mesh-checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
