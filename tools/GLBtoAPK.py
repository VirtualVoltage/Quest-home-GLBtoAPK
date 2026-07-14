#!/usr/bin/env python3
# GLBtoAPK.py — convert a glTF 2.0 .glb/.gltf model into a Quest Home *source* APK.
#
# This is the whole glb2home conversion engine as one dependency-free script (stdlib only:
# json + zipfile + struct + base64). The editor calls it when a .glb/.gltf is dropped on the
# window, then loads the APK it produces exactly as if that APK had been dropped instead.
#
# It emits the precise nesting the V79 loader expects:
#
#   <out>.apk  (zip)
#     assets/scene.zip  (zip, STORED)
#       _WORLD_MODEL.gltf.ovrscene  (zip, STORED)
#         V9.gltf     glTF JSON
#         V9.bin      the single repacked geometry buffer
#         tex<i>.png / tex<i>.jpg   textures, referenced by uri
#       _BACKGROUND_LOOP.<ext>      optional ambient audio loop
#
# What it fixes on the way through:
#   * Wrong-UV black meshes — the loader samples the base texture with TEXCOORD_0 ONLY. Models that
#     bind it to another UV set (baseColorTexture.texCoord = 3/7/...) render black. Each primitive's
#     real UV set is remapped into TEXCOORD_0.
#   * Embedded textures — the loader only decodes images that have a `uri`, so GLB-embedded images
#     are externalized to real PNG/JPEG files in the .ovrscene.
#   * The buffer is repacked into one V9.bin (node hierarchy, skins and animations are preserved).
#
# Texture policy: PNG and JPEG only. Anything else (WebP, KTX/ASTC, GIF, BMP, DDS) is a hard error
# rather than a silently blank texture — re-export the model with PNG/JPEG textures.
#
# CLI:  python3 GLBtoAPK.py <input.glb> [-o output.apk] [--audio loop.ogg]
# Exit: 0 on success; 1 with the reason on stderr on any failure.

import argparse
import base64
import json
import os
import struct
import sys
import zipfile

GLB_MAGIC = b"glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

AUDIO_EXTS = (".ogg", ".wav", ".mp3", ".flac")


class ConvertError(Exception):
    """Any failure the user needs to see. Message is printed verbatim."""


def read_file(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""


def data_uri_bytes(uri):
    """Decode a `data:...;base64,...` URI. Non-base64 data URIs yield b'' (as in glb2home)."""
    marker = "base64,"
    i = uri.find(marker)
    if i < 0:
        return b""
    return base64.b64decode(uri[i + len(marker):], validate=False)


def sniff_image(b):
    """Identify an image by magic bytes. Mirrors g2h::sniffImage."""
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if b.startswith(b"\xFF\xD8\xFF"):
        return "jpg"
    if len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b.startswith(b"\xABKTX 11"):
        return "ktx"
    if len(b) >= 12 and b[4:10] == b"KTX 20":
        return "ktx2"
    if b.startswith(b"GIF8"):
        return "gif"
    if b.startswith(b"BM"):
        return "bmp"
    if b.startswith(b"DDS "):
        return "dds"
    return "unknown"


def load_container(path):
    """Parse a .glb or .gltf into (root_json, buffers, base_dir)."""
    raw = read_file(path)
    if not raw:
        raise ConvertError("cannot read input file: " + path)
    base_dir = os.path.dirname(os.path.abspath(path))

    glb_bin = b""
    is_glb = len(raw) > 12 and raw[0:4] == GLB_MAGIC
    if is_glb:
        text = None
        off = 12
        while off + 8 <= len(raw):
            length, ctype = struct.unpack_from("<II", raw, off)
            off += 8
            if off + length > len(raw):
                break
            if ctype == CHUNK_JSON:
                text = raw[off:off + length].decode("utf-8", "replace")
            elif ctype == CHUNK_BIN:
                glb_bin = raw[off:off + length]
            off += length + (-length % 4)   # chunks are 4-byte aligned
        if not text:
            raise ConvertError("GLB has no JSON chunk")
    else:
        text = raw.decode("utf-8", "replace")

    try:
        root = json.loads(text)
    except ValueError:
        raise ConvertError("failed to parse glTF JSON")
    if "meshes" not in root:
        raise ConvertError("glTF has no meshes")

    buffers = []
    for i, b in enumerate(root.get("buffers", [])):
        uri = b.get("uri", "")
        if not uri:
            if is_glb and i == 0:
                buffers.append(glb_bin)
            else:
                raise ConvertError(
                    "buffer[%d] has no uri and is not the GLB binary chunk" % i)
        elif uri.startswith("data:"):
            buffers.append(data_uri_bytes(uri))
        else:
            data = read_file(os.path.join(base_dir, uri))
            if not data:
                raise ConvertError("cannot read external buffer: " + uri)
            buffers.append(data)
    return root, buffers, base_dir


def image_bytes(root, buffers, base_dir, img, index):
    """Pull an image's bytes. Returns (data, source_bufferView_or_-1)."""
    uri = img.get("uri")
    if uri is not None:
        if uri.startswith("data:"):
            return data_uri_bytes(uri), -1
        return read_file(os.path.join(base_dir, uri)), -1

    if "bufferView" in img:
        bvi = int(img["bufferView"])
        bv = root["bufferViews"][bvi]
        buf = int(bv["buffer"])
        off = int(bv.get("byteOffset", 0))
        length = int(bv["byteLength"])
        if buf < 0 or buf >= len(buffers) or off + length > len(buffers[buf]):
            raise ConvertError("image bufferView out of range")
        return buffers[buf][off:off + length], bvi

    return b"", -1


def remap_buffer_views(node, bv_map):
    """Rewrite every `bufferView` index in `node` through bv_map (in place)."""
    if isinstance(node, dict):
        for key, val in node.items():
            if key == "bufferView" and isinstance(val, int) and not isinstance(val, bool):
                if val < 0 or val >= len(bv_map) or bv_map[val] < 0:
                    raise ConvertError("accessor references a texture bufferView (unexpected)")
                node[key] = bv_map[val]
            else:
                remap_buffer_views(val, bv_map)
    elif isinstance(node, list):
        for item in node:
            remap_buffer_views(item, bv_map)


def build_zip(entries):
    """entries: list of (name, data, store). Returns the zip as bytes."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data, store in entries:
            z.writestr(name, data,
                       compress_type=zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def material_texcoord(root, mat_idx):
    """Which UV set does this material's base (or emissive) texture actually sample?"""
    mats = root.get("materials")
    if mat_idx is None or mat_idx < 0 or not mats or mat_idx >= len(mats):
        return 0
    m = mats[mat_idx]
    pbr = m.get("pbrMetallicRoughness")
    if pbr and "baseColorTexture" in pbr:
        return int(pbr["baseColorTexture"].get("texCoord", 0))
    if "emissiveTexture" in m:
        return int(m["emissiveTexture"].get("texCoord", 0))
    return 0


def convert(input_path, output_path=None, audio_path=None):
    """The conversion. Returns a dict of stats. Raises ConvertError on failure."""
    if not input_path:
        raise ConvertError("no input file")
    low = input_path.lower()
    if not (low.endswith(".glb") or low.endswith(".gltf")):
        raise ConvertError("input must be .glb or .gltf")
    if not output_path:
        output_path = os.path.splitext(input_path)[0] + ".apk"

    root, buffers, base_dir = load_container(input_path)
    stats = {"meshes": 0, "textures": 0, "uv_fixes": 0, "audio": False, "bytes": 0,
             "output": output_path}

    # ── 1. Externalize every image to a real PNG/JPEG file inside the .ovrscene ──
    ovr = []                      # (name, data, store)
    image_bvs = set()
    for i, img in enumerate(root.get("images", [])):
        data, src_bv = image_bytes(root, buffers, base_dir, img, i)
        if not data:
            raise ConvertError("image[%d] has no readable data" % i)
        fmt = sniff_image(data)
        if fmt not in ("png", "jpg"):
            raise ConvertError(
                "unsupported texture format in image[%d]: %s — only PNG and JPEG are supported. "
                "Re-export the model with PNG/JPEG textures." % (i, fmt))
        fname = "tex%d.%s" % (i, fmt)
        ovr.append((fname, data, True))
        if src_bv >= 0:
            image_bvs.add(src_bv)
        root["images"][i] = {"uri": fname}
        stats["textures"] += 1

    # ── 2. Repack every non-image bufferView into one 4-byte-aligned V9.bin ──
    new_bin = bytearray()
    new_bvs = []
    bv_map = []
    old_bvs = root.get("bufferViews", [])
    if old_bvs:
        bv_map = [-1] * len(old_bvs)
        for i, bv in enumerate(old_bvs):
            if i in image_bvs:
                continue        # the image's bytes now live as a standalone file
            buf = int(bv["buffer"])
            off = int(bv.get("byteOffset", 0))
            length = int(bv["byteLength"])
            if buf < 0 or buf >= len(buffers) or off + length > len(buffers[buf]):
                raise ConvertError("bufferView[%d] out of range" % i)
            while len(new_bin) % 4:
                new_bin.append(0)
            new_off = len(new_bin)
            new_bin += buffers[buf][off:off + length]
            nv = {"buffer": 0, "byteOffset": new_off, "byteLength": length}
            if "byteStride" in bv:
                nv["byteStride"] = int(bv["byteStride"])
            if "target" in bv:
                nv["target"] = int(bv["target"])
            bv_map[i] = len(new_bvs)
            new_bvs.append(nv)

    root["bufferViews"] = new_bvs
    if "accessors" in root:
        remap_buffer_views(root["accessors"], bv_map)
    root["buffers"] = [{"byteLength": len(new_bin), "uri": "V9.bin"}]

    # ── 3. Fold each primitive's REAL UV set into TEXCOORD_0 (the only one the loader reads) ──
    meshes = root["meshes"]
    stats["meshes"] = len(meshes)
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes")
            if not attrs or "material" not in prim:
                continue
            tc = material_texcoord(root, int(prim["material"]))
            if tc <= 0:
                continue
            want = "TEXCOORD_%d" % tc
            if want in attrs:
                attrs["TEXCOORD_0"] = attrs[want]
                stats["uv_fixes"] += 1

    # ── 4. Nest the zips: .ovrscene -> scene.zip -> .apk ──
    gltf_json = json.dumps(root, separators=(",", ":")).encode("utf-8")
    ovr.append(("V9.gltf", gltf_json, False))
    ovr.append(("V9.bin", bytes(new_bin), False))
    ovr_zip = build_zip(ovr)

    scene = [("_WORLD_MODEL.gltf.ovrscene", ovr_zip, True)]
    if audio_path:
        ab = read_file(audio_path)
        if not ab:
            raise ConvertError("cannot read audio file: " + audio_path)
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in AUDIO_EXTS:
            raise ConvertError("audio must be .ogg/.wav/.mp3/.flac")
        scene.append(("_BACKGROUND_LOOP" + ext, ab, True))
        stats["audio"] = True
    scene_zip = build_zip(scene)

    apk_bytes = build_zip([("assets/scene.zip", scene_zip, True)])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(apk_bytes)
    except OSError as e:
        raise ConvertError("cannot write output: %s (%s)" % (output_path, e))
    stats["bytes"] = len(apk_bytes)
    return stats


def main(argv):
    ap = argparse.ArgumentParser(
        prog="GLBtoAPK.py",
        description="Convert a .glb/.gltf model into a Quest Home source APK.")
    ap.add_argument("input", help="the .glb or .gltf model to convert")
    ap.add_argument("-o", "--output", help="output .apk path (default: <input>.apk)")
    ap.add_argument("--audio", help="optional ambient loop (.ogg/.wav/.mp3/.flac)")
    args = ap.parse_args(argv)

    try:
        r = convert(args.input, args.output, args.audio)
    except ConvertError as e:
        sys.stderr.write("[GLBtoAPK] error: %s\n" % e)
        return 1
    except Exception as e:                                    # malformed glTF -> a clean message
        sys.stderr.write("[GLBtoAPK] error: %s: %s\n" % (type(e).__name__, e))
        return 1

    sys.stderr.write(
        "[GLBtoAPK] %s -> %s (%d meshes, %d textures, %d UV remaps, %s, %.1f MB)\n"
        % (os.path.basename(args.input), r["output"], r["meshes"], r["textures"],
           r["uv_fixes"], "audio" if r["audio"] else "no audio", r["bytes"] / 1048576.0))
    print(r["output"])                                        # stdout = the APK path, for callers
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
