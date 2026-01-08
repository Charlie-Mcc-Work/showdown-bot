#!/usr/bin/env python3
"""Generate placeholder icons for the browser extension."""

from pathlib import Path

def create_png_icon(size: int, output_path: Path):
    """Create a simple PNG icon with a Pokeball-inspired design."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fallback: create a minimal valid PNG manually
        create_minimal_png(size, output_path)
        return

    # Create image with transparency
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a Pokeball-inspired icon
    center = size // 2
    radius = size // 2 - 2

    # Red top half
    draw.pieslice(
        [center - radius, center - radius, center + radius, center + radius],
        180, 360,
        fill='#E53935'
    )

    # White bottom half
    draw.pieslice(
        [center - radius, center - radius, center + radius, center + radius],
        0, 180,
        fill='#FFFFFF'
    )

    # Center stripe
    stripe_height = size // 8
    draw.rectangle(
        [0, center - stripe_height // 2, size, center + stripe_height // 2],
        fill='#212121'
    )

    # Center circle (outer)
    circle_radius = size // 5
    draw.ellipse(
        [center - circle_radius, center - circle_radius,
         center + circle_radius, center + circle_radius],
        fill='#212121'
    )

    # Center circle (inner) - represents AI/coach
    inner_radius = size // 8
    draw.ellipse(
        [center - inner_radius, center - inner_radius,
         center + inner_radius, center + inner_radius],
        fill='#4FC3F7'  # Cyan for AI theme
    )

    # Save
    img.save(output_path, 'PNG')
    print(f"Created {output_path}")


def create_minimal_png(size: int, output_path: Path):
    """Create a minimal valid PNG without PIL (simple colored square)."""
    import struct
    import zlib

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)

    # IHDR
    ihdr_data = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)  # 8-bit RGB

    # IDAT - simple red/white pattern (Pokeball-ish)
    raw_data = b''
    for y in range(size):
        raw_data += b'\x00'  # Filter byte
        for x in range(size):
            # Simple circle check
            dx, dy = x - size//2, y - size//2
            dist_sq = dx*dx + dy*dy
            radius_sq = (size//2 - 2) ** 2

            if dist_sq > radius_sq:
                # Outside circle - transparent (black)
                raw_data += bytes([30, 30, 46])  # Dark background
            elif abs(dy) < size // 8:
                # Center stripe
                if dx*dx + dy*dy < (size//5)**2:
                    # Center button
                    raw_data += bytes([79, 195, 247])  # Cyan
                else:
                    raw_data += bytes([33, 33, 33])  # Dark
            elif dy < 0:
                # Top half - red
                raw_data += bytes([229, 57, 53])
            else:
                # Bottom half - white
                raw_data += bytes([255, 255, 255])

    compressed = zlib.compress(raw_data, 9)

    # Build PNG
    png = b'\x89PNG\r\n\x1a\n'
    png += png_chunk(b'IHDR', ihdr_data)
    png += png_chunk(b'IDAT', compressed)
    png += png_chunk(b'IEND', b'')

    output_path.write_bytes(png)
    print(f"Created {output_path} (minimal PNG)")


def main():
    extension_dir = Path(__file__).parent.parent / "extension"
    extension_dir.mkdir(exist_ok=True)

    create_png_icon(48, extension_dir / "icon48.png")
    create_png_icon(128, extension_dir / "icon128.png")

    print("\nIcons created successfully!")
    print(f"Location: {extension_dir}")


if __name__ == "__main__":
    main()
