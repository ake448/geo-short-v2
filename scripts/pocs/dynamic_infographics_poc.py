"""
dynamic_infographics_poc.py — Isolated POC for Gemini-designed infographics.

This script tests the "Primitive Command" approach: Gemini emits a list of 
simple drawing instructions (rect, text, line), and our code translates 
them into pixels via PIL. This avoids hardcoded templates.
"""
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor

# 1080x1920 CANVAS
W, H = 1080, 1920

def _draw_shadow(img: Image.Image, box: Tuple[int, int, int, int], radius: int, alpha: int = 80):
    """Draw a blurred drop shadow for a rectangle."""
    # Create a separate layer for the shadow
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    # Draw the shadow box (dark gray/black)
    sd.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, alpha))
    # Blur it
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    # Composite onto the main image
    return Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")

def _draw_gradient_rect(draw: ImageDraw.Draw, box: Tuple[int, int, int, int], color_hex: str, radius: int):
    """Draw a rectangle with a subtle top-to-bottom darkening gradient."""
    # Parse color
    c = ImageColor.getrgb(color_hex)
    x0, y0, x1, y1 = box
    h = y1 - y0
    
    # Draw base rect (rounded)
    draw.rounded_rectangle(box, radius=radius, fill=color_hex)
    
    # Simple line-by-line darkening overlay
    for row in range(y0, y1):
        frac = (row - y0) / h
        alpha = int(frac * 60) # Darken up to 60/255 at the bottom
        draw.line([x0, row, x1, row], fill=(0, 0, 0, alpha))


def _font(size: int, *, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, max(10, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, x: int, y: int, content: Any, color: str, size: int, align: str):
    align = (align or "left").lower()
    fnt = _font(size, bold=int(size) >= 48)
    if align == "center":
        draw.text((x, y), str(content), fill=color, font=fnt, anchor="ma", align="center")
    elif align == "right":
        draw.text((x, y), str(content), fill=color, font=fnt, anchor="ra", align="right")
    else:
        draw.text((x, y), str(content), fill=color, font=fnt, anchor="la")


def render_spec(spec: Dict[str, Any], out_path: Path):
    # Using Urban Atlas V2 base colors
    bg_color = "#0a0e27" 
    accent_yellow = "#FFD700"
    white = "#f5f7ff"

    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Optional: Draw a grid for POC visualization
    for x in range(0, W, 100):
        draw.line([x, 0, x, H], fill="#1a2046", width=1)
    for y in range(0, H, 100):
        draw.line([0, y, W, y], fill="#1a2046", width=1)

    instructions = spec.get("instructions", [])
    for inst in instructions:
        cmd = inst.get("cmd")
        args = inst.get("args", [])
        
        try:
            if cmd == "rect":
                # x, y, w, h, color_hex, radius, shadow=False, gradient=False
                x, y, w, h, color, radius = args[:6]
                has_shadow = args[6] if len(args) > 6 else False
                has_gradient = args[7] if len(args) > 7 else False
                
                box = (x, y, x+w, y+h)
                
                if has_shadow:
                    img = _draw_shadow(img, (x+6, y+6, x+w+6, y+h+6), radius)
                    draw = ImageDraw.Draw(img) # Re-draw on the new composite
                
                if has_gradient:
                    _draw_gradient_rect(draw, box, color, radius)
                else:
                    draw.rounded_rectangle(box, radius=radius, fill=color)
            
            elif cmd == "text":
                # x, y, content, color, size, align
                x, y, content, color, size, align = args
                _draw_text(draw, x, y, content, color, size, align)
                
            elif cmd == "line":
                # x1, y1, x2, y2, color, width
                x1, y1, x2, y2, color, width = args
                draw.line([x1, y1, x2, y2], fill=color, width=width)
                
            elif cmd == "icon":
                # x, y, name, color, size
                x, y, name, color, size = args
                # Placeholder for icon rendering (e.g. SVG or PNG shapes)
                draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], outline=color, width=max(3, size // 14))
                letter = str(name or "?")[0].upper()
                draw.text((x, y), letter, fill=color, font=_font(max(16, size // 2), bold=True), anchor="mm")
        except Exception as e:
            print(f"Error executing command {cmd}: {e}")

    img.save(out_path)
    return out_path

if __name__ == "__main__":
    test_cases = {
        "universal_comparison": {
            "instructions": [
                {"cmd": "text", "args": [100, 200, "COMPARISON", "#FFD700", 60, "left"]},
                # rect args: [x,y,w,h,hex,rad,shadow,gradient]
                {"cmd": "rect", "args": [100, 400, 880, 400, "#161b3d", 20, True, True]}, 
                {"cmd": "text", "args": [120, 450, "SUBJECT A", "#FFFFFF", 40, "left"]},
                {"cmd": "text", "args": [120, 520, "100.0", "#FFD700", 80, "left"]},
                {"cmd": "rect", "args": [100, 850, 400, 300, "#161b3d", 20, True, True]}, 
                {"cmd": "text", "args": [120, 880, "SUBJECT B", "#FFFFFF", 40, "left"]},
                {"cmd": "text", "args": [120, 950, "50.0", "#FFD700", 60, "left"]},
                {"cmd": "line", "args": [100, 1200, 980, 1200, "#FFD700", 4]},
                {"cmd": "text", "args": [540, 1300, "2.0X DIFFERENCE", "#FFD700", 100, "center"]}
            ]
        },
        "general_ranking": {
            "instructions": [
                {"cmd": "text", "args": [100, 200, "GLOBAL RANKINGS", "#FFD700", 60, "left"]},
                {"cmd": "rect", "args": [100, 350, 880, 1000, "#161b3d", 20, True, False]},
                {"cmd": "text", "args": [150, 450, "1. TOP TIER ITEM", "#FFFFFF", 50, "left"]},
                {"cmd": "text", "args": [150, 550, "2. SECONDARY ITEM", "#FFFFFF", 50, "left"]},
                {"cmd": "text", "args": [150, 650, "3. THIRD PLACE", "#FFFFFF", 50, "left"]},
                {"cmd": "line", "args": [150, 950, 850, 950, "#aab2d5", 2]},
                {"cmd": "text", "args": [150, 1100, "47. BOTTOM RANK", "#FFD700", 80, "left"]}
            ]
        },
        "geographic_highlight": {
            "instructions": [
                {"cmd": "text", "args": [100, 200, "REGIONAL FOCUS", "#FFD700", 60, "left"]},
                {"cmd": "rect", "args": [100, 300, 880, 800, "#161b3d", 10, False, True]}, # Map Area with gradient
                {"cmd": "text", "args": [540, 650, "(MAP DATA CONTEXT)", "#aab2d5", 30, "center"]},
                {"cmd": "icon", "args": [540, 550, "target", "#FFD700", 120]},
                {"cmd": "rect", "args": [100, 1150, 880, 250, "#161b3d", 15, True, True]},
                {"cmd": "text", "args": [140, 1220, "KEY STATISTIC", "#FFFFFF", 40, "left"]},
                {"cmd": "text", "args": [140, 1320, "1,234 UNITS", "#FFD700", 90, "left"]}
            ]
        }
    }
    
    for name, spec in test_cases.items():
        out = Path(f"poc_{name}.png")
        render_spec(spec, out)
        print(f"Rendered {name} to {out.absolute()}")
