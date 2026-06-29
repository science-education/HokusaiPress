import json
from .model import Document, RegionKind

def export_markdown(document: Document) -> str:
    """
    Export the document content to Markdown format.
    Regions are sorted geometrically (y, then x) and output as text or image placeholders.
    """
    md_lines = []
    
    for i, p in enumerate(document.pages):
        page_num_str = str(p.page_number) if p.page_number is not None else "Unnumbered"
        md_lines.append(f"## Page {i} ({page_num_str})\n")
        
        # Sort regions roughly by Y coordinate.
        # If there are layout_labels or more complex logical ordering, we would use them here.
        regions = sorted(p.regions, key=lambda r: (r.box.y0, r.box.x0))
        
        for r in regions:
            if r.kind == RegionKind.TEXT:
                text = r.ocr_text or ""
                if text.strip():
                    md_lines.append(f"{text}\n")
            elif r.kind in (RegionKind.FIGURE, RegionKind.PHOTO):
                box_info = f"({r.box.x0:.1f}, {r.box.y0:.1f}) - ({r.box.x1:.1f}, {r.box.y1:.1f})"
                kind_str = "Figure" if r.kind == RegionKind.FIGURE else "Photo"
                md_lines.append(f"![{kind_str} at {box_info}](image_layer_placeholder)\n")
        
        md_lines.append("\n---\n")
        
    return "\n".join(md_lines)
