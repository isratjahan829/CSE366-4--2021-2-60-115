import sys
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
d = Document(sys.argv[1]); st = d.styles
def style(name, size=None, bold=None, italic=None, color=(0, 0, 0), font='Times New Roman', align=None, spacing=None):
    try: s = st[name]
    except KeyError: return
    s.font.name = font
    rpr = s.element.get_or_add_rPr(); rf = rpr.find(qn('w:rFonts'))
    if rf is None: rf = OxmlElement('w:rFonts'); rpr.append(rf)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rf.set(qn(a), font)
    for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:cstheme', 'w:eastAsiaTheme'):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
    if size: s.font.size = Pt(size)
    if bold is not None: s.font.bold = bold
    if italic is not None: s.font.italic = italic
    if color is not None: s.font.color.rgb = RGBColor(*color)
    if align is not None: s.paragraph_format.alignment = align
    if spacing: s.paragraph_format.line_spacing = spacing
J = WD_ALIGN_PARAGRAPH.JUSTIFY
for n in ('Normal', 'Body Text', 'First Paragraph', 'Compact'):
    style(n, 12, align=J if n != 'Compact' else None, spacing=1.5 if n != 'Compact' else None)
style('Title', 17, True, align=WD_ALIGN_PARAGRAPH.CENTER); style('Subtitle', 12, False, True, (70, 70, 70), align=WD_ALIGN_PARAGRAPH.CENTER)
style('Heading 1', 14, True, False); style('Heading 2', 12.5, True, False); style('Heading 3', 12, True, True)
for n in ('Image Caption', 'Table Caption', 'Caption'):
    style(n, 10.5, False, False, align=J, spacing=1.15)
# tables: small font, borders
for t in d.tables:
    tblPr = t._tbl.tblPr
    b = OxmlElement('w:tblBorders')
    for e in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{e}'); el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4'); el.set(qn('w:color'), '808080'); b.append(el)
    tblPr.append(b)
    for i, row in enumerate(t.rows):
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0; p.paragraph_format.alignment = None
                for r in p.runs:
                    r.font.size = Pt(9); r.font.name = 'Times New Roman'
                    if i == 0: r.font.bold = True
for s in d.sections:
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(2.5)
d.save(sys.argv[2])
