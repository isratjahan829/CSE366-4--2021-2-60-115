import copy, re
import docx
from docx.shared import Inches
from docx.oxml.ns import qn
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def ptext(el):
    return "".join(x.text or "" for x in el.iter(qn("w:t")))


def find_p(doc, startswith, after=None):
    els = list(doc.element.body.iterchildren())
    start = els.index(after) + 1 if after is not None else 0
    for el in els[start:]:
        if el.tag == qn("w:p") and ptext(el).strip().startswith(startswith):
            return el
    raise KeyError(startswith)


def set_text(p, text, bold=None):
    runs = p.findall(qn("w:r"))
    rpr = None
    if runs:
        r0 = runs[0].find(qn("w:rPr"))
        rpr = copy.deepcopy(r0) if r0 is not None else None
    for r in list(p):
        if r.tag in (qn("w:r"), qn("w:hyperlink"), qn("w:ins"), qn("w:del"), qn("w:proofErr"), qn("w:bookmarkStart"), qn("w:bookmarkEnd")):
            p.remove(r)
    add_runs(p, text, rpr, bold)
    return p


def add_runs(p, text, rpr=None, bold=None):
    # **bold** segments supported
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        b = part.startswith("**")
        t = part[2:-2] if b else part
        r = etree.SubElement(p, qn("w:r"))
        rp = copy.deepcopy(rpr) if rpr is not None else etree.Element(qn("w:rPr"))
        if b or bold:
            if rp.find(qn("w:b")) is None:
                rp.insert(0, etree.Element(qn("w:b")))
        r.append(rp)
        te = etree.SubElement(r, qn("w:t"))
        te.text = t
        te.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def clone_after(ref, template, text):
    new = copy.deepcopy(template)
    set_text(new, text)
    ref.addnext(new)
    return new


def make_table(rows, widths, bold_rows=()):
    total = sum(widths)
    tbl = etree.Element(qn("w:tbl"))
    pr = etree.SubElement(tbl, qn("w:tblPr"))
    etree.SubElement(pr, qn("w:tblW"), {qn("w:w"): str(total), qn("w:type"): "dxa"})
    etree.SubElement(pr, qn("w:jc"), {qn("w:val"): "center"})
    b = etree.SubElement(pr, qn("w:tblBorders"))
    for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        etree.SubElement(b, qn(f"w:{side}"), {qn("w:val"): "single", qn("w:sz"): "4", qn("w:space"): "0", qn("w:color"): "auto"})
    mar = etree.SubElement(pr, qn("w:tblCellMar"))
    etree.SubElement(mar, qn("w:left"), {qn("w:w"): "60", qn("w:type"): "dxa"})
    etree.SubElement(mar, qn("w:right"), {qn("w:w"): "60", qn("w:type"): "dxa"})
    grid = etree.SubElement(tbl, qn("w:tblGrid"))
    for w in widths:
        etree.SubElement(grid, qn("w:gridCol"), {qn("w:w"): str(w)})
    for ri, row in enumerate(rows):
        tr = etree.SubElement(tbl, qn("w:tr"))
        if ri == 0:
            trpr = etree.SubElement(tr, qn("w:trPr")); etree.SubElement(trpr, qn("w:tblHeader"))
        for ci, (val, w) in enumerate(zip(row, widths)):
            tc = etree.SubElement(tr, qn("w:tc"))
            tcpr = etree.SubElement(tc, qn("w:tcPr"))
            etree.SubElement(tcpr, qn("w:tcW"), {qn("w:w"): str(w), qn("w:type"): "dxa"})
            if ri == 0:
                etree.SubElement(tcpr, qn("w:shd"), {qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): "D9D9D9"})
            p = etree.SubElement(tc, qn("w:p"))
            ppr = etree.SubElement(p, qn("w:pPr"))
            etree.SubElement(ppr, qn("w:spacing"), {qn("w:before"): "20", qn("w:after"): "20"})
            etree.SubElement(ppr, qn("w:jc"), {qn("w:val"): "left" if ci == 0 else "center"})
            rpr = etree.Element(qn("w:rPr"))
            etree.SubElement(rpr, qn("w:sz"), {qn("w:val"): "18"})
            etree.SubElement(rpr, qn("w:szCs"), {qn("w:val"): "18"})
            add_runs(p, str(val), rpr, bold=(ri == 0 or ri in bold_rows))
    return tbl


def add_image_after(doc, ref, path, width_in=6.0, center_template=None):
    para = doc.add_paragraph()
    para.alignment = 1
    para.add_run().add_picture(path, width=Inches(width_in))
    el = para._p
    ref.addnext(el)
    return el
