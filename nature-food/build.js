const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, ShadingType, PageBreak, LineNumberRestartFormat, Footer, PageNumber,
  HeadingLevel,
} = require("docx");
const C = require("./content.js");
const FIG = "/home/user/CSE366-4--2021-2-60-115/nature-food/figures/";
const FONT = "Arial";

// ---- inline markup parser: **bold**, *italic*, ^sup^, [[placeholder]] ----
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("[[")) out.push(new TextRun({ text: "[" + t.slice(2, -2) + "]", shading: { type: ShadingType.CLEAR, fill: "FFF200", color: "auto" }, ...base }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 160, line: 480 }, ...opts.para });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---- word counts ----
const strip = (s) => s.replace(/\^[^^]+\^/g, "").replace(/\[\[[^\]]+\]\]/g, "").replace(/[*]/g, "");
const wc = (s) => strip(s).split(/\s+/).filter(Boolean).length;
const mainWords = C.sections.reduce((a, s) => a + s.paras.reduce((b, p) => b + wc(p), 0), 0);
const absWords = wc(C.abstract);
console.log("Abstract words:", absWords, "Main text words:", mainWords);

// ---- tables ----
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(text, w, { bold = false, shade = false, align = AlignmentType.LEFT, top = false, bottom = false } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: top ? border : none, bottom: bottom ? border : none, left: none, right: none },
    shading: shade ? { fill: "EEF3FA", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align, children: runs(text, { size: 17, bold }) })],
  });
}
function table(widths, header, rows, sectionRows = new Set()) {
  const total = widths.reduce((a, b) => a + b, 0);
  const hdr = new TableRow({ tableHeader: true, children: header.map((h, i) =>
    cell(h, widths[i], { bold: true, top: true, bottom: true, align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((v, i) =>
    cell(v, widths[i], { bold: sectionRows.has(ri), shade: sectionRows.has(ri), bottom: ri === rows.length - 1,
      align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] });
}
// Table 1: sample characteristics (report Table 5)
const t1rows = [
  ["**Age (years)**", "", "", "", ""],
  ["10–19", "3.5", "5.0", "5.9", "100.0"], ["20–29", "17.5", "29.2", "67.0", "–"], ["30–39", "28.6", "32.5", "22.9", "–"],
  ["40–49", "23.5", "20.5", "3.4", "–"], ["50–59", "14.9", "9.4", "0.3", "–"], ["≥60", "12.0", "3.4", "0.5", "–"],
  ["**Sex**", "", "", "", ""], ["Male", "62.0", "0.7", "1.2", "43.4"], ["Female", "38.0", "99.3", "98.8", "56.6"],
  ["**Residence**", "", "", "", ""], ["Rural", "58.4", "58.4", "65.1", "63.2"], ["Urban", "21.4", "21.4", "17.7", "19.5"], ["Slum", "20.2", "20.2", "17.2", "17.4"],
  ["**Division**", "", "", "", ""], ["Barisal", "9.2", "9.2", "10.0", "12.2"], ["Chattogram", "18.4", "18.4", "16.9", "15.2"],
  ["Dhaka", "22.8", "22.8", "22.4", "21.8"], ["Khulna", "14.5", "14.5", "19.5", "18.0"], ["Mymensingh", "11.2", "11.2", "11.7", "10.6"],
  ["Rajshahi", "10.4", "10.4", "7.4", "8.7"], ["Rangpur", "7.2", "7.2", "4.8", "7.2"], ["Sylhet", "6.4", "6.4", "7.4", "6.3"],
  ["**Education**", "", "", "", ""], ["No formal education", "28.7", "25.4", "7.5", "1.9"], ["Primary", "33.6", "32.9", "32.4", "38.6"],
  ["Secondary", "29.9", "35.8", "51.6", "57.3"], ["Higher secondary or above", "7.8", "5.9", "8.5", "2.2"],
];
const t1sec = new Set(t1rows.map((r, i) => (r[0].startsWith("**") ? i : -1)).filter((i) => i >= 0));
t1rows.forEach((r, i) => { if (t1sec.has(i)) r[0] = r[0].replace(/\*\*/g, ""); });
const table1 = table([3600, 1400, 1400, 1400, 1400], ["Characteristic", "Food purchasers (n = 5,000)", "Cooks (n = 5,000)", "Caregivers (n = 581)", "Adolescents (n = 2,473)"], t1rows, t1sec);

// Table 2: adjusted ORs (high vs low), CI from OR and SE (delta method)
const ci = (or, se) => { if (or == null) return "–"; const s = se / or; const lo = Math.exp(Math.log(or) - 1.96 * s), hi = Math.exp(Math.log(or) + 1.96 * s);
  const f = (x) => (x >= 10 ? x.toFixed(1) : x.toFixed(2)); return `${f(or)} (${f(lo)}–${f(hi)})`; };
const R = (label, p, c, a) => [label, ci(...p), ci(...c), ci(...a)];
const NA = [null, null];
const t2rows = [
  ["Education (ref. none)", "", "", ""],
  R("Primary", [3.932, 0.607], [3.040, 0.503], NA),
  R("Secondary", [9.749, 1.653], [5.736, 1.002], [5.045, 0.797]),
  R("Higher secondary or above", [41.38, 11.11], [23.22, 7.807], [28.57, 20.48]),
  ["Wealth quintile (ref. poorest)", "", "", ""],
  R("Poorer", [1.428, 0.244], [1.964, 0.323], [2.504, 0.604]),
  R("Middle", [1.324, 0.209], [1.906, 0.294], [1.696, 0.377]),
  R("Richer", [2.275, 0.444], [3.721, 0.74], [1.107, 0.278]),
  R("Richest", [2.509, 0.593], [4.833, 1.19], [3.332, 0.977]),
  ["Other characteristics", "", "", ""],
  R("Female (ref. male)", [0.846, 0.286], NA, [2.409, 0.347]),
  R("Urban (ref. rural)", [1.020, 0.171], [0.888, 0.159], [0.598, 0.125]),
  R("Adequate dietary diversity (ref. inadequate)", [0.888, 0.102], [0.889, 0.101], [1.036, 0.162]),
  R("Normal weight (ref. underweight)", [1.716, 0.317], [1.147, 0.19], [3.046, 0.514]),
  R("Overweight (ref. underweight)", [2.110, 0.454], [1.584, 0.297], [6.165, 3.667]),
];
const t2sec = new Set([0, 4, 9]);
const table2 = table([3900, 1800, 1800, 1700], ["Characteristic", "Food purchasers", "Cooks", "Adolescents"], t2rows, t2sec);

// ---- assemble ----
const children = [];
children.push(new Paragraph({ children: runs("Article", { size: 20, color: "52514E" }), spacing: { after: 120 } }));
children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(C.title)] }));
children.push(P(C.authors));
C.affiliations.forEach((a, i) => children.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: String(i + 1), superScript: true }), ...runs(" " + a, { size: 20 })] })));
children.push(P("Correspondence: " + C.corresponding, { para: { spacing: { before: 120, after: 120 } } }));
children.push(P("**Running title:** " + C.runningTitle));
children.push(P(`**Word counts:** abstract ${absWords}; main text (introduction, results and discussion) ${mainWords}; display items: 5 figures and 2 tables.`));
children.push(pageBreak());

children.push(H1("Abstract"));
children.push(P(C.abstract));
children.push(pageBreak());

C.sections.forEach((s, i) => {
  if (i === 0) children.push(H1("Main"));
  if (s.h) children.push(H1(s.h));
  if (s.h2) children.push(H2(s.h2));
  s.paras.forEach((p) => children.push(P(p)));
});

children.push(H1("Methods"));
C.methods.forEach((s) => { children.push(H2(s.h2)); s.paras.forEach((p) => children.push(P(p))); });

children.push(H1("Data availability")); children.push(P(C.dataAvailability));
children.push(H1("Code availability")); children.push(P(C.codeAvailability));

children.push(H1("References"));
C.references.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 100, line: 360 }, indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: `${i + 1}.\t` }), ...runs(r)], tabStops: [{ type: "left", position: 440 }] })));

children.push(H1("Acknowledgements")); children.push(P(C.acknowledgements));
children.push(H1("Author contributions")); children.push(P(C.contributions));
children.push(H1("Competing interests")); children.push(P(C.competing));
children.push(H1("Inclusion and ethics statement")); children.push(P(C.inclusion));

// Tables
children.push(pageBreak());
children.push(new Paragraph({ spacing: { after: 120 }, children: runs("**Table 1 | Characteristics of respondents, by role (%).**") }));
children.push(table1);
children.push(P("Weighted percentages. One respondent per role per household; the same person could be both food purchaser and cook. Caregivers were the primary caregivers of children aged 6–23 months.", { run: { size: 18 }, para: { spacing: { before: 120, line: 276 } } }));
children.push(pageBreak());
children.push(new Paragraph({ spacing: { after: 120 }, children: runs("**Table 2 | Factors associated with high basic nutrition knowledge (rural and urban samples).**") }));
children.push(table2);
children.push(P("Adjusted odds ratios (95% confidence intervals) for being in the highest versus lowest tertile of basic nutrition knowledge, from multinomial logistic regression models fitted separately for food purchasers (n = 2,836 with complete data), cooks (n = 2,843) and adolescents (n = 2,042). Models also included age group and division, and, for purchasers and cooks, self-rated health. Confidence intervals were derived from reported standard errors. –, not estimated (variable not in model, or sparse data). Cooks were 99.3% female, so sex was not modelled.", { run: { size: 18 }, para: { spacing: { before: 120, line: 276 } } }));

// Figures
C.figures.forEach((f) => {
  children.push(pageBreak());
  const width = 620, height = Math.round(width * f.h / f.w);
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + f.file), transformation: { width, height } })] }));
  children.push(new Paragraph({ spacing: { before: 160, after: 80, line: 360 }, children: runs("**" + f.title + "** " + f.legend) }));
});

const doc = new Document({
  creator: "Study team", title: C.title,
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 34, bold: true }, paragraph: { spacing: { before: 0, after: 240 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1300, right: 1300 } },
      lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS },
    },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children,
  }],
});
const out = "/home/user/CSE366-4--2021-2-60-115/nature-food/NatureFood_Manuscript_Final_Draft.docx";
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
