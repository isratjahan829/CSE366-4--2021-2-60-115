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
children.push(P(`**Word counts:** abstract ${absWords}; main text (introduction, results and discussion) ${mainWords}; display items: 5 figures and 1 table; Extended Data: 6 tables.`));
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
children.push(new Paragraph({ spacing: { after: 120 }, children: runs("**Table 1 | Factors associated with high basic nutrition knowledge (rural and urban samples).**") }));
children.push(table2);
children.push(P("Adjusted odds ratios (95% confidence intervals) for being in the highest versus lowest tertile of basic nutrition knowledge, from multinomial logistic regression models fitted separately for food purchasers (n = 2,836 with complete data), cooks (n = 2,843) and adolescents (n = 2,042). Models also included age group and division, and, for purchasers and cooks, self-rated health. Confidence intervals were derived from reported standard errors. –, not estimated (variable not in model, or sparse data). Cooks were 99.3% female, so sex was not modelled.", { run: { size: 18 }, para: { spacing: { before: 120, line: 276 } } }));

// Figures
C.figures.forEach((f) => {
  children.push(pageBreak());
  const width = 620, height = Math.round(width * f.h / f.w);
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + f.file), transformation: { width, height } })] }));
  children.push(new Paragraph({ spacing: { before: 160, after: 80, line: 360 }, children: runs("**" + f.title + "** " + f.legend) }));
});


// ================= Extended Data =================
const SR = JSON.parse(fs.readFileSync("/home/user/CSE366-4--2021-2-60-115/nature-food/analysis/secondary_results.json", "utf8"));
const note = (t) => P(t, { run: { size: 18 }, para: { spacing: { before: 120, line: 276 } } });
const edTitle = (t) => new Paragraph({ spacing: { after: 120 }, children: runs(t) });
const fmtCI = (a) => `${a[0].toFixed(1)} (${a[1].toFixed(1)} to ${a[2].toFixed(1)})`;
children.push(pageBreak());
children.push(H1("Extended Data"));
// ED1
children.push(edTitle("**Extended Data Table 1 | Characteristics of respondents, by role (%).**"));
children.push(table1);
children.push(note("Weighted percentages. One respondent per role per household; the same person could be both food purchaser and cook. Caregivers were the primary caregivers of children aged 6–23 months."));
// ED2 inequality
children.push(pageBreak());
children.push(edTitle("**Extended Data Table 2 | Slope and relative indices of inequality in basic nutrition knowledge (rural and urban samples).**"));
const gl = { purchasers: "Food purchasers", cooks: "Cooks", adolescents: "Adolescents" };
const ed2 = [];
for (const g of ["purchasers", "cooks", "adolescents"]) {
  const q = SR.inequality[g];
  ed2.push([gl[g], q.edu_SII_high[0].toFixed(1), q.edu_SII_high[1].toFixed(1), q.wealth_SII_high[0].toFixed(1), q.wealth_SII_high[1].toFixed(1),
    q.edu_SII_low[0].toFixed(1), q.wealth_SII_low[0].toFixed(1)]);
}
children.push(table([1900, 1250, 1250, 1250, 1250, 1250, 1250], ["Respondent type", "Education SII, high (pp)", "Education RII, high", "Wealth SII, high (pp)", "Wealth RII, high", "Education SII, low (pp)", "Wealth SII, low (pp)"], ed2));
children.push(note("SII, slope index of inequality: absolute difference (percentage points, pp) in the predicted share of respondents in the highest (\"high\") or lowest (\"low\") knowledge tertile between the top and bottom of the education or wealth distribution. RII, relative index of inequality: ratio of predicted prevalence at the top to that at the bottom. Estimated by weighted least-squares regression of group prevalence on ridit scores (Methods). Negative SII for \"low\" indicates that low knowledge is concentrated at the bottom of the distribution."));
// ED3 knowledge–diet
children.push(pageBreak());
children.push(edTitle("**Extended Data Table 3 | Association between basic nutrition knowledge and diet (rural and urban samples).**"));
const ed3 = [];
for (const g of ["purchasers", "cooks", "adolescents"]) {
  const k = SR.knowledge_diet_RD[g];
  ed3.push([gl[g], `${k.n_adequate} / ${k.n_inadequate}`, fmtCI(k.RD_high_adequate_vs_inadequate), fmtCI(k.RD_low_adequate_vs_inadequate), fmtCI(k.RD_high_discretionary_vs_not),
    `${SR.MDE_OR[g][0].toFixed(2)} / ${SR.MDE_OR[g][1].toFixed(2)}`]);
}
const po = SR.pooled_OR_dd;
ed3.push(["Pooled adjusted OR (adequate dietary diversity)", "", `${po.OR.toFixed(2)} (${po.lo.toFixed(2)} to ${po.hi.toFixed(2)})`, `I² = ${po.I2}%; Q = ${po.Q}`, "", `${SR.MDE_pooled[0].toFixed(2)} / ${SR.MDE_pooled[1].toFixed(2)}`]);
children.push(table([1900, 1250, 1600, 1600, 1600, 1450], ["Respondent type", "n adequate / inadequate diversity", "RD in high knowledge, adequate vs inadequate diversity (pp, 95% CI)", "RD in low knowledge, adequate vs inadequate diversity (pp, 95% CI)", "RD in high knowledge, discretionary vs none (pp, 95% CI)", "Minimum detectable OR (80% power)"], ed3));
children.push(note("RD, risk difference in the percentage of respondents in the highest or lowest knowledge tertile. Confidence intervals inflate binomial variance by a design effect of 2.5. The pooled odds ratio combines the adjusted odds ratios for adequate dietary diversity (high versus low knowledge) from the three respondent-specific multinomial models by inverse-variance weighting (Methods). Minimum detectable odds ratios (lower / upper) are for two-sided α = 0.05 and 80% power given each model's standard error."));
// ED4 handwashing
children.push(pageBreak());
children.push(edTitle("**Extended Data Table 4 | Handwashing with soap at critical times: COVID-19-attributed adoption and estimated pre- and post-pandemic prevalence (%).**"));
const hwRaw = { rural_urban: { "After defecation": [92.6, 7.1, 99.6], "Before eating": [80.9, 18.8, 99.7], "After cleaning child's faeces": [86.8, 13.7, 93.1], "Before cooking": [81.1, 18.7, 98.1], "After returning home": [44.3, 54.3, 98.6], "Before feeding a child": [76.7, 29.7, 100.0] },
  slum: { "After defecation": [90.4, 9.6, 97.9], "Before eating": [73.5, 26.1, 100.0], "After cleaning child's faeces": [82.9, 20.7, 100.0], "Before cooking": [79.0, 20.2, 100.0], "After returning home": [28.2, 70.1, 99.4], "Before feeding a child": [74.3, 30.8, 100.0] } };
const ed4 = []; const ed4sec = new Set();
for (const [st, lbl] of [["rural_urban", "Rural–urban (n = 3,989)"], ["slum", "Slum (n = 1,011)"]]) {
  ed4sec.add(ed4.length); ed4.push([lbl, "", "", "", "", "", ""]);
  for (const [k, v] of Object.entries(SR.handwashing[st])) {
    const r = hwRaw[st][k];
    ed4.push([k, v.current.toFixed(1), r[0].toFixed(1), r[1].toFixed(1), r[2].toFixed(1), v.pre.toFixed(1), `${v.gain_pp.toFixed(1)}`]);
  }
}
children.push(table([2300, 1100, 1300, 1300, 1300, 1200, 1100], ["Critical time", "Current", "Of practitioners: usual before pandemic", "Of practitioners: adopted due to COVID-19 messaging", "Of adopters: will continue", "Estimated pre-pandemic", "Change (pp)"], ed4, ed4sec));
children.push(note("Current prevalence and the three conditional shares are as reported by respondents. Estimated pre-pandemic prevalence = current × share whose practice predated the pandemic; where the two conditional shares summed to more than 100% they were rescaled (Methods). Projected post-pandemic prevalence was within 1 percentage point of current prevalence for every item because almost all adopters intended to continue."));
// ED5 slum vs RU
children.push(pageBreak());
children.push(edTitle("**Extended Data Table 5 | Diets and hygiene in slum versus rural–urban households.**"));
const ed5 = Object.entries(SR.slum_vs_ru).map(([k, v]) => [k, v.ru.toFixed(1), v.slum.toFixed(1), fmtCI(v.RD)]);
children.push(table([4300, 1500, 1500, 2300], ["Indicator", "Rural–urban (%)", "Slum (%)", "Difference, slum − rural–urban (pp, 95% CI)"], ed5));
children.push(note("Sample sizes: adults, rural–urban 3,989 and slum 1,011; adolescents, 2,044 and 429; children aged 6–23 months, 481 and 100. Confidence intervals inflate binomial variance by a design effect of 2.5. Discretionary foods: savoury or fried snacks, sweets or sugar-sweetened beverages in the previous 24 hours."));
// ED6 IYCF
children.push(pageBreak());
children.push(edTitle("**Extended Data Table 6 | Infant and young child feeding knowledge of caregivers and diets of children aged 6–23 months (%).**"));
const ed6 = [
  ["Caregiver knowledge (rural–urban, n = 481): answered incorrectly", "", ""],
  ["How often should a baby breastfeed?", "42.0", "–"],
  ["How long after birth should breastfeeding start?", "35.3", "–"],
  ["Should infants <6 months be given water in hot weather?", "34.5", "–"],
  ["What to do if breast milk seems insufficient? (incorrect / partially correct)", "20.8 / 76.9", "–"],
  ["What to do with colostrum?", "2.3", "–"],
  ["Up to what age should a child be breastfed?", "15.6", "–"],
  ["Child food groups consumed in previous 24 h", "Rural–urban (n = 481)", "Slum (n = 100)"],
  ["Breast milk", "87.9", "89.0"], ["Grains, white roots and tubers", "88.4", "90.0"], ["Legumes and nuts", "45.9", "46.0"],
  ["Dairy", "23.1", "11.0"], ["Flesh foods", "44.3", "39.0"], ["Eggs", "40.9", "28.0"],
  ["Vitamin A-rich fruits and vegetables", "33.3", "27.0"], ["Other fruits and vegetables", "33.5", "30.0"],
  ["Minimum dietary diversity (≥5 of 8 groups)", "40.7", "31.0"],
];
children.push(table([5200, 2200, 2200], ["Item", "Rural–urban", "Slum"], ed6, new Set([0, 7])));
children.push(note("In slums, 48.0%, 36.0% and 46.0% of caregivers answered incorrectly on breastfeeding frequency, timing of initiation and water for young infants, respectively. Caregiver knowledge level was associated with division only (high knowledge in Chattogram versus Barisal, odds ratio 3.6)."));

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
