// Builds the Nature Food manuscript (.docx) from content.js, figures/ and analysis/microdata_results.json
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, ShadingType, PageBreak, LineNumberRestartFormat, Footer, PageNumber, HeadingLevel,
} = require("docx");
const ROOT = __dirname;
const C = require(path.join(ROOT, "content.js"));
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "analysis/microdata_results.json"), "utf8"));
const FIG = path.join(ROOT, "figures") + "/";
const FONT = "Arial";

function runs(text, base = {}) {
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|\[\[[^\]]+\]\])/g; let last = 0, m;
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
const strip = (s) => s.replace(/\^[^^]+\^/g, "").replace(/\[\[[^\]]+\]\]/g, "").replace(/[*]/g, "");
const wc = (s) => strip(s).split(/\s+/).filter(Boolean).length;
const mainWords = C.sections.reduce((a, s) => a + s.paras.reduce((b, p) => b + wc(p), 0), 0);
const absWords = wc(C.abstract);
console.log("Abstract words:", absWords, "Main text words:", mainWords);

// ---------- table helpers ----------
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(text, w, { bold = false, shade = false, align = AlignmentType.LEFT, top = false, bottom = false } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: top ? border : none, bottom: bottom ? border : none, left: none, right: none },
    shading: shade ? { fill: "EEF3FA", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align, children: runs(String(text), { size: 17, bold }) })],
  });
}
function table(widths, header, rows, sectionRows = new Set()) {
  const total = widths.reduce((a, b) => a + b, 0);
  const hdr = new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { bold: true, top: true, bottom: true, align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((v, i) => cell(v, widths[i], { bold: sectionRows.has(ri), shade: sectionRows.has(ri), bottom: ri === rows.length - 1, align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] });
}
const f2 = (x) => (x == null ? "–" : Number(x).toFixed(2));
const f1 = (x) => (x == null ? "–" : Number(x).toFixed(1));
const ci = (e, d = 2) => (e ? `${Number(e.est).toFixed(d)} (${Number(e.lo).toFixed(d)} to ${Number(e.hi).toFixed(d)})` : "–");
const pv = (p) => (p == null ? "" : p < 0.001 ? "<0.001" : Number(p).toFixed(3));
const note = (t) => P(t, { run: { size: 18 }, para: { spacing: { before: 120, line: 276 } } });
const tTitle = (t) => new Paragraph({ spacing: { after: 120 }, children: runs(t) });
const roles = [["purch", "Food purchasers"], ["cook", "Cooks"], ["adol", "Adolescents"]];

// ---------- Table 1: knowledge–diet ----------
const M2 = R.M2_knowledge_diet;
const t1 = []; const t1sec = new Set();
for (const [s, lab] of [["RU", "Rural–urban households"], ["Slum", "Slum households"]]) {
  t1sec.add(t1.length); t1.push([lab, "", "", "", ""]);
  for (const [r, n] of roles) {
    const x = M2[s][r];
    t1.push([`${n} (n = ${x.dds_adj.n.toLocaleString("en-US")})`, ci(x.dds_crude), ci(x.dds_adj), ci(x.mdd_adj_OR), ci(x.disc_adj_OR)]);
  }
}
const table1 = table([2500, 1800, 1800, 1700, 1800], ["Respondent", "Dietary diversity, crude (food groups)", "Dietary diversity, adjusted (food groups)", "Minimum dietary diversity, adjusted OR", "Discretionary foods, adjusted OR"], t1, t1sec);

// ---------- assemble ----------
const children = [];
children.push(new Paragraph({ children: runs("Article", { size: 20, color: "52514E" }), spacing: { after: 120 } }));
children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(C.title)] }));
children.push(P(C.authors));
C.affiliations.forEach((a, i) => children.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: String(i + 1), superScript: true }), ...runs(" " + a, { size: 20 })] })));
children.push(P("Correspondence: " + C.corresponding, { para: { spacing: { before: 120, after: 120 } } }));
children.push(P("**Running title:** " + C.runningTitle));
children.push(P(`**Word counts:** abstract ${absWords}; main text (introduction, results and discussion) ${mainWords}; display items: 5 figures and 1 table (placed in the text at first citation); Extended Data: 8 tables and 1 figure.`));
children.push(pageBreak());
children.push(H1("Abstract")); children.push(P(C.abstract)); children.push(pageBreak());
// figures and Table 1 are placed immediately after the paragraph that first cites them
const pngSize = (f) => { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; };
const placed = new Set();
function figureBlock(f) {
  const [w, h] = pngSize(FIG + f.file); const width = 615, height = Math.round(width * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 240, after: 80 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + f.file), transformation: { width, height } })] }),
    new Paragraph({ spacing: { before: 60, after: 280, line: 300 }, children: runs("**" + f.title + "** " + f.legend, { size: 18 }) }),
  ];
}
function table1Block() {
  return [new Paragraph({ spacing: { before: 240, after: 120 }, keepNext: true, children: runs("**Table 1 | Association of basic nutrition knowledge (per 10 points) with diet.**") }), table1,
    note("Estimates per 10-point higher knowledge score (0–100) with 95% confidence intervals. Dietary diversity: number of ten food groups consumed in the previous 24 hours (weighted linear regression). Minimum dietary diversity: five or more groups. Discretionary foods: savoury or fried snacks, sweets or sugar-sweetened beverages in the previous 24 hours (weighted logistic regression; OR, odds ratio). Adjusted models include wealth quintile, log food expenditure per person, food insecurity (HFIAS category), household internet access, division and (rural–urban only) urban residence. Standard errors clustered by primary sampling unit; rural–urban and slum samples analysed separately.")];
}
C.sections.forEach((sec, i) => {
  if (i === 0) children.push(H1("Main")); if (sec.h) children.push(H1(sec.h)); if (sec.h2) children.push(H2(sec.h2));
  sec.paras.forEach((p) => {
    children.push(P(p));
    for (const f of C.figures) {
      const re = new RegExp(f.ref.replace(".", "\\.") + "(?![0-9])");
      if (!placed.has(f.ref) && re.test(p)) { placed.add(f.ref); children.push(...figureBlock(f)); }
    }
    if (!placed.has("Table 1") && /Table 1(?![0-9])/.test(p.replace(/Extended Data Table 1/g, ""))) { placed.add("Table 1"); children.push(...table1Block()); }
  });
});
C.figures.forEach((f) => { if (!placed.has(f.ref)) console.warn("figure not cited:", f.ref); });
children.push(H1("Methods"));
C.methods.forEach((s) => { children.push(H2(s.h2)); s.paras.forEach((p) => children.push(P(p))); });
children.push(H1("Data availability")); children.push(P(C.dataAvailability));
children.push(H1("Code availability")); children.push(P(C.codeAvailability));
children.push(H1("References"));
C.references.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 100, line: 360 }, indent: { left: 440, hanging: 440 }, children: [new TextRun({ text: `${i + 1}.\t` }), ...runs(r)], tabStops: [{ type: "left", position: 440 }] })));
children.push(H1("Acknowledgements")); children.push(P(C.acknowledgements));
children.push(H1("Author contributions")); children.push(P(C.contributions));
children.push(H1("Competing interests")); children.push(P(C.competing));
children.push(H1("Inclusion and ethics statement")); children.push(P(C.inclusion));

// ================= Extended Data =================
children.push(pageBreak()); children.push(H1("Extended Data"));
// ED1 sample characteristics
const Dd = R.descriptives;
const e1 = []; const e1s = new Set();
const sec = (lab) => { e1s.add(e1.length); e1.push([lab, "", ""]); };
sec("Sample");
e1.push(["Households (primary sampling units)", `${Dd.RU.n_hh.toLocaleString("en-US")} (${Dd.RU.n_psu})`, `${Dd.Slum.n_hh.toLocaleString("en-US")} (${Dd.Slum.n_psu})`]);
e1.push(["Separate food purchaser interviewed, n", Dd.RU.bn_purch_n.toLocaleString("en-US"), Dd.Slum.bn_purch_n.toLocaleString("en-US")]);
e1.push(["Cook was also the main food purchaser, %", f1(Dd.RU.purch_is_cook_share), f1(Dd.Slum.purch_is_cook_share)]);
e1.push(["Adolescents interviewed, n", Dd.RU.bn_adol_n.toLocaleString("en-US"), Dd.Slum.bn_adol_n.toLocaleString("en-US")]);
sec("Household characteristics (%)");
for (const k of ["Secure", "Mild", "Moderate", "Severe"]) e1.push([`Food insecurity (HFIAS): ${k.toLowerCase()}`, f1(Dd.RU.hfias[k]), f1(Dd.Slum.hfias[k])]);
e1.push(["Household internet access", f1(Dd.RU.internet), f1(Dd.Slum.internet)]);
e1.push(["Receives any social-safety-net benefit", f1(Dd.RU.safety_net), f1(Dd.Slum.safety_net)]);
e1.push(["Handwashing place with water and soap (observed)", f1(Dd.RU.hw_station_water_soap), f1(Dd.Slum.hw_station_water_soap)]);
e1.push(["Food stored near chemicals (observed)", f1(Dd.RU.food_near_chem_obs), f1(Dd.Slum.food_near_chem_obs)]);
sec("Knowledge (score 0–100, mean; % scoring ≥60)");
for (const [r, n] of roles) e1.push([`Basic nutrition — ${n.toLowerCase()}`, `${f1(Dd.RU[`bn_${r}_mean`])}; ${f1(Dd.RU[`bn_${r}_ge60`])}`, `${f1(Dd.Slum[`bn_${r}_mean`])}; ${f1(Dd.Slum[`bn_${r}_ge60`])}`]);
e1.push(["Food safety — cooks", `${f1(Dd.RU.foodsafety_know)}; ${f1(Dd.RU.fs_ge60)}`, `${f1(Dd.Slum.foodsafety_know)}; ${f1(Dd.Slum.fs_ge60)}`]);
e1.push(["Cooking practices — cooks (mean)", f1(Dd.RU.cooking_know), f1(Dd.Slum.cooking_know)]);
e1.push(["Infant feeding — caregivers (mean)", f1(Dd.RU.iycf_know), f1(Dd.Slum.iycf_know)]);
sec("Diets, previous 24 h");
for (const [r, n] of roles) {
  e1.push([`${n}: dietary diversity, mean groups; % ≥5 groups`, `${f2(Dd.RU[`dds_${r}_mean`])}; ${f1(Dd.RU[`mdd_${r}`])}`, `${f2(Dd.Slum[`dds_${r}_mean`])}; ${f1(Dd.Slum[`mdd_${r}`])}`]);
  e1.push([`${n}: any discretionary food / SSB / sweets / snacks, %`, `${f1(Dd.RU[`anydisc_${r}`])} / ${f1(Dd.RU[`ssb_${r}`])} / ${f1(Dd.RU[`sweets_${r}`])} / ${f1(Dd.RU[`snacks_${r}`])}`, `${f1(Dd.Slum[`anydisc_${r}`])} / ${f1(Dd.Slum[`ssb_${r}`])} / ${f1(Dd.Slum[`sweets_${r}`])} / ${f1(Dd.Slum[`snacks_${r}`])}`]);
}
e1.push(["Children 6–23 months: minimum dietary diversity, %", f1(Dd.RU.child_mdd), f1(Dd.Slum.child_mdd)]);
children.push(tTitle("**Extended Data Table 1 | Sample characteristics, knowledge and diets (survey-weighted).**"));
children.push(table([5200, 2200, 2200], ["Characteristic", "Rural–urban", "Slum"], e1, e1s));
children.push(note("SSB, sugar-sweetened beverages (including sweetened tea). HFIAS, Household Food Insecurity Access Scale."));

// ED2 knowledge correlates
children.push(pageBreak());
const M1 = R.M1_knowledge_correlates_RU;
const terms = [["C(wealth_q)[T.2.0]", "Wealth quintile 2 (ref. poorest)"], ["C(wealth_q)[T.3.0]", "Wealth quintile 3"], ["C(wealth_q)[T.4.0]", "Wealth quintile 4"], ["C(wealth_q)[T.5.0]", "Wealth quintile 5 (richest)"],
  ["urban", "Urban (ref. rural)"], ["internet", "Household internet access"], ["msg30_{r}", "Saw nutrition/food-safety message, previous 30 days"],
  ["C(hfias_cat)[T.Mild]", "Mild food insecurity (ref. secure)"], ["C(hfias_cat)[T.Moderate]", "Moderate food insecurity"], ["C(hfias_cat)[T.Severe]", "Severe food insecurity"],
  ["C(division)[T.Chattogram]", "Chattogram (ref. Barishal)"], ["C(division)[T.Dhaka]", "Dhaka"], ["C(division)[T.Khulna]", "Khulna"], ["C(division)[T.Mymensingh]", "Mymensingh"],
  ["C(division)[T.Rajshahi]", "Rajshahi"], ["C(division)[T.Rangpur]", "Rangpur"], ["C(division)[T.Sylhet]", "Sylhet"]];
const e2 = terms.map(([t, lab]) => [lab, ...[["purch", "purch"], ["cook", "cook"], ["adol", "adol"], ["foodsafety", "cook"]].map(([k, r]) => ci(M1[k][t.replace("{r}", r)], 1))]);
children.push(tTitle("**Extended Data Table 2 | Correlates of knowledge scores (rural–urban households).**"));
children.push(table([2900, 1700, 1700, 1700, 1700], ["Characteristic", "Basic nutrition, purchasers", "Basic nutrition, cooks", "Basic nutrition, adolescents", "Food safety, cooks"], e2));
children.push(note("Adjusted differences in knowledge score (points, 0–100) with 95% confidence intervals from survey-weighted linear regression including all variables shown. Standard errors clustered by primary sampling unit."));

// ED3 effect modification
children.push(pageBreak());
const e3 = []; const e3s = new Set();
for (const [s, lab] of [["RU", "Rural–urban"], ["Slum", "Slum"]]) {
  e3s.add(e3.length); e3.push([lab, "", "", "", "", "", ""]);
  for (const [r, n] of roles) {
    const x = M2[s][r];
    e3.push([n, ci(x.slope_secure_mild), ci(x.slope_mod_severe), pv(x.interaction_p), f2(x.slope_nonpoor.est), ci(x.slope_poor), pv(x.poor_interaction_p)]);
  }
}
children.push(tTitle("**Extended Data Table 3 | Association of knowledge with dietary diversity by food security and wealth.**"));
children.push(table([1700, 1500, 1500, 900, 1100, 1500, 900], ["Respondent", "Food-secure or mild", "Moderate or severe insecurity", "P interaction", "Wealth Q3–Q5", "Wealth Q1–Q2", "P interaction"], e3, e3s));
children.push(note("Difference in dietary diversity (food groups) per 10-point higher knowledge within each stratum, with 95% confidence intervals, from survey-weighted linear models with a knowledge × stratum interaction, adjusted for the covariates in Table 1."));

// ED4 decomposition
children.push(pageBreak());
const dec = R.decomposition_RU; const labs = ["Knowledge", "Food insecurity", "Food expenditure", "Internet access", "Urban residence"];
const e4 = labs.map((l) => [l, `${f2(dec.cook.components[l])} (${f1(dec.cook.share_pct[l])}%)`, `${f2(dec.purch.components[l])} (${f1(dec.purch.share_pct[l])}%)`]);
e4.push(["Unexplained (including division)", `${f2(dec.cook.unexplained)} (${f1(100 * dec.cook.unexplained / dec.cook.gap)}%)`, `${f2(dec.purch.unexplained)} (${f1(100 * dec.purch.unexplained / dec.purch.gap)}%)`]);
e4.unshift(["Gap in dietary diversity, richest − poorest quintile", f2(dec.cook.gap), f2(dec.purch.gap)]);
children.push(tTitle("**Extended Data Table 4 | Decomposition of the rich–poor gap in dietary diversity (rural–urban households).**"));
children.push(table([4400, 2600, 2600], ["Component", "Cooks: food groups (% of gap)", "Purchasers: food groups (% of gap)"], e4));
children.push(note("Contribution of each factor = pooled regression coefficient × difference in weighted means between the richest and poorest wealth quintiles. Model includes knowledge (per 10 points), HFIAS score, log food expenditure per person, internet access, urban residence and division."));

// ED5 food safety practice & handwashing
children.push(pageBreak());
const FS = R.foodsafety_practice, HW = R.handwashing;
const e5 = []; const e5s = new Set();
e5s.add(e5.length); e5.push(["Observed practice", "", ""]);
e5.push(["Handwashing place with water and soap, %", f1(FS.RU.station_ws_overall), f1(FS.Slum.station_ws_overall)]);
e5.push(["  poorest / richest quintile, %", `${f1(FS.RU.station_ws_by_wealth["1"])} / ${f1(FS.RU.station_ws_by_wealth["5"])}`, `${f1(FS.Slum.station_ws_by_wealth["1"])} / ${f1(FS.Slum.station_ws_by_wealth["5"])}`]);
e5.push(["  adjusted OR, richest vs poorest quintile", ci(FS.RU.station_wealthQ5_OR), ci(FS.Slum.station_wealthQ5_OR)]);
e5.push(["  adjusted OR per 10-point food-safety knowledge", ci(FS.RU.station_OR_per10), ci(FS.Slum.station_OR_per10)]);
e5.push(["Food stored near chemicals: cook knows / does not know it should be avoided, %", `${f1(FS.RU.near_chem_obs_if_knows)} / ${f1(FS.RU.near_chem_obs_if_not)}`, `${f1(FS.Slum.near_chem_obs_if_knows)} / ${f1(FS.Slum.near_chem_obs_if_not)}`]);
e5s.add(e5.length); e5.push(["Handwashing with soap: knows the critical time / reports practice / estimated pre-pandemic, %", "", ""]);
for (const [k, lab] of [["after_latrine", "After defecation"], ["before_eating", "Before eating"], ["before_cooking", "Before cooking"], ["after_returning_home", "After returning home"]]) {
  const a = HW.RU[k], b = HW.Slum[k];
  e5.push([lab, `${f1(a.know)} / ${f1(a.report)} / ${f1(a.report * a.pre_share / 100)}`, `${f1(b.know)} / ${f1(b.report)} / ${f1(b.report * b.pre_share / 100)}`]);
}
children.push(tTitle("**Extended Data Table 5 | Food-safety knowledge, observed practice and handwashing.**"));
children.push(table([5200, 2200, 2200], ["Indicator", "Rural–urban", "Slum"], e5, e5s));
children.push(note("Observed indicators were recorded by enumerators. Estimated pre-pandemic prevalence = reported prevalence × share of practitioners reporting the practice predated the COVID-19 pandemic. OR, odds ratio (95% confidence interval) from survey-weighted logistic regression adjusted for wealth quintile, residence, division and drinking-water source."));

// ED6 item-level knowledge
children.push(pageBreak());
const IT = R.items_RU;
const itemLabels = { foodgroups: "Food groups in a balanced diet", unhealthy_foods: "Unhealthy foods (categories)", harms_ncd: "NCD consequences of unhealthy foods", balanced_needed: "Problems if balanced diet not eaten", energy_foods: "Energy-giving foods", bodybuilding_foods: "Body-building foods", protective_foods: "Protective foods", vitC_foods: "Vitamin C-rich foods", heard_anaemia: "Heard of iron-deficiency anaemia", vitA_foods: "Vitamin A-rich foods", iron_foods: "Iron-rich foods", arsenic_source: "Source of arsenic", pregnancy_diet: "Diet in pregnancy", lactation_diet: "Diet in lactation", trans_fat: "Trans-fatty acids", healthy_diet: "Features of a healthy diet" };
const e6 = Object.entries(itemLabels).map(([k, l]) => [l, ...roles.map(([r]) => `${f1(IT[r][k].mean_credit)} (${f1(IT[r][k].none)})`)]);
children.push(tTitle("**Extended Data Table 6 | Item-level basic nutrition knowledge (rural–urban households).**"));
children.push(table([3800, 1900, 1900, 1900], ["Item", "Food purchasers", "Cooks", "Adolescents"], e6));
children.push(note("Mean credit (%) with the percentage receiving no credit in parentheses. Survey-weighted."));

// ED7 scoring key
children.push(pageBreak());
const key = [
  ["Basic nutrition (16 items)", ""],
  ["Food groups in balanced diet", "Fraction of 8 groups named"],
  ["Unhealthy foods", "Fraction of 3 categories (salty/fried; sweet; sugar-sweetened beverages) with ≥1 item named"],
  ["NCD consequences of unhealthy foods", "Fraction of 4 (NCD/kidney failure, obesity, high blood pressure, diabetes)"],
  ["Problems if balanced diet not eaten; heard of anaemia", "Yes = 1"],
  ["Energy-giving / body-building / protective foods", "Fraction of correct groups: grains and potatoes, oils and fats / fish–meat–egg–pulses, milk / vegetables, fruits"],
  ["Vitamin C-rich foods", "Amlaki, lemon or orange = 1"],
  ["Vitamin A-rich foods", "Fraction of 3 correct options"],
  ["Iron-rich foods", "Fraction of 3 correct options (flesh foods, pulses, leafy vegetables) minus 1/3 if potato named"],
  ["Source of arsenic", "Shallow tubewell = 1"],
  ["Diet in pregnancy; diet in lactation", "Fraction of 6 recommended practices"],
  ["Trans-fatty acids", "'Bad for health' or 'should avoid' = 1; not heard = 0"],
  ["Features of a healthy diet", "Fraction of 12 dietary-guideline features"],
  ["Food safety (12 items, cooks)", ""],
  ["Store food away from chemicals; clean hands before cooking; wash before eating; separate raw and cooked; clean utensils", "Correct answer = 1"],
  ["Safety of vegetables; fruits", "Fraction of 3 correct practices"],
  ["Safety of cooked food; prevent contamination", "Fraction of 5 correct practices"],
  ["Utensil cleaning method", "Hot soapy water after each use, or soap and ash = 1"],
  ["Disease transmission sources", "Fraction of 6"],
  ["Reheating cooked food", "Once = 1"],
  ["Cooking practices (8 items, cooks)", ""],
  ["Keep rice gruel; wash vegetables whole; keep boiling water; lid on; eat within 1 h; cover food; avoid stale food; iodised salt", "Correct answer = 1"],
  ["Infant feeding (11 items, caregivers)", ""],
  ["Breastfeed after birth; initiate within 1 h; give colostrum; on demand; exclusive breastfeeding; increase breastfeeding if milk insufficient; no water <6 months; liquids and foods from 6 months; breastfeed ≥24 months; no other food in first days", "Correct answer = 1"],
];
const keyS = new Set(key.map((r, i) => (r[1] === "" ? i : -1)).filter((i) => i >= 0));
children.push(tTitle("**Extended Data Table 7 | Knowledge scoring key.**"));
children.push(table([4800, 4800], ["Item", "Credit"], key, keyS));
children.push(note("Domain score = mean item credit × 100. Basic nutrition scores required at least 14 of 16 items to be non-missing."));

// ED8 comparison with published report
children.push(pageBreak());
const cmp = [
  ["Cooks: reheat cooked food only once, %", "42.3", "42.1"],
  ["Cooks: know not to store food near chemicals, %", "47.8", "54.2"],
  ["Cooks: store raw and cooked food separately, %", "81.7", "85.1"],
  ["Cooks: wash vegetables whole before cutting, %", "24.6", "20.2"],
  ["Cooks: keep rice gruel, %", "75.1", "78.4"],
  ["Reported handwashing with soap before eating, %", "96.5", "97.8"],
  ["Food purchasers: inadequate dietary diversity, %", "57.3", "60.6"],
  ["Children 6–23 months: minimum dietary diversity, %", "40.7", "37.2"],
  ["Slum food purchasers: normal BMI, %", "11.3", "67.1"],
];
children.push(tTitle("**Extended Data Table 8 | Comparison with estimates published in the survey report.**"));
children.push(table([5600, 2000, 2000], ["Indicator (rural–urban unless stated)", "Survey report", "This analysis"], cmp));
children.push(note("Differences reflect the larger analytical sample (5,239 versus 5,000 households), harmonised indicator definitions (for example, WHO 2021 child dietary diversity including breast milk) and correction of a tabulation error in the report's slum anthropometry table."));

// ED Fig 1
children.push(pageBreak());
{ const f = "EDFig1_screening.png"; const b = fs.readFileSync(FIG + f); const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: b, transformation: { width: 615, height: Math.round(615 * h / w) } })] }));
  children.push(new Paragraph({ spacing: { before: 120, line: 300 }, children: runs("**Extended Data Fig. 1 | Performance of brief household screening models.** **a**, Cumulative share of affected households identified when households are screened in order of predicted risk, from penalised logistic regression with five-fold cross-validation grouped by primary sampling unit; the dashed line shows random selection and the vertical line the top 30%. AUC, area under the receiver operating characteristic curve. **b**, Calibration: weighted mean predicted risk against observed prevalence by decile of predicted risk. Predictors: 28 household characteristics recordable in about two minutes (Methods). Survey weights normalised within sample; rural–urban and slum households combined.", { size: 18 }) }));
}
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
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1300, right: 1300 } }, lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children,
  }],
});
const out = path.join(ROOT, "NatureFood_Manuscript_Final_Draft.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
