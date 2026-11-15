import docx, sys
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

FIG = "/home/user/CSE366-4--2021-2-60-115/HyperDiag-Thesis/results/figures/"
OUT = sys.argv[1]
d = docx.Document()
sec = d.sections[0]
sec.left_margin = sec.right_margin = Inches(0.9); sec.top_margin = sec.bottom_margin = Inches(0.8)

def set_font(style_or_run, size=None, bold=None, color=None):
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run.element, "get_or_add_rPr") else style_or_run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), "Nirmala UI")
    lang = rpr.find(qn("w:lang"))
    if lang is None: lang = OxmlElement("w:lang"); rpr.append(lang)
    lang.set(qn("w:val"), "en-US"); lang.set(qn("w:bidi"), "bn-BD")

for sname, sz in [("Normal", 11), ("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 12), ("Title", 22)]:
    st = d.styles[sname]; st.font.size = Pt(sz)
    st.element.get_or_add_rPr()
    rpr = st.element.rPr
    rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), "Nirmala UI")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
    szcs = OxmlElement("w:szCs"); szcs.set(qn("w:val"), str(int(sz * 2))); rpr.append(szcs)
    if sname.startswith("Heading") or sname == "Title":
        st.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)
d.styles["Normal"].paragraph_format.space_after = Pt(5)
d.styles["Normal"].paragraph_format.line_spacing = 1.2

def P(text, bold_lead=None, style=None, italic=False, indent=None):
    p = d.add_paragraph(style=style)
    if indent: p.paragraph_format.left_indent = Inches(indent)
    parts = text.split("**")
    for i, t in enumerate(parts):
        if not t: continue
        r = p.add_run(t); r.bold = (i % 2 == 1); r.italic = italic
    return p

def B(text):  # bullet
    return P(text, style="List Bullet")

def H(text, lvl): return d.add_heading(text, lvl)

def box(title, text, fill="EAF2FB"):
    t = d.add_table(rows=1, cols=1); t.style = "Table Grid"
    c = t.cell(0, 0)
    tcpr = c._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill); tcpr.append(shd)
    c.paragraphs[0].add_run(title).bold = True
    for line in text.split("\n"):
        pp = c.add_paragraph(); parts = line.split("**")
        for i, s in enumerate(parts):
            if s: pp.add_run(s).bold = (i % 2 == 1)
    d.add_paragraph()

def table(rows, widths=None, bold_last=False):
    t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""
            r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(9.5)
            if i == 0 or (bold_last and i == len(rows) - 1):
                r.bold = True
            if i == 0:
                tcpr = c._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "D9E2F3"); tcpr.append(shd)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows: row.cells[j].width = Inches(w)
    d.add_paragraph()

def fig(name, width=6.2, cap=None):
    d.add_picture(FIG + name, width=Inches(width))
    d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if cap: P(cap, italic=True).alignment = WD_ALIGN_PARAGRAPH.CENTER

def item(label, what, how, result, meaning, viva=None):
    H(label, 3)
    P("**কী দেখায়:** " + what)
    P("**কীভাবে পড়বে:** " + how)
    P("**তোমার রেজাল্ট:** " + result)
    P("**এর মানে:** " + meaning)
    if viva: box("ভাইভায় বলার লাইন (English)", viva, "FFF4E0")

# ---------------------------------------------------------------- title
d.add_heading("HyperDiag থিসিস: Results অংশের সহজ বাংলা গাইড", 0)
P("এই গাইডে ড্রাফটের (HyperDiag_PreDefence_Draft_v2.docx) প্রতিটা টেবিল আর ফিগার একটা একটা করে সহজ বাংলায় ব্যাখ্যা করা হয়েছে। "
  "প্রতিটার জন্য চারটা জিনিস আছে: **কী দেখায়**, **কীভাবে পড়বে**, **তোমার রেজাল্ট** আর **এর মানে**। "
  "সাথে ভাইভায় বলার জন্য একটা ছোট ইংরেজি লাইন। টেবিল আর ফিগারের নম্বর ড্রাফটের সাথে হুবহু মেলে।")
box("সবচেয়ে আগে এই ৫টা কথা মনে রাখো",
    "১. ডেটাসেটে ২৩% রেকর্ড ডুপ্লিকেট। সাধারণ random split করলে test-এর **৪৩.৯%** কেস train-এও থাকে, তাই আগের কাজগুলোর accuracy বেশি দেখায়।\n"
    "২. আমরা এমনভাবে ভাগ করেছি যে কোনো লক্ষণ-প্যাটার্ন train আর test দুই জায়গায় পড়ে না (leakage = ০%)।\n"
    "৩. একই লক্ষণে একাধিক রোগ থাকায় কোনো মডেল top-1-এ **৮৯.৮৯%**-এর বেশি পেতে পারবে না (Bayes ceiling)।\n"
    "৪. HyperDiag: Top-1 **৮৪.৫৭%**, Top-5 **৯৭.৮২%**, MRR **৯০.৩৬%**। Random Forest আর Logistic Regression-এর চেয়ে নিশ্চিতভাবে ভালো, "
    "অন্য নিউরাল মডেলগুলোর সমান।\n"
    "৫. HyperDiag-এর বিশেষ দিক: probability সবচেয়ে নির্ভরযোগ্য (temperature scaling-এর পর), ব্যাখ্যা দিতে পারে, আর এর ভুলগুলো ব্যাখ্যাযোগ্য।")

# ---------------------------------------------------------------- glossary
H("অংশ ০: আগে কিছু শব্দের মানে জেনে নাও", 1)
P("Results পড়তে এই শব্দগুলো বারবার আসবে। প্রথমবার পড়ার সময় এই অংশটা মন দিয়ে পড়ো, পরে দরকার হলে ফিরে এসো।")
table([["শব্দ", "সহজ মানে", "ভালো কোনটা"],
       ["Train / Validation / Test set", "Train দিয়ে মডেল শেখে, validation দিয়ে সেরা মডেল বাছা হয়, test দিয়ে শেষে পরীক্ষা। Test-এর রোগী মডেল আগে কখনো দেখেনি।", "—"],
       ["Top-1 accuracy", "মডেলের ১ নম্বর উত্তরটা কত % রোগীর ক্ষেত্রে সঠিক।", "বেশি"],
       ["Top-5 accuracy", "সঠিক রোগটা মডেলের প্রথম ৫টা উত্তরের মধ্যে কত % ক্ষেত্রে আছে। ডাক্তারের ‘সম্ভাব্য রোগের তালিকা’র মতো।", "বেশি"],
       ["MRR", "সঠিক রোগ তালিকার কত উপরে আছে। ১ নম্বরে থাকলে ১, ২ নম্বরে ০.৫, ৩ নম্বরে ০.৩৩ ... সব রোগীর গড়।", "বেশি"],
       ["NDCG@10", "MRR-এর মতোই, তবে প্রথম ১০টার মধ্যে অবস্থান ধরে নম্বর দেয়।", "বেশি"],
       ["Mean / Median rank", "সঠিক রোগ গড়ে / মাঝামাঝি কত নম্বরে আসে। ১ মানে একদম প্রথমে।", "কম"],
       ["Precision", "মডেল যখন বলে ‘রোগ X’, তখন কত % সত্যিই X।", "বেশি"],
       ["Recall", "যাদের আসলে রোগ X, তাদের কত % মডেল ধরতে পেরেছে।", "বেশি"],
       ["F1", "Precision আর Recall-এর মিশ্রণ (harmonic mean)। দুটোই ভালো হলে F1 ভালো।", "বেশি"],
       ["Macro-F1", "প্রতিটা রোগের F1 আলাদা করে বের করে গড়। বিরল আর সাধারণ রোগ সমান গুরুত্ব পায়, তাই বিরল রোগে খারাপ করলে এটা কমে যায়।", "বেশি"],
       ["Weighted-F1", "F1-এর গড়, তবে যে রোগের রোগী বেশি তার গুরুত্ব বেশি।", "বেশি"],
       ["Cohen's κ (kappa), MCC", "মডেল আর সঠিক উত্তরের মিল, যেটুকু ভাগ্যক্রমে মিলতে পারত সেটুকু বাদ দিয়ে।", "বেশি"],
       ["Log-loss", "মডেল সঠিক রোগকে কত সম্ভাবনা দিয়েছে তার জরিমানা। সঠিক রোগকে কম সম্ভাবনা দিলে বেশি জরিমানা।", "কম"],
       ["Brier score", "পূর্বাভাসের সম্ভাবনা আর আসল উত্তরের গড় বর্গ-দূরত্ব।", "কম"],
       ["ECE (calibration error)", "মডেল ‘৮০% নিশ্চিত’ বললে সত্যিই কি ৮০% বার ঠিক হয়? এই অমিলের মাপ।", "কম"],
       ["mean ± std", "৩টা আলাদা seed (র‍্যান্ডম শুরু) দিয়ে ৩ বার ট্রেন করে গড় ± কতটা ওঠানামা করে।", "std কম = স্থিতিশীল"],
       ["95% CI [a, b]", "Confidence interval: আসল মান ৯৫% নিশ্চয়তায় a থেকে b-এর মধ্যে।", "সরু = নিশ্চিত"],
       ["p-value", "পার্থক্যটা শুধু ভাগ্যক্রমে হওয়ার সম্ভাবনা। p < 0.05 হলে পার্থক্যকে ‘significant’ (আসল) ধরা হয়।", "ছোট p = আসল পার্থক্য"],
       ["Holm correction", "অনেকগুলো তুলনা একসাথে করলে ভুল করে ‘significant’ পাওয়ার সম্ভাবনা বাড়ে। Holm সেই ভুল ঠেকাতে p-value বাড়িয়ে দেয়।", "—"],
       ["Effect size", "পার্থক্যটা কত বড় (ছোট/মাঝারি/বড়)। p-value শুধু বলে পার্থক্য আছে কিনা; effect size বলে কতটা।", "বড় = গুরুত্বপূর্ণ"],
       ["Baseline", "তুলনার জন্য অন্য মডেল (Naive Bayes, Random Forest, LogReg, MLP, GCN, GAT, HGNN)।", "—"],
       ["Ablation", "মডেলের একটা একটা অংশ সরিয়ে দেখা ফলাফলে কী পরিবর্তন হয়, তাতে বোঝা যায় কোন অংশ কাজের।", "—"],
       ], [1.6, 4.0, 1.0])

# ---------------------------------------------------------------- chapter 3
H("অংশ ১: ডেটা নিয়ে ফলাফল (Chapter 3)", 1)
item("Table 1: Data audit",
     "ডেটাসেটের ‘স্বাস্থ্য পরীক্ষা’: কতগুলো রো, ডুপ্লিকেট, ফাঁকা মান, একই লক্ষণে একাধিক রোগ ইত্যাদি।",
     "প্রতিটা সারি একটা বৈশিষ্ট্য, পাশে তার মান।",
     "২,৪৬,৯৪৫টা রো, ৭৭৩টা রোগ, ৩৭৭টা লক্ষণ। কোনো ফাঁকা মান নেই। কিন্তু **৫৭,২৯৮টা রো (২৩.২%) হুবহু ডুপ্লিকেট**। "
     "**১২,৬৩৪টা লক্ষণ-প্যাটার্ন একাধিক রোগের সাথে আছে**। ৪৯টা লক্ষণ কখনো ব্যবহারই হয়নি। Bayes ceiling (ডুপ্লিকেট বাদে): Top-1 ৮৯.৫৮%, Top-5 ৯৯.৪৩%।",
     "ডেটা পরিষ্কার দেখালেও এর মধ্যে বড় সমস্যা লুকিয়ে আছে। একই লক্ষণে একাধিক রোগ থাকায় কোনো মডেলই ১০০% পেতে পারবে না।",
     "\"The audit showed 23% duplicate rows and 12,634 symptom patterns shared by several diseases, which caps the achievable top-1 accuracy at about 90%.\"")
item("Figure 2: Class distribution, symptoms per case",
     "তিনটা ছোট ছবি: (ক) প্রতিটা রোগে কতজন রোগী, (খ) একজন রোগীর কয়টা লক্ষণ, (গ) সবচেয়ে কমন ১৫টা লক্ষণ।",
     "(ক) বাঁ দিকে বেশি রোগীর রোগ, ডান দিকে কম; y-অক্ষ log scale। (খ) কলাম যত উঁচু, তত বেশি রোগীর ঐ সংখ্যক লক্ষণ।",
     "কিছু রোগে ১,২০০+ রোগী, কিছু রোগে মাত্র ১ জন (imbalance অনুপাত ১২১৯:১)। গড়ে একজন রোগীর ৫.৩টা লক্ষণ।",
     "রোগগুলোর রোগী-সংখ্যায় বিশাল অসমতা আছে, তাই বিরল রোগ ভালো শেখানো কঠিন। এজন্যই class weighting ব্যবহার করা হয়েছে।")
item("Table 2 ও Figure 3: Leakage under three protocols",
     "তিনভাবে ডেটা ভাগ করলে test-এর কত % কেস train-এ আগেই থাকে (leakage), আর তাতে accuracy কেমন দেখায়।",
     "(A) সাধারণ random split, (B) ডুপ্লিকেট বাদ দিয়ে random split, (C) আমাদের grouped split। ‘Leaked test cases’ যত বেশি, পরীক্ষা তত অসৎ।",
     "(A)-তে **৪৩.৯% leakage**, MLP-এর Top-1 ৮৬.০২%। (B)-তে ১৫.১% leakage। (C)-তে **০% leakage**, MLP-এর Top-1 ৮৪.০২%।",
     "সাধারণ পদ্ধতিতে প্রায় অর্ধেক test কেস আসলে মডেল আগেই দেখেছে, তাই accuracy বেশি দেখায়। "
     "(B)-তে accuracy (C)-র চেয়েও কম কারণ সেখানে একই লক্ষণের কিন্তু ভিন্ন রোগের কেসগুলো train আর test-এ ভাগ হয়ে যায়। "
     "এটাই তোমার থিসিসের সবচেয়ে বড় novelty: **সৎ মূল্যায়নের পদ্ধতি**।",
     "\"With a random split, 43.9% of test cases are exact copies of training cases. Our grouped split removes this leakage completely.\"")
fig("fig02_leakage_protocols.png", 6.0, "(Figure 3) বাঁয়ে: কোন পদ্ধতিতে কত leakage; ডানে: পদ্ধতি অনুযায়ী accuracy কেমন দেখায়")
item("Figure 4 ও Figure 5: Hypergraph",
     "Figure 4: প্রতিটা রোগ-hyperedge-এ কয়টা লক্ষণ, প্রতিটা লক্ষণ কয়টা hyperedge-এ আছে। Figure 5: পাঁচটা কাছাকাছি রোগের hypergraph-এর ছবি।",
     "Figure 5-এ রঙিন বর্গ = রোগ (hyperedge), বৃত্ত = লক্ষণ। গাঢ় বৃত্ত মানে লক্ষণটা একাধিক রোগে আছে। রেখা যত মোটা, ঐ রোগে লক্ষণটা তত সাধারণ।",
     "৬৫৮টা রোগ-hyperedge (গড়ে ৭.৭টা লক্ষণ) আর ২৮৪টা co-occurrence hyperedge (প্রতিটায় ~৫টা লক্ষণ)।",
     "একটা hyperedge একসাথে একটা রোগের সব লক্ষণকে বাঁধে। গাঢ় (shared) লক্ষণগুলোই রোগ চেনা কঠিন করে; সেখানে পুরো লক্ষণ-সেট একসাথে দেখা দরকার।")
fig("fig04_hypergraph_network.png", 5.6, "(Figure 5) বর্গ = রোগ, বৃত্ত = লক্ষণ; গাঢ় বৃত্ত = একাধিক রোগে থাকা লক্ষণ")

# ---------------------------------------------------------------- 4.1
H("অংশ ২: মূল তুলনা (Section 4.1)", 1)
item("Table 3: Comparative performance (সবচেয়ে গুরুত্বপূর্ণ টেবিল)",
     "৮টা মডেলের Top-1, Top-3, Top-5, Top-10, MRR আর Macro-F1, leakage-মুক্ত test set-এ। শেষ সারি Bayes ceiling (সর্বোচ্চ সম্ভব)।",
     "প্রতিটা সংখ্যা %; ‘± ’-এর পরের সংখ্যা ৩টা seed-এর ওঠানামা। Naive Bayes আর Random Forest-এ ± নেই কারণ ওগুলো একবারই চালানো।",
     "", "", None)
table([["মডেল", "Top-1", "Top-5", "MRR", "Macro-F1"],
       ["Naive Bayes", "84.31", "97.67", "90.15", "76.43"], ["Random Forest", "82.80", "97.36", "89.11", "71.02"],
       ["Logistic Regression", "84.24", "97.62", "90.08", "75.11"], ["MLP", "84.63", "97.84", "90.38", "81.71"],
       ["GCN", "84.48", "97.80", "90.28", "81.12"], ["GAT", "84.50", "97.84", "90.31", "81.35"],
       ["HGNN", "84.51", "97.82", "90.30", "81.28"], ["HyperDiag (তোমার মডেল)", "84.57", "97.82", "90.36", "81.16"],
       ["Bayes ceiling (সর্বোচ্চ সম্ভব)", "89.89", "99.55", "—", "—"]], [2.4, 1, 1, 1, 1])
P("**কীভাবে বুঝবে:**")
B("HyperDiag ১০০ জন রোগীর মধ্যে প্রায় **৮৫ জনের** ক্ষেত্রে প্রথম উত্তরেই সঠিক রোগ বলে, আর প্রায় **৯৮ জনের** ক্ষেত্রে সঠিক রোগটা প্রথম ৫টার মধ্যে থাকে।")
B("সর্বোচ্চ সম্ভব Top-1 হলো ৮৯.৮৯%। HyperDiag তার **৯৪.১%** পর্যন্ত পৌঁছেছে।")
B("সব নিউরাল মডেল (MLP, GCN, GAT, HGNN, HyperDiag) প্রায় একই জায়গায়: Top-1-এ পার্থক্য মাত্র ০.১৫ পয়েন্টের মধ্যে।")
B("Macro-F1-এ নিউরাল মডেলরা (~৮১%) Naive Bayes (৭৬%) আর Random Forest (৭১%)-এর চেয়ে অনেক ভালো। মানে বিরল রোগেও এরা ভালো করে।")
P("**এর মানে:** এই ডেটাসেট প্রায় ‘সর্বোচ্চ সীমায়’ পৌঁছে গেছে, তাই আর্কিটেকচার বদলে accuracy খুব একটা বাড়ে না। "
  "HyperDiag ক্লাসিক্যাল মডেলের চেয়ে ভালো এবং সেরা নিউরাল মডেলের সমান। এর আসল সুবিধা অন্য জায়গায় (নিচে Table 7, Figure 15–18 দেখো)।")
box("ভাইভায় বলার লাইন (English)", "\"HyperDiag reaches 84.6% top-1, 97.8% top-5 and 90.4% MRR, which is 94% of the attainable top-1 accuracy. "
    "It matches the strongest neural baselines and clearly outperforms the classical ones.\"", "FFF4E0")

item("Table 4, 5, 6: বিস্তারিত মেট্রিক (২০টা কলাম)",
     "Table 3-এর বাইরের আরও মেট্রিক: Table 4 = র‍্যাঙ্কিং (NDCG, mean/median rank), Table 5 = শ্রেণিবিভাগ (Precision, Recall, κ, MCC), "
     "Table 6 = সম্ভাবনার মান ও খরচ (Log-loss, Brier, ECE, প্যারামিটার, সময়)।",
     "প্রতিটা কলামে সেরা মান **বোল্ড**। ‘কম ভালো’ কলামগুলো (rank, log-loss, Brier, ECE, সময়) খেয়াল করো।",
     "HyperDiag: NDCG@10 ৯২.৫৭%, median rank **১** (অর্ধেকের বেশি রোগীর সঠিক রোগ একদম প্রথমে), mean rank ১.৪১, "
     "Macro-Precision ৮৩.০২%, Macro-Recall ৮২.৪০%, κ ও MCC ৮৪.৫২%। প্যারামিটার ৪.১৬ লাখ, ট্রেনিং ২৫৯ সেকেন্ড, ১০০০ রোগীর ফল দিতে ১৫.৫ মিলিসেকেন্ড।",
     "সব মেট্রিকেই একই ছবি: HyperDiag সেরা নিউরাল মডেলগুলোর সমান। ট্রেনিংয়ে একটু বেশি সময় লাগে, কিন্তু ব্যবহারের সময় খুব দ্রুত।")
item("Figure 6: Multi-view performance profile",
     "চারটা ছবি একসাথে: (a) radar chart, (b) CMC curve, (c) Macro-F1 ও তার confidence interval, (d) accuracy বনাম গতি।",
     "(a) বাইরের দিকে থাকা মানে ভালো; মানগুলো মডেলগুলোর মধ্যে ০ থেকে ১-এ রূপান্তর করা, তাই ছোট পার্থক্যও বড় দেখায়। "
     "(b) x-অক্ষ = তালিকার দৈর্ঘ্য k, y = Top-k accuracy; লাইন যত উপরে তত ভালো। (c) বিন্দু = মান, দাগ = 95% CI। (d) উপরে-বাঁয়ে থাকা সবচেয়ে ভালো (বেশি accurate, দ্রুত)।",
     "(b)-তে k = ৫-এ সব ভালো মডেল ~৯৮%, k = ১০-এ ~৯৯.৪%। (c)-তে নিউরাল মডেলগুলোর CI একে অপরের উপর পড়ে (overlap), মানে তাদের পার্থক্য নিশ্চিত না।",
     "তালিকা লম্বা করলে (৫টা রোগ বললে) প্রায় সব রোগীর সঠিক রোগ ধরা পড়ে, যা ডাক্তারের differential diagnosis-এর সাথে মেলে।")
fig("fig06_performance_profile.png", 6.2, "(Figure 6)")
item("Figure 7 ও Table 7: Calibration ও Temperature scaling",
     "মডেলের দেওয়া ‘সম্ভাবনা’ বিশ্বাসযোগ্য কিনা। মডেল ৯০% নিশ্চিত বললে সত্যিই কি ৯০% বার ঠিক হয়?",
     "Figure 7: প্রতিটা বার একটা নিশ্চয়তা-স্তর। বার কোণাকুণি দাগ ছুঁলে নিখুঁত। বার দাগের **উপরে** = মডেল কম আত্মবিশ্বাসী (underconfident), নিচে = অতি-আত্মবিশ্বাসী। "
     "লাল দাগকাটা অংশ = ভুলের পরিমাণ। Table 7: একটা সংখ্যা T দিয়ে probability ঠিক করার আগে আর পরে।",
     "নিউরাল মডেলগুলো underconfident (ECE ~১৫%), কারণ class weighting আর label smoothing। Temperature scaling-এর পর HyperDiag-এর "
     "ECE **১৫.৯২% → ১.৫৭%**, Log-loss **০.৫৩২ → ০.৩৩২**, Brier ০.২০৬ → ০.১৭১। নিউরাল মডেলগুলোর মধ্যে এগুলো **সবচেয়ে ভালো**। "
     "এতে র‍্যাঙ্কিং বদলায় না, তাই accuracy একই থাকে।",
     "এটা তোমার মডেলের একটা আসল সুবিধা: সঠিক calibration-এর পর HyperDiag-এর দেওয়া সম্ভাবনা সবচেয়ে বিশ্বাসযোগ্য। "
     "ডাক্তারকে ‘এই রোগের সম্ভাবনা ৭০%’ বললে সেটা যেন সত্যিই ৭০% হয়, এটাই চাই।",
     "\"After temperature scaling, HyperDiag has the lowest log-loss and ECE among the discriminative models, so its probabilities are the most trustworthy.\"")
fig("fig07_calibration.png", 6.2, "(Figure 7) বার দাগের উপরে মানে মডেল নিজের ব্যাপারে কম নিশ্চিত")
item("Figure 8: Learning dynamics",
     "ট্রেনিংয়ের সময় প্রতি epoch-এ loss (ভুলের পরিমাণ) আর validation MRR কীভাবে বদলেছে।",
     "বাঁয়ে loss নামতে থাকা ভালো; ডানে MRR উঠে স্থির হওয়া ভালো।",
     "সব মডেলের loss কমেছে আর MRR ~৯০%-এ এসে স্থির হয়েছে। হঠাৎ ওঠানামা বা overfitting দেখা যায়নি।",
     "ট্রেনিং ঠিকঠাক হয়েছে, মডেলগুলো স্থিতিশীলভাবে শিখেছে।")

# ---------------------------------------------------------------- 4.2 stats
H("অংশ ৩: স্ট্যাটিস্টিক্যাল অ্যানালাইসিস (Section 4.2)", 1)
P("এই অংশের প্রশ্ন একটাই: **মডেলগুলোর মধ্যে যে পার্থক্য দেখছি, সেটা কি আসল, নাকি ভাগ্যক্রমে?** "
  "সংখ্যা সামান্য বেশি হলেই মডেল ভালো বলা যায় না; পরিসংখ্যান দিয়ে প্রমাণ লাগে।")
item("Table 8: McNemar test (HyperDiag বনাম প্রতিটা মডেল)",
     "একই রোগীদের উপর দুটো মডেলের তুলনা: কতজনকে শুধু HyperDiag ঠিক বলেছে, কতজনকে শুধু অন্য মডেল ঠিক বলেছে।",
     "‘Ours-only correct’ বনাম ‘Baseline-only correct’ তুলনা করো। ‘McNemar p (Holm)’ < 0.05 হলে পার্থক্য আসল। "
     "Cohen's h = effect size (০.২-এর নিচে মানে খুব ছোট)। শেষ দুই কলাম = পার্থক্য ও তার 95% CI।",
     "Random Forest-এর বিরুদ্ধে HyperDiag ২,২৯২ জনকে একা ঠিক বলেছে, RF মাত্র ১,৬১২ জন; p < 0.001, **significant**। "
     "Logistic Regression: p = ০.০০৯৮, **significant**। Naive Bayes: p = ০.০৬৯ (Holm-এর পর), significant **না**। "
     "MLP, GCN, GAT, HGNN: p = ১, **কোনো পার্থক্য নেই**।",
     "HyperDiag নিশ্চিতভাবে RF আর LogReg-এর চেয়ে ভালো; বাকিদের সমান। Naive Bayes-এর সাথে পার্থক্য সীমানায়: correction ছাড়া significant (p = ০.০১৪), correction-এর পর না।",
     "\"After Holm correction, HyperDiag is significantly better than Random Forest and Logistic Regression and statistically tied with the others.\"")
item("Figure 9: All-pairs McNemar heatmap",
     "৮টা মডেলের প্রতিটা জোড়ার McNemar test একটা ছকে।",
     "সারির মডেল বনাম কলামের মডেল। **লাল** = সারির মডেল ভালো, **নীল** = কলামের মডেল ভালো, সাদা/ফ্যাকাশে = পার্থক্য নেই। "
     "*** = খুব নিশ্চিত পার্থক্য, n.s. = significant না।",
     "RF আর LogReg-এর সারি পুরো নীল (সবার চেয়ে খারাপ)। বাকি মডেলগুলোর মধ্যে প্রায় সব ঘর ‘n.s.’।",
     "ভালো মডেলগুলো একটা দলে, পরস্পরের মধ্যে আসল পার্থক্য নেই।")
item("Figure 10: Critical-difference (CD) diagram",
     "প্রতিটা রোগের F1 ধরে মডেলগুলোকে ১ থেকে ৮ র‍্যাঙ্ক দেওয়া হয়েছে, তারপর গড় র‍্যাঙ্ক। Friedman test দেখে সব মডেল সমান কিনা।",
     "স্কেলে বাঁয়ে = ভালো (র‍্যাঙ্ক ১)। উপরের ছোট দাগটা CD (০.৪১): দুটো মডেলের গড় র‍্যাঙ্কের পার্থক্য এর চেয়ে কম হলে তারা আলাদা না। "
     "মোটা কালো বার দিয়ে যুক্ত মডেলগুলোর মধ্যে significant পার্থক্য নেই।",
     "Friedman χ² = ৩৮৪.০, p = ৬.৬×১০⁻⁷⁹: সব মডেল সমান না। গড় র‍্যাঙ্ক: GAT ৩.৯৪, MLP ৩.৯৮, HGNN ৪.০৯, GCN ৪.১৬, **HyperDiag ৪.২১**, "
     "Naive Bayes ৪.৭৮, LogReg ৫.০৭, RF ৫.৭৬। পাঁচটা নিউরাল মডেল একটা বারে যুক্ত।",
     "নিউরাল মডেলগুলো একটা দল (সমান), আর তারা ক্লাসিক্যাল মডেলগুলোর চেয়ে ভালো। HyperDiag আর Naive Bayes-এর র‍্যাঙ্ক-পার্থক্য (০.৫৭) CD-র চেয়ে বেশি, "
     "তাই প্রতিটা রোগের F1-এর হিসাবে HyperDiag Naive Bayes-এর চেয়ে ভালো।")
fig("fig10_critical_difference.png", 6.0, "(Figure 10) বাঁয়ে = ভালো; কালো বারে যুক্ত মডেলরা পরস্পরের সমান")
item("Table 9: Wilcoxon signed-rank test",
     "প্রতিটা রোগের F1-এ HyperDiag বনাম অন্য মডেল: কয়টা রোগে জিতেছে, কয়টায় সমান, কয়টায় হেরেছে।",
     "Wins/Ties/Losses = রোগের সংখ্যা। r_rb = effect size (ধনাত্মক মানে HyperDiag ভালো)। Effect কলামে negligible/small/medium/large।",
     "Random Forest-এর বিরুদ্ধে ৩৮৮টা রোগে জয়, ১৬০টায় হার, **large** effect (r = ০.৫৬)। LogReg-এর বিরুদ্ধে **medium** (০.৪২)। "
     "Naive Bayes-এর বিরুদ্ধে ২৭৬ জয় বনাম ২১৩ হার, **small** effect (০.২৪), significant। নিউরাল মডেলদের বিরুদ্ধে **negligible** (নগণ্য)।",
     "রোগ-ভিত্তিক হিসাবেও একই কথা: ক্লাসিক্যাল মডেলের চেয়ে HyperDiag স্পষ্টভাবে ভালো, নিউরাল মডেলদের সমান।")
item("Figure 11: Bootstrap forest plot",
     "HyperDiag বিয়োগ প্রতিটা মডেল: Top-1, Top-5, MRR, NDCG@10-এ কত পয়েন্ট পার্থক্য, 95% CI সহ।",
     "মাঝের দাগ = ০ (কোনো পার্থক্য নেই)। দাগের **ডানে** পুরো interval থাকলে HyperDiag নিশ্চিতভাবে ভালো। রঙিন = significant, ধূসর = না।",
     "RF-এর বিরুদ্ধে Top-1-এ +১.৮ পয়েন্ট [১.৪৮, ২.১১], LogReg-এর বিরুদ্ধে +০.৩৬, Naive Bayes-এর বিরুদ্ধে +০.২৯ [০.০৭, ০.৫৪]। "
     "নিউরাল মডেলগুলোর interval ০ পার হয়।",
     "ছবিতে এক নজরে দেখা যায় কার বিরুদ্ধে জয় নিশ্চিত, কার সাথে সমান।")

# ---------------------------------------------------------------- partial / rare / ablation
H("অংশ ৪: বিশেষ পরিস্থিতিতে মডেল (Sections 4.3–4.5)", 1)
item("Table 10 ও Figure 12: অল্প লক্ষণ বললে (Partial symptoms)",
     "রোগী সব লক্ষণ না বলে অল্প কয়টা বললে মডেল কেমন করে। প্রতিটা test রোগীর লক্ষণ থেকে ৮০%, ৬০%, ৪০%, ২০% রেখে পরীক্ষা।",
     "কলাম = কত % লক্ষণ রাখা হয়েছে; ঘরের মান = Top-5 accuracy। Heatmap-এ গাঢ় নীল = ভালো।",
     "সব লক্ষণে HyperDiag ৯৭.৮%, ৪০%-এ ৭৯.৪%, ২০%-এ ৪৩.৬%। ২০%-এ Naive Bayes ৫২.৯%, LogReg ৫০.২%, MLP ৪৪.৬%, GCN/GAT/HGNN ~৪২.৮%।",
     "লক্ষণ কমলে সবারই accuracy কমে। খুব কম লক্ষণে (২০%) Naive Bayes সবার চেয়ে ভালো, কারণ নিউরাল মডেলরা ‘কোন লক্ষণ নেই’ সেটার উপরও নির্ভর করতে শেখে। "
     "গ্রাফ-ভিত্তিক মডেলগুলোর মধ্যে HyperDiag সবচেয়ে ভালো। এটা তোমার থিসিসের **limitation**, আর ভবিষ্যতের কাজের দিক।",
     "\"With very few reported symptoms, Naive Bayes is more robust than all discriminative models, including ours; this is a limitation we report honestly.\"")
fig("fig12_partial_symptoms.png", 6.2, "(Figure 12) বাঁ থেকে ডানে লক্ষণ কমছে; গাঢ় নীল = ভালো")
item("Table 11 ও Figure 13: বিরল বনাম সাধারণ রোগ",
     "যে রোগের ট্রেনিং ডেটা কম (বিরল, ≤২০ জন) আর যেটার বেশি (>৪০০ জন), সেগুলোতে কোন মডেল কেমন।",
     "Table 11: প্রতিটা গ্রুপের রোগগুলোর গড় recall@1। Figure 13(a): প্রতিটা বিন্দু একটা রোগ, x = ট্রেনিং কেস (log), y = F1; লাইন = গড়।",
     "বিরল রোগে: GAT ৭৭.২%, MLP ৭৬.৩%, HGNN ৭৫.৮%, GCN ৭৪.৮%, **HyperDiag ৭৩.২%**, Naive Bayes ৫১.৩%, LogReg ৪৮.৪%, RF ৩৬.৮%। "
     "সাধারণ রোগে সবাই ~৮২–৮৫%।",
     "বিরল রোগে নিউরাল মডেলরা ক্লাসিক্যালদের চেয়ে অনেক ভালো (class weighting কাজ করেছে)। নিউরালদের মধ্যে HyperDiag একটু পিছিয়ে, "
     "তবে বিরল রোগে test কেস খুব কম, তাই এই পার্থক্য খুব নির্ভরযোগ্য না।")
item("Table 12 ও Figure 14: Ablation study",
     "HyperDiag-এর একেকটা অংশ সরিয়ে আবার ট্রেন করলে ফলাফল কতটা বদলায়।",
     "প্রথম সারি = পুরো মডেল। নিচের প্রতিটা সারি = একটা অংশ বাদ। Figure 14-এ বার = পুরো মডেলের তুলনায় পরিবর্তন; লাল = উল্লেখযোগ্য কমতি।",
     "বেশিরভাগ অংশ সরালে পরিবর্তন ±০.১ পয়েন্টের মধ্যে, যা seed-এর ওঠানামার মতোই। শুধু **prototype scoring** সরালে Macro-F1 **৮১.১৬ → ৭৯.২৫** "
     "(−১.৯ পয়েন্ট) আর ফল অনেক অস্থির হয় (std ০.১৩ → ১.০৭)।",
     "‘প্রতিটা রোগকে তার নিজের hyperedge দিয়ে score করা’ (prototype scoring) HyperDiag-এর সবচেয়ে কার্যকর অংশ। "
     "অন্য attention অংশগুলো accuracy বাড়ায় না, তবে ব্যাখ্যা দেওয়ার কাজে লাগে।",
     "\"The ablation shows that disease-hyperedge prototype scoring is the component that matters most: removing it lowers macro-F1 by 1.9 points.\"")

# ---------------------------------------------------------------- interpretability
H("অংশ ৫: ব্যাখ্যাযোগ্যতা ও ভুল বিশ্লেষণ (Sections 4.6–4.7)", 1)
item("Figure 15: Case studies",
     "তিনজন test রোগীর জন্য: বাঁয়ে কোন লক্ষণকে মডেল কত গুরুত্ব দিয়েছে (γ), ডানে মডেলের প্রথম ৫টা সম্ভাব্য রোগ।",
     "বাঁয়ের বার যত লম্বা, সেই লক্ষণ সিদ্ধান্তে তত বেশি ভূমিকা রেখেছে। ডানে সবুজ বার = আসল রোগ।",
     "তিনটা উদাহরণেই আসল রোগ ১ নম্বরে। যেমন একজন রোগীর জন্য মডেল ৭৯% সম্ভাবনা দিয়ে chronic sinusitis বলেছে, আর সবচেয়ে গুরুত্ব দিয়েছে "
     "sinus congestion লক্ষণকে (০.৫৯); ২ নম্বরে acute sinusitis (৪%)। আরেকজনের জন্য yeast infection (৮২%), মূল লক্ষণ pelvic pain।",
     "মডেল শুধু উত্তর দেয় না, কেন দিল সেটাও দেখায়। ডাক্তারের আস্থার জন্য এটা জরুরি।")
fig("fig15_case_studies.png", 5.8, "(Figure 15) বাঁয়ে: লক্ষণের গুরুত্ব; ডানে: প্রথম ৫টা রোগ (সবুজ = আসল)")
item("Figure 16: Hyperedge-এর ভেতরের attention",
     "একটা রোগের hyperedge-এর ভেতরে মডেল কোন লক্ষণকে বেশি গুরুত্ব দেয় (নীল), আর আসলে ঐ রোগে লক্ষণটা কত সাধারণ (কমলা)।",
     "নীল আর কমলা বার একই রকম হলে মডেলের attention বাস্তবতার সাথে মেলে।",
     "গড় Spearman ρ = **০.৮৭** (৯৯.৪% রোগে ধনাত্মক)। মানে মডেলের attention প্রায় পুরোপুরি বাস্তবের লক্ষণ-প্রাদুর্ভাবের সাথে মেলে।",
     "মডেলের ‘মনোযোগ’ চিকিৎসাগতভাবে যুক্তিসঙ্গত।")
item("Table 13: Faithfulness (deletion) test",
     "মডেলের দেখানো ব্যাখ্যা কি সত্যিকারের? সবচেয়ে গুরুত্বপূর্ণ লক্ষণটা মুছে দিলে ফল কতটা বদলায়, তা এলোমেলো বা সবচেয়ে কম গুরুত্বপূর্ণ লক্ষণ মোছার সাথে তুলনা।",
     "‘Mean drop’ যত বড়, ঐ লক্ষণ মোছায় তত ক্ষতি। ব্যাখ্যা সত্যি হলে: most attended > random > least attended।",
     "সবচেয়ে গুরুত্বপূর্ণ লক্ষণ মুছলে সঠিক রোগের সম্ভাবনা কমে **০.১৫৫**, এলোমেলো লক্ষণে **০.০৯৫**, সবচেয়ে কম গুরুত্বপূর্ণে **০.০৬১**। "
     "Top-1 নামে ৮৭.৩% থেকে যথাক্রমে ৬৮.২%, ৭৫.৫%, ৭৯.৭%-এ।",
     "ক্রমটা ঠিক যেমন হওয়া উচিত, তাই HyperDiag-এর ব্যাখ্যা বিশ্বস্ত (faithful); শুধু সুন্দর দেখানোর জন্য না।",
     "\"Deleting the most-attended symptom lowers the true-disease probability 1.6 times more than deleting a random one, so the explanations are faithful.\"")
item("Figure 17: t-SNE",
     "মডেল ভেতরে রোগ (a) আর লক্ষণগুলোকে (b) কীভাবে সাজিয়েছে, ২-মাত্রার ম্যাপে।",
     "কাছাকাছি বিন্দু = মডেলের চোখে একই রকম। রং (a)-তে ট্রেনিং কেসের সংখ্যা, (b)-তে লক্ষণটা কয়টা hyperedge-এ আছে।",
     "(a)-তে রোগগুলো স্পষ্ট কয়েকটা গুচ্ছে (cluster) ভাগ হয়েছে; একই গুচ্ছের রোগগুলোর লক্ষণ-সেট সাধারণত কাছাকাছি।",
     "মডেল নিজে থেকেই সম্পর্কিত রোগগুলোকে কাছাকাছি রেখেছে, অর্থাৎ অর্থপূর্ণ গঠন শিখেছে।")
item("Figure 18: Error analysis (ভুল বিশ্লেষণ)",
     "HyperDiag কোন রোগের সাথে কোন রোগ গুলিয়ে ফেলে, আর সেই রোগগুলোর লক্ষণ কতটা মেলে।",
     "(a) নেটওয়ার্ক: প্রতিটা রেখা দুটো রোগ যা গুলিয়ে যায়; মোটা রেখা = বেশি ভুল। "
     "(b) ECDF: লাল লাইন (গুলিয়ে যাওয়া জোড়া) ডানে = তাদের লক্ষণ বেশি মেলে; ধূসর = এলোমেলো রোগ-জোড়া।",
     "গুলিয়ে যাওয়া রোগ-জোড়ার লক্ষণ গড়ে **৫৩%** মেলে (Jaccard ০.৫২৯), এলোমেলো জোড়ায় মাত্র **৩%**। "
     "সবচেয়ে বেশি ভুল: noninfectious বনাম infectious gastroenteritis (৬৬ বার, লক্ষণ ৮৫% এক), acute bronchitis বনাম COPD (৫৬), "
     "cholecystitis বনাম gallstone (৪৭), psychotic disorder বনাম schizophrenia (৩৯)।",
     "মডেলের ভুলগুলো এলোমেলো না। যে রোগগুলোর লক্ষণ প্রায় এক, শুধু সেগুলোই গুলিয়ে যায়, যা ডাক্তারের জন্যও শুধু লক্ষণ দেখে আলাদা করা কঠিন। "
     "এটা Bayes ceiling-এর সাথেও মেলে।",
     "\"The errors are not random: confused disease pairs share 53% of their symptoms on average, versus 3% for random pairs.\"")
fig("fig19_error_analysis.png", 6.2, "(Figure 18) বাঁয়ে: কোন রোগ কোনটার সাথে গুলিয়ে যায়; ডানে: গুলিয়ে যাওয়া রোগগুলোর লক্ষণ বেশি মেলে")
item("Table 14: Comparison with relevant research",
     "আগের গবেষণার ধরনগুলোর সাথে তোমার কাজের তুলনা: পদ্ধতি, গঠন, কাজ আর ব্যাখ্যাযোগ্যতা।",
     "শেষ সারি তোমার কাজ।",
     "আগের কাজ: flat feature বা pairwise graph, single-label। তোমার: hypergraph, top-k differential, ব্যাখ্যাযোগ্য।",
     "তোমার কাজ কোথায় নতুন, সেটা এক নজরে দেখায়।")

# ---------------------------------------------------------------- viva Q&A
H("অংশ ৬: ভাইভায় সম্ভাব্য প্রশ্ন ও উত্তর", 1)
qa = [
 ("তোমার মডেল তো MLP-কে হারাতে পারেনি, তাহলে novelty কী?",
  "আমার প্রধান অবদান তিনটা: (১) ডেটাসেটের leakage ধরা আর leakage-মুক্ত মূল্যায়ন পদ্ধতি, যা দেখায় আগের accuracy-গুলো বেশি দেখানো। (২) Bayes ceiling বের করা, "
  "যা দেখায় এই ডেটায় সর্বোচ্চ কত সম্ভব; সব মডেল ceiling-এর কাছে পৌঁছে গেছে, তাই বড় পার্থক্য সম্ভবই না। (৩) HyperDiag সেরা মডেলদের সমান accuracy দিয়েও "
  "ব্যাখ্যা দিতে পারে, সবচেয়ে নির্ভরযোগ্য probability দেয়, আর এর ভুলগুলো hyperedge-overlap দিয়ে ব্যাখ্যা করা যায়।"),
 ("Bayes ceiling কী?",
  "একই লক্ষণ-সেট যদি একাধিক রোগে দেখা যায়, তাহলে লক্ষণ দেখে কেউই নিশ্চিত হতে পারবে না কোনটা। সব এমন কেসে সবচেয়ে সাধারণ রোগটা বলাই সর্বোচ্চ সম্ভব। "
  "এভাবে হিসাব করে test set-এ Top-1-এর সর্বোচ্চ সীমা ৮৯.৮৯%।"),
 ("Top-1-এর বদলে Top-5 আর MRR কেন?",
  "ডাক্তার একটা রোগ না বলে সম্ভাব্য রোগের তালিকা ভাবেন (differential diagnosis)। Top-5 দেখে সঠিক রোগ তালিকায় আছে কিনা, MRR দেখে কত উপরে আছে।"),
 ("Holm correction কেন দিলে?",
  "আমি ৭টা মডেলের সাথে তুলনা করেছি। একসাথে অনেক test করলে ভাগ্যক্রমে কোনো একটা ‘significant’ আসার সম্ভাবনা বাড়ে। Holm সেটা নিয়ন্ত্রণ করে। "
  "এজন্য Naive Bayes-এর সাথে পার্থক্য (p = ০.০১৪) correction-এর পর significant থাকে না (p = ০.০৬৯)।"),
 ("অল্প লক্ষণে Naive Bayes ভালো কেন?",
  "Naive Bayes প্রতিটা লক্ষণকে আলাদাভাবে প্রমাণ হিসেবে দেখে। নিউরাল মডেলরা ‘কোন লক্ষণ নেই’ সেটাকেও তথ্য হিসেবে শেখে; অনেক লক্ষণ না বললে তাই ভুল পথে যায়। "
  "এটা আমার limitation, আর ভবিষ্যতে generative-discriminative মিশ্র মডেল দিয়ে সমাধান করা যেতে পারে।"),
 ("Temperature scaling কী, accuracy বদলায় না কেন?",
  "সব score-কে একটা সংখ্যা T দিয়ে ভাগ করা হয়, T validation set থেকে বের করা। এতে সম্ভাবনাগুলো ঠিক হয়, কিন্তু রোগগুলোর ক্রম একই থাকে, তাই Top-k আর MRR বদলায় না।"),
]
for q, a in qa:
    P("**প্রশ্ন: " + q + "**")
    P("উত্তর: " + a, indent=0.25)

d.save(OUT); print("saved", OUT)
