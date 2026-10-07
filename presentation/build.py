# -*- coding: utf-8 -*-
"""Builds the project presentation from the Slidesgo "Child Meningitis Case Study" template.

    python presentation/build.py        -> presentation/Cardio-Risk.pptx

The template's green (theme colour dk2) is swapped for a cardiac red, then every slide is created from
one of the template's layouts so its arches and dots carry over. Icons are copied from the template's
own icon sheets and recoloured. Morph transitions run on every slide after the first; shapes whose
names start with "!!" are matched by Morph across slides (the model bars of slide 7 grow into the
leakage bars of slide 11).

Numbers come from the executed report (report.ipynb). The scorecard, its risk mapping and the
command-line example are read from the saved model (outputs/model.joblib), so they cannot drift.
Speaker notes hold the spoken script; the build prints its length at 140 words a minute.
"""
import copy
import io
import json
import pathlib
import re
import sys
import zipfile

from lxml import etree
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import cardio  # noqa: E402

OUT = HERE / "Cardio-Risk.pptx"
FIG = HERE / "figures"
RENDER = HERE / "render"
RENDER.mkdir(exist_ok=True)

NAVY, BLUE, GREY, PALE, WHITE = "0D3C4C", "99D0E2", "ADBAC1", "ADD9E8", "FFFFFF"
RED = "E05A4F"              # replaces the template's green everywhere (theme colour dk2)
DKRED = "B23A2E"            # red for small text on white (contrast)
SOFT = "EEF6F9"             # light blue tint for cards
MUTED = "4F6A73"            # secondary text
HEAD, BODY, MONO = "Cabin", "Nanum Gothic", "Consolas"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"


# ------------------------------------------------------------------ template with the red theme
def red_template() -> io.BytesIO:
    src = zipfile.ZipFile(HERE / "template.pptx")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "ppt/theme/theme2.xml":
                data = data.replace(b'<a:dk2><a:srgbClr val="99BB6E"/>', b'<a:dk2><a:srgbClr val="%s"/>' % RED.encode())
                assert RED.encode() in data, "theme colour not found"
            dst.writestr(item, data)
    out.seek(0)
    return out


prs = Presentation(red_template())
TPL = list(prs.slides)
ICONIDX = json.load(open(HERE / "assets" / "iconidx.json"))


def find_shape(slide, sid):
    for el in slide.shapes._spTree.iter():
        if el.tag.endswith("}cNvPr") and el.get("id") == str(sid):
            return copy.deepcopy(el.getparent().getparent())
    raise KeyError(sid)


ICONS = {
    "heart": "34-51", "stethoscope": "34-71", "monitor": "34-97", "pills": "34-58", "flask": "34-63",
    "lungs": "34-88", "heartcheck": "34-78", "record": "34-95", "hospital": "34-87", "crossheart": "34-69",
    "apple": "34-53", "search": "34-30", "clipboard": "34-45", "scale": "34-3", "trophy": "34-36",
    "idea": "34-37", "hourglass": "34-50", "bars": "35-36", "trend": "35-45", "target": "35-5",
    "laptop": "35-14", "checklist": "39-62", "docsearch": "39-61", "medkit": "34-52", "hands": "34-61",
    "people": "39-21", "database": "39-36", "code": "39-9",
}
ICON_XML = {}
for name, key in ICONS.items():
    ICON_XML[name] = find_shape(TPL[int(key.split("-")[0]) - 1], ICONIDX[key])
THANKS_DECOR = [find_shape(TPL[19], sid) for sid in (1624, 1625, 1626, 1638)]

sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)
LAYOUT = {l.name: l for l in prs.slide_layouts}


# ------------------------------------------------------------------ helpers
def new(layout):
    return prs.slides.add_slide(LAYOUT[layout])


def ph(slide, idx, text_, size=None, color=None, bold=None, align=None):
    p_ = slide.placeholders[idx]
    tf = p_.text_frame
    lines = text_ if isinstance(text_, list) else [text_]
    tf.text = lines[0]
    for extra in lines[1:]:
        tf.add_paragraph().text = extra
    for para in tf.paragraphs:
        pPr = para._p.get_or_add_pPr()
        pPr.set("marL", "0")
        pPr.set("indent", "0")
        for b in pPr.findall("{%s}buNone" % A) + pPr.findall("{%s}buChar" % A):
            pPr.remove(b)
        bn = etree.SubElement(pPr, "{%s}buNone" % A)
        if pPr.find("{%s}defRPr" % A) is not None:
            pPr.find("{%s}defRPr" % A).addprevious(bn)
        if align is not None:
            para.alignment = align
        for r in para.runs:
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = RGBColor.from_string(color)
            if bold is not None:
                r.font.bold = bold
    return p_


def drop_empty_placeholders(slide):
    for p_ in list(slide.placeholders):
        if not p_.has_text_frame or not p_.text_frame.text.strip():
            p_._element.getparent().remove(p_._element)


def _nostyle(shape):
    st = shape._element.find("{%s}style" % P)
    if st is not None:
        shape._element.remove(st)


def fill_paras(tf, paras, size, color, font, bold, align, space=4, line=None):
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p_ = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_.alignment = align
        if i:
            p_.space_before = Pt(space)
        if line:
            p_.line_spacing = line
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p_.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = RGBColor.from_string(o.get("color", color))


def text(slide, x, y, w, h, paras, size=13, color=NAVY, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, space=4, line=None):
    """paras: str, or list of str / list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    fill_paras(tf, paras, size, color, font, bold, align, space, line)
    return tb


def box(slide, x, y, w, h, fill=SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None, line=None, radius=0.12):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    _nostyle(s)
    if name:
        s.name = name
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line:
        s.line.color.rgb = RGBColor.from_string(line)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.margin_left = s.text_frame.margin_right = Inches(0.08)
    s.text_frame.margin_top = s.text_frame.margin_bottom = Inches(0.03)
    return s


def label_in(shape, paras, size=12, color=WHITE, bold=True, font=BODY, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    fill_paras(tf, paras, size, color, font, bold, align)


def icon(slide, key, x, y, size=0.42, color=NAVY, name=None):
    el = copy.deepcopy(ICON_XML[key])
    for sf in el.iter("{%s}solidFill" % A):
        for c in list(sf):
            sf.remove(c)
        etree.SubElement(sf, "{%s}srgbClr" % A).set("val", color)
    xfrm = el.find(".//{%s}xfrm" % A)
    off, ext = xfrm.find("{%s}off" % A), xfrm.find("{%s}ext" % A)
    ow, oh = int(ext.get("cx")), int(ext.get("cy"))
    scale = Inches(size) / max(ow, oh)
    w, h = int(ow * scale), int(oh * scale)
    off.set("x", str(int(Inches(x) + (Inches(size) - w) / 2)))
    off.set("y", str(int(Inches(y) + (Inches(size) - h) / 2)))
    ext.set("cx", str(w))
    ext.set("cy", str(h))
    nid = max(int(e.get("id")) for e in slide.shapes._spTree.iter() if e.tag.endswith("}cNvPr")) + 1
    for e in el.iter():
        if e.tag.endswith("}cNvPr"):
            e.set("id", str(nid))
            nid += 1
    el.find(".//{%s}cNvPr" % P).set("name", name or ("icon " + key))
    slide.shapes._spTree.append(el)


def circle_icon(slide, key, x, y, d=0.62, fill=NAVY, color=WHITE, name=None):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL, name=name)
    icon(slide, key, x + d * 0.22, y + d * 0.22, d * 0.56, color)
    return c


def title(slide, t):
    ph(slide, 0, t)


SCRIPT = []


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t
    SCRIPT.append(t)


def figure(slide, name, x, y, w=None, h=None, crop=(0, 0, 1, 1), shape_name=None):
    """A report figure, cropped (fractions left, top, right, bottom) and placed by width or height."""
    im = Image.open(FIG / name)
    W, H = im.size
    l, t, r, b = crop
    cut = im.crop((int(l * W), int(t * H), int(r * W), int(b * H)))
    path = RENDER / ("crop_" + name if crop != (0, 0, 1, 1) else name)
    cut.save(path)
    ratio = cut.width / cut.height
    if w is not None:
        h = w / ratio
    else:
        w = h * ratio
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    if shape_name:
        pic.name = shape_name
    return w, h


def logo_plate():
    src = Image.open(HERE / "assets" / "lu-logo-src.png").convert("RGBA")
    src = src.resize((int(460 * src.width / src.height), 460), Image.LANCZOS)
    pad, rad = 34, 30
    w, h = src.width + pad * 2, src.height + pad * 2
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=rad, fill=255)
    plate = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    plate.paste(src, (pad, pad), src)
    plate.putalpha(mask)
    out = RENDER / "lu-logo-plate.png"
    plate.save(out)
    return out, w / h


# ------------------------------------------------------------------ numbers from the saved model
BUNDLE = cardio.load_final(ROOT / "outputs" / "model.joblib")
CARD = BUNDLE["points"].table()
if list(BUNDLE["points"].factors) != cardio.CLINICAL_FACTORS:
    # a model saved before the clinical scorecard: its points are not the ones the report recommends
    if "--draft" not in sys.argv:
        raise SystemExit("outputs/model.joblib holds an old scorecard: fetch the latest VM run's model, "
                         "or build with --draft to check the layout only")
    print("DRAFT: old scorecard, numbers on slides 10 and 13 are placeholders")
    labels = [cardio.POINTS_LABELS[f] for f in cardio.CLINICAL_FACTORS]
    CARD = CARD[CARD["factor"].isin(labels)]
EXAMPLE = dict(age=62, sex="male", height=172, weight=98, systolic=165, diastolic=100, cholesterol=3)
EXAMPLE_RESULT = cardio.assess(BUNDLE, cardio.patient(**EXAMPLE))
RISK_AT = BUNDLE["points"].risk_by_total([0, 10, 20, 30, 40])

# ================================================================== SLIDES
S = []

# 1 ---------------------------------------------------------------- title
s = new("TITLE"); S.append(s)
ph(s, 0, "Detecting Cardiovascular Disease from a Routine Check-up", size=32)
t0 = s.placeholders[0]; t0.left, t0.top, t0.width, t0.height = Inches(1.3), Inches(1.4), Inches(7.4), Inches(1.6)
ph(s, 1, "Patient's Sickness Prediction · 70,000 medical examinations", size=13)
t1 = s.placeholders[1]; t1.left, t1.top, t1.width, t1.height = Inches(1.5), Inches(3.05), Inches(7.0), Inches(0.45)
plate, ratio = logo_plate()
lh = 1.05
s.shapes.add_picture(str(plate), Inches(5 - lh * ratio / 2), Inches(0.18), Inches(lh * ratio), Inches(lh)).name = "!!logo"
text(s, 2.0, 3.85, 6.0, 0.8, [
    [("Hadi Ghazi", {"font": HEAD, "bold": True, "size": 17})],
    [("Lebanese University", {"size": 12.5, "color": DKRED, "bold": True})],
], align=PP_ALIGN.CENTER, space=3, name="!!byline")
notes(s, "Good morning. My project is topic two, patient's sickness prediction. The question is simple: from "
         "the values a doctor records at a routine check-up, can a model tell whether a patient has "
         "cardiovascular disease? And just as important: how good is it really, compared with what doctors "
         "already use?")

# 2 ---------------------------------------------------------------- why
s = new("TITLE_ONLY"); S.append(s)
title(s, "Why this matters")
box(s, 0.79, 1.35, 3.25, 3.75, fill=SOFT)
circle_icon(s, "heart", 1.04, 1.6, d=0.7, fill=RED)
text(s, 1.04, 2.5, 2.8, 0.6, "17.9 million", size=30, font=HEAD, bold=True, color=DKRED)
text(s, 1.04, 3.12, 2.8, 0.9, "deaths a year from cardiovascular disease, the world's leading cause of death (WHO)",
     size=12.5)
text(s, 1.04, 4.12, 2.8, 0.8, "Many could be delayed if patients at risk were found early, at an ordinary check-up.",
     size=12, color=MUTED)
text(s, 4.45, 1.35, 4.75, 0.3, "OBJECTIVE", size=10.5, bold=True, color=DKRED)
text(s, 4.45, 1.62, 4.75, 0.75, "Build a model that detects cardiovascular disease from one examination, "
     "and measure honestly how good it is", size=14.5, font=HEAD, bold=True)
qs = [("bars", "Q1 · What does machine learning add?", "Compared with a blood-pressure rule and a standard risk score"),
      ("database", "Q2 · How much does data handling matter?", "Cleaning, and the mistakes that inflate accuracy"),
      ("stethoscope", "Q3 · Can the predictions be trusted?", "Calibration, explanations, every kind of patient")]
for i, (ic, h, b) in enumerate(qs):
    y = 2.6 + i * 0.83
    circle_icon(s, ic, 4.45, y, d=0.58, name="!!q%d" % i)
    text(s, 5.2, y - 0.02, 4.0, 0.3, h, size=13.5, font=HEAD, bold=True)
    text(s, 5.2, y + 0.29, 4.0, 0.3, b, size=11.5, color=MUTED)
notes(s, "Cardiovascular diseases are the world's leading cause of death: about 17.9 million people a year. Many "
         "of those deaths can be delayed if patients at risk are found early. So the objective is a model that "
         "uses one examination's values to detect the disease. Three questions guide the evaluation. First, what "
         "does machine learning add over what a doctor already has: a blood-pressure threshold and a standard "
         "risk score? Second, how much do data-handling choices change the result? Third, can the predictions "
         "be trusted?")

# 3 ---------------------------------------------------------------- the data
s = new("TITLE_ONLY"); S.append(s)
title(s, "The data: 70,000 check-ups")
for i, (num, lab) in enumerate([("70,000", "patients"), ("11", "values per examination"),
                                ("50 / 50", "with and without CVD")]):
    x = 0.79 + i * 2.88
    box(s, x, 1.3, 2.6, 1.15, fill=SOFT)
    text(s, x, 1.38, 2.6, 0.6, num, size=28, font=HEAD, bold=True, color=DKRED, align=PP_ALIGN.CENTER)
    text(s, x, 1.98, 2.6, 0.35, lab, size=12, align=PP_ALIGN.CENTER, color=MUTED)
groups = [("stethoscope", "Measured", ["age", "sex", "height", "weight", "systolic pressure", "diastolic pressure"]),
          ("flask", "Lab result, 3 levels", ["cholesterol", "glucose"]),
          ("clipboard", "Self-reported, yes / no", ["smoking", "alcohol", "physical activity"])]
for i, (ic, h, items) in enumerate(groups):
    x = 0.79 + i * 2.88
    circle_icon(s, ic, x, 2.72, d=0.5)
    text(s, x + 0.62, 2.79, 2.0, 0.35, h, size=13, font=HEAD, bold=True)
    for j, item in enumerate(items):
        c = box(s, x + (j % 2) * 1.32, 3.38 + (j // 2) * 0.47, 1.25, 0.37, fill=SOFT, radius=0.3)
        label_in(c, item, size=10.5, color=NAVY, bold=False)
text(s, 0.79, 4.85, 8.4, 0.3, "Kaggle cardiovascular disease dataset (Ulianova, 2019). Its source and the definition "
     "of \"disease\" are not documented.", size=10.5, color=MUTED)
notes(s, "The dataset has seventy thousand patients. For each, eleven values: age, sex, height, weight, the two "
         "blood-pressure readings, cholesterol and glucose in three levels, and three yes-or-no answers about "
         "smoking, alcohol and physical activity. Half the patients have cardiovascular disease. Two warnings "
         "from the start: we don't know where the data comes from, and we don't know how disease was defined. "
         "I come back to both at the end.")

# 4 ---------------------------------------------------------------- cleaning
s = new("TITLE_ONLY"); S.append(s)
title(s, "2% of the records are impossible")
figure(s, "fig1_blood_pressure_cleaning.png", 0.62, 1.28, w=5.55, crop=(0, 0.085, 1, 1))
patterns = [("1000 → 100", "an extra zero"), ("12 → 120", "cmHg instead of mmHg"),
            ("−120 → 120", "a stray minus sign"), ("80/120 → 120/80", "the two readings swapped")]
text(s, 6.45, 1.3, 2.8, 0.3, "TYPING ERRORS WITH A PATTERN", size=10.5, bold=True, color=DKRED)
for i, (a, b) in enumerate(patterns):
    y = 1.65 + i * 0.6
    c = box(s, 6.45, y, 1.55, 0.42, fill=NAVY, radius=0.3)
    label_in(c, a, size=11.5, font=HEAD)
    text(s, 8.1, y, 1.15, 0.42, b, size=10.5, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.79, 3.95, 8.43, 1.1, fill=SOFT)
text(s, 1.0, 4.05, 2.4, 0.9, [[("r = 0.05", {"color": MUTED})]], size=30, font=HEAD, bold=True,
     anchor=MSO_ANCHOR.MIDDLE, name="!!r_raw")
box(s, 3.2, 4.38, 0.42, 0.26, fill=RED, shape=MSO_SHAPE.RIGHT_ARROW)
text(s, 3.85, 4.05, 2.4, 0.9, "r = 0.43", size=30, font=HEAD, bold=True, color=DKRED,
     anchor=MSO_ANCHOR.MIDDLE, name="!!r_clean")
text(s, 6.2, 4.12, 2.9, 0.85, "How closely systolic pressure tracks the diagnosis, before and after cleaning",
     size=11.5, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "The first finding came before any model. About two percent of the records hold blood pressures no "
         "living person can have: sixteen thousand, minus one hundred and fifty, zero. They are not random. "
         "Most are typing errors with a pattern: an extra zero, centimetres of mercury instead of millimetres, a "
         "stray minus sign, or the two readings swapped. I compared dropping these records with repairing them "
         "by rule. Either way the effect is dramatic: in the raw data systolic pressure barely correlates with "
         "the diagnosis, 0.05. After cleaning it is the strongest predictor, 0.43.")

# 5 ---------------------------------------------------------------- what the data shows
s = new("TITLE_ONLY"); S.append(s)
title(s, "What the clean data shows")
figure(s, "fig2_disease_rate_by_factor.png", 0.95, 1.22, h=4.05, crop=(0, 0.06, 1, 1))
notes(s, "Here is the clean data, on the training patients only. The disease rate climbs steadily with systolic "
         "pressure, from about a quarter below 120 to 86 percent at 160 and above. It almost triples with age, "
         "rises with weight, and three quarters of patients with very high cholesterol have the disease. The "
         "lifestyle answers barely move it, and two point the wrong way: smokers and drinkers have slightly "
         "less disease.")

# 6 ---------------------------------------------------------------- smoking paradox
s = new("TITLE_ONLY"); S.append(s)
title(s, "The smoking paradox")
figure(s, "fig3_smoking_paradox.png", 0.62, 1.3, w=5.0, crop=(0, 0.085, 1, 1))
text(s, 0.79, 4.45, 4.8, 0.6, "Men only: women hardly smoke in this data (1.8% vs 22% of men)", size=11, color=MUTED)
box(s, 5.95, 1.3, 3.27, 3.8, fill=SOFT)
circle_icon(s, "hospital", 6.2, 1.5, d=0.6, fill=RED)
text(s, 6.95, 1.6, 2.15, 0.45, "The sick quitter", size=15, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
text(s, 6.2, 2.28, 2.85, 1.2, "Diagnosis and answers come from the same visit. Patients already diagnosed have "
     "usually been told to quit, so they answer \"no\".", size=12)
for i, t in enumerate(["Detection, not prediction", "Associations, not causes"]):
    c = box(s, 6.2, 3.65 + i * 0.6, 2.8, 0.45, fill=NAVY, radius=0.3)
    label_in(c, t, size=12, font=HEAD)
notes(s, "That is the smoking paradox. Smoking causes heart disease, so I checked for a confounder. Almost only "
         "men smoke in this data, so I compared within men, age group by age group. The paradox survives: "
         "smokers have less disease at every age. The likely reason is how the data was collected. Diagnosis "
         "and answers come from the same visit, and patients who were diagnosed have usually been told to quit. "
         "Two consequences: this is detection of existing disease, not prediction of future disease, and every "
         "effect we find is an association, not a cause.")

# 7 ---------------------------------------------------------------- models vs baselines (morph source)
AX0, AXW = 3.55, 5.0                       # x of 50% accuracy and width for 50 -> 100%


def ax(v):
    return AX0 + (v - 50) / 50 * AXW


def bars(slide, rows, top=1.38, step=0.47, axis_label=True, inside_above=101):
    for i, (key, lab, acc, extra, col) in enumerate(rows):
        y = top + i * step
        text(slide, 0.79, y, 2.65, 0.36, lab, size=12, font=HEAD, bold=True, align=PP_ALIGN.RIGHT,
             anchor=MSO_ANCHOR.MIDDLE, name="!!lab_" + key)
        box(slide, AX0, y + 0.05, ax(acc) - AX0, 0.26, fill=col, shape=MSO_SHAPE.RECTANGLE, name="!!bar_" + key)
        if acc > inside_above:   # long bar: the value sits inside its end, in white
            text(slide, ax(acc) - 1.0, y, 0.92, 0.36, "%.1f%%" % acc, size=11, bold=True, color=WHITE,
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, name="!!val_" + key)
        else:
            text(slide, ax(acc) + 0.08, y, 2.3, 0.36, [[("%.1f%%" % acc, {"bold": True}), (extra, {"color": MUTED})]],
                 size=11, anchor=MSO_ANCHOR.MIDDLE, name="!!val_" + key)
    base = top + len(rows) * step + 0.02
    box(slide, AX0, top - 0.05, 0.012, base - top + 0.05, fill=GREY, shape=MSO_SHAPE.RECTANGLE, name="!!axis")
    if axis_label:
        for v in (50, 75, 100):
            text(slide, ax(v) - 0.4, base + 0.02, 0.8, 0.25, "%d%%" % v, size=9.5, align=PP_ALIGN.CENTER,
                 color=MUTED, name="!!tick%d" % v)


s = new("TITLE_ONLY"); S.append(s)
title(s, "What machine learning adds")
bars(s, [("a", "Gradient boosting", 73.6, "  AUC 0.804", NAVY),
         ("b", "Random forest", 73.4, "  AUC 0.803", NAVY),
         ("c", "Neural network", 73.4, "  AUC 0.803", NAVY),
         ("d", "Logistic regression", 72.6, "  AUC 0.793", NAVY),
         ("e", "Points score (4 factors)", 72.3, "  AUC 0.793", RED),
         ("f", "Rule: systolic ≥ 130", 71.3, "  AUC 0.712", GREY),
         ("g", "Framingham, as published", 64.6, "  AUC 0.693", GREY)])
text(s, 0.79, 4.95, 8.4, 0.3, "Accuracy on 13,722 test patients never used for training or tuning", size=10.5,
     color=MUTED, align=PP_ALIGN.CENTER)
notes(s, "Now the models. Everything was tuned with five-fold cross-validation on the training patients, and "
         "these accuracies are on 13,722 test patients the models never saw. At the bottom, the Framingham score "
         "as published: 64.6 percent. It was built for a different question, a first event within ten years, and "
         "it does not transfer. The guideline rule, systolic at least 130, already gets 71.3. Logistic regression "
         "72.6. Random forest, the neural network and gradient boosting are tied, around 73.5. The best is "
         "gradient boosting: 73.6 percent, AUC 0.804. Machine learning helps, but by about two points.")

# 8 ---------------------------------------------------------------- test results
s = new("TITLE_ONLY"); S.append(s)
title(s, "On patients the model has never seen")
stats = [("0.804", "ROC AUC  (95% CI 0.797-0.811)"), ("73.6%", "accuracy, as in cross-validation"),
         ("+301", "diseased patients found, at the rule's specificity")]
for i, (n, l) in enumerate(stats):
    y = 1.35 + i * 1.22
    box(s, 0.79, y, 4.05, 1.02, fill=SOFT)
    text(s, 1.0, y + 0.08, 1.75, 0.85, n, size=28, font=HEAD, bold=True, color=DKRED, anchor=MSO_ANCHOR.MIDDLE,
         name="!!stat%d" % i)
    text(s, 2.75, y + 0.08, 1.95, 0.85, l, size=12, anchor=MSO_ANCHOR.MIDDLE)
figure(s, "fig8_roc_test.png", 5.25, 1.25, h=3.95, crop=(0, 0.045, 1, 1))
notes(s, "On unseen patients, gradient boosting reaches an AUC of 0.804, with a confidence interval from 0.797 to "
         "0.811, and 73.6 percent accuracy, the same as in cross-validation, so it did not overfit. The most "
         "concrete way to see the gain: at exactly the rule's specificity, the model finds 301 more of the 6,785 "
         "diseased patients. And the learning curve is flat, so more data of the same kind would not help: the "
         "ceiling is set by what the examination records.")

# 9 ---------------------------------------------------------------- what drives it
s = new("TITLE_ONLY"); S.append(s)
title(s, "What drives the prediction")
IMPORTANCE = [("blood pressure", 0.210), ("age", 0.036), ("cholesterol", 0.030), ("body size", 0.007),
              ("physical activity", 0.002), ("glucose", 0.002), ("smoking", 0.001), ("alcohol", 0.001), ("sex", 0.000)]
for i, (lab, v) in enumerate(IMPORTANCE):
    y = 1.32 + i * 0.3
    text(s, 0.79, y, 1.55, 0.26, lab, size=11, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE,
         bold=i == 0, font=HEAD if i == 0 else BODY)
    if v > 0:
        box(s, 2.45, y + 0.04, max(v / 0.21 * 2.75, 0.02), 0.18, fill=RED if i == 0 else NAVY,
            shape=MSO_SHAPE.RECTANGLE)
    text(s, 2.45 + v / 0.21 * 2.75 + 0.07, y, 0.6, 0.26, "%.3f" % v, size=10, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.79, 4.1, 5.1, 0.95, [[("Drop in test AUC when a factor is scrambled. ", {"bold": True}),
     ("Blood pressure dominates; glucose, the lifestyle answers and sex add almost nothing.", {})]], size=11.5)
box(s, 6.15, 1.3, 3.07, 3.8, fill=SOFT)
figure(s, "fig11_partial_dependence.png", 6.25, 1.42, w=2.87, crop=(0.0, 0.07, 0.27, 0.52))
text(s, 6.35, 3.62, 2.7, 1.4, [[("A switch, not a slope: ", {"bold": True, "color": DKRED}),
     ("predicted risk jumps from about 35% to 84% between 125 and 150 mmHg, at the guideline's thresholds.", {})]],
     size=11.5)
notes(s, "What does the model rely on? Scramble the blood-pressure features and the AUC drops by 0.21. Age and "
         "cholesterol follow, at 0.036 and 0.030. Glucose, the lifestyle answers and sex add almost nothing. "
         "The shape matters too: predicted risk stays around a third below 125, jumps to 84 percent by 150, and "
         "then stays flat. It works like a switch at the guideline's thresholds, which is exactly what a "
         "straight-line model cannot capture.")

# 10 --------------------------------------------------------------- points score
s = new("TITLE_ONLY"); S.append(s)
title(s, "A score a doctor can add up")
factor_icon = {"Age (years)": "hourglass", "Systolic pressure (mmHg)": "heart", "BMI (kg/m²)": "scale",
               "Cholesterol": "flask"}
for i, (factor, g) in enumerate(CARD.groupby("factor", sort=False)):
    y = 1.3 + i * 0.78
    c = box(s, 0.79, y, 2.05, 0.62, fill=NAVY, radius=0.2)
    icon(s, factor_icon[factor], 0.9, y + 0.12, 0.38, WHITE)
    text(s, 1.36, y, 1.45, 0.62, factor.replace(" (", "\n("), size=10.5, font=HEAD, bold=True, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE)
    for j, (band, pts) in enumerate(zip(g["band"], g["points"])):
        x = 2.98 + j * 1.25
        box(s, x, y, 1.12, 0.62, fill=SOFT)
        text(s, x, y + 0.04, 1.12, 0.25, band.replace("well above normal", "well above").replace("above normal", "above"),
             size=10, color=MUTED, align=PP_ALIGN.CENTER)
        text(s, x, y + 0.26, 1.12, 0.34, "%d" % pts, size=17, font=HEAD, bold=True,
             color=DKRED if pts else MUTED, align=PP_ALIGN.CENTER)
risk_line = "   ·   ".join("%d points → %.0f%%" % (k, 100 * v) for k, v in RISK_AT.items())
text(s, 0.79, 4.45, 8.43, 0.3, risk_line, size=12, bold=True, align=PP_ALIGN.CENTER)
text(s, 0.79, 4.82, 8.43, 0.3, "AUC 0.793 on the test patients, against 0.804 for the best model. The lifestyle "
     "answers are left out: they would earn negative points.", size=11, color=MUTED, align=PP_ALIGN.CENTER)
notes(s, "Since the gains are small, I also built something a doctor can use with a pen. Four factors, and each "
         "band earns whole points: age, systolic pressure, BMI and cholesterol. I left out the lifestyle answers, "
         "because in this data they would earn negative points, and no one can hand out a score that rewards "
         "smoking. Add up the points and the total maps to a risk. This card reaches an AUC of 0.793, the same as "
         "logistic regression on all fifteen features, and only 0.011 below the best model.")

# 11 --------------------------------------------------------------- leakage (morph target)
s = new("TITLE_ONLY"); S.append(s)
title(s, "How 90% accuracy happens")
bars(s, [("a", "Honest: unseen patients", 73.6, "  gradient boosting", NAVY),
         ("b", "Luckiest of 50 random splits", 73.4, "", GREY),
         ("c", "SMOTE before the split", 86.7, "", RED),
         ("d", "Scored on its own training data", 99.9, "", RED)], top=1.45, step=0.62, inside_above=80)
for v, lab, side in [(88.09, "88% (Omkari 2024)", "left"), (94.5, "94.5% (Zafar 2026)", "right")]:
    box(s, ax(v) - 0.07, 4.14, 0.14, 0.14, fill=DKRED, shape=MSO_SHAPE.ISOSCELES_TRIANGLE)
    if side == "left":
        text(s, ax(v) - 1.65, 4.33, 1.6, 0.25, lab, size=10, bold=True, color=DKRED, align=PP_ALIGN.RIGHT)
    else:
        text(s, ax(v) + 0.05, 4.33, 1.6, 0.25, lab, size=10, bold=True, color=DKRED)
text(s, 0.79, 4.75, 8.43, 0.45, [[("Same data, same models; only the evaluation changes. ", {"bold": True}),
     ("Red markers: accuracies published for this dataset.", {"color": MUTED})]], size=12, align=PP_ALIGN.CENTER)
notes(s, "Back to the second question. Papers on this same dataset report 88 and even 94.5 percent. I reproduced "
         "three common mistakes. Choosing the luckiest of 50 random splits barely matters: 73.4. Oversampling "
         "with SMOTE before splitting puts near-copies of training patients into the test set: 86.7 percent. "
         "Scoring a model on its own training patients gives 99.9. Same data, same models, only the evaluation "
         "changed. I cannot know what those papers did, but on this data an accuracy far above 74 percent "
         "should be read as a sign of leakage.")

# 12 --------------------------------------------------------------- trust
s = new("TITLE_ONLY"); S.append(s)
title(s, "Can the predictions be trusted?")
figure(s, "fig13_calibration.png", 0.62, 1.25, h=3.95, crop=(0, 0.06, 1, 1))
trust = [("scale", "Calibrated", "patients given 20% have CVD about 20% of the time"),
         ("people", "Women and men alike", "AUC 0.803 and 0.808"),
         ("hourglass", "Weaker over 60", "AUC 0.72, against 0.84 under 50"),
         ("target", "1 in 5 sick? 44% of flags right", "a reason to look closer, not a diagnosis")]
for i, (ic, h, b) in enumerate(trust):
    y = 1.32 + i * 0.95
    circle_icon(s, ic, 5.15, y, d=0.55, fill=RED if i == 3 else NAVY)
    text(s, 5.85, y - 0.03, 3.4, 0.32, h, size=13.5, font=HEAD, bold=True)
    text(s, 5.85, y + 0.29, 3.4, 0.5, b, size=11.5, color=MUTED)
notes(s, "Third question: can the risks be trusted? The model is well calibrated: patients given 20 percent have "
         "the disease about 20 percent of the time. Women and men are served equally. It is weaker for patients "
         "over 60, an AUC of 0.72 against 0.84 under 50, because most older patients have high pressure and most "
         "have the disease. And in a check-up population where one patient in five is sick, only 44 percent of "
         "flagged patients would really have it. So a positive result means: worth a closer look, not a diagnosis.")

# 13 --------------------------------------------------------------- the tool
s = new("TITLE_ONLY"); S.append(s)
title(s, "The deliverable: a working tool")
cmd = ("$ python cardio.py predict --age 62 --sex male --height 172 --weight 98 "
       "--systolic 165 --diastolic 100 --cholesterol 3")
r = EXAMPLE_RESULT
out_lines = ["predicted risk of cardiovascular disease: %.0f%%" % (100 * r["risk"]),
             "verdict: %s" % r["verdict"],
             "screening (cut-off %.2f): %s" % (r["screening_cutoff"], r["screening"]),
             "clinical points score: %d points, risk %.0f%%" % (r["points"], 100 * r["points_risk"])]
out_lines += ["   %+d  %s" % (v, k) for k, v in r["points_breakdown"].items()]
term = box(s, 0.79, 1.3, 8.43, 2.5, fill=NAVY, radius=0.06)
text(s, 1.0, 1.45, 8.0, 2.6, [[(cmd, {"color": BLUE, "bold": True})]] + [[(l, {})] for l in out_lines],
     size=10.5, font=MONO, color=WHITE, space=3)
term2 = box(s, 0.79, 4.0, 8.43, 0.75, fill=SOFT, radius=0.1)
text(s, 1.0, 4.06, 8.0, 0.65, [[("$ python cardio.py predict ... --systolic 1400 ...", {"color": NAVY, "bold": True})],
     [("rejected: implausible examination values: ap_hi out of range", {"color": DKRED})]], size=10.5, font=MONO,
     space=2)
notes(s, "The deliverable is a working tool. The Python file trains and saves the model, then checks a new patient "
         "from the command line. It rejects impossible readings, with the same rules as the cleaning, and returns "
         "the risk, the verdict at two cut-offs, one for accuracy and one for screening, and the clinical points "
         "with their breakdown. Here a 62-year-old man with stage 2 hypertension and high cholesterol gets "
         "%.0f percent." % (100 * r["risk"]))

# 14 --------------------------------------------------------------- conclusions
s = new("TITLE_ONLY"); S.append(s)
title(s, "Conclusions")
concl = [("bars", "Q1 · A real but small gain", "+2.3 points over the guideline rule; the ceiling of about 74% is "
          "set by what the examination records"),
         ("database", "Q2 · Handling beats algorithms", "Cleaning rescues linear models; leakage can manufacture "
          "almost any accuracy"),
         ("stethoscope", "Q3 · Trustworthy, within limits", "Calibrated and equal for both sexes; weaker over 60; "
          "associations, not causes")]
for i, (ic, h, b) in enumerate(concl):
    x = 0.79 + i * 2.88
    box(s, x, 1.3, 2.6, 2.75, fill=SOFT)
    circle_icon(s, ic, x + 0.2, 1.5, d=0.58, name="!!q%d" % i)
    text(s, x + 0.2, 2.2, 2.25, 0.6, h, size=13.5, font=HEAD, bold=True)
    text(s, x + 0.2, 2.82, 2.25, 1.2, b, size=11.5)
text(s, 0.79, 4.25, 8.43, 0.8, [[("Limitations: ", {"bold": True, "color": DKRED}),
     ("detects existing disease, cannot forecast · undocumented source and diagnosis · no treatment data · "
      "a 50/50 sample · one dataset, no external validation", {})]], size=11.5)
notes(s, "To conclude. Machine learning adds something real but small: two points over the guideline rule, and the "
         "ceiling of about 74 percent is set by the data. Data handling matters more than the algorithm: "
         "cleaning rescues linear models, and leakage can manufacture almost any accuracy. And the predictions "
         "are trustworthy within limits. The main limitations: this detects existing disease and cannot "
         "forecast, the data's source and diagnosis definition are unknown, and treatment is not recorded. The "
         "next step would be data from a known clinic, followed over time.")

# 15 --------------------------------------------------------------- thanks
s = new("BLANK_1_1_1_1_1_1_2"); S.append(s)
ph(s, 0, "Thank you")
ph(s, 1, ["Questions?", "hadigghazi@gmail.com"])
for el in THANKS_DECOR:
    el = copy.deepcopy(el)
    nid = max(int(e.get("id")) for e in s.shapes._spTree.iter() if e.tag.endswith("}cNvPr")) + 1
    for e in el.iter():
        if e.tag.endswith("}cNvPr"):
            e.set("id", str(nid))
            nid += 1
    s.shapes._spTree.insert(2, el)
notes(s, "Thank you. I am happy to take questions.")

# 16 --------------------------------------------------------------- backup: test results
TABLE_STYLE = dict(head=NAVY, odd=SOFT, even=WHITE)


def table(slide, rows, widths, y=1.3, size=10.5, row_h=0.34):
    t = slide.shapes.add_table(len(rows), len(widths), Inches(0.79), Inches(y), Inches(sum(widths)),
                               Inches(row_h * len(rows))).table
    for c, wdt in enumerate(widths):
        t.columns[c].width = Inches(wdt)
    for r_, row in enumerate(rows):
        t.rows[r_].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = t.cell(r_, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(
                TABLE_STYLE["head"] if r_ == 0 else (TABLE_STYLE["odd"] if r_ % 2 else TABLE_STYLE["even"]))
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.text = ""
            fill_paras(cell.text_frame, val, size, WHITE if r_ == 0 else NAVY, HEAD if r_ == 0 else BODY, r_ == 0,
                       PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT)


s = new("TITLE_ONLY"); S.append(s)
title(s, "Backup: every candidate on the test set")
table(s, [("Model", "Test AUC", "95% CI", "CV AUC", "Accuracy", "Sensitivity", "Specificity"),
          ("Gradient boosting", "0.804", "0.797-0.811", "0.801", "73.6%", "68.9%", "78.2%"),
          ("Random forest", "0.803", "0.796-0.810", "0.799", "73.4%", "68.5%", "78.1%"),
          ("Neural network (MLP)", "0.803", "0.795-0.810", "0.799", "73.4%", "71.0%", "75.7%"),
          ("Points score (all factors)", "0.795", "0.788-0.803", "-", "72.5%", "64.3%", "80.6%"),
          ("Logistic regression", "0.793", "0.785-0.800", "0.790", "72.6%", "66.8%", "78.2%"),
          ("Framingham form, refitted", "0.780", "0.773-0.788", "0.780", "71.4%", "66.3%", "76.3%"),
          ("Systolic pressure alone", "0.754", "0.746-0.761", "0.751", "71.3%", "61.8%", "80.6%"),
          ("Rule: systolic ≥ 130 mmHg", "0.712", "0.704-0.719", "0.712", "71.3%", "61.8%", "80.6%"),
          ("Framingham, as published", "0.693", "0.684-0.701", "0.696", "64.6%", "74.0%", "55.5%")],
      [2.6, 0.9, 1.15, 0.85, 0.95, 1.0, 0.98])
notes(s, "Backup. Every candidate scored once on the test patients, with bootstrap intervals for the AUC and the "
         "cross-validated AUC for comparison. Gradient boosting's lead over random forest and the neural network "
         "is under 0.002: statistically detectable, clinically meaningless.")

# 17 --------------------------------------------------------------- backup: cleaning strategies
s = new("TITLE_ONLY"); S.append(s)
title(s, "Backup: cleaning strategies")
table(s, [("Trained on", "Gradient boosting AUC", "Accuracy", "Logistic regression AUC", "Accuracy"),
          ("Raw data (69,976 records)", "0.804", "73.6%", "0.776", "71.8%"),
          ("Implausible records dropped (68,581)", "0.804", "73.6%", "0.793", "72.6%"),
          ("Entry errors repaired (69,838)", "0.805", "73.6%", "0.793", "72.6%")],
      [3.35, 1.55, 0.95, 1.65, 0.93], row_h=0.42, size=11)
text(s, 0.79, 3.25, 8.43, 1.5, ["All six models are scored on the same test patients, those whose records pass "
     "every check.", "Trees split on the order of values, so absurd readings simply land in the top branch. A "
     "straight-line model is pulled by them: its blood-pressure coefficient shrinks towards zero."], size=12, space=8)
notes(s, "Backup. Gradient boosting is unaffected by the cleaning strategy; logistic regression loses 0.017 AUC on "
         "the raw data. Repairing and dropping give the same result.")

# 18 --------------------------------------------------------------- backup: subgroups
s = new("TITLE_ONLY"); S.append(s)
title(s, "Backup: who the model serves well")
table(s, [("Group", "Patients", "Share with CVD", "Mean predicted", "AUC", "Sensitivity", "Specificity"),
          ("Women", "8,858", "49.2%", "48.7%", "0.803", "68.0%", "78.6%"),
          ("Men", "4,864", "49.9%", "50.7%", "0.808", "70.4%", "77.6%"),
          ("Age under 50", "4,361", "36.8%", "36.3%", "0.835", "59.0%", "91.4%"),
          ("Age 50-59", "6,900", "50.8%", "51.6%", "0.768", "64.2%", "77.7%"),
          ("Age 60 and over", "2,461", "67.9%", "66.5%", "0.718", "88.1%", "34.7%"),
          ("Everyone", "13,722", "49.4%", "49.4%", "0.804", "68.9%", "78.2%")],
      [2.0, 1.0, 1.3, 1.3, 0.85, 1.0, 0.98], row_h=0.38, size=11)
notes(s, "Backup. Calibrated in every group: the mean predicted risk is within 1.5 points of the observed share. "
         "Discrimination falls with age, and at the 0.5 cut-off older patients are mostly flagged.")

# ------------------------------------------------------------------ tidy, morph, save
for s in S:
    drop_empty_placeholders(s)

NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
MORPH = ('<mc:AlternateContent xmlns:mc="{mc}" xmlns:p="{p}">'
         '<mc:Choice xmlns:p159="http://schemas.microsoft.com/office/powerpoint/2015/09/main" Requires="p159">'
         '<p:transition xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" spd="slow" p14:dur="{dur}">'
         '<p159:morph option="byObject"/></p:transition></mc:Choice>'
         '<mc:Fallback><p:transition xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" spd="slow" p14:dur="{dur}">'
         '<p:fade/></p:transition></mc:Fallback></mc:AlternateContent>')
SLOW = {10: 1800}                         # 0-based: the bars growing into the leakage bars
for n, s in enumerate(S):
    if n == 0:
        continue
    sld = s._element
    for old in sld.findall("{%s}transition" % P) + sld.findall("{%s}AlternateContent" % NS_MC):
        sld.remove(old)
    el = etree.fromstring(MORPH.format(mc=NS_MC, p=P, dur=SLOW.get(n, 900)))
    timing = sld.find("{%s}timing" % P)
    (timing.addprevious(el) if timing is not None else sld.append(el))

prs.save(OUT)
main = SCRIPT[:15]
words = sum(len(re.findall(r"\S+", t)) for t in main)
print("saved %s - %d slides (%d main + %d backup)" % (OUT.name, len(S), 15, len(S) - 15))
print("spoken script: %d words, about %.1f minutes at 140 words a minute" % (words, words / 140))
lines = ["# Spoken script", "", "Generated by build.py from the speaker notes (%d words, about %.1f minutes "
         "at 140 words a minute). Edit the notes in build.py, not this file." % (words, words / 140), ""]
for i, (sl, t) in enumerate(zip(S, SCRIPT), 1):
    n = len(re.findall(r"\S+", t))
    head = sl.shapes.title.text_frame.text if sl.shapes.title is not None else "Slide %d" % i
    lines += ["## %d. %s%s" % (i, head.replace("\n", " "), "" if i <= 15 else " (backup)"),
              "", "*%d words, about %.0f seconds*" % (n, n / 140 * 60) if i <= 15 else "*backup*", "", t, ""]
    if i <= 15:
        print("  slide %2d: %3d words  %4.0f s" % (i, n, n / 140 * 60))
(HERE / "script.md").write_text("\n".join(lines), encoding="utf-8")
print("written: script.md")
