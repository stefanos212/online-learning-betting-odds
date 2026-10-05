/* Defence deck generator.
 *
 * Colour code, used consistently on every slide and chart:
 *   accent1 blue  = our algorithm        accent2 red = the market / bookmakers
 *   accent3 green = a positive result    accent4 amber = a caution or rejection
 *
 * Build:  node build_deck.js      (from inside docs/)
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require(path.join(
  process.env.USERPROFILE, ".claude", "skills", "synced",
  "85b13cb6-e143-48b5-a86f-fa8612fb7466_11c9cbb5-1789-498a-b989-2ac2851b294d",
  "pptx", "scripts", "apply_theme.js"));

const D = JSON.parse(fs.readFileSync("../deck_data.json", "utf8"));

const THEME = {
  name: "OCO Defence",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "111827", lt1: "FFFFFF", dk2: "16213E", lt2: "F2F5F9",
    accent1: "0F4C81", accent2: "C1121F", accent3: "2F7D5C",
    accent4: "D98324", accent5: "5C6B8A", accent6: "94A3B8",
    hlink: "0F4C81", folHlink: "5C6B8A",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.3 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Stefanos Mousadis";
pres.title = "Online Convex Optimization based tuning of Mixture of Experts";
const C = pres.SchemeColor;

const W = 13.3, H = 7.5, M = 0.7;        // slide size and margin

/* ----------------------------------------------------------- layouts ---- */
pres.defineSlideMaster({
  title: "TITLE", background: { color: C.text2 },   // dk2: scheme "background2" is the LIGHT tint
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 1.0, y: 2.3, w: 11.3, h: 1.9,
        fontSize: 40, bold: true, color: C.background1, align: "center", valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 1.0, y: 4.35, w: 11.3, h: 1.6,
        fontSize: 17, color: "CBD5E1", align: "center", valign: "top" }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "SECTION", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "num", type: "body", x: 1.0, y: 2.5, w: 2.0, h: 1.2,
        fontSize: 68, bold: true, color: "6FA3D6", align: "left", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 3.1, y: 2.5, w: 9.2, h: 1.2,
        fontSize: 36, bold: true, color: C.background1, align: "left", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 3.1, y: 3.8, w: 9.2, h: 1.2,
        fontSize: 16, color: "CBD5E1", align: "left", valign: "top" }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.85,
        fontSize: 30, bold: true, color: C.text2, align: "left", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 1.5, w: W - 2 * M, h: 5.2,
        fontSize: 16, color: C.text1, align: "left", valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.6, y: 6.95, color: C.accent6, fontSize: 11 },
});

pres.defineSlideMaster({
  title: "CANVAS", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.85,
        fontSize: 30, bold: true, color: C.text2, align: "left", valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.6, y: 6.95, color: C.accent6, fontSize: 11 },
});

pres.defineSlideMaster({
  title: "STATEMENT", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 1.0, y: 1.9, w: 11.3, h: 2.4,
        fontSize: 34, bold: true, color: C.background1, align: "left", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 1.0, y: 4.5, w: 11.3, h: 1.8,
        fontSize: 17, color: "CBD5E1", align: "left", valign: "top" }, text: "" } },
  ],
});

/* ------------------------------------------------------------ helpers --- */
const chartBase = {
  showLegend: false, showTitle: false,
  catAxisLabelColor: "5C6B8A", valAxisLabelColor: "5C6B8A",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "E2E8F0", size: 1 },
  catGridLine: { style: "none" },
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11,
};

function card(slide, x, y, w, h, fill, name) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: "E2E8F0", width: 1 }, objectName: name,
  });
}

function numCircle(slide, x, y, n, color) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y, w: 0.62, h: 0.62, fill: { color },
    objectName: `circle-${n}`,
  });
  slide.addText(String(n), {
    x, y, w: 0.62, h: 0.62, align: "center", valign: "middle",
    fontSize: 20, bold: true, color: "FFFFFF", margin: 0, isTextBox: true,
  });
}

function stat(slide, x, y, w, value, label, color) {
  slide.addText(value, { x, y, w, h: 1.0, fontSize: 48, bold: true, color,
    align: "left", valign: "bottom", margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 1.0, w, h: 0.9, fontSize: 13, color: C.accent5,
    align: "left", valign: "top", margin: 0, isTextBox: true });
}

function bullets(slide, items, opts) {
  slide.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1 },
  })), Object.assign({ fontSize: 16, color: C.text1, paraSpaceAfter: 10,
    isTextBox: true }, opts));
}

const S = [];   // collect (slide, notes) so notes are added uniformly

/* =========================================================== SLIDES ===== */

pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Opening" });
s.addText("Online Convex Optimization based tuning\nof Mixture of Experts\nfor Sports Outcome Prediction",
  { placeholder: "title" });
s.addText([{ text: "Stefanos Mousadis", options: { bold: true, fontSize: 19, color: "FFFFFF", breakLine: true } },
           { text: "Supervisor: Prof. Thrasyvoulos Spyropoulos", options: { breakLine: true } },
           { text: "School of Electrical and Computer Engineering · Technical University of Crete", options: {} }],
  { placeholder: "body" });
s.addNotes("Greeting; name the problem in one sentence before the first slide change.");

/* --- the question ------------------------------------------------------- */
s = pres.addSlide({ masterName: "STATEMENT", sectionTitle: "Opening" });
s.addText([{ text: "Can a mixture of bookmakers beat\nthe bookmakers themselves?", options: {} }],
  { placeholder: "title" });
s.addText("No model of football. No team form, no injuries, no home advantage. Only the prices the market publishes, combined online, match by match.",
  { placeholder: "body" });
s.addNotes("The whole thesis in one question. Stress what is NOT used.");

/* --- why the baseline is hard ------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Opening" });
s.addText("The baseline is unusually strong", { placeholder: "title" });
stat(s, M, 1.7, 3.6, "26", "bookmakers, each already a\nprofessional forecaster", C.accent1);
stat(s, 4.7, 1.7, 3.6, "76,584", "matches, 22 leagues,\n10 seasons", C.accent1);
stat(s, 8.8, 1.7, 3.8, "4.4 → 6.0%", "the margin they charge,\nand it rose over the decade", C.accent2);
card(s, M, 4.3, W - 2 * M, 1.9, "F2F5F9", "note");
s.addText([{ text: "Beating a bookmaker is not beating a naive baseline. ", options: { bold: true } },
           { text: "Each expert is a firm with analysts, a risk desk and a live order flow correcting its prices. The published odds already aggregate everything the market knows." }],
  { x: M + 0.35, y: 4.55, w: W - 2 * M - 0.7, h: 1.4, fontSize: 15, color: C.text1,
    valign: "middle", isTextBox: true });
s.addNotes("Set expectations: a 0.001 improvement in log-loss against this baseline is large.");

/* --- why non-trivial ---------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Opening" });
s.addText("Three obstacles that shape every design choice", { placeholder: "title" });
const obstacles = [
  ["Partial coverage", "No bookmaker quotes every match. Coverage runs from 6.9% to 99.8%, so the standard expert-advice algorithms do not apply as written."],
  ["Odds are not probabilities", "They carry a margin, so they sum to more than one. Removing it is a modelling decision, not arithmetic."],
  ["The best log-loss is not the objective", "A better forecast only pays if the gain exceeds the margin. Accuracy and profit are different questions."],
];
obstacles.forEach((o, i) => {
  const y = 1.65 + i * 1.65;
  numCircle(s, M, y + 0.1, i + 1, C.accent1);
  s.addText(o[0], { x: M + 0.95, y, w: 11.0, h: 0.45, fontSize: 19, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  s.addText(o[1], { x: M + 0.95, y: y + 0.48, w: 11.0, h: 0.95, fontSize: 15,
    color: C.accent5, margin: 0, isTextBox: true });
});
s.addNotes("These three recur all the way through: they explain Sleeping Experts, de-vigging and the Kelly stage.");

/* --- contributions ------------------------------------------------------ */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Opening" });
s.addText("What this thesis contributes", { placeholder: "title" });
const contribs = [
  ["Three OCO stages, end to end", "Weighting, calibration and stake size, each a convex online problem, each causal.", C.accent1],
  ["Every comparison tested", "Moving block bootstrap on every claim, because consecutive matches are not independent.", C.accent1],
  ["A deployment test", "A real date cut: what a model fitted through 2024 does in 2025-26.", C.accent1],
  ["Negative results that bound the method", "Nine ideas tried and rejected, including one that overturns how the main result should be read.", C.accent4],
];
contribs.forEach((c0, i) => {
  const x = M + (i % 2) * 6.1, y = 1.7 + Math.floor(i / 2) * 2.35;
  card(s, x, y, 5.7, 2.0, "F2F5F9", `contrib-${i}`);
  s.addShape(pres.ShapeType.ellipse, { x: x + 0.35, y: y + 0.35, w: 0.5, h: 0.5,
    fill: { color: c0[2] }, objectName: `dot-${i}` });
  s.addText(c0[0], { x: x + 1.05, y: y + 0.3, w: 4.3, h: 0.6, fontSize: 17, bold: true,
    color: C.text2, margin: 0, valign: "middle", isTextBox: true });
  s.addText(c0[1], { x: x + 0.35, y: y + 1.0, w: 5.0, h: 0.85, fontSize: 14,
    color: C.accent5, margin: 0, isTextBox: true });
});
s.addNotes("Flag the fourth one now; it is the part examiners usually find most interesting.");

/* =================================================== 1. THE FRAMEWORK === */
pres.addSection({ title: "Framework" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Framework" });
s.addText("01", { placeholder: "num" });
s.addText("The framework", { placeholder: "title" });
s.addText("Online learning, regret, and the reduction that makes partial coverage tractable", { placeholder: "body" });

s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Framework" });
s.addText("Online learning: no training phase", { placeholder: "title" });
card(s, M, 1.6, 5.7, 2.4, "F2F5F9", "loop");
s.addText("The protocol", { x: M + 0.35, y: 1.8, w: 5.0, h: 0.4, fontSize: 16, bold: true,
  color: C.text2, margin: 0, isTextBox: true });
bullets(s, ["predict from rounds 1..t−1 only", "the outcome is revealed", "a loss is incurred", "the state is updated"],
  { x: M + 0.35, y: 2.25, w: 5.0, h: 1.6, fontSize: 14, color: C.text1 });
card(s, 6.9, 1.6, 5.7, 2.4, "F2F5F9", "regret");
s.addText("Regret", { x: 7.25, y: 1.8, w: 5.0, h: 0.4, fontSize: 16, bold: true,
  color: C.text2, margin: 0, isTextBox: true });
s.addText("How much worse we did than the best FIXED weight vector, chosen with hindsight over the whole history.",
  { x: 7.25, y: 2.25, w: 5.0, h: 1.5, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addText([{ text: "Every prediction is out of sample by construction. ", options: { bold: true } },
           { text: "There is no train/test split to argue about, and the bounds hold for ANY sequence, including an adversarial one. That is the methodological gain over a batch model." }],
  { x: M, y: 4.4, w: W - 2 * M, h: 1.2, fontSize: 16, color: C.text1, isTextBox: true });
s.addText("Keep the word \"fixed\" in mind. It returns at the end, and it is where the strongest theory fails.",
  { x: M, y: 5.8, w: W - 2 * M, h: 0.6, fontSize: 15, italic: true, color: C.accent2, isTextBox: true });
s.addNotes("Plant the static-comparator idea here. It pays off in the negative results and in the closing section.");

/* --- sleeping experts, with a small diagram ----------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Framework" });
s.addText("Sleeping Experts: the reduction that makes it work", { placeholder: "title" });
s.addText("Not every bookmaker quotes every match, so at round t only the awake subset A(t) predicts.",
  { x: M, y: 1.45, w: W - 2 * M, h: 0.4, fontSize: 16, color: C.text1, isTextBox: true });
const cols = 12, rows = 5, cw = 0.42, ch = 0.42, gx = M, gy = 2.1;
const awakeGrid = [
  [1,1,1,1,1,1,1,1,1,1,1,1],
  [0,0,1,1,1,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,0,0,0,0,0,0],
  [1,1,1,1,1,1,1,1,1,1,1,1],
  [0,0,0,0,1,1,1,1,0,0,1,1],
];
for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
  pres.ShapeType && s.addShape(pres.ShapeType.rect, {
    x: gx + c * (cw + 0.07), y: gy + r * (ch + 0.09), w: cw, h: ch,
    fill: { color: awakeGrid[r][c] ? "0F4C81" : "E2E8F0" },
    line: { color: "FFFFFF", width: 1 }, objectName: `cell-${r}-${c}`,
  });
}
s.addText("experts", { x: gx - 0.02, y: gy + rows * (ch + 0.09) + 0.05, w: 2.0, h: 0.3,
  fontSize: 12, color: C.accent5, margin: 0, isTextBox: true });
s.addText("rounds  →", { x: gx + 4.2, y: gy + rows * (ch + 0.09) + 0.05, w: 2.0, h: 0.3,
  fontSize: 12, color: C.accent5, margin: 0, isTextBox: true });
card(s, 6.9, 2.0, 5.7, 3.1, "F2F5F9", "rules");
bullets(s, [
  "each expert keeps its own persistent state",
  "the prediction is renormalised over A(t) alone",
  "a sleeping expert is FROZEN, not decayed",
  "regret is measured over each expert's own awake rounds",
], { x: 7.25, y: 2.25, w: 5.0, h: 2.6, fontSize: 15 });
s.addText("A bookmaker covering 7% of matches therefore does not dilute the guarantee of the rest.",
  { x: M, y: 5.5, w: W - 2 * M, h: 0.5, fontSize: 15, italic: true, color: C.accent5, isTextBox: true });
s.addNotes("The blue cells are quotes. Point at a gap: that expert's weight does not move while it sleeps.");

/* --- the three algorithms ----------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Framework" });
s.addText("Three algorithms, one reduction", { placeholder: "title" });
const algos = [
  ["Hedge", "Multiplicative weights.\nη sits in an exponent over\none round's loss.", "O(√(T log N))"],
  ["OGD", "Gradient step, then project\nback onto the simplex.\nη multiplies one gradient.", "O(√T)"],
  ["FTRL", "Re-derives weights from the\nWHOLE accumulated loss.\nη multiplies all of it.", "O(√T)"],
];
algos.forEach((a, i) => {
  const x = M + i * 4.05;
  card(s, x, 1.7, 3.75, 3.4, "F2F5F9", `algo-${i}`);
  s.addText(a[0], { x: x + 0.3, y: 1.95, w: 3.2, h: 0.5, fontSize: 22, bold: true,
    color: C.accent1, margin: 0, isTextBox: true });
  s.addText(a[1], { x: x + 0.3, y: 2.55, w: 3.2, h: 1.6, fontSize: 14, color: C.text1,
    margin: 0, isTextBox: true });
  s.addText(a[2], { x: x + 0.3, y: 4.3, w: 3.2, h: 0.5, fontSize: 15, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
});
s.addText("The difference in where η sits is not cosmetic: it is why the three need step sizes tuned in opposite directions, and why only one of them prefers a step that never decays.",
  { x: M, y: 5.4, w: W - 2 * M, h: 0.9, fontSize: 15, color: C.text1, isTextBox: true });
s.addNotes("Do not derive the bounds. The committee knows them; the point is the structural difference.");

/* --- de-vig -------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Framework" });
s.addText("From odds to probabilities", { placeholder: "title" });
s.addText("Published odds imply probabilities that sum to more than one. The excess is the bookmaker's margin.",
  { x: M, y: 1.45, w: W - 2 * M, h: 0.4, fontSize: 16, color: C.text1, isTextBox: true });
const devig = [
  ["Proportional", "Divide by the sum. Treats the margin as a flat tax on every outcome.", C.accent1],
  ["Shin", "Models the margin as protection against informed bettors. The parameter z IS the estimated share of informed money.", C.accent2],
];
devig.forEach((v, i) => {
  const x = M + i * 6.1;
  card(s, x, 2.1, 5.7, 2.2, "F2F5F9", `devig-${i}`);
  s.addText(v[0], { x: x + 0.35, y: 2.35, w: 5.0, h: 0.45, fontSize: 19, bold: true,
    color: v[2], margin: 0, isTextBox: true });
  s.addText(v[1], { x: x + 0.35, y: 2.85, w: 5.0, h: 1.3, fontSize: 14, color: C.text1,
    margin: 0, isTextBox: true });
});
s.addText("Both are carried through the whole pipeline, so no result rests on one choice. Shin adds nothing measurable once calibration is in place (the two correct the same bias).",
  { x: M, y: 4.6, w: W - 2 * M, h: 0.9, fontSize: 15, color: C.text1, isTextBox: true });
s.addNotes("Note for later: z being a social quantity is the hinge of the closing section.");

/* --- microstructure ------------------------------------------------------ */
s = pres.addSlide({ masterName: "STATEMENT", sectionTitle: "Framework" });
s.addText("Theory that earns its place makes a prediction you can lose", { placeholder: "title" });
s.addText("Microstructure (Kyle; Glosten–Milgrom; Shin) says the margin is protection against informed traders, and that price movement is how private information becomes public. It follows that closing prices should beat opening ones MORE where the line actually moved. That is testable, and §7 tests it.",
  { placeholder: "body" });
s.addNotes("This is the methodological spine: theory is kept where it predicts a measurement.");

/* ================================================= 2. THE THREE STAGES == */
pres.addSection({ title: "Stages" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Stages" });
s.addText("02", { placeholder: "num" });
s.addText("Three OCO stages", { placeholder: "title" });
s.addText("From published odds to a bet, with every step online and causal", { placeholder: "body" });

s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Stages" });
s.addText("The same recipe, three different unknowns", { placeholder: "title" });
const stages = [
  ["Weighting", "the weight vector w", "Hedge / OGD / FTRL over the awake experts", C.accent1],
  ["Calibration", "the temperature s", "Online temperature scaling; the loss is convex in s", C.accent1],
  ["Stake size", "the Kelly multiplier λ", "Online Newton Step; the loss is exp-concave in λ", C.accent3],
];
stages.forEach((st, i) => {
  const y = 1.7 + i * 1.55;
  numCircle(s, M, y + 0.15, i + 1, st[3]);
  s.addText(st[0], { x: M + 0.95, y, w: 2.6, h: 0.5, fontSize: 20, bold: true,
    color: C.text2, margin: 0, valign: "middle", isTextBox: true });
  s.addText(st[1], { x: M + 3.6, y, w: 3.0, h: 0.5, fontSize: 16, italic: true,
    color: st[3], margin: 0, valign: "middle", isTextBox: true });
  s.addText(st[2], { x: M + 6.8, y, w: 5.1, h: 0.9, fontSize: 14, color: C.accent5,
    margin: 0, valign: "middle", isTextBox: true });
});
s.addText("Convex loss, causal update, projection onto the feasible set. Only the unknown changes, and with it, which algorithm is the right one.",
  { x: M, y: 6.1, w: W - 2 * M, h: 0.7, fontSize: 15, color: C.text1, isTextBox: true });
s.addNotes("Emphasise the uniformity: this is what makes the thesis one piece of work rather than three.");

/* --- stage 3, the comparator point --------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Stages" });
s.addText("Stage 3: what you learn decides whether it works", { placeholder: "title" });
const kelly = [
  ["Learn the stake f directly", "Lost 9 of 20 paired comparisons, with zero wins.",
   "The comparator \"best constant f\" does not contain the closed-form Kelly rule at all: that rule produces a different f every round. It also throws away the probability the first two stages exist to produce.", C.accent2],
  ["Learn the multiplier λ in f = λ·f*", "Ties on closing targets, wins on opening ones.",
   "The round-specific information stays inside f*, and \"best constant λ\" is exactly the quantity the fixed ¼ pins down.", C.accent3],
];
kelly.forEach((k, i) => {
  const y = 1.65 + i * 2.5;
  card(s, M, y, W - 2 * M, 2.25, i ? "EEF6F1" : "FBF0F0", `kelly-${i}`);
  s.addText(k[0], { x: M + 0.35, y: y + 0.2, w: 7.0, h: 0.45, fontSize: 18, bold: true,
    color: k[3], margin: 0, isTextBox: true });
  s.addText(k[1], { x: 8.0, y: y + 0.2, w: 4.2, h: 0.45, fontSize: 15, bold: true,
    color: C.text2, margin: 0, align: "right", isTextBox: true });
  s.addText(k[2], { x: M + 0.35, y: y + 0.75, w: W - 2 * M - 0.7, h: 1.3, fontSize: 14,
    color: C.text1, margin: 0, isTextBox: true });
});
s.addText("Same data, same test, same bets. Only the object of learning changed.",
  { x: M, y: 6.7, w: W - 2 * M, h: 0.5, fontSize: 15, italic: true, color: C.accent5, isTextBox: true });
s.addNotes("This is the clearest single lesson of the thesis: a regret bound is only as useful as its comparator.");

/* ==================================================== 3. DATA/PROTOCOL == */
pres.addSection({ title: "Protocol" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Protocol" });
s.addText("03", { placeholder: "num" });
s.addText("Data and protocol", { placeholder: "title" });
s.addText("What is measured, and the three rules that keep the measurement honest", { placeholder: "body" });

s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Protocol" });
s.addText("Three rules, each one earned", { placeholder: "title" });
const rules = [
  ["Select on a chronological prefix", "Hyperparameters are chosen on the first 75% of rounds and the tail is reported. Selecting on the tail is selecting on the test set, and here the two criteria disagree."],
  ["Score on ONE common round set", "Series with different coverage are fitted on their full support but scored only where all of them are defined. A log-loss over a different set of matches is not a comparable number."],
  ["Match the panel to the target's phase", "Betting into an opening price, the entire closing side leaves the training panel: no closing price exists yet at that moment."],
];
rules.forEach((r, i) => {
  const y = 1.65 + i * 1.72;
  numCircle(s, M, y + 0.1, i + 1, C.accent1);
  s.addText(r[0], { x: M + 0.95, y, w: 11.0, h: 0.45, fontSize: 18, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  s.addText(r[1], { x: M + 0.95, y: y + 0.48, w: 11.0, h: 1.05, fontSize: 14.5,
    color: C.accent5, margin: 0, isTextBox: true });
});
s.addNotes("Each rule exists because breaking it produced a wrong conclusion at some point.");

/* ========================================================== 4. RESULTS == */
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Results" });
s.addText("04", { placeholder: "num" });
s.addText("Results", { placeholder: "title" });
s.addText("Where the mixture stands, and what the measurements refuse to support", { placeholder: "body" });

/* --- step tuning chart --------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("The theoretical step is wrong, in both directions", { placeholder: "title" });
s.addChart(pres.ChartType.line, [
  { name: "OGD", labels: D.ogd_mult.x.map(String), values: D.ogd_mult.val },
], Object.assign({}, chartBase, {
  x: M, y: 1.6, w: 7.3, h: 4.2,
  chartColors: ["0F4C81"], lineSize: 3, lineSmooth: false,
  valAxisMinVal: 0.9975, valAxisMaxVal: 0.9995,
  catAxisTitle: "step multiplier (1 = textbook schedule)", showCatAxisTitle: true,
  catAxisTitleColor: "5C6B8A", catAxisTitleFontSize: 12, catAxisTitleFontFace: "+mn-lt",
}));
card(s, 8.3, 1.6, 4.3, 4.2, "F2F5F9", "step-note");
s.addText("OGD wants ×100", { x: 8.65, y: 1.85, w: 3.6, h: 0.45, fontSize: 18, bold: true,
  color: C.accent1, margin: 0, isTextBox: true });
s.addText("A genuine interior optimum: a plateau from ×75 to ×110, clearly worse at ×200 and beyond.",
  { x: 8.65, y: 2.35, w: 3.6, h: 1.0, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addText("Hedge and FTRL want LESS", { x: 8.65, y: 3.5, w: 3.6, h: 0.45, fontSize: 16, bold: true,
  color: C.accent2, margin: 0, isTextBox: true });
s.addText("×0.25 and ×0.05. The worst-case bound M = log(1/ε) ≈ 13.8 assumes a loss the data never produces; the largest observed is 3.76.",
  { x: 8.65, y: 3.98, w: 3.6, h: 1.6, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addNotes("A lesson worth saying aloud: the first sweep stopped at ×1 and called it optimal. ×1 was the grid's edge, not a minimum.");

/* --- constant step ------------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("A step that never decays: right for exactly one algorithm", { placeholder: "title" });
s.addChart(pres.ChartType.bar, [
  { name: "decaying", labels: ["Hedge", "OGD", "FTRL"], values: [0.99883, 0.99765, 0.99889] },
  { name: "constant", labels: ["Hedge", "OGD", "FTRL"], values: [0.99892, 0.99795, 0.99879] },
], Object.assign({}, chartBase, {
  x: M, y: 1.7, w: 7.0, h: 3.9, barDir: "col", barGapWidthPct: 60,
  chartColors: ["0F4C81", "C1121F"], showLegend: true, legendPos: "t",
  legendColor: "5C6B8A", legendFontSize: 12, legendFontFace: "+mn-lt",
  valAxisMinVal: 0.997, valAxisMaxVal: 0.9992,
}));
card(s, 8.0, 1.7, 4.6, 3.9, "F2F5F9", "const-note");
s.addText("Why only FTRL", { x: 8.35, y: 1.95, w: 3.9, h: 0.45, fontSize: 18, bold: true,
  color: C.accent3, margin: 0, isTextBox: true });
s.addText("FTRL's η multiplies the entire accumulated loss, not one round's gradient. It is the one of the three for which \"non-decaying\" means something qualitatively different.",
  { x: 8.35, y: 2.45, w: 3.9, h: 1.5, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addText("−0.000096,  p = 0.001", { x: 8.35, y: 4.0, w: 3.9, h: 0.4, fontSize: 17, bold: true,
  color: C.accent3, margin: 0, isTextBox: true });
s.addText("and the two schemes score IDENTICALLY on the training prefix, so the selection did not favour it to flatter the tail.",
  { x: 8.35, y: 4.45, w: 3.9, h: 1.0, fontSize: 13.5, color: C.accent5, margin: 0, isTextBox: true });
s.addNotes("Lower is better on this chart. Hedge and OGD lose significantly; FTRL wins significantly.");

/* --- ranking ------------------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("Where the mixture lands", { placeholder: "title" });
const rk = D.ranking.slice(0, 9);
s.addChart(pres.ChartType.bar, [
  { name: "calibrated log-loss", labels: rk.map(r => r.name), values: rk.map(r => r.cal) },
], Object.assign({}, chartBase, {
  x: M, y: 1.6, w: 7.6, h: 4.5, barDir: "bar",
  chartColors: rk.map(r => (r.type === "algorithm" ? "0F4C81" : "C1121F")),
  valAxisMinVal: 0.9965, valAxisMaxVal: 1.0012,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00000",
  dataLabelColor: "5C6B8A",
}));
card(s, 8.6, 1.6, 4.0, 4.5, "F2F5F9", "rank-note");
s.addText("OGD is second", { x: 8.95, y: 1.85, w: 3.3, h: 0.4, fontSize: 18, bold: true,
  color: C.accent1, margin: 0, isTextBox: true });
bullets(s, [
  "beats the uniform average by 0.00141 (p < 0.001)",
  "beats Hedge, FTRL and B365C significantly",
  "PSC (Pinnacle) stays ahead by 0.00043",
  "calibration helps all four methods",
], { x: 8.95, y: 2.35, w: 3.3, h: 3.4, fontSize: 13.5 });
/* The bars are full-sample; the quoted tests are paired over the 71,818 rounds every
 * series covers, so the two sets of gaps are not the same number by construction. */
s.addText("lower is better · blue = algorithm · red = bookmaker · bars full-sample, tests paired on common rounds",
  { x: M, y: 6.25, w: 7.6, h: 0.35, fontSize: 12, italic: true, color: C.accent5, isTextBox: true });
s.addNotes("Lower is better. Note the caveat: at a 70% coverage floor only one closing bookmaker survives, so this is mostly a comparison against opening prices.");

/* --- panel / heterogeneity ----------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("Learning pays only where the experts differ", { placeholder: "title" });
s.addChart(pres.ChartType.bar, [
  { name: "opening (13)", labels: ["OGD", "Hedge", "FTRL", "Uniform"], values: [1.00076, 1.00088, 1.00088, 1.00080] },
  { name: "all (26)", labels: ["OGD", "Hedge", "FTRL", "Uniform"], values: [0.99745, 0.99842, 0.99843, 0.99886] },
  { name: "closing (13)", labels: ["OGD", "Hedge", "FTRL", "Uniform"], values: [0.99722, 0.99762, 0.99761, 0.99748] },
], Object.assign({}, chartBase, {
  x: M, y: 1.7, w: 7.4, h: 4.0, barDir: "col", barGapWidthPct: 50,
  chartColors: ["C1121F", "5C6B8A", "0F4C81"], showLegend: true, legendPos: "t",
  legendColor: "5C6B8A", legendFontSize: 12, legendFontFace: "+mn-lt",
  valAxisMinVal: 0.996, valAxisMaxVal: 1.002,
}));
card(s, 8.4, 1.7, 4.2, 4.0, "FBF0F0", "panel-note");
s.addText("The reversal", { x: 8.75, y: 1.95, w: 3.5, h: 0.4, fontSize: 18, bold: true,
  color: C.accent2, margin: 0, isTextBox: true });
s.addText("On the FULL panel all three algorithms beat the plain average significantly.\n\nOn a HOMOGENEOUS panel, Hedge and FTRL LOSE to it, in 8 tests out of 8.",
  { x: 8.75, y: 2.4, w: 3.5, h: 2.0, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addText("13 good closing experts and 13 systematically worse opening ones is a quality difference to learn. Remove it and adaptation is pure cost.",
  { x: 8.75, y: 4.4, w: 3.5, h: 1.2, fontSize: 13.5, italic: true, color: C.accent5,
    margin: 0, isTextBox: true });
s.addNotes("In my view the most generally useful finding: it tells anyone applying this elsewhere what to check first.");

/* --- information arrival -------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("The prediction from theory, tested", { placeholder: "title" });
const arr = D.arrival.filter(r => r.stratifier === "line movement").slice(0, 5);
s.addChart(pres.ChartType.bar, [
  { name: "closing advantage", labels: ["Q1\nquiet", "Q2", "Q3", "Q4", "Q5\nnoisy"],
    values: arr.map(r => Math.abs(Number(r.mean_diff))) },
], Object.assign({}, chartBase, {
  x: M, y: 1.7, w: 7.3, h: 4.1, barDir: "col", barGapWidthPct: 45,
  chartColors: ["C7D2DE", "9FB3C8", "6E8CA8", "3D6B8E", "0F4C81"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000",
  dataLabelColor: "5C6B8A",
}));
card(s, 8.3, 1.7, 4.3, 4.1, "F2F5F9", "arr-note");
s.addText("A 26-fold gradient", { x: 8.65, y: 1.95, w: 3.6, h: 0.4, fontSize: 18, bold: true,
  color: C.accent1, margin: 0, isTextBox: true });
s.addText("Microstructure predicted the closing advantage should concentrate where the line moved. It does: 0.00044 in the quiet fifth, 0.01158 in the noisy one.",
  { x: 8.65, y: 2.4, w: 3.6, h: 1.5, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
s.addText("In Q1 it is not even significant (p = 0.052).", { x: 8.65, y: 3.95, w: 3.6, h: 0.6,
  fontSize: 14, bold: true, color: C.accent2, margin: 0, isTextBox: true });
s.addText("\"Closing prices are better\" is not a general property of final prices. It is a property of the top quintile.",
  { x: 8.65, y: 4.6, w: 3.6, h: 1.1, fontSize: 13.5, italic: true, color: C.accent5,
    margin: 0, isTextBox: true });
s.addNotes("Control for the obvious objection: match difficulty is NOT monotone across the quintiles, so this is not 'harder matches, bigger gaps'.");

/* --- deployment ----------------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("Deployment test: a real date cut", { placeholder: "title" });
s.addText("Fitted through 31/12/2024, measured on 11,574 rounds of 2025-26.",
  { x: M, y: 1.45, w: 8.0, h: 0.4, fontSize: 16, color: C.text1, isTextBox: true });
const dep = [
  ["Every opening price", "beaten significantly", "−0.0019 to −0.0065, all p ≤ 0.019", C.accent3],
  ["The bookmakers' closing prices", "drawn level", "B365C p=0.193 · BWC p=0.361 · BFEC p=0.052", C.accent5],
  ["Betfair Exchange", "beats us significantly", "an exchange, not a bookmaker: a market price with no operator margin", C.accent2],
];
dep.forEach((d0, i) => {
  const y = 2.0 + i * 1.55;
  card(s, M, y, W - 2 * M, 1.35, i === 2 ? "FBF0F0" : "F2F5F9", `dep-${i}`);
  s.addText(d0[0], { x: M + 0.35, y: y + 0.18, w: 4.6, h: 0.45, fontSize: 17, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  s.addText(d0[1], { x: M + 5.1, y: y + 0.18, w: 3.4, h: 0.45, fontSize: 17, bold: true,
    color: d0[3], margin: 0, isTextBox: true });
  s.addText(d0[2], { x: M + 0.35, y: y + 0.72, w: W - 2 * M - 0.7, h: 0.5, fontSize: 13.5,
    color: C.accent5, margin: 0, isTextBox: true });
});
s.addText("The honest statement: we REACH the leading edge of the bookmaker market. The market without an intermediary is still ahead.",
  { x: M, y: 6.6, w: W - 2 * M, h: 0.6, fontSize: 15, bold: true, color: C.text2, isTextBox: true });
s.addNotes("Do not oversell. The Betfair line is the one an examiner will press on, so state it first.");

/* --- value betting --------------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("Better prediction does not mean profit", { placeholder: "title" });
/* Plotted as log10(final / starting bankroll): a log AXIS is dropped by some
 * renderers, and on a linear one PSC's wipe-out is invisible. Transformed, zero
 * is break-even and the sign carries the result. */
const vbMult = [38.52, 7.89, 1.81, 1.38, 1.09, 0.75, 0.58, 0.26, 0.0002];
s.addChart(pres.ChartType.bar, [
  { name: "log10 growth", labels: ["WHC", "IW", "BWC", "WH", "BW", "VC_BVC", "B365C", "PS", "PSC"],
    values: vbMult.map(v => Math.round(Math.log10(v) * 1000) / 1000) },
], Object.assign({}, chartBase, {
  x: M, y: 1.7, w: 7.3, h: 4.1, barDir: "col",
  chartColors: ["2F7D5C", "2F7D5C", "94A3B8", "94A3B8", "94A3B8", "94A3B8", "94A3B8", "94A3B8", "C1121F"],
  valAxisMinVal: -4, valAxisMaxVal: 2,
  valAxisTitle: "log10 (final bankroll / stake)", showValAxisTitle: true,
  valAxisTitleColor: "5C6B8A", valAxisTitleFontSize: 12, valAxisTitleFontFace: "+mn-lt",
}));
card(s, 8.3, 1.7, 4.3, 4.1, "F2F5F9", "vb-note");
s.addText("Two profits, one loss", { x: 8.65, y: 1.95, w: 3.6, h: 0.4, fontSize: 18, bold: true,
  color: C.text2, margin: 0, isTextBox: true });
s.addText("Significant gains against WHC (p<0.001) and IW (p=0.008), both recreational-facing houses. A significant LOSS against PSC, the best forecaster in the panel, i.e. exactly the opponent an inferior model should lose to.",
  { x: 8.65, y: 2.4, w: 3.6, h: 2.0, fontSize: 13.5, color: C.text1, margin: 0, isTextBox: true });
s.addText("The quality gained is smaller than the margin charged. The test works in both directions.",
  { x: 8.65, y: 4.5, w: 3.6, h: 1.1, fontSize: 13.5, italic: true, color: C.accent5,
    margin: 0, isTextBox: true });
s.addText("0 = break even · +1 = a ten-fold bankroll · green = significant profit · red = significant loss",
  { x: M, y: 5.95, w: 7.3, h: 0.35, fontSize: 12, italic: true, color: C.accent5, isTextBox: true });
s.addNotes("Bankrolls across rows are not comparable: the bet counts differ by an order of magnitude.");

/* --- the 41% ---------------------------------------------------------------- */
s = pres.addSlide({ masterName: "STATEMENT", sectionTitle: "Results" });
s.addText("41% of the margin over the uniform average is not learning", { placeholder: "title" });
s.addText("Under the Sleeping Experts reduction the weight vector moves even with a step of ZERO: the projection is re-done onto a simplex of different dimension whenever the awake set changes. A zero-step control separates the two. The number was real; the obvious reading of it was false, and this thesis made that reading before correcting it.",
  { placeholder: "body" });
s.addNotes("The methodological high point. Any comparison against a uniform baseline should now carry this control.");

/* --- negative results -------------------------------------------------------- */
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Results" });
s.addText("Nine ideas tried and rejected", { placeholder: "title" });
const negs = [
  ["Fixed-Share", "every α > 0 loses; the step already does the forgetting"],
  ["Per-expert step clock", "better on train, −0.00073 on the tail: textbook overfitting"],
  ["Markov kernels on the weights", "every kernel selects α = 0 and ties exactly"],
  ["Latent market regimes", "τ = 0 selected, belief sd 0.0000: the chain provably does nothing"],
  ["Bandit, one-point estimator", "alignment with the true gradient is +0.002; the horizon is too short"],
  ["Bayesian / MCMC Kelly", "6 ties, 2 significant losses, zero wins"],
  ["Per-league specialisation", "loses in all six leagues tested"],
  ["AdaGrad", "recovers 20% of the hand-tuned gain at best"],
  ["Joint step + calibration sweep", "no interaction at all, and train selection then picks a worse step"],
];
negs.forEach((n, i) => {                       // 3 x 3, so all nine fit
  const x = M + (i % 3) * 4.0, y = 1.75 + Math.floor(i / 3) * 1.6;
  s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.08, w: 0.26, h: 0.26,
    fill: { color: C.accent4 }, objectName: `neg-${i}` });
  s.addText(n[0], { x: x + 0.42, y, w: 3.3, h: 0.62, fontSize: 15, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  s.addText(n[1], { x: x + 0.42, y: y + 0.66, w: 3.3, h: 0.85, fontSize: 12.5,
    color: C.accent5, margin: 0, isTextBox: true });
});
s.addText("A pattern across them: once the step size is tuned, no further movement of mass over the experts buys anything. The step and the leak control the same quantity.",
  { x: M, y: 6.5, w: W - 2 * M, h: 0.7, fontSize: 15, color: C.text1, isTextBox: true });
s.addNotes("Do not rush this slide. The negative results are a real part of the contribution.");

/* ===================================================== 5. WHAT IT MEANS == */
pres.addSection({ title: "Meaning" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Meaning" });
s.addText("05", { placeholder: "num" });
s.addText("What the numbers mean", { placeholder: "title" });
s.addText("Why the ranking of forecasters turns out to be a ranking of business models", { placeholder: "body" });

s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Meaning" });
s.addText("The odds are not a property of the match", { placeholder: "title" });
const soc = [
  ["Remove Pinnacle and the \"probability\" changes", "What was measured was not knowledge about football but position in the market.", C.accent1],
  ["Pinnacle leads because it does not ban winners", "Low margin, and informed bettors are allowed to correct its price instead of being shut out.", C.accent1],
  ["The best forecaster is not a bookmaker at all", "Betfair Exchange: a market price carrying no operator margin.", C.accent1],
  ["The market got dearer, not better", "Overround 4.42% → 6.00% (p = 0.0007) while no series shows a significant trend in log-loss.", C.accent2],
];
soc.forEach((v, i) => {
  const y = 1.62 + i * 1.3;
  s.addText(v[0], { x: M, y, w: 6.4, h: 0.5, fontSize: 16, bold: true, color: v[2],
    margin: 0, valign: "middle", isTextBox: true });
  s.addText(v[1], { x: 7.3, y, w: 5.3, h: 0.9, fontSize: 13.5, color: C.accent5,
    margin: 0, valign: "middle", isTextBox: true });
});
s.addText("Shin's parameter z IS the estimated share of informed bettors, a social quantity, entering the method as a nuisance parameter to be removed.",
  { x: M, y: 6.3, w: W - 2 * M, h: 0.8, fontSize: 15, italic: true, color: C.text1, isTextBox: true });
s.addNotes("Keep this measured and evidential. Every line is one of our own results.");

s = pres.addSlide({ masterName: "STATEMENT", sectionTitle: "Meaning" });
s.addText("The edge exists where it cannot be used", { placeholder: "title" });
s.addText("The significant profits are against two recreational-facing houses. Against the one house that does NOT exclude winning players, the simulation loses significantly. The sharp house has the right price because it lets informed money correct it; the soft ones have the wrong price because they drive that money away. Account restrictions are not measured here: the data are prices, not customer outcomes.",
  { placeholder: "body" });
s.addNotes("State the limit of the evidence explicitly; it is the honest version and it is stronger.");

/* ======================================================== CONCLUSIONS ==== */
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Close" });
s.addText("Conclusions", { placeholder: "title" });
const concl = [
  ["The framework applies", "An online mixture of published odds beats the uniform average and almost every individual bookmaker, with no distributional assumption.", C.accent3],
  ["It reaches the market's edge", "Level with the sharpest bookmaker, but behind an exchange with no operator margin.", C.accent1],
  ["Part of the gain is geometry", "41% of the margin over uniform, isolated with a zero-step control. Future comparisons should carry one.", C.accent4],
  ["Heterogeneity is a precondition", "Learned weighting pays only when the experts are produced by different processes.", C.accent1],
  ["Better prediction is not profit", "The quality gained falls short of the margin, and the edge sits where access is controlled.", C.accent2],
];
concl.forEach((c0, i) => {
  const y = 1.6 + i * 1.03;
  s.addShape(pres.ShapeType.ellipse, { x: M, y: y + 0.1, w: 0.28, h: 0.28,
    fill: { color: c0[2] }, objectName: `cc-${i}` });
  // valign defaults to middle, so boxes of different height do not share a baseline
  s.addText(c0[0], { x: M + 0.48, y, w: 4.3, h: 0.45, fontSize: 16, bold: true,
    color: C.text2, margin: 0, valign: "top", isTextBox: true });
  s.addText(c0[1], { x: 5.65, y, w: 6.95, h: 0.85, fontSize: 13.5, color: C.accent5,
    margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("Five points, roughly fifteen seconds each. Do not add anything new here.");

s = pres.addSlide({ masterName: "CANVAS", sectionTitle: "Close" });
s.addText("Future work", { placeholder: "title" });
const fw = [
  ["A live odds feed", "Every result here says what the method WOULD have done. The three stages are online and causal, the state is one weight vector plus two scalars, so it runs in real time against a streaming API. That turns \"a statistical edge exists\" into \"an exploitable edge exists\", and records the price trajectory rather than two snapshots."],
  ["A second sport", "Basketball or tennis would show whether the framework generalises. The cost is a data pipeline; the algorithms do not change. A two-outcome sport is of particular interest."],
  ["Bayesian Model Averaging as a baseline", "An empirical point of comparison against the online algorithms, and more light on why the Aggregating Algorithm underperforms here."],
];
fw.forEach((f, i) => {
  const y = 1.7 + i * 1.75;
  numCircle(s, M, y + 0.05, i + 1, C.accent1);
  s.addText(f[0], { x: M + 0.95, y, w: 11.0, h: 0.45, fontSize: 18, bold: true,
    color: C.text2, margin: 0, isTextBox: true });
  s.addText(f[1], { x: M + 0.95, y: y + 0.46, w: 11.0, h: 1.15, fontSize: 13.5,
    color: C.accent5, margin: 0, isTextBox: true });
});
s.addNotes("The live feed is the one that answers the thesis's own main limitation.");

s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Close" });
s.addText("Thank you", { placeholder: "title" });
s.addText("Questions", { placeholder: "body" });
s.addNotes("Stop talking. Let them ask.");

/* ------------------------------------------------------------- write ----- */
(async () => {
  await pres.writeFile({ fileName: "defense_presentation.pptx" });
  await applyTheme("defense_presentation.pptx", THEME);
  console.log("wrote defense_presentation.pptx");
})();
