# AI image prompts for the five chapter figures

**Chapter:** AI in Bioreactor Design, Monitoring, and Scale-Up

## Read this first

**Check Elsevier's policy before you submit any of these images.** Elsevier's policy on generative AI says you may not use generative AI tools to create or alter images in a submitted manuscript. The only exception is when using AI is itself part of the research method. Book chapters follow the same policy. The safe way to use these prompts:

1. Use the AI image as a **visual draft or reference only**.
2. Redraw the final figure yourself in BioRender, Inkscape, Illustrator or PowerPoint.
3. For the submission, use the 600 dpi TIFF, PDF or SVG files in this folder, which are not AI-generated.

AI images are fine without these limits for slides, posters and the graphical abstract on a presentation.

**Tips for good results:**
- **Text labels.** Image models often misspell labels. Generate the image with **no text** first, using the "clean" line at the end of each prompt. Then add the labels yourself in PowerPoint or Inkscape.
- **Aspect ratio.** Use landscape 16:9 or 3:2. Upscale to at least 4000 px wide.
- **Fixing errors.** If something comes out wrong, repeat just that part in a follow-up message. Example: "Make the impeller a Rushton turbine with 6 flat blades".

## Global style block (paste at the start of every prompt)

```
Style: professional scientific illustration for an Elsevier textbook chapter, clean flat-vector with subtle 3D shading, white background, high detail, crisp edges, consistent color palette (deep blue #1F5FA8 for AI/data elements, teal #2A9D8F for biology/cells, warm orange #E76F51 for gradients/alerts, neutral greys #555555 for equipment), thin dark-grey outlines, sans-serif labels (Arial), generous white space, no photographic noise, no watermark, no decorative clutter, publication quality, 600 dpi look.
```

## Figure 1. Map of AI model families

```
[Global style block]

A scientific infographic showing a two-axis map of artificial-intelligence model families used for bioreactors.
Vertical axis (left, arrow pointing up): "Physics knowledge embedded" from low to high.
Horizontal axis (bottom, arrow pointing right): "Data needed" from low to high.
Place eight rounded-rectangle cards, each with a small iconic pictogram:
- top-left: "Mechanistic models" – icon of differential equations and a stirred tank;
- upper-middle, highlighted with a glowing deep-blue border and light-blue fill: "Hybrid (grey-box) models" – icon of a half gear / half neural-network brain;
- top-right: "PINNs & neural operators" – icon of a colorful CFD flow field inside a tank mesh;
- middle-left: "Gaussian process" – icon of a curve with a shaded confidence band;
- lower-middle: "Random forest / boosting" – icon of small decision trees;
- middle-right: "Reinforcement learning" – icon of an agent loop (agent -> action -> environment -> reward);
- bottom-left: "PLS chemometrics" – icon of a Raman spectrum with peaks;
- bottom-right: "Deep networks (LSTM, transformer)" – icon of layered neural nodes.
Under each card, a tiny grey tag naming the lifecycle stage: Design, Monitoring, Control or Scale-up.
A soft diagonal gradient background band from bottom-right to top-left labelled subtly "data-only -> knowledge-plus-data".
Landscape 3:2.
```
*Clean version:* add "Do not render any text; leave empty label areas on the cards and axes."

## Figure 2. Serial hybrid model of a fed-batch bioreactor

```
[Global style block]

A detailed scientific schematic of a serial hybrid (grey-box) model for a fed-batch bioreactor, flowing left to right.
Left: a realistic cut-away glass-and-stainless-steel stirred-tank bioreactor (Rushton impeller, sparger bubbles, pH and dissolved-oxygen probes, a feed pump with tubing), with small floating CHO cells visible in the liquid. Data cables run from the probes to the next block.
Block 1 "Process inputs": icons for thermometer (T), pH meter, O2 bubble (DO), pump (feed rate F), volume gauge (V).
Block 2 "Neural network" (highlighted deep blue): a clean feed-forward neural network with 3 layers of glowing nodes; outputs labelled μ, qS, qP (specific growth, substrate uptake, product formation rates).
Block 3 "Mass balances (known physics)": a light-grey panel showing three ordinary differential equations
dX/dt = μX − (F/V)X ; dS/dt = −qS·X + (F/V)(Sf − S) ; dP/dt = qP·X − (F/V)P
with a small integral/solver icon.
Block 4 "Predicted trajectories": a mini line chart with three smooth curves over time – biomass X rising (teal), substrate S falling (orange), product P rising (blue).
Bottom feedback loop: "Measured data" (lab sample tubes and Raman probe icon) -> "Training loop" (loss curve going down, gradient arrow) -> curved arrow back up to the Neural network, with a padlock icon on the mass-balance block showing "physics fixed".
Landscape 16:9.
```
*Clean version:* "Render equations and labels as empty placeholder boxes; no text."

## Figure 3. AI soft-sensor pipeline (Raman to control)

```
[Global style block]

A process-flow scientific illustration of real-time AI monitoring of a mammalian cell-culture bioreactor.
Far left: a 2,000 L single-use stirred-tank bioreactor in a clean GMP suite, with an in-line Raman probe (fiber-optic cable glowing faint blue), pH, DO, capacitance and off-gas sensors.
Arrow to "Pre-processing": a raw noisy Raman spectrum transforming into a clean baseline-corrected spectrum (before/after mini plots).
Arrow to "Soft-sensor model" (highlighted deep blue): a PLS / neural-network icon outputting three live gauges – glucose, lactate, viable cell density – each with a shaded uncertainty band.
Arrow to "State estimator": a Kalman-filter style diagram fusing fast online signals and sparse offline lab samples into one smooth estimated trajectory.
Down arrow to "Control action": a model-predictive-control panel adjusting a feed pump to hold glucose at a dashed set-point line.
Lower loop: from the soft-sensor model down to "Drift monitor" – a 2D scatter of spectra with a validated ellipse region and one red outlier point outside it with a small warning triangle – then left to "Recalibration" – lab analyzer, new reference samples, model version tags v1 -> v2 – and a curved arrow back up into the soft-sensor model.
Subtle dashboard aesthetic, realistic equipment, clean vector arrows.
Landscape 16:9.
```
*Clean version:* "No text; show gauges and plots without numbers."

## Figure 4. Regime-matched, AI-assisted scale-up

```
[Global style block]

A scientific workflow illustration of AI-assisted, regime-matched bioreactor scale-up in eight numbered steps arranged in a U-shaped path (top row left->right, then bottom row right->left).
1 "Candidate design": engineering sketch of a 50 m³ industrial stirred tank with three impellers, feed point at the top and sparger at the bottom.
2 "CFD library": the tank cross-section rendered as a colorful computational-fluid-dynamics contour map (glucose concentration high near the feed point in orange-red, low near the bottom in blue), with mesh lines visible and an hourglass icon ("days per run").
3 "AI surrogate" (highlighted deep blue): a neural-operator / graph-network icon instantly producing the same contour map, with a stopwatch icon ("seconds").
4 "Cell lifelines": hundreds of thin colored trajectories of virtual cells looping through the tank, one trajectory highlighted, plus an inset line plot of the glucose and oxygen concentration that single cell experiences over time (fluctuating zig-zag).
5 "Regime statistics" (highlighted): a stacked bar or pie chart showing the fraction of time cells spend in regimes such as "glucose-rich / O2-poor", "starved", "balanced".
6 "Scale-down runs": a small laboratory two-compartment system (a stirred tank connected to a plug-flow tube) or a pulsed-feed lab bioreactor reproducing those fluctuations.
7 "Transfer learning": a hybrid model icon receiving data from the lab and from a few plant runs, with a small "fine-tune" arrow.
8 "Verify at scale": the large plant tank with a prediction band and a green check / red cross go/no-go decision.
A return arrow from step 8 up to step 1 labelled "rank designs, next product".
Landscape 16:9, rich but uncluttered.
```
*Clean version:* "No text or numbers; numbered circles only."

## Figure 5. The DSDS conceptual framework

```
[Global style block]

A conceptual framework diagram for a textbook: the "Design–Sense–Decide–Scale (DSDS) loop".
Center: a glowing deep-blue 3D holographic digital twin of a stirred-tank bioreactor floating above a circuit-board platform, labelled "Hybrid digital twin", with tiny equations and neural-network nodes merging inside it.
Around it, four stage cards placed at 12, 3, 6 and 9 o'clock and connected clockwise by bold curved arrows forming a cycle:
- 12 o'clock "1 Design": icons of parallel mini-bioreactors (robotic high-throughput platform), a Bayesian-optimization surface, and a CFD tank.
- 3 o'clock "2 Sense": a Raman probe and a live spectrum with an uncertainty band and a drift-alert icon.
- 6 o'clock "3 Decide": a model-predictive-control dashboard and a reinforcement-learning agent behind a shield icon (safety filter).
- 9 o'clock "4 Scale": a small lab vessel and a large production tank linked by a "regime-matching" arrow and a transfer-learning icon.
Dashed blue lines connect each stage card to the central twin (two-way data exchange).
A curved arrow from "Scale" back to "Design" labelled "plant data".
Beneath the cycle, three wide horizontal foundation layers like building strata:
"Data layer" (database cylinders, FAIR data badge), "Trust layer" (uncertainty band, SHAP bar chart, version-control branches), "Governance layer" (regulatory shield, QbD / ICH Q13 / FDA document icons).
Balanced, symmetric composition, landscape 4:3.
```
*Clean version:* "No text; keep empty label plates on cards and layers."

## Optional: graphical abstract or cover image (no labels needed)

```
[Global style block]

A visually striking graphical abstract: on the left a small 2 L laboratory glass bioreactor, on the right a towering 50 m³ industrial steel fermenter; between them a luminous deep-blue neural-network bridge carrying data streams and tiny glowing cells. Inside the large tank, faint orange-to-blue concentration gradients swirl around the impellers. Above the bridge, a translucent holographic digital twin. Clean white-to-light-grey background, cinematic soft lighting, scientific realism, no text. Landscape 16:9.
```

## Files in this folder (ready for submission, not AI-generated)

| Figure | Raster, 600 dpi, about 16 cm wide | Vector |
| --- | --- | --- |
| 1 | Figure_1.tif / Figure_1.png | Figure_1.pdf / Figure_1.svg |
| 2 | Figure_2.tif / Figure_2.png | Figure_2.pdf / Figure_2.svg |
| 3 | Figure_3.tif / Figure_3.png | Figure_3.pdf / Figure_3.svg |
| 4 | Figure_4.tif / Figure_4.png | Figure_4.pdf / Figure_4.svg |
| 5 | Figure_5.tif / Figure_5.png | Figure_5.pdf / Figure_5.svg |

The titles were removed from inside the figures because each figure's caption in the chapter text does that job. You can edit the SVG files in Inkscape or Illustrator.
